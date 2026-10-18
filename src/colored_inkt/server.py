"""The viewer: a small web server (standard library only) for comparing filters on segments.

    colored-inkt serve [--port 6544] [--data ~/.colored-inkt]

Two tabs: Segments (one segment, any filter against a reference, with live colour controls, labels, pre-filters
and a depth scan) and Methods (what every filter does). Segments can be added from the page: layer images
uploaded from the browser, or a folder, file or Zarr URL the server can read.

The server has no login. It listens on 127.0.0.1 unless told otherwise; only expose it to people you trust, since
anyone who reaches it can add data and make it read paths on this computer.
"""

from __future__ import annotations

import json
import math
import secrets
import shutil
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from colored_inkt import render as rendering
from colored_inkt.store import NOTCH_WIDTHS, SPECKLE_SIZES, Store, check_depth, check_prefilter, lookup, resolve
from colored_inkt.visualizers import _ANCHORS, REGISTRY
from colored_inkt.volume import ome_voxel_um, open_volume, read_label

STATIC = Path(__file__).parent / "static"
UPLOAD_CHUNK = 1 << 20
STORE: Store | None = None


def _clean(value):
    """JSON-safe: NaN → None, tuples → lists."""
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def prefilter_of(query: dict) -> tuple[int, int]:
    """(notch, speckle) from ``?notch=…&speckle=…`` (0 = off)."""
    notch, speckle = int(query.get("notch", 0) or 0), int(query.get("speckle", 0) or 0)
    check_prefilter(notch, speckle)
    return notch, speckle


def depth_of(query: dict) -> dict:
    """{depth, fixed} from ``?depth=…&fixed=1``."""
    depth = int(query.get("depth", 0) or 0)
    check_depth(depth)
    return {"depth": depth, "fixed": str(query.get("fixed", "0")) in ("1", "true", "True")}


def visualizer_list() -> list[dict]:
    return [
        {"name": v.name, "short": rendering.short_name(v.name), "title": v.title, "family": v.family,
         "idea": v.idea, "animated": v.animated, "display": v.display, "variant_of": v.variant_of,
         "prefilter": {"notch": v.prefilter[0], "speckle": v.prefilter[1]}}
        for v in REGISTRY.values()
    ]


def _positive(body: dict, key: str) -> float | None:
    value = body.get(key)
    if value in (None, ""):
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def add_segment(store: Store, body: dict):
    """Register a segment from an upload (``upload`` token + file names) or a ``path`` / URL."""
    title = str(body.get("title") or "").strip() or "my segment"
    level = int(body.get("level", 0) or 0)
    side = 1 if int(body.get("side", 1)) >= 0 else -1
    binning = int(body.get("bin", 1) or 1)
    if binning not in (1, 2, 3, 4):
        raise ValueError("bin must be 1, 2, 3 or 4")
    crop = body.get("crop")
    crop = tuple(int(v) for v in crop) if crop else None
    label = None
    cleanup = None
    known = None
    if body.get("upload"):
        token = str(body["upload"])
        staging = upload_dir(store, token)
        names = [Path(n).name for n in body.get("files", [])]
        files = [staging / n for n in names]
        missing = [n for n, f in zip(names, files) if not f.exists()]
        if not files or missing:
            raise ValueError(f"uploaded files missing: {missing or 'none given'}")
        # One file is a whole stack (multi-page TIFF or .npy); several are one layer each, in the given order.
        volume = open_volume(files[0]) if len(files) == 1 else open_volume(files)
        if body.get("label_file"):
            label = read_label(staging / Path(body["label_file"]).name)
        origin = {"uploaded": names}
        cleanup = staging
    elif body.get("path"):
        path = str(body["path"]).strip()
        volume = open_volume(path, level=level)
        known = ome_voxel_um(path, level)
        if body.get("label_path"):
            label = read_label(str(body["label_path"]).strip(), level)
        origin = {"path": path, "level": level}
    else:
        raise ValueError("give files to upload or a path / URL")
    layer_um = _positive(body, "layer_um") or (known[0] if known else None)
    if layer_um is None:
        raise ValueError("give the layer spacing in µm (the scan's voxel size)")
    pixel_um = _positive(body, "pixel_um") or (known[1] if known and not body.get("layer_um") else layer_um)
    if label is not None and tuple(label.shape) != tuple(volume.shape[1:]):
        raise ValueError(f"the label is {tuple(label.shape)} px but the layers are {tuple(volume.shape[1:])} px")
    return store.add(title=title, source=volume, layer_um=layer_um, pixel_um=pixel_um, side=side, bin=binning,
                     crop=crop, label=label, origin=origin, cleanup=cleanup)


def upload_dir(store: Store, token: str) -> Path:
    if not token.isalnum():
        raise ValueError("bad upload token")
    return store.root / "uploads" / token


class Handler(BaseHTTPRequestHandler):
    server_version = "colored-inkt/1"

    def log_message(self, fmt, *args):  # quieter console
        if not self.path.startswith(("/seg/", "/segch/", "/api/")):
            super().log_message(fmt, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        self._send(json.dumps(_clean(payload)).encode(), "application/json", status)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _known(self, seg_id: str | None, need_ready: bool = True) -> bool:
        if seg_id not in STORE.segments:
            self._error(404, "unknown segment")
            return False
        if need_ready and not STORE.ready(seg_id):
            self._error(409, "segment not prepared")
            return False
        return True

    # --- GET -------------------------------------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        store = STORE
        try:
            if url.path in ("/", "/index.html"):
                return self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self._json({
                    "visualizers": visualizer_list(), "colormaps": _ANCHORS,
                    "prefilters": {"notch": NOTCH_WIDTHS, "speckle": SPECKLE_SIZES},
                    "recommended": list(rendering.DEFAULT_FILTERS),
                })
            if url.path == "/api/segments":
                return self._json({"segments": store.describe()})
            if url.path == "/api/starred":
                return self._json(store.load_starred())
            if url.path == "/api/depthscan":  # ?seg=…&vis=…&notch=…&speckle=…&fixed=…: which depths are ready
                seg_id, vis = query.get("seg"), query.get("vis")
                if not self._known(seg_id) or lookup(vis) is None:
                    return None if seg_id not in store.segments else self._error(404, "unknown filter")
                notch, speckle = resolve(vis, *prefilter_of(query))[1:]
                return self._json(store.depth_scan_status(seg_id, vis, notch, speckle, depth_of(query)["fixed"]))
            if url.path.startswith("/segch/"):
                # /segch/<segment>/<filter>.json (metadata) or /segch/<segment>/<filter>/<channel>.png
                parts = url.path.split("/")
                if not self._known(parts[2]):
                    return None
                if len(parts) == 4 and parts[3].endswith(".json"):
                    vis = parts[3].removesuffix(".json")
                    if lookup(vis) is None:
                        return self._error(404, "unknown filter")
                    return self._json(store.channel_meta(parts[2], vis, *prefilter_of(query), **depth_of(query)))
                if len(parts) == 5 and lookup(parts[3]) is not None:
                    channel = parts[4].removesuffix(".png")
                    pre, dep = prefilter_of(query), depth_of(query)
                    names = [c["name"] for c in store.channel_meta(parts[2], parts[3], *pre, **dep)["channels"]]
                    if channel not in names:
                        return self._error(404, "unknown channel")
                    return self._send(store.channel_png(parts[2], parts[3], channel, *pre, **dep), "image/png",
                                      cache=True)
                return self._error(404, "not found")
            if url.path.startswith("/seg/"):
                _, _, seg_id, filename = url.path.split("/", 3)
                if not self._known(seg_id):
                    return None
                if filename == "label.png":
                    path = store.folder(seg_id) / "label.png"
                    if not path.exists():
                        return self._error(404, "this segment has no labels")
                    return self._send(path.read_bytes(), "image/png")
                vis = filename.rsplit(".", 1)[0]
                if lookup(vis) is None:
                    return self._error(404, "unknown filter")
                payload, content_type = store.render(seg_id, vis, *prefilter_of(query),
                                                     cmap=query.get("cmap") or None, **depth_of(query))
                return self._send(payload, content_type, cache=True)
            return self._error(404, "not found")
        except ValueError as exc:  # a bad pre-filter, depth or colour map in the query
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    # --- POST ------------------------------------------------------------------------------------------------
    def do_POST(self):
        url = urlparse(self.path)
        store = STORE
        try:
            if url.path == "/api/upload/start":
                token = secrets.token_hex(8)
                upload_dir(store, token).mkdir(parents=True, exist_ok=True)
                return self._json({"upload": token})
            if url.path.startswith("/api/upload/"):  # /api/upload/<token>/<file name>, raw body
                _, _, _, token, name = url.path.split("/", 4)
                folder = upload_dir(store, token)
                if not folder.exists():
                    return self._error(404, "unknown upload")
                target = folder / Path(unquote(name)).name
                remaining = int(self.headers.get("Content-Length") or 0)
                with open(target, "wb") as out:
                    while remaining > 0:
                        chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                return self._json({"ok": True, "bytes": target.stat().st_size})
            body = self._body()
            if url.path == "/api/upload/cancel":
                shutil.rmtree(upload_dir(store, str(body.get("upload", ""))), ignore_errors=True)
                return self._json({"ok": True})
            if url.path == "/api/segments/add":
                seg = add_segment(store, body)
                return self._json({"id": seg.id, "segments": store.describe()})
            if url.path == "/api/segments/prepare":
                if not self._known(body.get("id"), need_ready=False):
                    return None
                store.prepare_async(body["id"])
                return self._json({"segments": store.describe()})
            if url.path == "/api/segments/side":
                if not self._known(body.get("id"), need_ready=False):
                    return None
                store.set_side(body["id"], int(body.get("side", 1)))
                return self._json({"segments": store.describe()})
            if url.path == "/api/segments/delete":
                if not self._known(body.get("id"), need_ready=False):
                    return None
                store.delete(body["id"])
                return self._json({"segments": store.describe()})
            if url.path == "/api/starred":
                return self._json(store.save_starred(body))
            if url.path == "/api/depthscan":  # {seg, vis, notch, speckle, fixed}: every depth in the background
                seg_id, vis = body.get("seg"), body.get("vis")
                if not self._known(seg_id):
                    return None
                if lookup(vis) is None:
                    return self._error(404, "unknown filter")
                notch, speckle = resolve(vis, *prefilter_of(body))[1:]
                fixed = bool(body.get("fixed"))
                store.depth_scan(seg_id, vis, notch, speckle, fixed)
                return self._json(store.depth_scan_status(seg_id, vis, notch, speckle, fixed))
            return self._error(404, "not found")
        except (ValueError, FileNotFoundError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")


def serve(host: str = "127.0.0.1", port: int = 6544, data: str | Path = "~/.colored-inkt") -> None:
    global STORE
    STORE = Store(Path(data))
    print(f"{len(STORE.segments)} segments in {STORE.root}, {len(REGISTRY)} filters", flush=True)
    print(f"Open http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
