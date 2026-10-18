"""The viewer's segments: the ten open-data examples and whatever users add, each prepared once and cached.

A segment lives in its own folder under the data directory (``--data``, default ``~/.colored-inkt``):

    segment.json     what it is: title, source, voxel sizes, crop, ink side, state
    ct.npy           the cropped (and binned) CT stack with a 48 px ring of context, in the source's layers and type
    label.png        optional ink-label overlay (yellow outline; dimmed where nobody labelled)
    products.npz     the surface products every filter renders from (``surface.compute_products``)
    v<version>/      cached renders, channels and calibrations per filter and pre-filter

Preparing reads the CT once (download for the examples, a copy for added segments), then finds the papyrus face
block by block. Filters render on the whole segment from the products and are calibrated on the segment itself
(label-free).
"""

from __future__ import annotations

import io
import json
import re
import shutil
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from colored_inkt import render as rendering
from colored_inkt import surface
from colored_inkt.visualizers import COLORMAPS, PRODUCTS, REGISTRY, calibrate, render, to_uint8
from colored_inkt.volume import Volume, open_volume, read_label

EXAMPLES = Path(__file__).with_name("examples.json")
MAX_SIDE = 4096  # largest segment side the viewer accepts (pixels after binning); memory grows with the area
DEFAULT_SIDE = 2048  # crop used when the source is larger and no crop is given
NOTCH_WIDTHS = (0, 4, 7, 12)  # fibre notch pre-filter half widths (degrees); 0 = off
SPECKLE_SIZES = (0, 30, 50, 90)  # despeckle pre-filter: largest speck removed (µm); 0 = off
DEPTH_MAX, DEPTH_STEP = 30, 2  # depth scan: offsets −30 … +30 planes (±72 µm) in steps of 2
LIVE_KINDS = ("scalar", "rgb", "overlay", "tints")  # recoloured in the browser from raw channels


@dataclass
class Segment:
    id: str
    title: str
    kind: str  # "example" or "added"
    layer_um: float  # spacing of the stored layers
    pixel_um: float  # in-plane pixel size of the stored CT (after binning)
    side: int  # +1: the inked face looks towards higher layer numbers
    h: int = 0
    w: int = 0
    layers: int = 0
    source: dict = field(default_factory=dict)  # where the CT came from (for the record and for examples)
    note: str = ""
    label: bool = False
    ink_fraction: float | None = None

    @property
    def geometry(self) -> surface.Geometry:
        return surface.Geometry(layer_um=self.layer_um, pixel_um=self.pixel_um, side=self.side)


def check_prefilter(notch: int, speckle: int) -> None:
    if notch not in NOTCH_WIDTHS:
        raise ValueError(f"notch must be one of {NOTCH_WIDTHS}")
    if speckle not in SPECKLE_SIZES:
        raise ValueError(f"speckle must be one of {SPECKLE_SIZES}")


def check_depth(depth: int) -> None:
    if depth % DEPTH_STEP or abs(depth) > DEPTH_MAX:
        raise ValueError(f"depth must be a multiple of {DEPTH_STEP} within ±{DEPTH_MAX}")


def depths() -> list[int]:
    return list(range(-DEPTH_MAX, DEPTH_MAX + 1, DEPTH_STEP))


def suffix(notch: int, speckle: int = 0, depth: int = 0, fixed: bool = False) -> str:
    """File-name suffix of products or a render: depth offset, then the pre-filters (despeckle, then notch)."""
    shift = f".{'layers' if fixed else 'depth'}{depth:+d}" if depth else ""
    return shift + (f".speckle{speckle}" if speckle else "") + (f".notch{notch}" if notch else "")


def lookup(name: str):
    """The filter called ``name`` (full or short name), or None."""
    try:
        return rendering.resolve(name)
    except KeyError:
        return None


def resolve(name: str, notch: int, speckle: int):
    """(filter, notch, speckle): a filter with its own pre-filters ignores the viewer's."""
    vis = lookup(name)
    if vis is None:
        raise KeyError(name)
    notch, speckle = rendering.prefilters(vis, notch, speckle)
    return vis, notch, speckle


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:48] or "segment"


def default_crop(h: int, w: int, side: int = DEFAULT_SIDE) -> tuple[int, int, int, int]:
    """The centre square of at most ``side`` px."""
    ch, cw = min(h, side), min(w, side)
    return (h - ch) // 2, (w - cw) // 2, ch, cw


class Store:
    """Segments on disk, their preparation (one at a time, in the background) and their renders."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        (self.root / "segments").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # one heavy computation at a time
        self.status: dict[str, str] = {}
        self._tiles: dict[tuple, object] = {}
        self._depth_tiles: dict[tuple, object] = {}
        self._scan = None
        self.segments: dict[str, Segment] = {}
        self._install_examples()
        for path in sorted((self.root / "segments").glob("*/segment.json")):
            seg = Segment(**json.loads(path.read_text()))
            self.segments[seg.id] = seg

    # --- Folders ------------------------------------------------------------------------------------------
    def folder(self, seg_id: str) -> Path:
        return self.root / "segments" / seg_id

    def products_path(self, seg_id: str, notch: int = 0, speckle: int = 0) -> Path:
        return self.folder(seg_id) / f"products{suffix(notch, speckle)}.npz"

    def save(self, seg: Segment) -> None:
        folder = self.folder(seg.id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "segment.json").write_text(json.dumps(asdict(seg), indent=1))
        self.segments[seg.id] = seg

    def _install_examples(self) -> None:
        """Register the ten open-data examples (nothing is downloaded until one is prepared)."""
        for ex in json.loads(EXAMPLES.read_text()):
            if (self.folder(ex["id"]) / "segment.json").exists():
                continue
            scale = 2 ** ex["level"]
            self.save(Segment(
                id=ex["id"], title=f"{ex['scroll']} · {ex['segment']}", kind="example",
                layer_um=ex["voxel_um"], pixel_um=round(ex["voxel_um"] * scale, 3), side=1,
                h=ex["box"][2], w=ex["box"][3], source=ex,
                note=f"open data, labelled ink {ex['ink_fraction']:.0%} of the square", label=True,
                ink_fraction=ex["ink_fraction"],
            ))

    # --- State --------------------------------------------------------------------------------------------
    def ready(self, seg_id: str) -> bool:
        return self.products_path(seg_id).exists()

    def describe(self) -> list[dict]:
        rows = []
        for seg in self.segments.values():
            state = "ready" if self.ready(seg.id) else self.status.get(seg.id, "not prepared")
            rows.append({**asdict(seg), "state": state,
                         "size_mm": [round(seg.h * seg.pixel_um / 1000, 1), round(seg.w * seg.pixel_um / 1000, 1)]})
        rows.sort(key=lambda r: (r["kind"] != "added", r["title"]))
        return rows

    def busy(self, seg_id: str) -> bool:
        return self.status.get(seg_id, "").split(" ")[0] in ("queued", "reading", "finding", "block", "copying")

    # --- Adding and preparing -----------------------------------------------------------------------------
    def add(self, *, title: str, source: Volume, layer_um: float, pixel_um: float, side: int = 1, bin: int = 1,
            crop: tuple[int, int, int, int] | None = None, label=None, origin: dict | None = None,
            cleanup: Path | None = None) -> Segment:
        """Register a segment from any readable stack and prepare it in the background.

        ``crop`` = (y, x, h, w) in source pixels; default: the centre ``DEFAULT_SIDE`` square after binning.
        ``label`` is an optional 2D ink label of the same size as the source layers (nonzero = ink).
        ``cleanup`` is removed once the CT has been copied (an upload's staging folder).
        """
        layers, height, width = source.shape
        if crop is None:
            cy, cx, ch, cw = default_crop(height // bin, width // bin)
            crop = (cy * bin, cx * bin, ch * bin, cw * bin)
        y, x, h, w = (int(v) for v in crop)
        h, w = min(h, height - y), min(w, width - x)
        if y < 0 or x < 0 or h <= 0 or w <= 0:
            raise ValueError(f"the crop {crop} lies outside the {height} × {width} px layers")
        hb, wb = h // bin // 2 * 2, w // bin // 2 * 2  # even sides: the notch filter's frequency grid needs it
        if max(hb, wb) > MAX_SIDE:
            raise ValueError(f"the crop is {hb} × {wb} px after binning; the viewer takes at most {MAX_SIDE} px "
                             f"per side. Crop it, bin it, or use the command line for large areas.")
        if hb < 64 or wb < 64:
            raise ValueError(f"the crop is only {hb} × {wb} px after binning; give at least 64 px per side")
        base = slug(title)
        seg_id, k = base, 2
        while seg_id in self.segments:
            seg_id, k = f"{base}-{k}", k + 1
        seg = Segment(id=seg_id, title=title, kind="added", layer_um=layer_um, pixel_um=round(pixel_um * bin, 4),
                      side=side, h=hb, w=wb, layers=layers,
                      source={**(origin or {}), "crop": [y, x, h, w], "bin": bin, "layers": layers,
                              "full_size": [height, width]},
                      note=f"{layers} layers at {layer_um:g} µm, crop {h} × {w} px at ({y}, {x})"
                           + (f", binned {bin}×" if bin > 1 else ""),
                      label=label is not None)
        self.save(seg)
        self.status[seg.id] = "queued"
        threading.Thread(target=self._prepare_quietly, args=(seg.id, source, (y, x, hb * bin, wb * bin), bin, label,
                                                             cleanup), daemon=True).start()
        return seg

    def prepare_async(self, seg_id: str) -> None:
        """Prepare a registered segment (download an example, or recompute the products of an added one)."""
        if self.ready(seg_id) or self.busy(seg_id):
            return
        self.status[seg_id] = "queued"
        threading.Thread(target=self._prepare_quietly, args=(seg_id,), daemon=True).start()

    def _prepare_quietly(self, seg_id: str, *args) -> None:
        try:
            self.prepare(seg_id, *args)
        except Exception as exc:
            self.status[seg_id] = f"failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()

    def prepare(self, seg_id: str, source: Volume | None = None, crop=None, bin: int = 1, label=None,
                cleanup: Path | None = None) -> None:
        """Copy the CT (if not there yet), draw the label overlay, compute the products (blocking)."""
        seg = self.segments[seg_id]
        with self.lock:
            folder = self.folder(seg_id)
            ct_path = folder / "ct.npy"
            if not ct_path.exists():
                if source is None and seg.kind == "example":
                    source, crop, bin, label = self._example_source(seg)
                if source is None:
                    raise ValueError("the CT of this segment is missing; add it again")
                self._copy_ct(seg, source, crop, bin, ct_path)
                if label is not None:
                    self._write_label(seg, label, crop, bin)
                if cleanup is not None:
                    shutil.rmtree(cleanup, ignore_errors=True)
            self.status[seg_id] = "finding the papyrus face, block 0"
            products = surface.compute_products(
                self._area(seg_id), progress=lambda i, n: self.status.__setitem__(seg_id, f"finding the papyrus face, block {i}/{n}"))
            self._save_products(seg_id, products)
            self.status.pop(seg_id, None)

    def _area(self, seg_id: str) -> surface.Area:
        """The segment's stored CT as an area (the stored ring of context is outside the box)."""
        seg = self.segments[seg_id]
        ct = np.load(self.folder(seg_id) / "ct.npy", mmap_mode="r")
        return surface.Area(_array_volume(ct), (surface.MARGIN, surface.MARGIN, seg.h, seg.w), seg.geometry)

    def _example_source(self, seg: Segment):
        ex = seg.source
        volume = open_volume(ex["ct_url"], level=ex["level"])
        label = {"ink": read_label(ex["ink_url"], ex["level"]), "supervision": read_label(ex["supervision_url"], ex["level"])}
        return volume, tuple(ex["box"]), 1, label

    def _copy_ct(self, seg: Segment, source: Volume, crop, bin: int, path: Path) -> None:
        """Read the crop and a ``MARGIN`` ring around it (zero outside the source), binned, strip by strip."""
        m = surface.MARGIN
        y, x = crop[0] - m * bin, crop[1] - m * bin
        h, w = seg.h + 2 * m, seg.w + 2 * m
        tmp = path.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=source.dtype, shape=(source.shape[0], h, w))
        strip = 256
        for r in range(0, h, strip):
            rh = min(strip, h - r)
            self.status[seg.id] = f"reading the CT, {int(100 * r / h)} %"
            raw = _retry(source.read, y + r * bin, x, rh * bin, w * bin)
            if bin > 1:
                values, _ = surface.bin_block(raw, bin)
                raw = np.rint(values).astype(source.dtype)
            out[:, r : r + rh] = raw[:, :rh, :w]
        out.flush()
        del out
        tmp.replace(path)
        seg.layers = source.shape[0]
        self.save(seg)

    def _write_label(self, seg: Segment, label, crop, bin: int) -> None:
        """RGBA overlay: ink outlined in yellow; where a supervision mask says nobody looked, dimmed."""
        from scipy import ndimage

        y, x, h, w = crop

        def cut(array):
            window = np.zeros((h, w), bool)
            y1, x1 = min(array.shape[0], y + h), min(array.shape[1], x + w)
            if y1 > y and x1 > x:
                window[: y1 - y, : x1 - x] = np.asarray(array[y:y1, x:x1]) > 0
            if bin > 1:
                window = window[: h // bin * bin, : w // bin * bin].reshape(h // bin, bin, w // bin, bin).mean(axis=(1, 3)) >= 0.5
            return window[: seg.h, : seg.w]

        if isinstance(label, dict):
            ink, supervised = cut(label["ink"]), cut(label["supervision"])
            ink &= supervised
        else:
            ink, supervised = cut(label), None
        edge = ink & ~ndimage.binary_erosion(ink, iterations=2)
        rgba = np.zeros(ink.shape + (4,), dtype=np.uint8)
        if supervised is not None:
            rgba[~supervised] = (0, 0, 0, 110)
        rgba[ink] = (255, 210, 0, 45)
        rgba[edge] = (255, 210, 0, 235)
        Image.fromarray(rgba, mode="RGBA").save(self.folder(seg.id) / "label.png")
        seg.label = True
        seg.ink_fraction = round(float(ink.mean()), 4)
        self.save(seg)

    def _save_products(self, seg_id: str, products: dict[str, np.ndarray], notch: int = 0, speckle: int = 0) -> None:
        path = self.products_path(seg_id, notch, speckle)
        tmp = path.with_suffix(".tmp.npz")
        # float16 holds 0–255 CT values to 1/8 and heights to 1/64 plane: plenty for display.
        np.savez(tmp, **{k: (v if k == "valid" else v.astype(np.float16)) for k, v in products.items()})
        tmp.replace(path)

    def set_side(self, seg_id: str, side: int) -> None:
        """Flip which face the ink is on: the products (and every cached render) are made again."""
        seg = self.segments[seg_id]
        if side not in (1, -1):
            raise ValueError("side must be +1 or -1")
        if self.busy(seg_id):
            raise ValueError("the segment is being prepared")
        if not (self.folder(seg_id) / "ct.npy").exists() and seg.kind != "example":
            raise ValueError("the CT of this segment is missing; add it again")
        with self.lock:
            seg.side = side
            self.save(seg)
            self._forget(seg_id)
            for path in self.folder(seg_id).glob("products*.npz"):
                path.unlink()
            for path in self.folder(seg_id).glob("v*/"):
                shutil.rmtree(path, ignore_errors=True)
        self.prepare_async(seg_id)

    def delete(self, seg_id: str) -> None:
        """Remove an added segment, or the downloaded data of an example (it can be prepared again)."""
        seg = self.segments[seg_id]
        if self.busy(seg_id):
            raise ValueError("the segment is being prepared")
        with self.lock:
            self._forget(seg_id)
            if seg.kind == "example":
                for path in self.folder(seg_id).iterdir():
                    if path.name != "segment.json":
                        shutil.rmtree(path) if path.is_dir() else path.unlink()
                seg.side = 1
                self.save(seg)
            else:
                shutil.rmtree(self.folder(seg_id), ignore_errors=True)
                self.segments.pop(seg_id, None)
            self.status.pop(seg_id, None)

    def _forget(self, seg_id: str) -> None:
        for cache in (self._tiles, self._depth_tiles):
            for key in [k for k in cache if k[0] == seg_id]:
                cache.pop(key)

    # --- Tiles --------------------------------------------------------------------------------------------
    def tile(self, seg_id: str, notch: int = 0, speckle: int = 0, depth: int = 0, fixed: bool = False):
        """The segment as a tile, pre-filtered (products cached on disk); ``depth`` ≠ 0 is computed from the CT."""
        check_prefilter(notch, speckle)
        check_depth(depth)
        if depth:
            return self._depth_tile(seg_id, notch, speckle, depth, fixed)
        key = (seg_id, notch, speckle)
        if key not in self._tiles:
            seg = self.segments[seg_id]
            path = self.products_path(seg_id, notch, speckle)
            if notch and not path.exists():
                self._save_products(seg_id, self._with_valid(seg_id, self.tile(seg_id, 0, speckle).notched(notch)),
                                    notch, speckle)
            elif speckle and not path.exists():
                self._save_products(seg_id, self._with_valid(seg_id, self.tile(seg_id, 0, 0).despeckled(speckle)),
                                    0, speckle)
            with np.load(path) as stored:
                products = {k: stored[k] for k in (*PRODUCTS, "valid")}
            tile = surface.tile_from_products(products, seg.geometry)
            if notch or speckle:  # keep the fibres of the unfiltered skin (a notched one has none left)
                tile.__dict__["fiber_orientations"] = self.tile(seg_id, 0, 0).fiber_orientations
            while len(self._tiles) >= 2:
                self._tiles.pop(next(iter(self._tiles)))
            self._tiles[key] = tile
        return self._tiles[key]

    def _with_valid(self, seg_id: str, tile) -> dict[str, np.ndarray]:
        with np.load(self.products_path(seg_id)) as stored:
            valid = stored["valid"]
        return {**tile.products(), "valid": valid}

    def _depth_tile(self, seg_id: str, notch: int, speckle: int, depth: int, fixed: bool):
        key = (seg_id, notch, speckle, depth, fixed)
        if key in self._depth_tiles:
            return self._depth_tiles[key]
        seg = self.segments[seg_id]
        if notch or speckle:
            plain = self._depth_tile(seg_id, 0, 0, depth, fixed)
            tile = rendering.prefiltered(plain, notch, speckle)
        else:
            products = surface.compute_products(self._area(seg_id), depth=depth, fixed=fixed)
            tile = surface.tile_from_products(products, seg.geometry)
        while len(self._depth_tiles) >= 3:
            self._depth_tiles.pop(next(iter(self._depth_tiles)))
        self._depth_tiles[key] = tile
        return tile

    def _calibration(self, seg_id: str, vis, notch: int = 0, speckle: int = 0) -> dict:
        """The filter's calibration on the segment (cached); despeckled filters keep the plain one."""
        if speckle:
            return self._calibration(seg_id, vis, notch, 0)
        path = self.folder(seg_id) / f"v{vis.version}" / f"{vis.name}{suffix(notch)}.calibration.json"
        if path.exists():
            return json.loads(path.read_text())
        calibration = calibrate(vis, [self.tile(seg_id, notch)])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(calibration))
        return calibration

    # --- Renders and channels -----------------------------------------------------------------------------
    def channel_meta(self, seg_id: str, vis_name: str, notch: int = 0, speckle: int = 0,
                     depth: int = 0, fixed: bool = False) -> dict:
        """The filter's raw channels on the segment, for live recolouring in the browser.

        Each channel is quantised to 16 bits over its 0.01–99.99 % range and stored as a lossless PNG (red = high
        byte, green = low byte). The returned metadata gives that range, the calibrated display range, the norm and
        a 1024-bin histogram of the quantised values (for percentile-based min / max in the viewer).
        """
        vis, notch, speckle = resolve(vis_name, notch, speckle)
        seg = self.segments[seg_id]
        base = self.folder(seg_id) / f"v{vis.version}" / "channels"
        tag = suffix(notch, speckle, depth, fixed)
        meta_path = base / f"{vis.name}{tag}.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        with self.lock:
            if meta_path.exists():
                return json.loads(meta_path.read_text())
            meta = {"visualizer": vis.name, "display": vis.display, "channels": [], "w": seg.w, "h": seg.h,
                    "notch": notch, "speckle": speckle, "depth": depth, "fixed": fixed}
            if vis.display["kind"] in LIVE_KINDS:
                tile = self.tile(seg_id, notch, speckle, depth, fixed)
                calibration = self._calibration(seg_id, vis, notch, speckle)
                base.mkdir(parents=True, exist_ok=True)
                for name, values in vis.channels(tile, calibration["params"]).items():
                    sample = tile.pixels(values, 4)
                    qlo, qhi = (float(v) for v in np.percentile(sample, [0.01, 99.99]))
                    if qhi <= qlo:
                        qhi = qlo + 1e-6
                    q = np.clip((values - qlo) / (qhi - qlo) * 65535 + 0.5, 0, 65535).astype(np.uint16)
                    packed = np.stack([(q >> 8).astype(np.uint8), (q & 255).astype(np.uint8),
                                       np.zeros(q.shape, np.uint8)], axis=-1)
                    Image.fromarray(packed).save(base / f"{vis.name}{tag}.{name}.png", format="PNG")
                    hist = np.bincount((tile.pixels(q) >> 6).ravel(), minlength=1024)
                    meta["channels"].append({
                        "name": name, "qlo": qlo, "qhi": qhi, "norm": vis.norm_of(name),
                        "range": calibration["ranges"][name], "hist": hist.tolist(),
                    })
            base.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta))
            return meta

    def channel_png(self, seg_id: str, vis_name: str, channel: str, notch: int = 0, speckle: int = 0,
                    depth: int = 0, fixed: bool = False) -> bytes:
        self.channel_meta(seg_id, vis_name, notch, speckle, depth, fixed)
        vis, notch, speckle = resolve(vis_name, notch, speckle)
        path = (self.folder(seg_id) / f"v{vis.version}" / "channels"
                / f"{vis.name}{suffix(notch, speckle, depth, fixed)}.{channel}.png")
        return path.read_bytes()

    def render(self, seg_id: str, vis_name: str, notch: int = 0, speckle: int = 0,
               cmap: str | None = None, depth: int = 0, fixed: bool = False) -> tuple[bytes, str]:
        """(image bytes, content type): PNG for a still, animated WebP at half size for an animation.

        Pixels without data are black. ``depth`` / ``fixed`` render as if the segment were off the face; the
        calibration stays the one of the true surface, so every depth is shown through the same colouring.
        """
        vis, notch, speckle = resolve(vis_name, notch, speckle)
        if cmap is not None and cmap not in COLORMAPS:
            raise ValueError(f"unknown colour map {cmap!r}")
        if not vis.recolor or cmap == vis.recolor:
            cmap = None
        ext = "webp" if vis.animated else "png"
        tag = suffix(notch, speckle, depth, fixed) + (f".cmap-{re.sub(r'[^A-Za-z0-9]+', '_', cmap)}" if cmap else "")
        path = self.folder(seg_id) / f"v{vis.version}" / f"{vis.name}{tag}.{ext}"
        content_type = f"image/{ext}"
        if path.exists():
            return path.read_bytes(), content_type
        with self.lock:
            if path.exists():
                return path.read_bytes(), content_type
            tile = self.tile(seg_id, notch, speckle, depth, fixed)
            if vis.animated:
                tile = _half(tile)  # 26 full-size float frames would need > 1 GB
                calibration = calibrate(vis, [tile])
            else:
                calibration = self._calibration(seg_id, vis, notch, speckle)
            if cmap:
                calibration = {**calibration, "params": {**calibration["params"], "cmap": cmap}}
            out = render(vis, tile, calibration)
            buffer = io.BytesIO()
            if vis.animated:
                images = [Image.fromarray(to_uint8(f)) for f in out]
                images[0].save(buffer, format="WEBP", save_all=True, append_images=images[1:],
                               duration=110, loop=0, quality=85, method=4)
            else:
                rgb = to_uint8(out)
                if tile.valid is not None:
                    rgb[~tile.valid] = 0
                Image.fromarray(rgb).save(buffer, format="PNG")
            payload = buffer.getvalue()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return payload, content_type

    # --- Depth scan: every offset of one filter, rendered in the background ---------------------------------
    def depth_output(self, seg_id: str, vis_name: str, notch: int, speckle: int, depth: int, fixed: bool) -> Path:
        vis, notch, speckle = resolve(vis_name, notch, speckle)
        tag = suffix(notch, speckle, depth, fixed)
        base = self.folder(seg_id) / f"v{vis.version}"
        if vis.display["kind"] in LIVE_KINDS:
            return base / "channels" / f"{vis.name}{tag}.json"
        return base / f"{vis.name}{tag}.{'webp' if vis.animated else 'png'}"

    def depth_scan_status(self, seg_id: str, vis_name: str, notch: int, speckle: int, fixed: bool) -> dict:
        ready = [d for d in depths() if self.depth_output(seg_id, vis_name, notch, speckle, d, fixed).exists()]
        scan = self._scan
        running = bool(scan and scan[0] == (seg_id, vis_name, notch, speckle, fixed) and scan[2].is_alive())
        return {"depths": depths(), "ready": ready, "running": running}

    def depth_scan(self, seg_id: str, vis_name: str, notch: int, speckle: int, fixed: bool) -> None:
        """Compute every depth of the filter in a background thread (nearest offsets first)."""
        key = (seg_id, vis_name, notch, speckle, fixed)
        scan = self._scan
        if scan and scan[0] == key and scan[2].is_alive():
            return
        if scan:
            scan[1].set()
        stop = threading.Event()
        live = lookup(vis_name).display["kind"] in LIVE_KINDS

        def run():
            for d in sorted(depths(), key=lambda d: (abs(d), d)):
                if stop.is_set():
                    return
                try:
                    if live:
                        self.channel_meta(seg_id, vis_name, notch, speckle, d, fixed)
                    else:
                        self.render(seg_id, vis_name, notch, speckle, depth=d, fixed=fixed)
                except Exception:
                    traceback.print_exc()
                    return

        thread = threading.Thread(target=run, daemon=True)
        self._scan = (key, stop, thread)
        thread.start()

    # --- Stars and saved display variants -------------------------------------------------------------------
    def starred_path(self) -> Path:
        return self.root / "starred.json"

    def load_starred(self) -> dict:
        path = self.starred_path()
        if path.exists():
            return json.loads(path.read_text())
        return {"starred": list(rendering.DEFAULT_FILTERS), "variants": []}

    def save_starred(self, payload: dict) -> dict:
        """Keep known filters only; a variant is {id, title, vis, spec}."""
        starred = [n for n in payload.get("starred", []) if lookup(n) is not None]
        variants = [
            {"id": str(v["id"]), "title": str(v.get("title") or v["id"]), "vis": v["vis"], "spec": v.get("spec", {})}
            for v in payload.get("variants", []) if v.get("vis") in REGISTRY and v.get("id")
        ]
        result = {"starred": list(dict.fromkeys(starred)), "variants": variants}
        self.starred_path().write_text(json.dumps(result, indent=1))
        return result


def _array_volume(array: np.ndarray) -> Volume:
    return Volume(tuple(array.shape), array.dtype, lambda y0, y1, x0, x1: np.asarray(array[:, y0:y1, x0:x1]),
                  "stored CT")


def _retry(fn, *args, attempts: int = 5):
    """Call ``fn``, retrying with a growing pause (the open-data bucket answers 503 "Slow Down" under load)."""
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))


def _half(tile):
    """The same tile at half the in-plane resolution (2×2 means)."""
    from colored_inkt.visualizers import Tile

    def down(a: np.ndarray) -> np.ndarray:
        h, w = a.shape[-2] // 2 * 2, a.shape[-1] // 2 * 2
        a = a[..., :h, :w]
        return a.reshape(a.shape[:-2] + (h // 2, 2, w // 2, 2)).mean(axis=(-3, -1))

    valid = None if tile.valid is None else down(tile.valid.astype(np.float32)) > 0.5
    return Tile.from_products({k: down(v) for k, v in tile.products().items()},
                              side=tile.side, xy_scale=tile.xy_scale * 2, valid=valid)
