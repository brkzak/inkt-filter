"""Command line: ``colored-inkt render`` (filtered images), ``colored-inkt serve`` (the viewer), ``colored-inkt filters``.

    colored-inkt render SOURCE -o OUT [--filters two_hue isoluminant lbp] [--box Y X H W] [--voxel-um 7.91] ...
    colored-inkt serve [--port 6544] [--data ~/.colored-inkt]
    colored-inkt filters [--all]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np


def _log(message: str) -> None:
    print(message, flush=True)


def cmd_filters(args) -> None:
    from colored_inkt import render as rendering
    from colored_inkt.visualizers import REGISTRY

    picked = REGISTRY.values() if args.all else [rendering.resolve(n) for n in rendering.DEFAULT_FILTERS]
    family = None
    for vis in picked:
        if args.all and vis.family != family:
            family = vis.family
            print(f"\n{family}")
        note = " (animated: viewer only)" if vis.animated else ""
        print(f"  {rendering.short_name(vis.name):24s} {vis.title}{note}")
        if not args.all:
            print(f"  {'':24s} {vis.idea}\n")


def cmd_render(args) -> None:
    from PIL import Image

    from colored_inkt import render as rendering
    from colored_inkt import surface
    from colored_inkt.volume import ome_voxel_um, open_volume

    names = [v.name for v in rendering.REGISTRY.values() if not v.animated] if args.filters == ["all"] else \
        [rendering.resolve(n).name for n in args.filters]
    volume = open_volume(args.source, level=args.level)
    layers, height, width = volume.shape
    voxel = ome_voxel_um(args.source, args.level)
    layer_um = args.layer_um or args.voxel_um or (voxel[0] if voxel else None)
    pixel_um = args.pixel_um or args.voxel_um or (voxel[1] if voxel else None)
    if layer_um is None or pixel_um is None:
        sys.exit("The voxel size is not in the data: give it with --voxel-um (e.g. 7.91 or 2.4).")
    geometry = surface.Geometry(layer_um=layer_um, pixel_um=pixel_um, bin=args.bin, side=args.side)
    y, x, h, w = args.box or (0, 0, height, width)
    b = args.bin
    # The area in binned pixels, with even sides (the notch filter's frequency grid needs them).
    box = (y // b, x // b, min(h, height - y) // b // 2 * 2, min(w, width - x) // b // 2 * 2)
    if box[2] < 64 or box[3] < 64:
        sys.exit(f"The area is only {box[2]} × {box[3]} px; give at least 64 px per side.")
    _log(f"{volume.description}: {layers} layers of {height} × {width} px")
    _log(f"area {box[2]} × {box[3]} px at {pixel_um * b:g} µm, layers {layer_um:g} µm"
         + (" (resampled to 2.4 µm planes)" if geometry.resamples else "") + f", ink side {args.side:+d}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out / ".colored-inkt-work"
    work.mkdir(parents=True, exist_ok=True)
    area = surface.Area(volume, box, geometry)
    shapes = surface.product_shapes(box[2], box[3])
    products = {k: np.lib.format.open_memmap(work / f"{k}.npy", mode="w+", dtype=np.float16, shape=s)
                for k, s in shapes.items()}
    products["valid"] = np.lib.format.open_memmap(work / "valid.npy", mode="w+", dtype=bool, shape=box[2:])
    start = time.time()
    step = max(1, (box[2] // surface.BLOCK + 1) * (box[3] // surface.BLOCK + 1) // 20)
    surface.compute_products(area, out=products, progress=lambda i, n: (i % step == 0 or i == n) and _log(
        f"finding the papyrus face: block {i}/{n} ({time.time() - start:.0f} s)"))
    if not np.asarray(products["valid"]).any():
        sys.exit("The area holds no data (every layer is 0 there).")
    written = []
    for name, rgb in rendering.render_area(products, geometry, names, notch=args.notch, speckle=args.despeckle,
                                           scratch=work, log=_log):
        Image.MAX_IMAGE_PIXELS = None
        path = out / f"{args.prefix}{rendering.short_name(name)}.png"
        image = Image.fromarray(rgb)
        image.save(path, compress_level=3)
        if args.preview:
            image.reduce(4).save(path.with_name(path.stem + "_quarter.jpg"), quality=90)
        written.append(path.name)
        _log(f"wrote {path} ({time.time() - start:.0f} s)")
    record = {"source": str(args.source), "level": args.level, "box_source_px": [y, x, h, w], "bin": b,
              "layer_um": layer_um, "pixel_um": pixel_um, "side": args.side, "notch": args.notch,
              "despeckle": args.despeckle, "filters": names, "images": written}
    (out / f"{args.prefix}render.json").write_text(json.dumps(record, indent=1))
    del products
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


def cmd_serve(args) -> None:
    from colored_inkt.server import serve

    serve(args.host, args.port, args.data)


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(prog="colored-inkt", description="Colour filters that make ink easier to see "
                                     "in CT surface volumes of the Herculaneum scrolls.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("render", help="write filtered images of a surface volume",
                       description="Write one PNG per filter for a surface volume (or a box of it).")
    p.add_argument("source", help="folder of layer images, multi-page .tif, .npy, or Zarr (path or https URL)")
    p.add_argument("-o", "--out", required=True, help="output folder")
    p.add_argument("-f", "--filters", nargs="+", default=["two_hue", "isoluminant", "lbp"],
                   help="filter names (see `colored-inkt filters --all`), or `all`; default: two_hue isoluminant lbp")
    p.add_argument("--voxel-um", type=float, help="voxel size in µm (layer spacing and pixel size); read from "
                                                  "OME-Zarr metadata when there")
    p.add_argument("--layer-um", type=float, help="layer spacing in µm, if it differs from the pixel size")
    p.add_argument("--pixel-um", type=float, help="in-plane pixel size in µm, if it differs from the layer spacing")
    p.add_argument("--level", type=int, default=0, help="pyramid level of a Zarr (default 0)")
    p.add_argument("--box", type=int, nargs=4, metavar=("Y", "X", "H", "W"),
                   help="only this box, in pixels of the source (default: everything)")
    p.add_argument("--bin", type=int, default=1, choices=(1, 2, 3, 4), help="average N × N pixels in-plane first")
    p.add_argument("--side", type=int, default=1, choices=(1, -1),
                   help="+1 if the inked face looks towards higher layer numbers (default), -1 if lower")
    p.add_argument("--notch", type=int, default=0, choices=(0, 4, 7, 12),
                   help="fibre notch pre-filter, ± degrees (filters with their own pre-filter ignore it)")
    p.add_argument("--despeckle", type=int, default=0, choices=(0, 30, 50, 90),
                   help="remove dense specks up to this size in µm first (two_hue always uses 50)")
    p.add_argument("--preview", action="store_true", help="also write a quarter-size JPEG of every image")
    p.add_argument("--prefix", default="", help="put this before every file name")
    p.add_argument("--work", help="folder for the intermediate files (default: OUT/.colored-inkt-work)")
    p.add_argument("--keep-work", action="store_true", help="keep the intermediate files")
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("serve", help="run the viewer in the browser")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this computer only)")
    p.add_argument("--port", type=int, default=6544)
    p.add_argument("--data", default="~/.colored-inkt", help="where segments and renders are kept")
    p.set_defaults(run=cmd_serve)

    p = sub.add_parser("filters", help="list the filters")
    p.add_argument("--all", action="store_true", help="all 79, not only the three recommended ones")
    p.set_defaults(run=cmd_filters)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
