"""Rendering filters on prepared surface products: names, pre-filters, calibration, tiling.

Every filter's colours go through a display range fixed once per area ("calibration", VC3D's window/level),
fitted on the area itself without labels, so a blank stretch of papyrus is not stretched into false structure.
Small areas are rendered in one piece. Large ones are rendered in 2048 px tiles with 256 px of context, feathered
together over a 256 px overlap, because some filters scale by statistics of their tile.
"""

from __future__ import annotations

import numpy as np

from colored_inkt import generalise, research  # noqa: F401  (importing them registers their filters)
from colored_inkt.surface import Geometry, tile_from_products
from colored_inkt.visualizers import REGISTRY, Tile, Visualizer, calibrate, render

DEFAULT_FILTERS = ("gen_two_hue", "res_isoluminant", "res_lbp")
ONE_PIECE = 2560  # areas up to this side are rendered without tiling
CORE, CONTEXT, OVERLAP = 2048, 256, 256
CAL_TILE, CAL_GRID = 1024, 3


def short_name(name: str) -> str:
    """``gen_two_hue`` → ``two_hue``: the name without its family prefix."""
    for prefix in ("gen_", "res_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def resolve(name: str) -> Visualizer:
    """The filter called ``name``, by its full or short name (``two_hue`` = ``gen_two_hue``)."""
    if name in REGISTRY:
        return REGISTRY[name]
    matches = [v for n, v in REGISTRY.items() if short_name(n) == name]
    if len(matches) != 1:
        raise KeyError(f"unknown filter {name!r}; `colored-inkt filters` lists them")
    return matches[0]


def prefilters(vis: Visualizer, notch: int = 0, speckle: int = 0) -> tuple[int, int]:
    """(notch °, despeckle µm) the filter renders with: its own if it names any (two_hue: despeckle 50 µm)."""
    return tuple(vis.prefilter) if any(vis.prefilter) else (notch, speckle)


def prefiltered(tile: Tile, notch: int, speckle: int) -> Tile:
    """``tile`` despeckled, then notch-filtered, keeping the fibre directions of the unfiltered skin."""
    if not notch and not speckle:
        return tile
    orientations = tile.fiber_orientations
    if speckle:
        tile = tile.despeckled(speckle)
    if notch:
        tile = tile.notched(notch)
    tile.__dict__["fiber_orientations"] = orientations
    return tile


def calibrate_filter(vis: Visualizer, tiles: list[Tile], notch: int = 0, speckle: int = 0) -> dict:
    """The filter's display calibration on ``tiles``.

    A despeckled filter keeps the calibration of the tiles *with* their specks: the specks held up the top of every
    density range, and refitting without them stretched ordinary papyrus into the glow.
    """
    del speckle
    return calibrate(vis, [prefiltered(t, notch, 0) for t in tiles])


def render_tile(vis: Visualizer, tile: Tile, calibration: dict, notch: int = 0, speckle: int = 0):
    """RGB float (H, W, 3) in [0, 1], or a list of frames for an animated filter."""
    return render(vis, prefiltered(tile, notch, speckle), calibration)


def calibration_windows(h: int, w: int) -> list[tuple[slice, slice]]:
    """A 3 × 3 grid of 1024 px windows over a large area (the whole area if it is small)."""
    if max(h, w) <= ONE_PIECE:
        return [(slice(0, h), slice(0, w))]
    boxes = []
    for i in range(CAL_GRID):
        for j in range(CAL_GRID):
            y0 = int(np.clip((i + 0.5) * h / CAL_GRID - CAL_TILE // 2, 0, max(0, h - CAL_TILE)))
            x0 = int(np.clip((j + 0.5) * w / CAL_GRID - CAL_TILE // 2, 0, max(0, w - CAL_TILE)))
            boxes.append((slice(y0, min(h, y0 + CAL_TILE)), slice(x0, min(w, x0 + CAL_TILE))))
    return boxes


def _starts(length: int) -> list[int]:
    step = CORE - OVERLAP
    starts = list(range(0, max(1, length - CORE) + 1, step))
    if starts[-1] + CORE < length:
        starts.append(length - CORE)
    return starts


def feather(n: int, first: bool, last: bool) -> np.ndarray:
    """1D blend weight of a tile side: ramps over OVERLAP where a neighbour overlaps it (never 0)."""
    ramp = (np.arange(OVERLAP) + 0.5) / OVERLAP
    weight = np.ones(n, dtype=np.float32)
    k = min(OVERLAP, n)
    if not first:
        weight[:k] = np.minimum(weight[:k], ramp[:k])
    if not last:
        weight[n - k :] = np.minimum(weight[n - k :], ramp[:k][::-1])
    return weight


def render_area(products: dict[str, np.ndarray], geometry: Geometry, names: list[str], *, notch: int = 0,
                speckle: int = 0, scratch=None, log=print):
    """Yield (name, uint8 RGB (H, W, 3)) for every named filter over the whole area; black where there is no data.

    ``notch`` / ``speckle`` pre-filter every filter that does not name its own pre-filters. One filter is held in
    memory at a time; with ``scratch`` (a folder) its accumulators are memory-mapped files there instead.
    """
    h, w = products["valid"].shape
    visualizers = [resolve(n) for n in names]
    animated = [v.name for v in visualizers if v.animated]
    if animated:
        raise ValueError(f"animated filters cannot be written as still images: {animated}")
    windows = calibration_windows(h, w)
    one_piece = max(h, w) <= ONE_PIECE
    rows, cols = ([0], [0]) if one_piece else (_starts(h), _starts(w))
    size_y, size_x = (h, w) if one_piece else (CORE, CORE)
    jobs = [(i, j, y, x) for i, y in enumerate(rows) for j, x in enumerate(cols)]
    valid = np.asarray(products["valid"], dtype=bool)
    for vis in visualizers:
        pre = prefilters(vis, notch, speckle)
        calibration = calibrate_filter(vis, [tile_from_products(products, geometry, win) for win in windows], *pre)
        if scratch is None:
            acc, total = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
        else:
            acc = np.lib.format.open_memmap(scratch / "acc.npy", mode="w+", dtype=np.float32, shape=(h, w, 3))
            total = np.lib.format.open_memmap(scratch / "total.npy", mode="w+", dtype=np.float32, shape=(h, w))
        for k, (i, j, y, x) in enumerate(jobs):
            th, tw = min(size_y, h - y), min(size_x, w - x)
            context = 0 if one_piece else CONTEXT
            y0, x0 = max(0, y - context), max(0, x - context)
            y1, x1 = min(h, y + th + context), min(w, x + tw + context)
            tile = tile_from_products(products, geometry, (slice(y0, y1), slice(x0, x1)))
            core = (slice(y - y0, y - y0 + th), slice(x - x0, x - x0 + tw))
            weight = np.outer(feather(th, i == 0, i == len(rows) - 1), feather(tw, j == 0, j == len(cols) - 1))
            acc[y : y + th, x : x + tw] += render_tile(vis, tile, calibration, *pre)[core] * weight[..., None]
            total[y : y + th, x : x + tw] += weight
            if not one_piece:
                log(f"{vis.name}: tile {k + 1}/{len(jobs)}")
        rgb = np.empty((h, w, 3), np.uint8)
        for y in range(0, h, 1024):  # row strips keep the float temporaries small
            strip = slice(y, y + 1024)
            values = acc[strip] / np.maximum(total[strip], 1e-6)[..., None]
            values[~valid[strip]] = 0
            rgb[strip] = (np.clip(values, 0, 1) * 255 + 0.5).astype(np.uint8)
        del acc, total
        yield vis.name, rgb
