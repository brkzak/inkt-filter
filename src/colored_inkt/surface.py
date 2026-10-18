"""From a surface volume to the surface products every filter reads.

The filters (``visualizers``, ``research``, ``generalise``) work on a ``Tile``: a block of the CT around the
segment, from which each pixel's papyrus face is found and the stack is resampled at fixed depths from it. A
segment area is too large for one tile, so this module

1. reads the area block by block (512 px blocks with 48 px of context, as the original viewer did),
2. bins each block in-plane if asked, marks the pixels that have data (``valid``: not zero in every layer),
3. resamples the layers to 2.4 µm spacing along the normal when the scan's layers are much coarser or finer
   (the filters' depth lengths are written in 2.4 µm planes; a 7.91 µm scan is sampled every 2.4 µm, which adds
   no detail but keeps every depth parameter at its physical size),
4. finds the face and keeps the compact products (``visualizers.PRODUCTS`` plus ``valid``), stitched over the area.

Rendering a filter then needs only the products (``Tile.from_products``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colored_inkt.visualizers import FLAT_D, PRODUCTS, Tile
from colored_inkt.volume import Volume

PLANE_UM = 2.4  # depth spacing the filters are written for
RESAMPLE_TOLERANCE = 0.1  # scans within 10 % of 2.4 µm are used as they are (the open data has 2.2–2.4 µm)
BLOCK = 512
MARGIN = 48  # context around a block; the widest face-search filters need ~40 px


@dataclass(frozen=True)
class Geometry:
    """How the stack's voxels map to physical lengths, and which way the ink faces.

    ``layer_um``: spacing of the layers (along the surface normal).
    ``pixel_um``: in-plane pixel size of the source.
    ``bin``: in-plane binning applied when reading (1 = none).
    ``side``: +1 if the inked face of the sheet looks towards higher layer numbers, -1 if towards lower.
    """

    layer_um: float
    pixel_um: float
    bin: int = 1
    side: int = 1

    def __post_init__(self) -> None:
        if self.layer_um <= 0 or self.pixel_um <= 0:
            raise ValueError(f"voxel sizes must be positive, got layer {self.layer_um} µm, pixel {self.pixel_um} µm")
        if self.bin < 1:
            raise ValueError(f"bin must be >= 1, got {self.bin}")
        if self.side not in (1, -1):
            raise ValueError(f"side must be +1 or -1, got {self.side}")

    @property
    def xy_scale(self) -> float:
        """In-plane pixel size (after binning) in units of 2.4 µm: ``Tile.xy_scale``."""
        return self.pixel_um * self.bin / PLANE_UM

    @property
    def resamples(self) -> bool:
        return abs(self.layer_um / PLANE_UM - 1) > RESAMPLE_TOLERANCE

    def planes(self, layers: int) -> np.ndarray:
        """Positions (in layer units) of the 2.4 µm planes, centred on the middle layer."""
        if not self.resamples:
            return np.arange(layers, dtype=np.float64)
        mid = (layers - 1) / 2
        half = int(np.floor(mid * self.layer_um / PLANE_UM))
        return mid + np.arange(-half, half + 1) * PLANE_UM / self.layer_um


def bin_block(raw: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """(binned stack float32, valid) of a raw (L, h, w) block; empty pixels (all layers 0) do not count."""
    valid = (raw > 0).any(axis=0)
    if factor == 1:
        return raw.astype(np.float32), valid
    layers, h, w = raw.shape
    hb, wb = h // factor, w // factor
    cells = raw[:, : hb * factor, : wb * factor].reshape(layers, hb, factor, wb, factor).astype(np.float32)
    weight = valid[: hb * factor, : wb * factor].reshape(hb, factor, wb, factor).astype(np.float32)
    count = weight.sum(axis=(1, 3))
    sums = (cells * weight[None]).sum(axis=(2, 4))
    return (sums / np.maximum(count, 1)[None]).astype(np.float32), count > 0


def resample_layers(stack: np.ndarray, geometry: Geometry) -> np.ndarray:
    """``stack`` (L, h, w) linearly resampled at 2.4 µm planes (unchanged when the scan is close to 2.4 µm)."""
    if not geometry.resamples:
        return stack
    z = geometry.planes(stack.shape[0])
    k = np.clip(np.floor(z).astype(np.intp), 0, stack.shape[0] - 2)
    f = (z - k).astype(np.float32)[:, None, None]
    return stack[k] * (1 - f) + stack[k + 1] * f


def shift_depth(stack: np.ndarray, depth: int, side: int) -> np.ndarray:
    """``stack`` as seen from a segment ``depth`` planes deeper into the sheet (edge planes repeat)."""
    planes = np.clip(np.arange(stack.shape[0]) - side * depth, 0, stack.shape[0] - 1)
    return stack[planes]


class Area:
    """A box of a volume, read block by block as float planes at 2.4 µm spacing.

    ``box`` = (y, x, h, w) in *binned* pixels of the volume.
    """

    def __init__(self, volume: Volume, box: tuple[int, int, int, int], geometry: Geometry):
        y, x, h, w = (int(v) for v in box)
        if h <= 0 or w <= 0:
            raise ValueError(f"empty box {box}")
        self.volume, self.box, self.geometry = volume, (y, x, h, w), geometry

    @property
    def shape(self) -> tuple[int, int]:
        return self.box[2], self.box[3]

    def read(self, y: int, x: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """(planes float32 0 … 255, valid) of the area-relative window (zero and invalid outside the volume)."""
        b = self.geometry.bin
        y0, x0 = self.box[0] + y, self.box[1] + x
        raw = self.volume.read(y0 * b, x0 * b, h * b, w * b)
        stack, valid = bin_block(raw, b)
        stack *= np.float32(self.volume.value_scale)
        return resample_layers(stack, self.geometry), valid


def block_products(stack: np.ndarray, valid: np.ndarray, geometry: Geometry, *, depth: int = 0,
                   fixed: bool = False) -> dict[str, np.ndarray]:
    """The surface products of one block (planes at 2.4 µm, with its context ring).

    ``depth`` ≠ 0 (planes, + = into the sheet) simulates a segment off the papyrus face: with ``fixed`` False the
    segment itself is moved and the face is searched again around it; with ``fixed`` True the face found stays and
    every product reads the layers ``depth`` deeper (a wrong face, or fixed layers as VC3D shows them).
    """
    if depth and not fixed:
        stack = shift_depth(stack, depth, geometry.side)
    tile = Tile(stack, side=geometry.side, xy_scale=geometry.xy_scale, valid=valid)
    if depth and fixed:
        tile.__dict__["height"] = tile.height - geometry.side * depth
    return tile.products()


def product_shapes(h: int, w: int) -> dict[str, tuple[int, ...]]:
    shapes = {k: ((len(FLAT_D), h, w) if k == "flat" else (h, w)) for k in PRODUCTS}
    return shapes


def compute_products(area: Area, *, out: dict[str, np.ndarray] | None = None, depth: int = 0, fixed: bool = False,
                     progress=None, skip=None) -> dict[str, np.ndarray]:
    """Surface products of the whole area, block by block, plus ``valid`` (bool).

    ``out`` may hold preallocated (memory-mapped) arrays to fill; ``skip(by, bx)`` may return True for blocks that
    are already done (resuming); ``progress(i, n)`` is called after each block.
    """
    h, w = area.shape
    if out is None:
        out = {k: np.zeros(s, np.float32) for k, s in product_shapes(h, w).items()}
        out["valid"] = np.zeros((h, w), bool)
    blocks = [(by, bx) for by in range(0, h, BLOCK) for bx in range(0, w, BLOCK)]
    for i, (by, bx) in enumerate(blocks):
        if skip is not None and skip(by, bx):
            continue
        bh, bw = min(BLOCK, h - by), min(BLOCK, w - bx)
        stack, valid = area.read(by - MARGIN, bx - MARGIN, bh + 2 * MARGIN, bw + 2 * MARGIN)
        core = (slice(MARGIN, MARGIN + bh), slice(MARGIN, MARGIN + bw))
        out["valid"][by : by + bh, bx : bx + bw] = valid[core]
        if valid[core].any():
            for name, values in block_products(stack, valid, area.geometry, depth=depth, fixed=fixed).items():
                out[name][..., by : by + bh, bx : bx + bw] = values[(..., *core)]
        if progress:
            progress(i + 1, len(blocks))
    return out


def tile_from_products(products: dict[str, np.ndarray], geometry: Geometry,
                       window: tuple[slice, slice] | None = None) -> Tile:
    """A ``Tile`` of (a window of) stored products, with its mask of pixels that have data."""
    window = window or (slice(None), slice(None))
    picked = {k: np.asarray(products[k][(..., *window)], dtype=np.float32) for k in PRODUCTS}
    valid = np.asarray(products["valid"][window], dtype=bool)
    if not valid.any():
        valid = None
    return Tile.from_products(picked, side=geometry.side, xy_scale=geometry.xy_scale, valid=valid)
