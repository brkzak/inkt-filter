"""Visualizers that make ink easier to see on raw CT surface volumes.

None of these is trained on ink labels. Each one turns a surface volume tile
(Z, H, W) into an RGB image (or a short animation) from a physical idea of how
ink differs from bare papyrus:

* Carbon ink is a thin deposit **on** the papyrus surface: slightly denser,
  slightly raised, and it fills the grooves between fibers.
* It dries and **cracks** ("crackle"): fine polygonal texture that does not
  follow the fiber directions.
* Papyrus texture is dominated by two **fiber families** at ~90° to each
  other; strokes and cracks have edges at every angle.

A visualizer has three parts, so that it ports to VC3D's renderer:

``fit(tiles)``        per-volume parameters from a few unlabeled tiles (a PCA
                      basis, a density threshold). Optional.
``channels(tile)``    float maps (H, W): per-pixel depth-column operations
                      along the surface normal, then 2D image filters.
``compose(ch, tile)`` RGB in [0, 1] from the channels after per-volume
                      normalisation (VC3D's window/level), or a list of
                      frames for an animation.

Depth convention: ``d`` is the distance in voxels from the papyrus surface on
the ink side, positive into the sheet (``d < 0`` is the air or gap in front
of the ink face). ``Tile.side`` says which way the ink face looks along z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------
# Colour maps (anchor tables, linearly interpolated; no matplotlib needed)
# ----------------------------------------------------------------------------
_ANCHORS = {
    "gray": [(0, 0, 0), (255, 255, 255)],
    "inferno": [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
                (245, 125, 21), (250, 193, 39), (252, 255, 164)],
    "magma": [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122),
              (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142), (38, 130, 142),
                (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37)],
    "cividis": [(0, 32, 77), (0, 52, 110), (57, 73, 108), (87, 94, 109), (112, 113, 115),
                (138, 134, 120), (166, 157, 117), (197, 181, 107), (230, 206, 90), (255, 233, 69)],
    "turbo": [(48, 18, 59), (70, 107, 227), (40, 187, 236), (49, 242, 153), (162, 252, 60),
              (237, 208, 58), (251, 128, 34), (208, 47, 5), (122, 4, 3)],
    "hot": [(0, 0, 0), (230, 0, 0), (255, 210, 0), (255, 255, 255)],
    "coolwarm": [(59, 76, 192), (141, 176, 254), (221, 221, 221), (244, 154, 123), (180, 4, 38)],
    "terrain": [(40, 60, 150), (60, 160, 170), (120, 190, 100), (230, 220, 140),
                (170, 110, 60), (250, 250, 250)],
    "ink": [(25, 20, 15), (90, 60, 35), (190, 140, 80), (250, 225, 170)],
    "sepia": [(20, 12, 6), (112, 66, 20), (204, 160, 100), (255, 244, 214)],
    "gray→yellow": [(30, 30, 30), (120, 120, 120), (255, 220, 0)],
    "blue→black→red": [(40, 120, 255), (0, 0, 0), (255, 60, 40)],
}
COLORMAPS = tuple(_ANCHORS)  # the viewer's colour schemes (sent to the browser)


def colormap(values: np.ndarray, name: str) -> np.ndarray:
    """Map [0, 1] values to RGB [0, 1] through a named anchor table."""
    anchors = np.asarray(_ANCHORS[name], dtype=np.float32) / 255.0
    x = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    i = np.minimum(x.astype(np.int32), len(anchors) - 2)
    f = (x - i)[..., None]
    return anchors[i] * (1 - f) + anchors[i + 1] * f


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    rgb = np.zeros(h.shape + (3,), dtype=np.float32)
    for k, (r, g, b) in enumerate(table):
        m = i == k
        rgb[m, 0], rgb[m, 1], rgb[m, 2] = r[m], g[m], b[m]
    return rgb


def gray(values: np.ndarray) -> np.ndarray:
    return np.repeat(np.clip(values, 0, 1)[..., None], 3, axis=-1)


def overlay(base: np.ndarray, heat: np.ndarray, cmap: str = "inferno", strength: float = 0.75):
    """Gray ``base`` with ``heat`` coloured on top where it is high."""
    alpha = (np.clip(heat, 0, 1) * strength)[..., None]
    return gray(base) * (1 - alpha) + colormap(heat, cmap) * alpha


# ----------------------------------------------------------------------------
# Small image helpers
# ----------------------------------------------------------------------------
# An axis whose Gaussian is at least this wide (σ in pixels) is convolved through the FFT instead of directly.
# The direct sum costs O(σ) per pixel: generalise's 1.5 mm local statistics (σ ≈ 160 px on a 9.4 µm scan) took
# ~0.4 s per 864 px image, ~0.04 s through the FFT. On such images the two break even near σ = 8 px (M5, 2026-10-17).
FFT_SIGMA_PX = 8.0


def gauss(a: np.ndarray, sigma) -> np.ndarray:
    """``scipy.ndimage.gaussian_filter(a, sigma, mode="nearest")``, with its wide axes summed through the FFT.

    Axis by axis as scipy does (axes with σ ≤ 1e-15 skipped, each pass rounded to ``a``'s dtype). An axis with
    σ ≥ ``FFT_SIGMA_PX`` uses the same kernel and the same edge (``_gauss_fft_axis``); the sum is then evaluated
    in float64 by the FFT, so it differs from scipy's direct float64 sum by round-off (~1e-16 of the input's
    magnitude) before the float32 rounding: at most one unit in the last place where the result is not tiny.
    """
    a = np.asarray(a)
    sigmas = [float(s) for s in np.broadcast_to(np.asarray(sigma, dtype=np.float64), (a.ndim,))]
    if not np.issubdtype(a.dtype, np.floating) or max(sigmas) < FFT_SIGMA_PX:
        return ndimage.gaussian_filter(a, sigma, mode="nearest")
    out = a
    for axis, s in enumerate(sigmas):
        if s <= 1e-15:
            continue
        if s >= FFT_SIGMA_PX:
            out = _gauss_fft_axis(out, s, axis)
        else:
            out = ndimage.gaussian_filter1d(out, s, axis=axis, mode="nearest")
    return out


def _gauss_fft_axis(a: np.ndarray, sigma: float, axis: int, truncate: float = 4.0) -> np.ndarray:
    """``ndimage.gaussian_filter1d(a, sigma, axis, mode="nearest")`` as a linear convolution by the FFT.

    The kernel is scipy's (radius ``int(truncate σ + 0.5)``, normalised to sum 1) and ``mode="nearest"`` is an
    edge padding by that radius, so only the summation differs. A non-negative input gives a non-negative output
    in the direct sum; the FFT's round-off can dip just below 0 where the true value is ~0 (far from any data), so
    such an output is clipped at 0: a square root of it (``generalise.local_z``) would otherwise be NaN.
    """
    from scipy import fft

    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 / (sigma * sigma) * x**2)
    kernel = kernel / kernel.sum()
    n = a.shape[axis]
    padded = np.pad(a.astype(np.float64), [(radius, radius) if i == axis else (0, 0) for i in range(a.ndim)],
                    mode="edge")
    length = fft.next_fast_len(n + 4 * radius, real=True)  # no wrap-around: padded + kernel - 1 ≤ length
    shape = [1] * a.ndim
    shape[axis] = -1
    full = fft.irfft(fft.rfft(padded, length, axis=axis) * fft.rfft(kernel, length).reshape(shape), length, axis=axis)
    # Output i is centred on padded sample i + radius, i.e. full-convolution index i + 2 * radius.
    out = np.take(full, np.arange(2 * radius, 2 * radius + n), axis=axis)
    if a.size and a.min() >= 0:
        np.maximum(out, 0, out=out)
    return out.astype(a.dtype)


def dog(a: np.ndarray, s1: float, s2: float) -> np.ndarray:
    return gauss(a, s1) - gauss(a, s2)


def sample_depth(volume: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear interpolation of ``volume`` (Z, H, W) at per-pixel depth ``z`` (H, W)."""
    depth = volume.shape[0]
    z = np.clip(z, 0, depth - 1.0001)
    z0 = np.floor(z).astype(np.intp)
    f = (z - z0).astype(np.float32)
    a = np.take_along_axis(volume, z0[None], 0)[0]
    b = np.take_along_axis(volume, z0[None] + 1, 0)[0]
    return a * (1 - f) + b * f


def structure_tensor(image: np.ndarray, sigma_grad: float, sigma_int: float):
    """(Jxx, Jxy, Jyy) of the smoothed gradient, integrated over ``sigma_int``."""
    gy = ndimage.gaussian_filter(image, sigma_grad, order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(image, sigma_grad, order=(0, 1), mode="nearest")
    return (gauss(gx * gx, sigma_int), gauss(gx * gy, sigma_int), gauss(gy * gy, sigma_int)), (gx, gy)


def dominant_orientations(image: np.ndarray, *, cross_tolerance_deg: float = 25.0, xy_scale: float = 1.0) -> np.ndarray:
    """The two strongest texture orientations (radians in [0, π)) of fiber texture.

    Measured in the Fourier power spectrum, where each fiber family is a line
    through the origin at its gradient orientation (perpendicular to the
    fibers). Per-pixel gradients would mix two crossing families into
    diagonals. The two layers of a sheet are laid roughly perpendicular, so
    the second family is searched within ``cross_tolerance_deg`` of the first
    plus 90°. ``xy_scale`` is the pixel size in units of 2.4 µm; the analysed
    band (7–96 µm periods) is kept in physical units.
    """
    h, w = image.shape
    detail = image - gauss(image, 8 / xy_scale)
    window = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2(detail * window))) ** 2
    fy, fx = np.mgrid[-(h // 2) : h - h // 2, -(w // 2) : w - w // 2].astype(np.float32)
    radius = np.hypot(fx / w, fy / h)  # cycles per pixel
    band = (radius > xy_scale / 40) & (radius < min(0.45, xy_scale / 3))
    theta = np.arctan2(fy / h, fx / w) % np.pi
    hist, _ = np.histogram(theta[band], bins=180, range=(0, np.pi), weights=power[band])
    hist = ndimage.gaussian_filter1d(hist, 2, mode="wrap")
    first = int(np.argmax(hist))
    distance = np.abs((np.arange(180) - first + 90) % 180 - 90)  # 0 … 90 from the first
    second = int(np.argmax(np.where(distance >= 90 - cross_tolerance_deg, hist, -1)))
    return np.deg2rad(np.array([first, second], dtype=np.float32) + 0.5)


def angular_distance(theta: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance between orientations modulo π (radians, 0 … π/2)."""
    d = np.abs((theta[..., None] - reference + np.pi / 2) % np.pi - np.pi / 2)
    return d.min(axis=-1)


def hillshade(elevation: np.ndarray, azimuth_deg: float, altitude_deg: float, exaggeration: float):
    """Lambert shading of a height field lit from (azimuth, altitude)."""
    gy, gx = np.gradient(elevation * exaggeration)
    norm = np.sqrt(gx * gx + gy * gy + 1)
    nx, ny, nz = -gx / norm, -gy / norm, 1 / norm
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    lx, ly, lz = np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)
    return np.clip(nx * lx + ny * ly + nz * lz, 0, 1)


# ----------------------------------------------------------------------------
# Tile: one surface-volume block with its papyrus surface
# ----------------------------------------------------------------------------
FLAT_D = np.arange(-8, 17)  # depths of the flattened stack
# Bump when any visualizer's output changes: keys the calibration and render caches. 5: wide Gaussians through the
# FFT (``gauss``; values change by round-off only).
VERSION = 5


PRODUCTS = ("mid_plane", "window_mean", "window_max", "coarse_height", "height", "flat")


class Tile:
    """A (Z, H, W) surface-volume block and lazily derived surface quantities.

    Depth is always in voxels of the volume's z axis (2.4 µm for every scan
    here, also in pyramid levels, which shrink only in-plane). ``xy_scale``
    is the in-plane pixel size in units of 2.4 µm (1 at level 0, 4 at level
    2); every in-plane filter length below is written for 2.4 µm pixels and
    divided by it, so a method looks at the same physical scale everywhere.

    ``valid`` (bool, H x W) marks the pixels that have data when part of the
    block is empty (a skewed segment rendered into square blocks, its empty
    pixels 0). The empty pixels then take the values of the nearest pixel
    with data (the volume, or every product of a tile made from products), so
    no filter sees a step to 0 at the edge of the data; statistics over the
    tile (``pixels``: percentiles, fits, means) use only the pixels with data,
    and the Fourier filters see the empty part as the mean (``filled``).
    None (or a mask without empty pixels) = every pixel, as before.
    """

    def __init__(self, ct: np.ndarray, *, side: int = 1, search: tuple[int, int] = (-24, 30),
                 xy_scale: float = 1.0, valid: np.ndarray | None = None):
        if ct.ndim != 3:
            raise ValueError("expected a (Z, H, W) block")
        self.ct = ct
        self.side = 1 if side >= 0 else -1
        self.search = search
        self.xy_scale = float(xy_scale)
        self.valid = _check_valid(valid, ct.shape[1:])

    @classmethod
    def from_products(cls, products: dict[str, np.ndarray], *, side: int = 1, xy_scale: float = 1.0,
                      valid: np.ndarray | None = None) -> "Tile":
        """A tile without its CT, from precomputed surface products (see ``PRODUCTS``)."""
        tile = cls.__new__(cls)
        tile.ct = None
        tile.side = 1 if side >= 0 else -1
        tile.search = None
        tile.xy_scale = float(xy_scale)
        tile.__dict__.update({k: np.asarray(products[k], dtype=np.float32) for k in PRODUCTS})
        tile.valid = _check_valid(valid, tile.height.shape)
        if tile.valid is not None:
            tile.__dict__.update({k: nearest_fill(tile.__dict__[k], tile.valid) for k in PRODUCTS})
        return tile

    def products(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in PRODUCTS}

    @property
    def shape2d(self) -> tuple[int, int]:
        return tuple(self.ct.shape[1:]) if self.ct is not None else tuple(self.height.shape)

    def xy(self, length: float) -> float:
        """An in-plane length given in 2.4 µm pixels, in this tile's pixels."""
        return length / self.xy_scale

    def xy_size(self, size: int) -> int:
        """An odd filter window of ``size`` 2.4 µm pixels, in this tile's pixels (≥ 3)."""
        n = int(round(size / self.xy_scale))
        return max(3, n | 1)

    def pixels(self, values: np.ndarray, step: int = 1) -> np.ndarray:
        """``values[..., ::step, ::step]`` restricted to the pixels with data: the sample for a statistic.

        Unchanged (not flattened) without a mask; with one, the in-plane axes become one axis of the pixels
        with data. Any leading axes (depth) are kept.
        """
        return data_pixels(values, self.valid, step)

    def valid_mean(self, image: np.ndarray) -> float:
        """Mean of an in-plane ``image`` over the pixels with data."""
        if self.valid is None or not self.valid.any():
            return float(image.mean())
        return float(image[self.valid].mean())

    def filled(self, image: np.ndarray) -> np.ndarray:
        """``image`` with the empty pixels set to the mean of the others (no step at the edge of the data)."""
        if self.valid is None or not self.valid.any():
            return image
        return np.where(self.valid, image, np.float32(self.valid_mean(image)))

    @property
    def mid(self) -> int:
        return self.ct.shape[0] // 2

    @cached_property
    def volume(self) -> np.ndarray:
        return self._fill(self.ct.astype(np.float32))

    @cached_property
    def _fill_index(self) -> tuple[np.ndarray, np.ndarray]:
        """(iy, ix) of the nearest pixel with data, for every pixel (``nearest_fill``), found once per mask."""
        _, (iy, ix) = ndimage.distance_transform_edt(~self.valid, return_indices=True)
        return iy, ix

    def _fill(self, values: np.ndarray) -> np.ndarray:
        """``nearest_fill(values, self.valid)``, reusing the tile's nearest-pixel index."""
        if self.valid is None or not self.valid.any():
            return values
        iy, ix = self._fill_index
        return values[..., iy, ix]

    @cached_property
    def coarse_height(self) -> np.ndarray:
        """z of the ink-side papyrus interface, smoothed over ~60 px.

        The interface is the steepest *fall* of the depth profile when moving
        from the sheet into the air or gap in front of the ink face.
        """
        smooth = gauss(self.volume, (2, self.xy(6), self.xy(6)))
        fall = -self.side * np.diff(smooth, axis=0)  # fall[k] between planes k and k+1
        lo, hi = self.search if self.side > 0 else (-self.search[1], -self.search[0])
        lo = max(0, self.mid + lo)
        hi = min(fall.shape[0], self.mid + hi)
        k = np.argmax(fall[lo:hi], axis=0) + lo + 0.5
        k = ndimage.median_filter(k, size=self.xy_size(31), mode="nearest")
        return gauss(k.astype(np.float32), self.xy(8))

    @cached_property
    def height(self) -> np.ndarray:
        """Sub-voxel interface depth: steepest fall within ±3 of the coarse surface."""
        light = gauss(self.volume, (1.0, self.xy(1.5), self.xy(1.5)))
        fall = -self.side * np.diff(light, axis=0)
        depth = fall.shape[0]
        offsets = np.arange(-3, 4)
        z = np.clip(np.round(self.coarse_height - 0.5).astype(np.intp)[None] + offsets[:, None, None], 1, depth - 2)
        values = np.take_along_axis(fall, z, 0)
        best = np.argmax(values, axis=0)
        zb = np.take_along_axis(z, best[None], 0)[0]
        f0 = np.take_along_axis(fall, (zb - 1)[None], 0)[0]
        f1 = np.take_along_axis(fall, zb[None], 0)[0]
        f2 = np.take_along_axis(fall, (zb + 1)[None], 0)[0]
        denom = f0 - 2 * f1 + f2
        shift = np.where(np.abs(denom) > 1e-6, 0.5 * (f0 - f2) / np.where(denom == 0, 1, denom), 0)
        fine = zb + 0.5 + np.clip(shift, -0.5, 0.5)
        return ndimage.median_filter(fine, size=self.xy_size(5), mode="nearest")

    @cached_property
    def elevation(self) -> np.ndarray:
        """Height of the ink face towards the viewer (voxels), lightly smoothed."""
        return gauss(self.side * self.height, self.xy(1.0))

    @cached_property
    def relief(self) -> np.ndarray:
        """Elevation minus its ~40 µm average: local relief without the tile-scale tilt."""
        return self.elevation - gauss(self.elevation, self.xy(16))

    @cached_property
    def shade_relief(self) -> np.ndarray:
        """Relief with height and in-plane distance in the same unit, for lighting."""
        return self.relief / self.xy_scale

    def at_depth(self, d: float, *, smooth: bool = False) -> np.ndarray:
        source = self.smooth_volume if smooth else self.volume
        return sample_depth(source, self.height - self.side * d)

    @cached_property
    def smooth_volume(self) -> np.ndarray:
        return gauss(self.volume, (0.7, self.xy(1.0), self.xy(1.0)))

    def _make_flat_plane(self, i: int) -> np.ndarray:
        """Flattened plane ``i`` (depth ``FLAT_D[i]``) computed from scratch."""
        return self.at_depth(float(FLAT_D[i]))

    def _flat_plane(self, i: int) -> np.ndarray:
        """Flattened plane ``i``: from ``flat`` once it exists, else computed on first use and kept.

        Most methods read a few planes (``band``, ``planes``) and never the whole stack; a derived tile
        (``notched``, ``despeckled``) then filters only those.
        """
        if "flat" in self.__dict__:
            return self.flat[i]
        cache = self.__dict__.setdefault("_flat_planes", {})
        if i not in cache:
            cache[i] = self._make_flat_plane(i)
        return cache[i]

    @cached_property
    def flat(self) -> np.ndarray:
        """(len(FLAT_D), H, W) stack resampled at fixed depths from the surface."""
        done = self.__dict__.pop("_flat_planes", {})
        return np.stack([done[i] if i in done else self._make_flat_plane(i) for i in range(len(FLAT_D))])

    def planes(self, d0: int, d1: int) -> np.ndarray:
        """Flattened planes d0 … d1 (inclusive) as a (k, H, W) stack: ``flat``'s slice, computing only those."""
        index = range(*slice(int(d0 - FLAT_D[0]), int(d1 - FLAT_D[0]) + 1).indices(len(FLAT_D)))
        if "flat" in self.__dict__ or not len(index):
            return self.flat[index.start : index.stop]
        return np.stack([self._flat_plane(i) for i in index])

    def band(self, d0: int, d1: int) -> np.ndarray:
        """Mean of the flattened planes d0 … d1 (inclusive)."""
        return self.planes(d0, d1).mean(axis=0)

    @cached_property
    def skin(self) -> np.ndarray:
        """The outer ~10 µm of the sheet, where an ink deposit lies."""
        return self.band(0, 4)

    @cached_property
    def body(self) -> np.ndarray:
        return self.band(8, 16)

    @cached_property
    def fiber_orientations(self) -> np.ndarray:
        return dominant_orientations(self.filled(self.skin), xy_scale=self.xy_scale)

    def notched(self, half_width_deg: float = 7.0) -> "Tile":
        """This tile with the fiber families notch-filtered out of every in-plane product.

        Each flattened plane, the baseline planes and the height map are
        filtered (``notch_filter``) with the fiber orientations of the original
        skin, before any visualizer runs. The coarse surface is kept. Empty
        pixels (``valid``) are the mean for the filter, then the nearest
        filtered pixel again (``from_products``). Each product is filtered when
        a method first reads it (``_DerivedTile``).
        """
        orientations = self.fiber_orientations

        def filtered(image):
            mean = self.valid_mean(image)
            return notch_filter(image - mean, orientations, half_width_deg=half_width_deg, valid=self.valid) + mean

        tile = _DerivedTile(self, filtered, keep=("coarse_height",))
        tile.__dict__["fiber_orientations"] = orientations
        return tile

    def despeckled(self, size_um: float = 50.0) -> "Tile":
        """This tile with small dense specks (mineral grains) removed from every in-plane product.

        A speck is a bright feature shorter than ``size_um`` in every direction
        that stands well above its surroundings (``despeckle``); fibers, strokes
        and larger patches are kept. The threshold is 0.5 × (p99 − p50) of the
        face slab, so it follows the scan's contrast (on PHerc0841 it changes
        ~6 % of the face voxels at 50 µm). Height maps are kept. Each product is
        despeckled when a method first reads it (``_DerivedTile``).
        """
        length = self.xy_size(size_um / 2.4)
        face = self.pixels(self.planes(-2, 6), 2)  # the planes d = −2 … 6, as the dense-tail fit
        threshold = 0.5 * float(np.percentile(face, 99) - np.percentile(face, 50))
        tile = _DerivedTile(self, lambda image: despeckle(image, length, threshold), keep=("coarse_height", "height"))
        tile.__dict__["fiber_orientations"] = self.fiber_orientations
        return tile

    def window(self, n: int) -> np.ndarray:
        """Planes mid-n … mid+n of the unflattened volume (what VC3D composites)."""
        return self.volume[max(0, self.mid - n) : self.mid + n + 1]

    @cached_property
    def mid_plane(self) -> np.ndarray:
        return self.volume[self.mid]

    @cached_property
    def window_mean(self) -> np.ndarray:
        return self.window(8).mean(axis=0)

    @cached_property
    def window_max(self) -> np.ndarray:
        return self.window(8).max(axis=0)


class _DerivedTile(Tile):
    """A tile whose surface products are an in-plane filter of another tile's, each filtered when first read.

    ``Tile.notched`` and ``Tile.despeckled`` filter every in-plane product, but a method reads only some of them
    (the generalisation methods read 11 of the 25 flattened planes and no baseline plane), and every product, every
    flattened plane, is filtered on its own. So each is filtered on first use and then filled from the nearest
    pixel with data, exactly as ``Tile.from_products`` fills the eagerly filtered products: the values are the
    same, only the unread products are never computed. ``keep`` names the products taken over unfiltered (still
    filled, as ``from_products`` fills them).
    """

    def __init__(self, parent: Tile, transform: Callable[[np.ndarray], np.ndarray], *, keep: tuple[str, ...]):
        self.ct = None
        self.side = parent.side
        self.search = None
        self.xy_scale = parent.xy_scale
        self.valid = parent.valid
        self._parent = parent
        self._transform = transform
        self._keep = keep

    def _derive(self, name: str, values: np.ndarray) -> np.ndarray:
        out = values if name in self._keep else self._transform(values)
        return self._fill(np.asarray(out, dtype=np.float32))

    @property
    def shape2d(self) -> tuple[int, int]:
        return self._parent.shape2d

    @property
    def _fill_index(self) -> tuple[np.ndarray, np.ndarray]:
        return self._parent._fill_index  # the same mask: no second distance transform

    def _make_flat_plane(self, i: int) -> np.ndarray:
        return self._derive("flat", self._parent._flat_plane(i))

    @cached_property
    def coarse_height(self) -> np.ndarray:
        return self._derive("coarse_height", self._parent.coarse_height)

    @cached_property
    def height(self) -> np.ndarray:
        return self._derive("height", self._parent.height)

    @cached_property
    def mid_plane(self) -> np.ndarray:
        return self._derive("mid_plane", self._parent.mid_plane)

    @cached_property
    def window_mean(self) -> np.ndarray:
        return self._derive("window_mean", self._parent.window_mean)

    @cached_property
    def window_max(self) -> np.ndarray:
        return self._derive("window_max", self._parent.window_max)


def _check_valid(valid: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    """The mask as bool, or None when it has no empty pixel (then the tile renders exactly as without one)."""
    if valid is None:
        return None
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != tuple(shape):
        raise ValueError(f"valid mask {valid.shape} does not match the tile's {tuple(shape)} pixels")
    return None if valid.all() else valid


def nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``values`` (..., H, W) with every empty pixel set to the value of the nearest pixel with data."""
    if not valid.any():
        return values
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return values[..., iy, ix]


def masked_gauss(a: np.ndarray, sigma, valid: np.ndarray | None) -> np.ndarray:
    """``gauss`` weighing only the pixels with data (normalised convolution); plain ``gauss`` without a mask."""
    if valid is None or not valid.any():
        return gauss(a, sigma)
    weight = valid.astype(np.float32)
    return gauss(a * weight, sigma) / np.maximum(gauss(weight, sigma), 1e-6)


def data_pixels(values: np.ndarray, valid: np.ndarray | None, step: int = 1) -> np.ndarray:
    """``values[..., ::step, ::step]`` on the pixels with data (``Tile.pixels``); as is without a mask."""
    strided = values[..., ::step, ::step]
    if valid is None:
        return strided
    keep = valid[::step, ::step]
    return strided[..., keep] if keep.any() else strided.reshape(*strided.shape[:-2], -1)


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
Channels = dict[str, np.ndarray]


@dataclass
class Visualizer:
    name: str
    title: str
    family: str
    idea: str
    channels: Callable[[Tile, dict], Channels]
    compose: Callable[[Channels, Tile, dict], np.ndarray | list[np.ndarray]]
    # "robust" → p1..p99 to 0..1; "diverging" → symmetric around the median; "unit" → as is;
    # (lo, hi) → that fixed range to 0..1 (the same on every region)
    norms: dict[str, str | tuple[float, float]] = field(default_factory=dict)
    fit: Callable[[list[Tile]], dict] | None = None
    animated: bool = False
    version: int = 0  # set from VERSION at registration
    variant_of: str | None = None  # the method this one is a variation of
    # A fixed-colour render whose colour map the server can swap (``params["cmap"]``):
    # the default map's name, else None.
    recolor: str | None = None
    # Pre-filters (notch, speckle) the method always renders with; (0, 0) = as the viewer asks.
    prefilter: tuple[int, int] = (0, 0)

    def __post_init__(self):
        self.version = self.version or VERSION

    def norm_of(self, channel: str) -> str | tuple[float, float]:
        return self.norms.get(channel, "robust")

    @property
    def display(self) -> dict:
        """How compose() colours the channels, for viewers that recolour them live.

        ``scalar`` (one channel through a colour map), ``rgb`` (three channels),
        ``overlay`` (heat over a gray base); anything else is ``image`` (only the
        rendered RGB can be adjusted; ``recolor`` names its colour map when the
        server can render it in another one) or ``animated``.
        """
        if self.animated:
            return {"kind": "animated"}
        display = dict(getattr(self.compose, "display", {"kind": "image"}))
        if self.recolor:
            display["recolor"] = self.recolor
        return display


REGISTRY: dict[str, Visualizer] = {}


def register(**kwargs):
    def wrap(fn_pair):
        channels, compose = fn_pair
        vis = Visualizer(channels=channels, compose=compose, **kwargs)
        REGISTRY[vis.name] = vis
        return vis

    return wrap


def _one(name: str, cmap: str | None = None):
    """compose() for a single channel: gray or a colour map."""

    def compose(ch, tile, params):
        return gray(ch[name]) if cmap is None else colormap(ch[name], cmap)

    compose.display = {"kind": "scalar", "channel": name, "cmap": cmap or "gray"}
    return compose


def _rgb(r: str, g: str, b: str):
    def compose(ch, tile, params):
        return np.stack([ch[r], ch[g], ch[b]], axis=-1)

    compose.display = {"kind": "rgb", "channels": [r, g, b]}
    return compose


def _overlay(base: str = "base", heat: str = "heat", cmap: str = "inferno", strength: float = 0.75):
    """compose() for heat coloured over a gray base."""

    def compose(ch, tile, params):
        return overlay(ch[base], ch[heat], cmap, strength)

    compose.display = {"kind": "overlay", "base": base, "heat": heat, "cmap": cmap, "strength": strength}
    return compose


# --- Baselines: what VC3D shows today ---------------------------------------
register(
    name="mid_slice", title="Mid slice", family="baseline",
    idea="The traced surface itself (the middle plane of the surface volume). VC3D's default view.",
)((lambda t, p: {"v": t.mid_plane}, _one("v")))

register(
    name="mean_composite", title="Mean composite ±8", family="baseline",
    idea="Mean of the 17 planes around the traced surface (VC3D composite 'mean').",
)((lambda t, p: {"v": t.window_mean}, _one("v")))

register(
    name="max_composite", title="Max composite ±8", family="baseline",
    idea="Maximum of the 17 planes around the traced surface (VC3D composite 'max').",
)((lambda t, p: {"v": t.window_max}, _one("v")))

# --- Surface-flattened density ----------------------------------------------
register(
    name="surface_skin", title="Surface skin", family="surface",
    idea="Each pixel's own papyrus surface is found (steepest fall into the gap on the ink side) and the "
         "outer ~10 µm of the sheet is averaged. Removes the waviness that makes fixed planes mix papyrus "
         "and air.",
)((lambda t, p: {"v": t.skin}, _one("v")))

register(
    name="skin_minus_body", title="Skin − body", family="surface",
    idea="Density of the outer skin minus the sheet 20–40 µm deeper. An ink deposit adds density on top, "
         "so ink should lean red.",
    norms={"v": "diverging"},
)((lambda t, p: {"v": t.skin - t.body}, _one("v", "coolwarm")))


def _edge_channels(t, p):
    flat = gauss(t.flat, (0, t.xy(1.0), t.xy(1.0)))
    i0 = int(-FLAT_D[0])
    drops = [flat[i0 + d] - flat[i0 + d - 2] for d in range(0, 7)]
    return {"v": gauss(np.max(drops, axis=0), t.xy(2.0))}


register(
    name="edge_sharpness", title="Edge sharpness", family="surface",
    idea="Steepest density step at the ink-side face (between the surface and 15 µm inside). The best "
         "simple ink predictor found by explainability at 9 µm; an ink layer makes the face crisper.",
)((_edge_channels, _one("v", "inferno")))

register(
    name="edge_sharpness_area", title="Edge sharpness, smoothed", family="surface",
    idea="Edge sharpness averaged over ~20 µm (σ 8 px): at 9 µm, heavy smoothing made this cue point at "
         "areas with writing rather than single strokes.",
)((lambda t, p: {"v": gauss(_edge_channels(t, p)["v"], t.xy(8))}, _one("v", "inferno")))

register(
    name="depth_rgb", title="Depth bands → RGB", family="surface",
    idea="False colour from three depth bands relative to the surface: red = just in front of the face, "
         "green = the skin, blue = the body. Extra material on the face shifts hue towards yellow/red.",
)((lambda t, p: {"r": t.band(-5, -1), "g": t.skin, "b": t.body}, _rgb("r", "g", "b")))


def _fringe_channels(t, p):
    flat = gauss(t.flat, (0.8, t.xy(1.0), t.xy(1.0)))
    i0 = int(-FLAT_D[0])
    return {"v": gauss(flat[i0 - 2] - 2 * flat[i0] + flat[i0 + 2], t.xy(1.5))}


register(
    name="fringe", title="Face curvature in depth (fringe)", family="surface",
    idea="Second derivative of density across the face. Phase-contrast CT puts bright/dark fringes on "
         "material boundaries; a thin dense ink layer changes that fringe.",
    norms={"v": "diverging"},
)((_fringe_channels, _one("v", "coolwarm")))


def _profiles(tile: Tile, d0: int = -6, d1: int = 14, stride: int = 1, data_only: bool = False) -> np.ndarray:
    """(pixels, depths): every pixel's density profile; ``data_only``: only the pixels with data (for fits)."""
    flat = gauss(tile.flat, (0, tile.xy(1.0), tile.xy(1.0)))
    i0, i1 = d0 - FLAT_D[0], d1 - FLAT_D[0] + 1
    block = tile.pixels(flat[i0:i1], stride) if data_only else flat[i0:i1, ::stride, ::stride]
    return block.reshape(block.shape[0], -1).T  # (pixels, depths)


def _pca_fit(tiles: list[Tile]) -> dict:
    rows = np.concatenate([_profiles(t, stride=4, data_only=True) for t in tiles])
    mean = rows.mean(axis=0)
    _, _, vt = np.linalg.svd(rows - mean, full_matrices=False)
    components = vt[:4]
    # A component's sign is arbitrary; make each one weigh the skin positively.
    skin = (np.arange(-6, 15) >= 0) & (np.arange(-6, 15) <= 4)
    components *= np.where(components[:, skin].sum(axis=1) < 0, -1, 1)[:, None]
    return {"mean": mean.tolist(), "components": components.tolist()}


def _pca_channels(first: int):
    def channels(t, p):
        h, w = t.shape2d
        rows = _profiles(t)
        proj = (rows - np.asarray(p["mean"])) @ np.asarray(p["components"]).T
        return {f"pc{k}": proj[:, k].reshape(h, w) for k in range(first, first + 3)}

    return channels


register(
    name="profile_pca", title="Depth-profile PCA 1–3", family="surface",
    idea="Every pixel's density profile across the face (−15 … +35 µm) projected on the volume's three "
         "main profile shapes, shown as RGB. Unsupervised: pixels with a different layer structure get "
         "a different colour.",
    fit=_pca_fit, norms={f"pc{k}": "diverging" for k in range(4)},
)((_pca_channels(0), _rgb("pc0", "pc1", "pc2")))

register(
    name="profile_pca_234", title="Depth-profile PCA 2–4", family="surface",
    idea="As PCA 1–3, without the first component (mostly overall brightness), so the shape of the "
         "profile, not its level, sets the colour.",
    fit=_pca_fit, norms={f"pc{k}": "diverging" for k in range(4)},
)((_pca_channels(1), _rgb("pc1", "pc2", "pc3")))

# --- Relief and lighting ------------------------------------------------------
RELIEF_EXAGGERATION = 2.5


register(
    name="relief_raking", title="Raking light", family="relief",
    idea="The ink face as a height field, lit by a low light (20° altitude) from 45°, oblique to both "
         "fiber families. Papyrologists use raking light to see ink that sits on the surface.",
)((lambda t, p: {"v": hillshade(t.shade_relief, 45, 25, RELIEF_EXAGGERATION)}, _one("v")))

register(
    name="relief_rgb_lights", title="Three coloured lights", family="relief",
    idea="The height field lit by red, green and blue raking lights from 45°, 165° and 285°. A flat face "
         "is gray; every slope gets the hue of the light it faces, so bumps and cracks become colour.",
)((
    lambda t, p: {
        "r": hillshade(t.shade_relief, 45, 25, RELIEF_EXAGGERATION),
        "g": hillshade(t.shade_relief, 165, 25, RELIEF_EXAGGERATION),
        "b": hillshade(t.shade_relief, 285, 25, RELIEF_EXAGGERATION),
    },
    _rgb("r", "g", "b"),
))

register(
    name="relief_detail", title="Relief detail", family="relief",
    idea="Height of the face above its own 15 µm average: raised deposits red, grooves and cracks blue.",
    norms={"v": "diverging"},
)((lambda t, p: {"v": t.elevation - gauss(t.elevation, t.xy(6))}, _one("v", "coolwarm")))

register(
    name="curvature", title="Surface curvature", family="relief",
    idea="Curvature of the face (negative Laplacian of height): ridges red, valleys blue. Brings out the "
         "crack network of dried ink.",
    norms={"v": "diverging"},
)((lambda t, p: {"v": -ndimage.gaussian_laplace(t.side * t.height, t.xy(1.5))}, _one("v", "coolwarm")))


def _relief_tint_compose(ch, t, p):
    shade = 0.35 + 0.65 * ch["shade"]
    return colormap(ch["h"], "terrain") * shade[..., None]


register(
    name="relief_tint", title="Hypsometric tint", family="relief",
    idea="Map-style relief: colour by height above the local average, shaded by raking light.",
    norms={"h": "diverging", "shade": "unit"},
)((
    lambda t, p: {
        "h": t.elevation - gauss(t.elevation, t.xy(12)),
        "shade": hillshade(t.shade_relief, 45, 30, RELIEF_EXAGGERATION),
    },
    _relief_tint_compose,
))


def _pbr_compose(ch, t, p):
    """Albedo from density, Lambert diffuse and Blinn-Phong specular from the height field."""
    elevation = t.shade_relief * RELIEF_EXAGGERATION
    gy, gx = np.gradient(elevation)
    norm = np.sqrt(gx * gx + gy * gy + 1)
    n = np.stack([-gx / norm, -gy / norm, 1 / norm], axis=-1)
    az, alt = np.deg2rad(45), np.deg2rad(30)
    light = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])
    half = light + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    diffuse = np.clip(n @ light, 0, 1)
    specular = np.clip(n @ half, 0, 1) ** 40
    density = ch["skin"]
    albedo = colormap(density, "ink")
    gloss = 0.15 + 0.85 * density  # denser (carbon-rich) material rendered glossier
    rgb = albedo * (0.25 + 0.75 * diffuse[..., None]) + (0.6 * gloss * specular)[..., None]
    return np.clip(rgb, 0, 1)


register(
    name="shaded_skin", title="Lit skin (PBR-lite)", family="relief",
    idea="Physically based shading: the skin density is the albedo and the gloss, the face's height "
         "field gives the normals, one raking light. Dense raised material catches highlights.",
)((lambda t, p: {"skin": t.skin}, _pbr_compose))


def _rotating_light_compose(ch, t, p):
    return [gray(0.15 + 0.85 * hillshade(t.shade_relief, az, 25, RELIEF_EXAGGERATION))
            for az in range(0, 360, 30)]


register(
    name="rotating_light", title="Rotating raking light (animated)", family="relief",
    idea="As raking light, but the light circles the sample (12 frames). Reflectance transformation "
         "imaging (RTI) of papyri works this way: relief pops out when the light moves.",
    animated=True,
)((lambda t, p: {}, _rotating_light_compose))


def _tail_fit(p_lo: float = 85, p_hi: float = 99.5):
    """fit(): the density window of the transfer function, as percentiles of the face slab."""

    def fit(tiles: list[Tile]) -> dict:
        values = np.concatenate([t.pixels(t.band(-2, 6), 4).ravel() for t in tiles])
        return {"lo": float(np.percentile(values, p_lo)), "hi": float(np.percentile(values, p_hi))}

    return fit


def _tail_steps(t: Tile, p: dict, d0: int, d1: int):
    """(v, opacity) per depth d0 … d1, front to back from the air side into the sheet."""
    lo, hi = p["lo"], p["hi"]
    i0 = int(-FLAT_D[0])
    flat = gauss(t.flat, (0, t.xy(0.7), t.xy(0.7)))
    for d in range(d0, d1 + 1):
        v = np.clip((flat[i0 + d] - lo) / (hi - lo), 0, 1)
        yield v, 0.35 * v * v


def _tail_compose(d0: int = -6, d1: int = 12):
    """Emission–absorption render of the slab d0 … d1 (inferno emission, or ``p["cmap"]``; skin behind)."""

    def compose(ch, t, p):
        cmap = p.get("cmap", "inferno")
        color = np.zeros(t.shape2d + (3,), dtype=np.float32)
        alpha = np.zeros(t.shape2d, dtype=np.float32)
        for v, a in _tail_steps(t, p, d0, d1):
            c = colormap(v, cmap)
            color += ((1 - alpha) * a)[..., None] * c
            alpha += (1 - alpha) * a
        background = gray(0.25 * ch["skin"])
        return color + (1 - alpha)[..., None] * background

    return compose


def _tail_glow(t: Tile, p: dict, d0: int = -6, d1: int = 12) -> np.ndarray:
    """Accumulated opacity of the dense-tail render: how much dense material the ray met."""
    alpha = np.zeros(t.shape2d, dtype=np.float32)
    for _, a in _tail_steps(t, p, d0, d1):
        alpha += (1 - alpha) * a
    return alpha


_dense_tail_fit = _tail_fit()
_dense_tail_compose = _tail_compose()


register(
    name="dense_tail_volume", title="Dense-tail volume render", family="relief",
    idea="Emission–absorption rendering of the slab across the face with a transfer function that is "
         "transparent for ordinary papyrus and glows only for the densest 15 % (the range an ink "
         "deposit would add).",
    fit=_dense_tail_fit, recolor="inferno",
)((lambda t, p: {"skin": t.skin}, _dense_tail_compose))

# --- Texture and frequency ----------------------------------------------------
def _crackle(t: Tile) -> np.ndarray:
    """Log energy of the skin's finest band (2–6 µm), averaged over ~12 µm."""
    return np.log1p(gauss(dog(t.skin, t.xy(0.8), t.xy(2.5)) ** 2, t.xy(5)))


register(
    name="crackle_energy", title="Crackle energy", family="texture",
    idea="Energy of the finest texture (2–6 µm band-pass) of the skin, laid over it as heat. Dried ink "
         "cracks into a fine network that bare papyrus lacks.",
)((
    lambda t, p: {"base": t.skin, "heat": _crackle(t)},
    _overlay(),
))


def _oblique_channels(t, p):
    skin = t.skin
    gy = ndimage.gaussian_filter(skin, t.xy(1.5), order=(1, 0), mode="nearest")
    gx = ndimage.gaussian_filter(skin, t.xy(1.5), order=(0, 1), mode="nearest")
    theta = np.arctan2(gy, gx) % np.pi
    off = angular_distance(theta, t.fiber_orientations)
    weight = np.clip((np.rad2deg(off) - 12) / 15, 0, 1)
    return {"base": skin, "heat": np.log1p(gauss((gx * gx + gy * gy) * weight, t.xy(4)))}


register(
    name="oblique_edges", title="Off-fiber edges", family="texture",
    idea="Edge energy, counting only edges at least ~20° away from both fiber families (found per tile "
         "from the orientation histogram). Fibers vanish; stroke outlines and cracks at other angles stay.",
)((_oblique_channels, _overlay()))


def _gabor_energies(t: Tile, wavelength: float = 10.0, sigma: float = 4.0, smooth: float = 6.0) -> dict:
    """Smoothed Gabor energy of the skin at four orientations relative to the first fiber family.

    Lengths in 2.4 µm pixels; keys fiber, diag1 (+45°), cross (+90°), diag2 (+135°).
    """
    skin = t.skin - gauss(t.skin, t.xy(wavelength))
    fiber = t.fiber_orientations[0]
    half = max(2, int(round(t.xy(3 * sigma))))
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    envelope = np.exp(-(x * x + y * y) / (2 * t.xy(sigma) ** 2))
    period = t.xy(wavelength)
    energy = {}
    for name, offset in (("fiber", 0.0), ("diag1", np.pi / 4), ("cross", np.pi / 2), ("diag2", 3 * np.pi / 4)):
        theta = fiber + offset
        u = x * np.cos(theta) + y * np.sin(theta)
        re = envelope * np.cos(2 * np.pi * u / period)
        im = envelope * np.sin(2 * np.pi * u / period)
        re -= re.mean()
        response = ndimage.convolve(skin, re, mode="nearest") ** 2 + ndimage.convolve(skin, im, mode="nearest") ** 2
        energy[name] = gauss(response, t.xy(smooth))
    return energy


def _gabor_rgb_channels(wavelength: float = 10.0, sigma: float = 4.0, smooth: float = 6.0):
    def channels(t, p):
        e = {k: np.log1p(v) for k, v in _gabor_energies(t, wavelength, sigma, smooth).items()}
        return {"r": e["diag1"], "g": e["diag2"], "b": 0.5 * (e["fiber"] + e["cross"])}

    return channels


_gabor_channels = _gabor_rgb_channels()


register(
    name="oblique_gabor_rgb", title="Gabor orientation RGB", family="texture",
    idea="Texture energy per orientation relative to the fibers (Gabor filters, 24 µm wavelength): "
         "blue = along either fiber family, red / green = the two diagonals. Fiber papyrus leans blue; "
         "strokes and cracks add red/green.",
)((_gabor_channels, _rgb("r", "g", "b")))


def _isotropy_at(sigma_int: float):
    def channels(t, p):
        return _isotropy_channels(t, p, sigma_int)

    return channels


def _isotropy_channels(t, p, sigma_int: float = 6.0):
    (jxx, jxy, jyy), _ = structure_tensor(t.skin, t.xy(1.0), t.xy(sigma_int))
    trace = jxx + jyy
    root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)
    coherence = root / (trace + 1e-6)
    energy = np.log1p(trace)
    return {"iso": 1 - coherence, "energy": energy}


def _isotropy_compose(ch, t, p):
    return colormap(ch["iso"], "viridis") * (0.4 + 0.6 * ch["energy"])[..., None]


register(
    name="isotropy", title="Texture isotropy", family="texture",
    idea="How directionless the local texture is (1 − structure-tensor coherence), brightness by texture "
         "energy. Bare papyrus is strongly directional; ink covering the fibers is not.",
)((_isotropy_channels, _isotropy_compose))


def notch_filter(image: np.ndarray, orientations: np.ndarray, *, half_width_deg: float = 7.0,
                 min_radius: int = 4, pad: int = 64, valid: np.ndarray | None = None) -> np.ndarray:
    """Delete the spectral lines of the fiber families (±``half_width_deg``) from ``image``.

    Fibers with gradient orientation θ put their energy on the spectral line at
    θ; frequencies within ``min_radius`` of the origin (the mean and the
    slowest variations) are kept. Mirror padding avoids wrap-around edges.
    Pixels outside ``valid`` (no data) are set to the mean of the valid ones
    first: the step to an empty corner would otherwise put its own lines in
    the spectrum and leave a block-wide offset that the kept low frequencies
    carry into the result.
    """
    if valid is not None and valid.any():
        image = np.where(valid, image, np.float32(image[valid].mean()))
    padded = np.pad(image, pad, mode="reflect")
    h, w = padded.shape
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    fy, fx = np.mgrid[-h // 2 : h - h // 2, -w // 2 : w - w // 2].astype(np.float32)
    radius = np.hypot(fx, fy)
    angle = np.arctan2(fy, fx) % np.pi
    off = angular_distance(angle, orientations)
    spectrum[(off < np.deg2rad(half_width_deg)) & (radius > min_radius)] = 0
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))[pad:-pad, pad:-pad]


def despeckle(image: np.ndarray, length: int, threshold: float) -> np.ndarray:
    """Replace bright specks shorter than ``length`` px in every direction by the local background.

    The union of grey openings by lines of ``length`` px (0°, 45°, 90°, 135°)
    keeps every bright feature that is long in some direction (fibers,
    strokes); what rises more than ``threshold`` above it is a speck. Specks
    and a 1 px rim are set to the square opening (the surrounding level);
    everything else is untouched.
    """
    line = np.ones((1, length), bool)
    diagonal = np.eye(length, dtype=bool)
    kept = np.max([ndimage.grey_opening(image, footprint=f)
                   for f in (line, line.T, diagonal, diagonal[::-1])], axis=0)
    specks = ndimage.binary_dilation(image - kept > threshold)
    return np.where(specks, ndimage.grey_opening(image, size=(length, length)), image)


def _notch_channels(t, p):
    residual = notch_filter(t.skin - t.valid_mean(t.skin), t.fiber_orientations, valid=t.valid)
    return {"v": gauss(residual, t.xy(0.8))}


register(
    name="fiber_notch", title="Fiber notch filter", family="texture",
    idea="Fourier filter that deletes the two spectral lines of the fiber families (±7°), leaving "
         "everything that does not run along a fiber: strokes, blobs, cracks.",
)((_notch_channels, _one("v")))


def _local_std(t: Tile, s1: float, s2: float) -> np.ndarray:
    """Local strength of the skin's s1→s2 band (2.4 µm pixels)."""
    return np.sqrt(gauss(dog(t.skin, t.xy(s1), t.xy(s2)) ** 2, t.xy(s2 + 1)))


register(
    name="multiscale_texture_rgb", title="Multiscale texture RGB", family="texture",
    idea="Local texture strength at three scales as RGB: red = fine (2–5 µm), green = medium (7–15 µm), "
         "blue = coarse (20–40 µm). Different surface finishes get different colours.",
)((
    lambda t, p: {"r": _local_std(t, 0.8, 2.0), "g": _local_std(t, 3, 6), "b": _local_std(t, 8, 16)},
    _rgb("r", "g", "b"),
))


def _clahe(image: np.ndarray, t: Tile, clip_limit: float) -> np.ndarray:
    """CLAHE over ~150 µm neighbourhoods."""
    from skimage import exposure

    kernel = max(8, int(round(t.xy(64))))
    return exposure.equalize_adapthist(np.clip(image, 0, 1), kernel_size=kernel, clip_limit=clip_limit)


def _clahe_compose(ch, t, p):
    return gray(_clahe(ch["v"], t, 0.02))


register(
    name="skin_clahe", title="Skin, local contrast (CLAHE)", family="texture",
    idea="The surface skin with contrast-limited adaptive histogram equalisation: every 150 µm "
         "neighbourhood uses its full gray range.",
)((lambda t, p: {"v": t.skin}, _clahe_compose))

register(
    name="stroke_dog", title="Stroke-scale band-pass", family="texture",
    idea="Difference of Gaussians at stroke width (σ 25 → 100 µm) of the skin: brighter or darker "
         "patches of about a pen stroke's size, with fiber and noise texture removed.",
    norms={"v": "diverging"},
)((lambda t, p: {"v": dog(t.skin, t.xy(10), t.xy(40))}, _one("v", "coolwarm")))

# --- Depth-coded ----------------------------------------------------------------
def _depth_of_max_channels(t, p):
    flat = gauss(t.flat, (0, t.xy(1.0), t.xy(1.0)))
    i0, i1 = -6 - FLAT_D[0], 12 - FLAT_D[0] + 1
    block = flat[i0:i1]
    return {"depth": np.argmax(block, axis=0).astype(np.float32) / (i1 - i0 - 1), "value": block.max(axis=0)}


register(
    name="depth_of_max_hue", title="Depth of brightest voxel", family="depth",
    idea="Hue = depth of the densest voxel across the face (red at the face → blue deep inside), "
         "brightness = its density. A deposit on the face pulls the maximum outward.",
    norms={"depth": "unit"},
)((_depth_of_max_channels, lambda ch, t, p: hsv_to_rgb(0.7 * ch["depth"], np.full_like(ch["value"], 0.85), ch["value"])))


def _topo_compose(ch, t, p):
    h = gauss(t.elevation, t.xy(2.0)) - gauss(t.elevation, t.xy(16))
    step = 1.0
    phase = (h / step) % 1.0
    line = np.clip(1 - np.minimum(phase, 1 - phase) / 0.08, 0, 1)
    tint = np.where(h[..., None] > 0, np.array([1.0, 0.45, 0.1]), np.array([0.2, 0.55, 1.0]))
    return gray(ch["v"]) * (1 - line[..., None]) + tint * line[..., None]


register(
    name="topo_contours", title="Topographic contours", family="depth",
    idea="The skin with contour lines of the face height every 2.4 µm (orange above, blue below the "
         "local average). Raised plateaus show as closed rings.",
)((lambda t, p: {"v": t.skin}, _topo_compose))


def _sweep_compose(ch, t, p):
    frames = []
    i0 = int(-FLAT_D[0])
    flat = t.flat
    lo, hi = np.percentile(t.pixels(flat[i0 - 4 : i0 + 10]), [1, 99])
    for d in list(range(-4, 10)) + list(range(8, -4, -1)):
        frames.append(gray((flat[i0 + d] - lo) / (hi - lo + 1e-6)))
    return frames


register(
    name="depth_sweep", title="Depth sweep through the face (animated)", family="depth",
    idea="Flattened planes from 10 µm in front of the face to 25 µm inside and back. Motion makes "
         "things that sit on the face (and vanish below it) easy to spot.",
    animated=True,
)((lambda t, p: {}, _sweep_compose))

# --- Fusion -----------------------------------------------------------------------
def _fusion_channels(t, p):
    return {
        "base": t.skin,
        "a": t.skin - t.body,
        "b": _edge_channels(t, p)["v"],
        "c": _isotropy_channels(t, p)["iso"],
        "d": _crackle(t),
    }


def _fusion_compose(ch, t, p):
    score = (ch["a"] + ch["b"] + ch["c"] + ch["d"]) / 4  # each 0…1 after normalisation
    score = gauss(score, t.xy(3))
    centered = np.clip((score - 0.5) * 2.5, -1, 1)
    luminance = _clahe(ch["base"], t, 0.015)
    hue = np.where(centered > 0, 0.08, 0.6)  # orange for ink-like, blue for not
    return hsv_to_rgb(hue, np.abs(centered) * 0.9, 0.15 + 0.85 * luminance)


register(
    name="ink_fusion", title="Fusion: gray skin + colour cues", family="fusion",
    idea="Familiar gray skin (CLAHE) as brightness; colour from the average of four cues (skin − body, "
         "edge sharpness, isotropy, crackle): orange where they agree on 'ink-like', blue where they "
         "agree on 'bare'. A hand-made mix, nothing fitted to labels.",
    norms={"a": "robust", "b": "robust", "c": "robust", "d": "robust"},
)((_fusion_channels, _fusion_compose))


# ----------------------------------------------------------------------------
# Variants of the promising candidates (picked by eye on the Segments tab)
# ----------------------------------------------------------------------------
# Each changes one structural parameter of its parent, or turns its colouring into one
# scalar channel so the viewer can recolour and threshold it live.

for _name, _title, _lo, _hi, _d0, _d1, _idea in (
    ("dense_tail_p70", "Dense tail · looser (top 30 %)", 70, 99.5, -6, 12,
     "As the dense-tail render, but the transfer function starts at the 70th percentile: more of the "
     "slab glows."),
    ("dense_tail_p95", "Dense tail · tighter (top 5 %)", 95, 99.8, -6, 12,
     "Only the densest 5 % of the face slab glows."),
    ("dense_tail_face", "Dense tail · face only", 85, 99.5, -3, 4,
     "The render restricted to the face itself (−7 … +10 µm), where an ink layer sits."),
    ("dense_tail_deep", "Dense tail · deep only (control)", 85, 99.5, 5, 14,
     "The render of the sheet 12–34 µm inside, away from the face. If ink shows here too, the cue is "
     "the sheet, not a surface deposit."),
):
    register(name=_name, title=_title, family="variant", variant_of="dense_tail_volume", idea=_idea,
             fit=_tail_fit(_lo, _hi), recolor="inferno")((lambda t, p: {"skin": t.skin}, _tail_compose(_d0, _d1)))


register(
    name="dense_tail_glow", title="Dense tail · glow (live)", family="variant", variant_of="dense_tail_volume",
    idea="How much dense material each ray through the face slab meets (the render's accumulated "
         "opacity), as heat over the gray skin. One channel, so colours and thresholds can be set live.",
    fit=_tail_fit(),
)((lambda t, p: {"base": t.skin, "heat": _tail_glow(t, p)}, _overlay(cmap="inferno", strength=0.85)))

for _name, _title, _wl, _sigma, _smooth, _idea in (
    ("gabor_fine", "Gabor RGB · fine (12 µm)", 5.0, 2.0, 4.0,
     "Gabor orientation RGB at half the wavelength: crack-scale texture."),
    ("gabor_coarse", "Gabor RGB · coarse (48 µm)", 20.0, 8.0, 10.0,
     "Gabor orientation RGB at twice the wavelength: stroke-edge scale."),
):
    register(name=_name, title=_title, family="variant", variant_of="oblique_gabor_rgb", idea=_idea)(
        (_gabor_rgb_channels(_wl, _sigma, _smooth), _rgb("r", "g", "b")))


def _gabor_ratio(wavelength: float = 10.0, sigma: float = 4.0, smooth: float = 6.0):
    def channels(t, p):
        e = _gabor_energies(t, wavelength, sigma, smooth)
        return {"v": np.log((e["diag1"] + e["diag2"] + 1e-3) / (e["fiber"] + e["cross"] + 1e-3))}

    return channels


_gabor_ratio_channels = _gabor_ratio()


register(
    name="gabor_offfiber_ratio", title="Gabor · off-fiber share (live)", family="variant",
    variant_of="oblique_gabor_rgb",
    idea="One number per pixel: log of diagonal over fiber-aligned Gabor energy. High (red) where the "
         "texture runs across the fibers, low (blue) on plain fiber texture. Recolour and threshold live.",
    norms={"v": "diverging"},
)((_gabor_ratio_channels, _one("v", "coolwarm")))


register(
    name="gabor_offfiber_ratio_coarse", title="Gabor · off-fiber share, coarse (live)", family="variant",
    variant_of="oblique_gabor_rgb",
    idea="The off-fiber share at 48 µm wavelength (stroke-edge scale). At 9.6 µm pixels the 24 µm "
         "version is close to the pixel size; this one is not.",
    norms={"v": "diverging"},
)((_gabor_ratio(20.0, 8.0, 10.0), _one("v", "coolwarm")))


def _gabor_hue_channels(t, p):
    e = _gabor_energies(t)
    cx, cy = e["fiber"] - e["cross"], e["diag1"] - e["diag2"]
    total = e["fiber"] + e["cross"] + e["diag1"] + e["diag2"] + 1e-6
    return {"angle": (np.arctan2(cy, cx) / (2 * np.pi)) % 1.0,  # doubled-angle orientation, 0 … 1
            "strength": np.hypot(cx, cy) / total, "energy": np.log1p(total)}


def _gabor_hue_compose(ch, t, p):
    return hsv_to_rgb(ch["angle"], np.clip(ch["strength"] * 1.2, 0, 1), 0.15 + 0.85 * ch["energy"])


register(
    name="gabor_orientation_hue", title="Gabor · orientation as hue", family="variant",
    variant_of="oblique_gabor_rgb",
    idea="The dominant texture orientation relative to the fibers as hue (red / cyan = along a fiber "
         "family, green / violet = the diagonals), how pronounced it is as saturation, texture energy "
         "as brightness.",
    norms={"angle": "unit", "strength": "unit"},
)((_gabor_hue_channels, _gabor_hue_compose))

for _name, _title, _sigma, _idea in (
    ("isotropy_fine", "Isotropy · fine (7 µm)", 3.0, "Texture isotropy over half the neighbourhood."),
    ("isotropy_coarse", "Isotropy · coarse (29 µm)", 12.0, "Texture isotropy over twice the neighbourhood."),
):
    register(name=_name, title=_title, family="variant", variant_of="isotropy", idea=_idea)(
        (_isotropy_at(_sigma), _isotropy_compose))

register(
    name="isotropy_live", title="Isotropy · one channel (live)", family="variant", variant_of="isotropy",
    idea="Isotropy alone (1 − coherence), without the energy shading, so colours and thresholds can be "
         "set live.",
)((lambda t, p: {"v": _isotropy_channels(t, p)["iso"]}, _one("v", "viridis")))

register(
    name="isotropy_energetic", title="Isotropy × energy (live)", family="variant", variant_of="isotropy",
    idea="Isotropy weighted by texture energy: directionless *and* textured (cracks, rough deposits), "
         "rather than directionless because smooth.",
)((lambda t, p: {"v": (lambda c: c["iso"] * c["energy"])(_isotropy_channels(t, p))}, _one("v", "viridis")))


def _cue_fit(cues):
    """fit(): p1 / p99 of every cue, so a cue average can be one channel."""

    def fit(tiles: list[Tile]) -> dict:
        ranges = {}
        for name, cue in cues.items():
            values = np.concatenate([t.pixels(cue(t, {}), 4).ravel() for t in tiles])
            lo, hi = np.percentile(values, [1, 99])
            ranges[name] = [float(lo), float(hi) if hi > lo else float(lo) + 1.0]
        return {"cues": ranges}

    return fit


def _cue_score(cues, t, p):
    score = np.zeros(t.shape2d, dtype=np.float32)
    for name, cue in cues.items():
        lo, hi = p["cues"][name]
        score += np.clip((cue(t, p) - lo) / (hi - lo), 0, 1)
    return gauss(score / len(cues), t.xy(3))


_FUSION_CUES = {
    "skin_minus_body": lambda t, p: t.skin - t.body,
    "edge": lambda t, p: _edge_channels(t, p)["v"],
    "isotropy": lambda t, p: _isotropy_channels(t, p)["iso"],
    "crackle": lambda t, p: _crackle(t),
}
_PROMISING_CUES = {
    "dense_tail": lambda t, p: _tail_glow(t, p["tail"]) if "tail" in p else _tail_glow(t, _tail_fit()([t])),
    "gabor_offfiber": lambda t, p: _gabor_ratio_channels(t, p)["v"],
    "isotropy": lambda t, p: _isotropy_channels(t, p)["iso"],
}


def _promising_fit(tiles):
    tail = _tail_fit()(tiles)
    fitted = _cue_fit({k: (lambda cue: lambda t, p: cue(t, {"tail": tail}))(c) for k, c in _PROMISING_CUES.items()})(tiles)
    return {**fitted, "tail": tail}


def _fused_compose(ch, t, p):
    """As ink_fusion: CLAHE skin as brightness, orange / blue by how far the score leans."""
    centered = np.clip((ch["score"] - 0.5) * 2.5, -1, 1)
    luminance = _clahe(ch["base"], t, 0.015)
    hue = np.where(centered > 0, 0.08, 0.6)
    return hsv_to_rgb(hue, np.abs(centered) * 0.9, 0.15 + 0.85 * luminance)


register(
    name="fusion_score", title="Fusion · score (live)", family="variant", variant_of="ink_fusion",
    idea="The fusion's cue average (skin − body, edge sharpness, isotropy, crackle; each scaled to its "
         "p1–p99 on the volume) as one channel: recolour and threshold it live.",
    fit=_cue_fit(_FUSION_CUES), norms={"v": "unit"},
)((lambda t, p: {"v": _cue_score(_FUSION_CUES, t, p)}, _one("v", "coolwarm")))

register(
    name="fusion_promising", title="Fusion of the promising cues", family="variant", variant_of="ink_fusion",
    idea="The fusion look, but the colour comes from the three other promising candidates: dense-tail "
         "glow, Gabor off-fiber share and isotropy.",
    fit=_promising_fit, norms={"score": "unit"},
)((lambda t, p: {"base": t.skin, "score": _cue_score(_PROMISING_CUES, t, p)}, _fused_compose))

register(
    name="fusion_promising_score", title="Promising cues · score (live)", family="variant",
    variant_of="ink_fusion",
    idea="The average of dense-tail glow, Gabor off-fiber share and isotropy as one channel: recolour "
         "and threshold it live.",
    fit=_promising_fit, norms={"v": "unit"},
)((lambda t, p: {"v": _cue_score(_PROMISING_CUES, t, p)}, _one("v", "coolwarm")))


# ----------------------------------------------------------------------------
# Normalisation and rendering
# ----------------------------------------------------------------------------
def channel_ranges(vis: Visualizer, tiles: list[Tile], params: dict) -> dict[str, list[float]]:
    """Per-volume display range of every channel (VC3D's window/level)."""
    gathered: dict[str, list[np.ndarray]] = {}
    for tile in tiles:
        for name, values in vis.channels(tile, params).items():
            gathered.setdefault(name, []).append(tile.pixels(values, 2).ravel())
    ranges = {}
    for name, chunks in gathered.items():
        values = np.concatenate(chunks)
        kind = vis.norm_of(name)
        if isinstance(kind, (tuple, list)):
            ranges[name] = [float(kind[0]), float(kind[1])]
        elif kind == "unit":
            ranges[name] = [0.0, 1.0]
        elif kind == "diverging":
            center = float(np.median(values))
            spread = float(np.percentile(np.abs(values - center), 98)) or 1.0
            ranges[name] = [center - spread, center + spread]
        else:
            lo, hi = np.percentile(values, [1, 99])
            ranges[name] = [float(lo), float(hi) if hi > lo else float(lo) + 1.0]
    return ranges


def calibrate(vis: Visualizer, tiles: list[Tile]) -> dict:
    params = vis.fit(tiles) if vis.fit else {}
    return {"params": params, "ranges": channel_ranges(vis, tiles, params), "version": vis.version}


def render(vis: Visualizer, tile: Tile, calibration: dict) -> np.ndarray | list[np.ndarray]:
    """RGB float [0, 1] (H, W, 3), or a list of such frames for an animated visualizer."""
    params = calibration["params"]
    channels = vis.channels(tile, params)
    normalised = {}
    for name, values in channels.items():
        lo, hi = calibration["ranges"].get(name, [0.0, 1.0])
        normalised[name] = np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)
    out = vis.compose(normalised, tile, params)
    if isinstance(out, list):
        return [np.clip(frame, 0, 1) for frame in out]
    return np.clip(out, 0, 1)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
