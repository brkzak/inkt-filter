"""Small image operations the filters share.

They come from the brute-force filter search of the original project (its ``bruteforce.py``), where the face
step, the p1–p99 scaling and the ridge detector were found to be the ingredients that kept winning. Only these
four functions are needed to run the filters; the search itself is not part of this package.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from colored_inkt.visualizers import data_pixels, gauss


def _px(t, um: float) -> float:
    """A length in µm as a filter sigma in this tile's pixels."""
    return t.xy(um / 2.4)


def _unit(v: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """``v`` scaled so its p1 … p99 (over the pixels with data, ``Tile.valid``) is 0 … 1 (label-free, per tile)."""
    lo, hi = np.percentile(data_pixels(v, valid, 4), [1, 99])
    return (v - lo) / (hi - lo if hi > lo else 1.0)


def _ridge(v: np.ndarray, sigma: float) -> np.ndarray:
    """Bright ridges and blobs of width ~``sigma`` (negative smaller Hessian eigenvalue, ≥ 0)."""
    hyy = ndimage.gaussian_filter(v, sigma, order=(2, 0), mode="nearest")
    hxx = ndimage.gaussian_filter(v, sigma, order=(0, 2), mode="nearest")
    hxy = ndimage.gaussian_filter(v, sigma, order=(1, 1), mode="nearest")
    smaller = (hxx + hyy) / 2 - np.sqrt(((hxx - hyy) / 2) ** 2 + hxy**2)
    return np.maximum(-smaller, 0) * sigma**2


def _edge(t, s, p):
    """The face step: the largest rise of density over ``step`` planes, for faces at depths ``s["d"]``."""
    # Only the planes the steps read (d0 - k … d1), smoothed in-plane only (σ 0 along depth: each plane on its
    # own), so a derived tile (Tile.despeckled) filters just those planes; the values are the whole stack's.
    (d0, d1), k = s["d"], s.get("step", 2)
    flat = gauss(t.planes(d0 - k, d1), (0, _px(t, 2.4), _px(t, 2.4)))
    return np.max([flat[d - d0 + k] - flat[d - d0] for d in range(d0, d1 + 1)], axis=0)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0, 1] → CIELAB (D65)."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)
