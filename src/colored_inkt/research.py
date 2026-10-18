"""Methods from the literature of neighbouring fields, applied to the surface volume (family ``research``).

The brute-force search of the original project kept converging on two label-free
ingredients:

* the **face step**: how steeply density rises into the sheet at the ink face;
* **skin − below**: the outer ~10 µm of the sheet minus the 10 µm under it.

Their average (each scaled to its p1–p99 on the region) is the *ink cue*
below. These methods take the cue, the raw planes or the depth profiles and
apply, or present them with, techniques from:

* multispectral palimpsest imaging: decorrelation stretch, pseudocolour, ICA;
* hyperspectral target and anomaly detection on the depth profile treated as
  a spectrum: MNF, CEM, SAM, local RX, k-means;
* document image processing: Sauvola binarisation, NL-means, cartoon–texture
  decomposition, Retinex;
* edge-preserving filtering: Perona–Malik, Kuwahara, rolling guidance,
  guided filter;
* feature detection: phase congruency, Frangi vesselness, LBP;
* archaeological LiDAR terrain visualisation, with the cue as the terrain:
  sky-view factor, red relief image map, multi-directional hillshade;
* perceptual presentation: stereo (wiggle and anaglyph), isoluminant colour,
  contours, posterisation.

Nothing here uses labels. Lengths are in µm and converted with the tile's
pixel size, so the methods look at the same physical scale at every level.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from colored_inkt.ops import _edge, _ridge, _unit
from colored_inkt.visualizers import (
    FLAT_D, Tile, _clahe, _one, _rgb, colormap, data_pixels, gauss, gray, hillshade, hsv_to_rgb, register,
)

FAMILY = "research"


def _px(t: Tile, um: float) -> float:
    return t.xy(um / 2.4)


def _memo(t: Tile, key: str, fn):
    """Per-tile cache of shared intermediate maps (the cue, the profile cube)."""
    store = t.__dict__.setdefault("_research", {})
    if key not in store:
        store[key] = fn()
    return store[key]


def face_step(t: Tile) -> np.ndarray:
    return _memo(t, "face_step", lambda: _edge(t, {"d": [0, 6], "step": 2}, {}))


def skin_below(t: Tile) -> np.ndarray:
    return _memo(t, "skin_below", lambda: t.band(1, 4) - t.band(4, 8))


def cue(t: Tile, smooth_um: float = 25.0) -> np.ndarray:
    """The ink cue: mean of the face step and skin − below, each p1–p99 scaled, smoothed ``smooth_um``."""
    def make():
        a = _unit(gauss(face_step(t), _px(t, smooth_um)), t.valid)
        b = _unit(gauss(skin_below(t), _px(t, smooth_um)), t.valid)
        return np.clip(0.5 * (a + b), -0.5, 1.5).astype(np.float32)

    return _memo(t, f"cue{smooth_um}", make)


def profiles(t: Tile, d0: int = -4, d1: int = 10) -> np.ndarray:
    """(D, H, W) density across the face, d0 … d1, lightly smoothed in-plane (the 'spectrum' per pixel)."""
    def make():
        i0 = int(-FLAT_D[0])
        return gauss(t.flat[i0 + d0 : i0 + d1 + 1], (0, _px(t, 2.4), _px(t, 2.4))).astype(np.float32)

    return _memo(t, f"profiles{d0}_{d1}", make)


def _rows(t: Tile, stride: int = 1, data_only: bool = False) -> np.ndarray:
    """(pixels, depths); ``data_only``: only the pixels with data (``Tile.pixels``), for the fits."""
    p = t.pixels(profiles(t), stride) if data_only else profiles(t)[:, ::stride, ::stride]
    return p.reshape(p.shape[0], -1).T  # (pixels, depths)


def _sample_rows(tiles: list[Tile], n: int = 60000) -> np.ndarray:
    rows = np.concatenate([_rows(t, 4, data_only=True) for t in tiles])
    rng = np.random.default_rng(0)
    return rows[rng.choice(len(rows), min(n, len(rows)), replace=False)].astype(np.float64)


def _project(t: Tile, mean, matrix) -> np.ndarray:
    """(k, H, W): every pixel's profile minus ``mean``, through the rows of ``matrix``."""
    h, w = t.shape2d
    proj = (_rows(t) - np.asarray(mean, np.float32)) @ np.asarray(matrix, np.float32).T
    return proj.T.reshape(-1, h, w)


def _shifted(padded: np.ndarray, pad: int, dy: int, dx: int, shape) -> np.ndarray:
    h, w = shape
    return padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]


def _horizon(elevation: np.ndarray, radius: int, directions: int = 8) -> np.ndarray:
    """(directions, H, W): the steepest elevation angle (radians) towards each direction, up to ``radius`` px."""
    pad = radius + 1
    padded = np.pad(elevation, pad, mode="edge")
    out = []
    for k in range(directions):
        angle = 2 * np.pi * k / directions
        best = np.full(elevation.shape, -np.inf, np.float32)
        for distance in range(1, radius + 1, max(1, radius // 16)):
            dy, dx = int(round(distance * np.sin(angle))), int(round(distance * np.cos(angle)))
            if dy == 0 and dx == 0:
                continue
            slope = (_shifted(padded, pad, dy, dx, elevation.shape) - elevation) / np.hypot(dy, dx)
            best = np.maximum(best, slope)
        out.append(np.arctan(best))
    return np.stack(out)


def _terrain(t: Tile, smooth_um: float = 40.0, radius_um: float = 400.0) -> tuple[np.ndarray, int]:
    """The cue as a terrain (heights in px, so slopes are moderate) and the horizon search radius in px."""
    radius = max(4, int(round(_px(t, radius_um))))
    return np.clip(cue(t, smooth_um), 0, 1) * radius / 4, radius


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He, Sun & Tang (2010): ``source`` smoothed with the edges of ``guide`` (box filters, O(N))."""
    def box(x):
        return ndimage.uniform_filter(x, size=2 * radius + 1, mode="nearest")

    mean_i, mean_p = box(guide), box(source)
    a = (box(guide * source) - mean_i * mean_p) / (box(guide * guide) - mean_i**2 + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """CIELAB (D65) → sRGB [0, 1], clipped."""
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200

    def inv(f):
        return np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29))

    xyz = np.stack([inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883], axis=-1)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]], np.float32)
    linear = np.clip(xyz @ m.T, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def _skin_l(t: Tile) -> np.ndarray:
    """The skin as a familiar gray base (CLAHE), 0 … 1."""
    return _memo(t, "skin_l", lambda: _clahe(np.clip(_unit(t.skin, t.valid), 0, 1), t, 0.015).astype(np.float32))


# ----------------------------------------------------------------------------
# 1–3: multispectral palimpsest imaging (depth bands as spectral bands)
# ----------------------------------------------------------------------------
def _bands(t: Tile):
    return {"r": t.band(-4, -1), "g": t.band(0, 4), "b": t.band(6, 12)}


def _dstretch_fit(tiles):
    x = np.concatenate([np.stack([t.pixels(v, 4).ravel() for v in _bands(t).values()], axis=1) for t in tiles])
    mean, cov = x.mean(axis=0), np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    target = np.sqrt(values.mean())  # every principal axis stretched to the same spread
    transform = vectors @ np.diag(target / np.sqrt(np.maximum(values, 1e-9))) @ vectors.T
    return {"mean": mean.tolist(), "transform": transform.tolist()}


def _dstretch_channels(t, p):
    bands = _bands(t)
    x = np.stack(list(bands.values()), axis=-1) - np.asarray(p["mean"], np.float32)
    y = x @ np.asarray(p["transform"], np.float32).T
    return {k: y[..., i] for i, k in enumerate(bands)}


register(
    name="res_dstretch", title="Decorrelation stretch of depth bands", family=FAMILY,
    idea="Gillespie, Kahle & Walker (1986); DStretch (Harman) for faint rock-art pigments. Three depth bands "
         "(in front of the face, the skin, 15–30 µm deep) as RGB, decorrelated and stretched so every "
         "independent direction gets the same contrast. Differences that all bands share (fibers, "
         "overall density) turn gray; what differs between depths turns into strong colour.",
    fit=_dstretch_fit, norms={"r": "robust", "g": "robust", "b": "robust"},
)((_dstretch_channels, _rgb("r", "g", "b")))

register(
    name="res_pseudocolor", title="Palimpsest pseudocolour (deep red, face cyan)", family=FAMILY,
    idea="Easton, Christens-Barry & Knox (2011), the Archimedes Palimpsest: one band that shows only the "
         "substrate goes to red, one that shows substrate and writing to green and blue. Here red = the "
         "sheet 15–30 µm deep (no ink can be there), green + blue = the skin. Fibers are in both and look "
         "gray; extra density only at the face (ink) turns cyan, missing density red.",
)((lambda t, p: {"deep": t.band(6, 12), "face": t.band(0, 3)}, _rgb("deep", "face", "face")))


def _ica_fit(tiles):
    from sklearn.decomposition import FastICA

    x = _sample_rows(tiles)
    ica = FastICA(n_components=6, whiten="unit-variance", random_state=0, max_iter=600, tol=1e-3)
    sources = ica.fit_transform(x)
    kurtosis = (sources**4).mean(axis=0) / (sources**2).mean(axis=0) ** 2 - 3
    order = np.argsort(-np.abs(kurtosis))[:3]
    skew = (sources[:, order] ** 3).mean(axis=0)
    unmixing = ica.components_[order] * np.sign(skew)[:, None]
    return {"mean": ica.mean_.tolist(), "unmixing": unmixing.tolist()}


register(
    name="res_ica", title="Independent components of the depth profile", family=FAMILY,
    idea="Tonazzini, Bedini & Salerno (2004), ICA for document restoration and palimpsests. Every pixel's "
         "density profile across the face is a mixture of independent sources (fiber layers, the surface, "
         "ink, noise). FastICA unmixes six; the three most non-Gaussian (sparse sources such as ink) are "
         "shown as RGB.",
    fit=_ica_fit, norms={"r": "diverging", "g": "diverging", "b": "diverging"},
)((lambda t, p: dict(zip("rgb", _project(t, p["mean"], p["unmixing"]))), _rgb("r", "g", "b")))


# ----------------------------------------------------------------------------
# 4–8: hyperspectral methods on the depth profile
# ----------------------------------------------------------------------------
def _mnf_fit(tiles):
    from scipy.linalg import eigh

    signal = np.cov(_sample_rows(tiles), rowvar=False)
    noise = []
    for t in tiles:  # noise: half the difference of horizontal neighbours (Green et al. 1988)
        p = profiles(t)[:, ::4, :]
        pairs = ((p[:, :, 1::8] - p[:, :, :-1:8]) / np.sqrt(2)).reshape(p.shape[0], -1).T
        if t.valid is not None:  # both neighbours with data (the filled empty part has no noise)
            pairs = pairs[(t.valid[::4, 1::8] & t.valid[::4, :-1:8]).ravel()]
        noise.append(pairs)
    noise = np.cov(np.concatenate(noise).astype(np.float64), rowvar=False) + 1e-6 * np.eye(signal.shape[0])
    values, vectors = eigh(signal, noise)
    vectors = vectors[:, np.argsort(-values)][:, :4].T
    face = slice(4, 9)  # d 0 … 4 within d −4 … 10
    vectors *= np.where(vectors[:, face].sum(axis=1) < 0, -1, 1)[:, None]
    mean = _sample_rows(tiles, 20000).mean(axis=0)
    return {"mean": mean.tolist(), "vectors": vectors[1:4].tolist()}


register(
    name="res_mnf", title="Minimum noise fraction of the depth profile", family=FAMILY,
    idea="Green, Berman, Switzer & Craig (1988), the standard first step of hyperspectral analysis. Like "
         "PCA, but the components are ordered by signal-to-noise instead of variance (noise estimated "
         "from neighbouring pixels), so low-contrast but spatially coherent layers rank above pixel noise. "
         "Components 2–4 as RGB (the first is overall brightness).",
    fit=_mnf_fit, norms={"r": "diverging", "g": "diverging", "b": "diverging"},
)((lambda t, p: dict(zip("rgb", _project(t, p["mean"], p["vectors"]))), _rgb("r", "g", "b")))


def _target_fit(tiles):
    """Label-free target: the mean profile of the 2 % most ink-like pixels by the cue; background statistics."""
    rows, top = [], []
    for t in tiles:
        r = _rows(t, 4, data_only=True)
        c = t.pixels(cue(t), 4).ravel()
        rows.append(r)
        top.append(r[c >= np.percentile(c, 98)])
    rows = np.concatenate(rows).astype(np.float64)
    mean = rows.mean(axis=0)
    target = np.concatenate(top).mean(axis=0) - mean
    cov = np.cov(rows, rowvar=False) + 1e-6 * np.eye(rows.shape[1])
    inv = np.linalg.inv(cov)
    w = inv @ target / (target @ inv @ target)
    return {"mean": mean.tolist(), "target": target.tolist(), "w": w.tolist()}


register(
    name="res_cem", title="Matched filter (CEM) for the ink profile", family=FAMILY,
    idea="Harsanyi (1993), constrained energy minimisation: the linear filter that passes a target "
         "spectrum with gain 1 and suppresses everything else as much as possible. The target is the mean "
         "depth profile of the 2 % most ink-like pixels by the ink cue (label-free); the background is "
         "the whole region.",
    fit=_target_fit,
)((lambda t, p: {"v": gauss(_project(t, p["mean"], [p["w"]])[0], _px(t, 10))}, _one("v", "inferno")))


def _sam_channels(t, p):
    h, w = t.shape2d
    rows = _rows(t) - np.asarray(p["mean"], np.float32)
    target = np.asarray(p["target"], np.float32)
    cos = rows @ target / (np.linalg.norm(rows, axis=1) * np.linalg.norm(target) + 1e-6)
    return {"v": gauss(cos.reshape(h, w), _px(t, 15))}


register(
    name="res_sam", title="Spectral angle to the ink profile", family=FAMILY,
    idea="Kruse et al. (1993), spectral angle mapper: the angle between a pixel's (mean-removed) depth "
         "profile and the ink target profile. Only the shape of the profile counts, not its brightness, "
         "so overall density differences between areas drop out.",
    fit=_target_fit,
)((_sam_channels, _one("v", "magma")))


def _rx_fit(tiles):
    residuals = []
    for t in tiles:
        p = profiles(t)
        local = gauss(p, (0, _px(t, 150), _px(t, 150)))
        residuals.append(t.pixels(p - local, 4).reshape(p.shape[0], -1).T)
    r = np.concatenate(residuals).astype(np.float64)
    return {"inv": np.linalg.inv(np.cov(r, rowvar=False) + 1e-6 * np.eye(r.shape[1])).tolist()}


def _rx_channels(t, p):
    prof = profiles(t)
    r = prof - gauss(prof, (0, _px(t, 150), _px(t, 150)))
    flat = r.reshape(r.shape[0], -1)
    distance = ((np.asarray(p["inv"], np.float32) @ flat) * flat).sum(axis=0)
    sign = np.sign(r[4:9].mean(axis=0)).ravel()  # denser at the face than around: positive
    return {"v": gauss((np.log1p(distance) * sign).reshape(t.shape2d), _px(t, 10))}


register(
    name="res_rx_local", title="Local RX anomaly of the depth profile", family=FAMILY,
    idea="Reed & Yu (1990), the RX anomaly detector with a local background: how far each pixel's depth "
         "profile lies from the mean profile of its 0.3 mm surroundings, in units of the region's "
         "residual covariance (Mahalanobis). Signed: denser than the surroundings at the face is warm.",
    fit=_rx_fit, norms={"v": "diverging"},
)((_rx_channels, _one("v", "coolwarm")))

_CLUSTER_COLOURS = np.array([(40, 40, 45), (70, 90, 120), (90, 130, 110), (150, 140, 90), (210, 150, 60),
                             (250, 220, 120)], np.float32) / 255


def _kmeans_fit(tiles):
    from sklearn.cluster import MiniBatchKMeans

    x = _sample_rows(tiles)
    km = MiniBatchKMeans(n_clusters=6, random_state=0, n_init=3, batch_size=4096).fit(x)
    order = np.argsort(km.cluster_centers_[:, 4:9].mean(axis=1))  # by face density
    return {"centers": km.cluster_centers_[order].tolist()}


def _kmeans_compose(ch, t, p):
    h, w = t.shape2d
    centers = np.asarray(p["centers"], np.float32)
    rows = _rows(t)
    distances = (rows**2).sum(axis=1, keepdims=True) - 2 * rows @ centers.T + (centers**2).sum(axis=1)
    label = np.argmin(distances, axis=1).reshape(h, w)
    label = ndimage.median_filter(label, size=t.xy_size(12))
    return _CLUSTER_COLOURS[label] * (0.55 + 0.45 * ch["skin"][..., None])


register(
    name="res_kmeans", title="Depth-profile classes (k-means)", family=FAMILY,
    idea="Unsupervised classification as in hyperspectral mapping: k-means groups the depth profiles into "
         "six classes, coloured dark → yellow by their face density and shaded by the skin. Areas whose "
         "layer structure differs (ink, gaps, bare fibers) become patches of their own colour.",
    fit=_kmeans_fit,
)((lambda t, p: {"skin": t.skin}, _kmeans_compose))


# ----------------------------------------------------------------------------
# 9–13: document and natural-image processing on the cue or the skin
# ----------------------------------------------------------------------------
def _cartoon(image: np.ndarray, sigma: float) -> np.ndarray:
    """Buades, Le, Morel & Vese (2010): the cartoon part (texture of scale < sigma removed)."""
    def ltv(x):
        gy, gx = np.gradient(x)
        return gauss(np.hypot(gx, gy), sigma)

    low = gauss(image, sigma)
    reduction = (ltv(image) - ltv(low)) / (ltv(image) + 1e-6)
    weight = np.clip((reduction - 0.25) / 0.25, 0, 1)  # texture where the low-pass removed much variation
    return weight * low + (1 - weight) * image


register(
    name="res_cartoon", title="Cartoon of the skin (fiber texture removed)", family=FAMILY,
    idea="Buades, Le, Morel & Vese (2010), fast cartoon + texture decomposition: where low-pass filtering "
         "removes most of the local variation the area is texture (the fiber weave) and is replaced by its "
         "smooth version; edges that survive the low-pass (stroke outlines) are kept sharp. The raw skin, "
         "without the fibers.",
)((lambda t, p: {"v": _cartoon(t.skin, _px(t, 30))}, _one("v")))


def _retinex(t, p):
    u = np.clip(_unit(t.skin, t.valid), 0.15, 1.5)  # dark specks would dominate the log
    return {"v": np.mean([np.log(u) - np.log(gauss(u, _px(t, s))) for s in (15, 60, 150)], axis=0)}


register(
    name="res_retinex", title="Multi-scale Retinex of the skin", family=FAMILY,
    idea="Jobson, Rahman & Woodell (1997): the log of the skin over its local average at three scales "
         "(15, 60, 150 µm), averaged. Removes slow density trends (thickness, the scan's shading) so equal "
         "local differences look equal everywhere.",
)((_retinex, _one("v")))


def _nlmeans(t, p):
    import cv2

    u = (np.clip(_unit(cue(t, 10), t.valid), 0, 1) * 255).astype(np.uint8)
    window = max(3, int(round(_px(t, 30))) | 1)
    return {"v": cv2.fastNlMeansDenoising(u, None, h=18, templateWindowSize=window,
                                          searchWindowSize=3 * window).astype(np.float32)}


register(
    name="res_nlmeans", title="Non-local means of the ink cue", family=FAMILY,
    idea="Buades, Coll & Morel (2005): each pixel is replaced by the mean of pixels whose surrounding "
         "patch looks alike, anywhere nearby, not just its neighbours. Repeated stroke edges reinforce each "
         "other; isolated noise does not. Applied to the ink cue (face step + skin − below).",
)((_nlmeans, _one("v")))


def _perona_malik(t, p, iterations: int = 25, kappa: float = 0.12, step: float = 0.2):
    u = np.clip(_unit(cue(t, 10), t.valid), -0.5, 1.5)
    for _ in range(iterations):
        flows = []
        for axis in (0, 1):
            forward = np.diff(u, axis=axis, append=np.take(u, [-1], axis=axis))
            backward = -np.diff(u, axis=axis, prepend=np.take(u, [0], axis=axis))
            flows += [np.exp(-((forward / kappa) ** 2)) * forward, np.exp(-((backward / kappa) ** 2)) * backward]
        u = u + step * sum(flows)
    return {"v": u}


register(
    name="res_perona_malik", title="Anisotropic diffusion of the ink cue", family=FAMILY,
    idea="Perona & Malik (1990): heat diffusion that slows down at strong edges. Noise and weak texture "
         "inside and outside strokes are smoothed away; the stroke outline, a strong edge, stops the "
         "diffusion and stays sharp.",
)((_perona_malik, _one("v")))


def _kuwahara(t, p):
    u = cue(t, 10)
    r = max(1, int(round(_px(t, 60))))
    size = r + 1
    mean = ndimage.uniform_filter(u, size=size, mode="nearest")
    var = ndimage.uniform_filter(u * u, size=size, mode="nearest") - mean**2
    pad = r
    pm, pv = np.pad(mean, pad, mode="edge"), np.pad(var, pad, mode="edge")
    half = r // 2 + 1
    offsets = [(-half, -half), (-half, half), (half, -half), (half, half)]
    means = np.stack([_shifted(pm, pad, dy, dx, u.shape) for dy, dx in offsets])
    variances = np.stack([_shifted(pv, pad, dy, dx, u.shape) for dy, dx in offsets])
    return {"v": np.take_along_axis(means, np.argmin(variances, axis=0)[None], 0)[0]}


register(
    name="res_kuwahara", title="Kuwahara filter of the ink cue", family=FAMILY,
    idea="Kuwahara (1976), the painterly edge-preserving filter: every pixel takes the mean of the "
         "quietest of its four 60 µm quadrants. Areas become flat patches with crisp borders, so strokes "
         "read as shapes.",
)((_kuwahara, _one("v")))


def _rolling_guidance(t, p, iterations: int = 4):
    source = cue(t, 5)
    radius = max(1, int(round(_px(t, 40))))
    guide = gauss(source, _px(t, 60))  # small structures (fibers, specks) removed first
    for _ in range(iterations):  # then edges of larger structures recovered
        guide = guided_filter(guide, source, radius, 0.02)
    return {"v": guide}


register(
    name="res_rolling_guidance", title="Rolling guidance filter of the ink cue", family=FAMILY,
    idea="Zhang, Shen, Xu & Jia (2014), rolling guidance: a Gaussian first removes every structure smaller "
         "than 60 µm (fibers, specks), then guided filtering with the original iteratively brings back the "
         "exact edges of what is larger (strokes). Scale-aware rather than contrast-aware smoothing.",
)((_rolling_guidance, _one("v")))


register(
    name="res_guided", title="Skin − below, guided by the face step", family=FAMILY,
    idea="He, Sun & Tang (2010), the guided filter as joint filtering: the noisier cue (skin − below) is "
         "smoothed over 50 µm, but with the edges of the cleaner one (the face step). Values from one "
         "measurement, outlines from the other.",
)((lambda t, p: {"v": guided_filter(np.clip(_unit(gauss(face_step(t), _px(t, 10)), t.valid), -0.5, 1.5),
                                    np.clip(_unit(skin_below(t), t.valid), -0.5, 1.5),
                                    max(1, int(round(_px(t, 50)))), 0.01)}, _one("v")))


# ----------------------------------------------------------------------------
# 14–17: features
# ----------------------------------------------------------------------------
def phase_congruency(image: np.ndarray, wavelengths: list[float], k: float = 2.0,
                     valid: np.ndarray | None = None) -> np.ndarray:
    """Kovesi's monogenic phase congruency: feature strength independent of local contrast, 0 … 1."""
    h, w = image.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1
    riesz = (1j * fx - fy) / radius
    spectrum = np.fft.fft2(image - data_pixels(image, valid).mean())
    sum_an = np.zeros((h, w))
    sum_f = np.zeros((h, w))
    sum_h1 = np.zeros((h, w))
    sum_h2 = np.zeros((h, w))
    tau = None
    for i, wavelength in enumerate(wavelengths):
        log_gabor = np.exp(-(np.log(radius * wavelength) ** 2) / (2 * np.log(0.55) ** 2))
        log_gabor[0, 0] = 0
        even = np.real(np.fft.ifft2(spectrum * log_gabor))
        odd = np.fft.ifft2(spectrum * log_gabor * riesz)
        amplitude = np.sqrt(even**2 + np.real(odd) ** 2 + np.imag(odd) ** 2)
        sum_an += amplitude
        sum_f += even
        sum_h1 += np.real(odd)
        sum_h2 += np.imag(odd)
        if i == 0:
            tau = np.median(data_pixels(sum_an, valid)) / np.sqrt(np.log(4))
    mult = wavelengths[1] / wavelengths[0] if len(wavelengths) > 1 else 2.0
    total_tau = tau * (1 - (1 / mult) ** len(wavelengths)) / (1 - 1 / mult)
    threshold = total_tau * np.sqrt(np.pi / 2) + k * total_tau * np.sqrt((4 - np.pi) / 2)
    energy = np.sqrt(sum_f**2 + sum_h1**2 + sum_h2**2)
    return np.maximum(energy - threshold, 0) / (sum_an + 1e-6)


register(
    name="res_phase_congruency", title="Phase congruency of the ink cue", family=FAMILY,
    idea="Morrone & Owens (1987), Kovesi (1999/2003): features are where the Fourier components of the "
         "image are in phase, independent of how strong the contrast is. Outlines of faint strokes count "
         "as much as those of strong ones. Monogenic version over 40–320 µm wavelengths.",
)((lambda t, p: {"v": gauss(phase_congruency(cue(t, 5), [_px(t, w) for w in (40, 80, 160, 320)], valid=t.valid), _px(t, 8))},
   _one("v", "inferno")))


def _frangi(t, p):
    from skimage.filters import frangi

    sigmas = [_px(t, s) for s in (40, 80, 160)]
    return {"v": frangi(cue(t, 5), sigmas=sigmas, black_ridges=False)}


register(
    name="res_frangi", title="Stroke-ness (Frangi vesselness) of the ink cue", family=FAMILY,
    idea="Frangi, Niessen, Vincken & Viergever (1998), multi-scale vesselness from the Hessian: high on "
         "bright elongated bands 80–320 µm wide, the width of pen strokes, low on blobs and on the much "
         "thinner fibers.",
)((_frangi, _one("v", "inferno")))


def _lbp(t, p):
    from skimage.feature import local_binary_pattern

    u8 = (np.clip(_unit(t.skin, t.valid), 0, 1) * 255).astype(np.uint8)
    radius = max(1, int(round(_px(t, 12))))
    var = local_binary_pattern(u8, P=8, R=radius, method="var")
    return {"v": gauss(np.log1p(np.nan_to_num(var)), _px(t, 30))}


register(
    name="res_lbp", title="Local binary pattern contrast of the skin", family=FAMILY,
    idea="Ojala, Pietikäinen & Mäenpää (2002), rotation-invariant LBP variance: texture contrast measured "
         "on a circle around each pixel, independent of the texture's direction. A covered, smoother "
         "(or a cracked, rougher) surface differs from the open fiber weave.",
)((_lbp, _one("v", "viridis")))


def _sauvola_compose(ch, t, p):
    from skimage.filters import threshold_sauvola

    c = ch["cue"]
    window = max(3, min(int(round(_px(t, 600))), min(c.shape) - 2) | 1)  # odd, smaller than the image
    mask = (c > threshold_sauvola(c, window_size=window, k=0.15)) & (c > np.percentile(t.pixels(c), 85))
    mask = ndimage.binary_opening(mask, iterations=max(1, int(round(_px(t, 15)))))
    edge = mask & ~ndimage.binary_erosion(mask, iterations=max(1, int(round(_px(t, 10)))))
    rgb = gray(0.15 + 0.7 * ch["base"])
    rgb[mask] = rgb[mask] * 0.55 + np.array([1.0, 0.65, 0.1]) * 0.45
    rgb[edge] = (1.0, 0.8, 0.2)
    return rgb


register(
    name="res_sauvola", title="Sauvola binarisation of the ink cue", family=FAMILY,
    idea="Sauvola & Pietikäinen (2000), the classic adaptive binarisation for degraded documents: a pixel is "
         "'ink' if the cue exceeds a threshold set by the mean and spread of its 0.6 mm surroundings. "
         "Shown as amber areas with outlines over the skin: a candidate transcription layer, not a "
         "measurement.",
    norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 30)}, _sauvola_compose))


# ----------------------------------------------------------------------------
# 18–20: archaeological terrain visualisation, the cue as terrain
# ----------------------------------------------------------------------------
def _svf(t, p):
    terrain, radius = _terrain(t)
    horizon = _horizon(terrain, radius)
    return {"v": 1 - np.mean(np.sin(np.maximum(horizon, 0)), axis=0)}


register(
    name="res_skyview", title="Sky-view factor of the ink cue as terrain", family=FAMILY,
    idea="Zakšek, Oštir & Kokalj (2011), the sky-view factor used to find faint earthworks in LiDAR: the "
         "share of the sky visible from each point of a terrain. Here the ink cue is the terrain; raised "
         "plateaus (strokes) see the whole sky and are bright, the ground between them is shaded by "
         "them. Direction-free, unlike hillshading.",
)((_svf, _one("v")))


def _rrim_channels(t, p):
    terrain, radius = _terrain(t)
    positive = np.mean(np.pi / 2 - _horizon(terrain, radius), axis=0)
    negative = np.mean(np.pi / 2 - _horizon(-terrain, radius), axis=0)
    gy, gx = np.gradient(terrain)
    return {"openness": (positive - negative) / 2, "slope": np.hypot(gx, gy)}


def _rrim_compose(ch, t, p):
    return hsv_to_rgb(np.zeros_like(ch["slope"]), 0.85 * ch["slope"], 0.1 + 0.9 * ch["openness"])


register(
    name="res_rrim", title="Red relief image map of the ink cue", family=FAMILY,
    idea="Chiba, Kaneta & Suzuki (2008), the Red Relief Image Map of Japanese archaeological survey: "
         "brightness = positive minus negative openness (ridges light, hollows dark), red saturation = "
         "slope. Reads like a relief without a light direction. The ink cue is the terrain.",
)((_rrim_channels, _rrim_compose))


def _mdow_channels(t, p):
    terrain, _ = _terrain(t, 80)
    gy, gx = np.gradient(terrain)
    aspect = np.arctan2(gy, gx)
    total, weight_sum = 0, 0
    for azimuth in (225, 270, 315, 360):
        weight = np.sin(aspect - np.deg2rad(azimuth)) ** 2 + 1e-3
        total = total + weight * hillshade(terrain, azimuth, 35, 3.0)
        weight_sum = weight_sum + weight
    return {"v": total / weight_sum}


register(
    name="res_mdow", title="Multi-directional hillshade of the ink cue", family=FAMILY,
    idea="Mark (1992), the multi-directional oblique-weighted hillshade of cartography: four lights from "
         "the north-west quadrant, each weighted by how oblique it is to the local slope, so no ridge is "
         "lost in its own shadow. The ink cue becomes an embossed relief, letters raised.",
    norms={"v": "unit"},
)((_mdow_channels, _one("v")))


# ----------------------------------------------------------------------------
# 21–27: presenting the cue to the eye
# ----------------------------------------------------------------------------
def _views(t: Tile, ch: dict, disparity_um: float = 30.0):
    """Left and right views of the gray skin with the ink cue as depth (horizontal parallax)."""
    h, w = t.shape2d
    shift = (np.clip(ch["cue"], 0, 1) - 0.5) * _px(t, disparity_um)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    left = ndimage.map_coordinates(ch["base"], [yy, xx + shift], order=1, mode="nearest")
    right = ndimage.map_coordinates(ch["base"], [yy, xx - shift], order=1, mode="nearest")
    return left, right


def _anaglyph_compose(ch, t, p):
    left, right = _views(t, ch)
    return np.stack([left, right, right], axis=-1)


register(
    name="res_anaglyph", title="Red/cyan anaglyph: ink cue as depth", family=FAMILY,
    idea="Stereoscopy: the familiar gray skin, but each pixel is shifted sideways by the ink cue, red view "
         "for the left eye, cyan for the right. With red/cyan glasses ink-like areas float in front of the "
         "papyrus while the texture stays untouched. Depth is a separate perceptual channel from "
         "brightness and colour.",
    norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 40)}, _anaglyph_compose))


def _wiggle_compose(ch, t, p):
    left, right = _views(t, ch, 40.0)
    middle = 0.5 * (left + right)
    return [gray(f) for f in (left, middle, right, middle)]


register(
    name="res_wiggle", title="Wiggle stereo: ink cue as depth (animated)", family=FAMILY,
    idea="Wiggle stereoscopy (no glasses): the two stereo views alternate, so the ink cue is seen as motion "
         "parallax. Ink-like areas shift against the papyrus while it rocks; the human visual system "
         "picks up coherent motion of faint shapes far better than faint contrast.",
    animated=True, norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 40)}, _wiggle_compose))


def _isoluminant_compose(ch, t, p):
    lightness = 30 + 55 * ch["base"]
    strength = np.clip((ch["cue"] - 0.5) * 2, 0, 1)
    chroma = 55 * strength
    hue = np.deg2rad(70)  # orange-yellow
    lab = np.stack([lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1)
    return lab_to_rgb(lab.astype(np.float32))


register(
    name="res_isoluminant", title="Isoluminant colour: skin texture + ink cue as hue", family=FAMILY,
    idea="Perceptual colour (CIELAB; Ware, Information Visualization): lightness is the skin exactly as it "
         "is, the ink cue only adds chroma at constant lightness. The texture a papyrologist reads is not "
         "altered; ink-like areas are tinted amber on top of it.",
    norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 30)}, _isoluminant_compose))


def _contour_compose(ch, t, p):
    rgb = gray(0.1 + 0.8 * ch["base"])
    c = ch["cue"]
    colours = [(0.3, 0.6, 1.0), (1.0, 0.75, 0.2), (1.0, 0.3, 0.2)]
    for level, colour in zip(np.percentile(t.pixels(c, 4), [75, 88, 95]), colours):
        above = c > level
        line = above ^ ndimage.binary_erosion(above, iterations=max(1, int(round(_px(t, 20)))))
        rgb[line] = colour
    return rgb


register(
    name="res_contours", title="Ink-cue contour lines over the skin", family=FAMILY,
    idea="Cartographic isolines: the skin unchanged, with three contour lines of the smoothed ink cue (its "
         "75th, 88th and 95th percentile: blue, amber, red). Nested rings mark where the cue peaks; the "
         "eye follows lines better than faint shading.",
    norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 60)}, _contour_compose))

_POSTER = np.array([(25, 25, 35), (55, 65, 95), (80, 120, 120), (190, 160, 80), (250, 215, 110)], np.float32) / 255


def _posterize_compose(ch, t, p):
    c = ch["cue"]
    levels = np.percentile(t.pixels(c, 4), [50, 75, 88, 95])
    index = np.searchsorted(levels, c)
    return _POSTER[index] * (0.7 + 0.3 * ch["base"][..., None])


register(
    name="res_posterize", title="Posterised ink cue (five classes)", family=FAMILY,
    idea="Posterisation / cel shading: the smoothed ink cue in five flat classes by its percentiles "
         "(50, 75, 88, 95), lightly shaded by the skin. Flat regions with hard borders let the eye compare "
         "shapes instead of shades, as on a choropleth map.",
    norms={"base": "unit"},
)((lambda t, p: {"base": _skin_l(t), "cue": cue(t, 50)}, _posterize_compose))


def _ridge_overlay(t, p):
    return {"base": _skin_l(t), "heat": _ridge(cue(t, 15), _px(t, 120))}


register(
    name="res_ridge_glow", title="Stroke ridges glowing over the skin", family=FAMILY,
    idea="Lindeberg (1998) scale-space ridge detection, shown as a glow: the Hessian ridge strength of the "
         "ink cue at stroke width (120 µm), as heat over the unchanged skin. The brute-force search's best "
         "ingredient presented as an overlay rather than as the image itself.",
    norms={"base": "unit"},
)((_ridge_overlay, lambda ch, t, p: gray(ch["base"]) * (1 - 0.8 * ch["heat"][..., None])
   + colormap(ch["heat"], "inferno") * 0.8 * ch["heat"][..., None]))
