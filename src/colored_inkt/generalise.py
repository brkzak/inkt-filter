"""Methods that should work on every segment (family ``generalisation``).

The per-segment analysis of the starred methods (README, "Why filters don't
generalise") found two different physical signatures of ink:

* **density**: on PHerc0009B, 0139, 0500P2, 0814, 0841 and 1667 the ink adds
  density in the outer 2–10 µm of the sheet (the face step and skin − below
  of ``research.cue``); the skin's fine texture there is unchanged or
  *smoother* (the deposit fills the fibre weave);
* **crackle**: on PHercParis4 there is hardly any extra density, but the skin
  under the ink is *rougher* and *less fibre-oriented*: the cracked ink layer
  that was first seen by eye in that scroll ("crackle"). LBP contrast picks
  this up; every density method misses it.

Each cue below is standardised against its own ~1.5 mm neighbourhood (minus
the local mean, over the local spread), so both signatures are in the same
units (local z-scores) on every segment and scan, and slow density or
contrast changes across the sheet drop out. The two are then combined
(attempt 1) or shown as two hues (attempts 2 and 3). Nothing
here uses labels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from colored_inkt.research import _memo, _px, _ridge_overlay, _skin_l, face_step, lab_to_rgb, skin_below
from colored_inkt.visualizers import VERSION, Tile, colormap, data_pixels, dog, gauss, gray, masked_gauss, register

FAMILY = "generalisation"
SMOOTH_UM = 40.0  # every cue is smoothed this much before standardising
LOCAL_UM = 1500.0  # neighbourhood of the local standardisation
DENSITY_UM, CRACKLE_UM = 20.0, 100.0  # display smoothing: crackle is only coherent at stroke scale


def local_z(x: np.ndarray, t: Tile, um: float = LOCAL_UM) -> np.ndarray:
    """``x`` minus its local mean over its local standard deviation (Gaussian windows of ``um``).

    The window (1.5 mm) is wider than a block; with a mask the local statistics weigh only the pixels with
    data (``masked_gauss``), so the empty part of a block does not set them.
    """
    sigma = _px(t, um)
    centred = x - masked_gauss(x, sigma, t.valid)
    return (centred / (np.sqrt(masked_gauss(centred**2, sigma, t.valid)) + 1e-6)).astype(np.float32)


def density(t: Tile) -> np.ndarray:
    """Ink as extra density at the face: mean local z of the face step and skin − below."""
    def make():
        s = _px(t, SMOOTH_UM)
        return 0.5 * (local_z(gauss(face_step(t), s), t) + local_z(gauss(skin_below(t), s), t))

    return _memo(t, "gen_density", make)


def crackle(t: Tile) -> np.ndarray:
    """Ink as a rough, unoriented skin: mean local z of fine roughness and of (−) fibre coherence."""
    def make():
        skin = t.skin
        rough = np.log(np.sqrt(gauss(dog(skin, t.xy(1), t.xy(4)) ** 2, t.xy(12))) + 1e-3)
        gy, gx = np.gradient(gauss(skin, t.xy(2)))
        jxx, jyy, jxy = (gauss(v, t.xy(16)) for v in (gx * gx, gy * gy, gx * gy))
        coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-9)
        s = _px(t, SMOOTH_UM)
        return 0.5 * (local_z(gauss(rough, s), t) - local_z(gauss(coherence, s), t))

    return _memo(t, "gen_crackle", make)


def _unit_spread(x: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    return (x / (data_pixels(x, valid).std() + 1e-6)).astype(np.float32)


def shown(t: Tile) -> tuple[np.ndarray, np.ndarray]:
    """(density, crackle) smoothed for display and scaled to unit spread on the tile, so they weigh the same."""
    return _memo(t, "gen_shown", lambda: (_unit_spread(gauss(density(t), _px(t, DENSITY_UM)), t.valid),
                                          _unit_spread(gauss(crackle(t), _px(t, CRACKLE_UM)), t.valid)))


DENSITY_FLOOR, CRACKLE_FLOOR = (1.0, 2.8), (0.6, 2.2)  # display ranges (units of each cue's spread)
# Cleaning (as in document binarisation): hysteresis thresholds, then components smaller than a stroke go.
DENSITY_KEEP, CRACKLE_KEEP = (1.0, 1.8, 0.15), (0.9, 1.6, 0.3)  # low, high (spreads), smallest area (mm²)


def kept(x: np.ndarray, t: Tile, low: float, high: float, area_mm2: float) -> np.ndarray:
    """Soft mask (0 … 1) of the stroke-sized components of ``x`` > ``low`` that reach ``high``."""
    from skimage.filters import apply_hysteresis_threshold

    components, _ = ndimage.label(apply_hysteresis_threshold(x, low, high))
    sizes = np.bincount(t.pixels(components).ravel(), minlength=components.max() + 1)  # area with data
    big = sizes >= area_mm2 * 1e6 / (2.4 * t.xy_scale) ** 2
    big[0] = False  # the background
    return gauss(big[components].astype(np.float32), _px(t, 10))


def heats(t: Tile) -> tuple[np.ndarray, np.ndarray]:
    """(density, crackle) as 0 … 1 heat over their own floors (the same on every segment), cleaned."""
    def make():
        d, c = shown(t)
        return (_heat(d, *DENSITY_FLOOR) * kept(d, t, *DENSITY_KEEP),
                _heat(c, *CRACKLE_FLOOR) * kept(c, t, *CRACKLE_KEEP))

    return _memo(t, "gen_heats", make)


def _heat(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def _glow_compose(ch, t, p):
    heat = ch["heat"][..., None]
    return gray(ch["base"]) * (1 - 0.8 * heat) + colormap(ch["heat"], "inferno") * 0.8 * heat


_glow_compose.display = {"kind": "overlay", "base": "base", "heat": "heat", "cmap": "inferno", "strength": 0.8}

register(
    name="gen_combined", title="Generalisation attempt 1: density or crackle glow", family=FAMILY,
    idea="Both physical signatures of ink in one glow over the unchanged skin: extra density at the face "
         "(the ink on most scrolls) or a rough, unoriented skin (crackle: the ink on PHercParis4), whichever "
         "is stronger. Each cue is a z-score against its own 1.5 mm surroundings, scaled to unit spread, "
         "and starts to glow above a fixed floor (density 1.0, crackle 0.6 spreads), so a segment without "
         "ink stays mostly gray. Renders on the despeckled skin (mineral grains also look rough).",
    norms={"base": "unit", "heat": "unit"}, prefilter=(0, 50),
)((lambda t, p: {"base": _skin_l(t), "heat": np.maximum(*heats(t))}, _glow_compose))


def cleaned(t: Tile) -> tuple[np.ndarray, np.ndarray]:
    """(density, crackle) in spreads, outside their stroke-sized components capped at the display floor.

    Through the floors (``DENSITY_FLOOR``, ``CRACKLE_FLOOR``) they look like ``heats``; as raw values the
    viewer can show their histograms and move the floors (below a floor the uncleaned cue reappears).
    """
    def make():
        out = []
        for x, floor, keep in zip(shown(t), (DENSITY_FLOOR, CRACKLE_FLOOR), (DENSITY_KEEP, CRACKLE_KEEP)):
            k = kept(x, t, *keep)
            out.append((k * x + (1 - k) * np.minimum(x, floor[0])).astype(np.float32))
        return tuple(out)

    return _memo(t, "gen_cleaned", make)


TINTS = ({"channel": "density", "color": "#eda047", "label": "density"},  # CIELAB hue 70°: amber
         {"channel": "crackle", "color": "#29abcd", "label": "crackle"})  # hue 230°: blue
TINT_CHROMA = 60.0  # CIELAB chroma of a tint at full heat and intensity 1


def tint_vector(color: str, intensity: float = 1.0) -> np.ndarray:
    """(a, b) CIELAB chroma that a tint adds at full heat: the colour's hue, ``TINT_CHROMA`` × intensity."""
    from colored_inkt.ops import to_lab

    rgb = np.array([[int(color[i : i + 2], 16) for i in (1, 3, 5)]], np.float32) / 255
    ab = to_lab(rgb)[0, 1:].astype(np.float64)
    norm = np.hypot(*ab)
    return np.zeros(2) if norm < 1.0 else ab / norm * TINT_CHROMA * intensity  # a gray: no hue, no tint


def tints_compose(ch, t, p):
    """Isoluminant tints (CIELAB): lightness from the gray base, each tint adds chroma in its own hue.

    The viewer reproduces this live (``display`` kind ``tints``) with editable colours, intensities and
    ranges; keep the two in step.
    """
    lightness = 30 + 55 * ch["base"]
    ab = sum(ch[tint["channel"]][..., None] * tint_vector(tint["color"]).astype(np.float32) for tint in TINTS)
    return lab_to_rgb(np.concatenate([lightness[..., None], ab], axis=-1).astype(np.float32))


tints_compose.display = {"kind": "tints", "base": "base", "lightness": [30, 55], "chroma": TINT_CHROMA,
                         "tints": [dict(t) for t in TINTS]}

register(
    name="gen_two_hue", title="Generalisation attempt 2: density amber, crackle blue", family=FAMILY,
    idea="The two signatures of ink as two hues at constant lightness (CIELAB) over the unchanged skin: "
         "extra density at the face tints amber, a rough, unoriented skin (crackle) tints blue; where both "
         "occur the tints mix towards gray-violet. Whichever way a scroll's ink shows, its letters are "
         "coloured, and the hue tells which signature it is. Same cues and floors as attempt 1, despeckled. "
         "In the viewer each tint's colour, intensity and range (in spreads of the cue) can be changed.",
    norms={"base": "unit", "density": DENSITY_FLOOR, "crackle": CRACKLE_FLOOR}, prefilter=(0, 50),
    version=VERSION + 1,  # channels are raw cues since the editable tints (renders cached under v5, at VERSION 4)
)((lambda t, p: dict(zip(("base", "density", "crackle"), (_skin_l(t), *cleaned(t)))), tints_compose))


def _ridge_crackle_compose(ch, t, p):
    heat = ch["ridges"][..., None]
    rgb = gray(ch["base"]) * (1 - 0.8 * heat) + colormap(ch["ridges"], "inferno") * 0.8 * heat
    blue = np.array([0.15, 0.65, 1.0], np.float32)
    tint = 0.55 * ch["crackle"][..., None] * (1 - heat)
    return rgb * (1 - tint) + blue * tint


def _ridge_crackle(t, p):
    glow = _ridge_overlay(t, p)
    return {"base": glow["base"], "ridges": glow["heat"], "crackle": heats(t)[1]}


register(
    name="gen_ridge_crackle", title="Generalisation attempt 3: stroke ridges glowing, crackle blue", family=FAMILY,
    idea="The presentation that read best by eye on the density scrolls (stroke ridges of the ink cue glowing "
         "over the skin, as 'Stroke ridges glowing over the skin') with the second signature added: "
         "stroke-sized areas of rough, unoriented skin (crackle, the ink of PHercParis4) tinted blue. "
         "Despeckled.",
    norms={"base": "unit", "ridges": "robust", "crackle": "unit"}, prefilter=(0, 50),
)((_ridge_crackle, _ridge_crackle_compose))
