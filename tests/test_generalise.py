import numpy as np
import pytest

from colored_inkt import generalise as G
from colored_inkt import visualizers as V
from tests.test_visualizers import synthetic_sheet

NAMES = [n for n, v in V.REGISTRY.items() if v.family == G.FAMILY]


def _tile_with_skin(skin: np.ndarray) -> V.Tile:
    """A level-2 tile (9.6 µm pixels) whose every flattened plane is ``skin``."""
    h, w = skin.shape
    flat = np.repeat(skin[None], len(V.FLAT_D), axis=0)
    zeros = np.zeros((h, w), np.float32)
    products = {"mid_plane": skin, "window_mean": skin, "window_max": skin, "coarse_height": zeros,
                "height": zeros, "flat": flat}
    return V.Tile.from_products(products, xy_scale=4)


def test_three_generalisation_attempts_are_registered():
    titles = [V.REGISTRY[n].title for n in NAMES]
    assert len(NAMES) == 3 and all(t.startswith(f"Generalisation attempt {i + 1}:") for i, t in enumerate(titles))


@pytest.mark.parametrize("name", NAMES)
def test_generalisation_method_renders(name):
    tile = V.Tile(synthetic_sheet(size=128)[0])
    vis = V.REGISTRY[name]
    for t in (tile, tile.despeckled(vis.prefilter[1])):
        out = V.render(vis, t, V.calibrate(vis, [t]))
        assert out.shape == (128, 128, 3) and np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_local_z_removes_a_slow_trend_and_scale():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    tile = _tile_with_skin(np.zeros((256, 256), np.float32))
    z = G.local_z(50 + x * 0.5 + 10 * rng.normal(size=(256, 256)).astype(np.float32), tile, um=300)
    inner = z[40:-40, 40:-40]
    assert abs(inner.mean()) < 0.1 and 0.8 < inner.std() < 1.2


def test_crackle_is_high_on_a_rough_unoriented_patch_among_fibres():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    skin = 100 + 10 * np.sin(2 * np.pi * y / 4)  # horizontal fibres
    patch = (np.abs(y - 128) < 30) & (np.abs(x - 128) < 30)
    skin[patch] = 100 + 14 * rng.normal(size=patch.sum())  # cracked, unoriented
    c = G.crackle(_tile_with_skin(skin.astype(np.float32)))
    assert c[patch].mean() - c[~patch].mean() > 1.0


def test_kept_drops_specks_and_keeps_strokes():
    x = np.zeros((256, 256), np.float32)
    x[100:110, 40:200] = 3  # a stroke: 0.1 × 1.5 mm at 9.6 µm
    x[30:33, 30:33] = 3  # a speck
    mask = G.kept(x, _tile_with_skin(x), 1.0, 2.0, 0.1)
    assert mask[105, 120] > 0.9 and mask[31, 31] < 0.1


def test_default_tints_keep_the_amber_and_blue_hues_at_full_chroma():
    for tint, hue in zip(G.TINTS, (70, 230)):
        a, b = G.tint_vector(tint["color"])
        assert np.hypot(a, b) == pytest.approx(G.TINT_CHROMA)
        assert np.degrees(np.arctan2(b, a)) % 360 == pytest.approx(hue, abs=1.0)
    assert np.allclose(G.tint_vector("#808080"), 0)  # a gray has no hue: no tint


def test_two_hue_channels_are_raw_cues_shown_through_fixed_ranges():
    vis = V.REGISTRY["gen_two_hue"]
    tile = V.Tile(synthetic_sheet(size=128)[0]).despeckled(50)
    calibration = V.calibrate(vis, [tile])
    assert calibration["ranges"]["density"] == list(G.DENSITY_FLOOR)
    assert calibration["ranges"]["crackle"] == list(G.CRACKLE_FLOOR)
    assert vis.display["kind"] == "tints" and [t["channel"] for t in vis.display["tints"]] == ["density", "crackle"]
    d, c = G.cleaned(tile)
    shown_d, _ = G.shown(tile)
    assert (d <= np.maximum(shown_d, G.DENSITY_FLOOR[0]) + 1e-5).all()  # cleaning only lowers values
