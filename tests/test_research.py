import numpy as np
import pytest

from colored_inkt import research as R
from colored_inkt import visualizers as V
from tests.test_visualizers import synthetic_sheet

NAMES = [n for n, v in V.REGISTRY.items() if v.family == R.FAMILY]


def test_at_least_twenty_research_methods():
    assert len(NAMES) >= 20


@pytest.mark.parametrize("name", NAMES)
def test_research_method_renders(name):
    tile = V.Tile(synthetic_sheet(size=128)[0])
    vis = V.REGISTRY[name]
    out = V.render(vis, tile, V.calibrate(vis, [tile]))
    frames = out if isinstance(out, list) else [out]
    for frame in frames:
        assert frame.shape == (128, 128, 3) and np.isfinite(frame).all()


def test_lab_round_trip():
    from colored_inkt.ops import to_lab

    rgb = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    np.testing.assert_allclose(R.lab_to_rgb(to_lab(rgb)), rgb, atol=2e-3)


def test_guided_filter_keeps_the_guides_edge():
    guide = np.zeros((64, 64), np.float32)
    guide[:, 32:] = 1
    noisy = guide + np.random.default_rng(0).normal(0, 0.3, guide.shape).astype(np.float32)
    out = R.guided_filter(guide, noisy, 4, 1e-3)
    assert out[:, 36:].mean() - out[:, :28].mean() > 0.9 and out[:, 36:].std() < 0.1


def test_phase_congruency_marks_a_step_regardless_of_its_contrast():
    image = np.zeros((128, 128))
    image[:, 64:] = 1
    weak, strong = (R.phase_congruency(a * image, [4, 8, 16]) for a in (0.1, 10))
    assert weak[:, 60:68].max() > 0.5 and abs(weak[:, 60:68].max() - strong[:, 60:68].max()) < 0.05
