"""Tests for the stand-alone parts: reading volumes, the 2.4 µm geometry, tiled rendering, the viewer's store,
the server's add-segment checks and the command line. All on small synthetic stacks, without network."""

import io
import json
import time

import numpy as np
import pytest
import tifffile
from PIL import Image

from colored_inkt import render as rendering
from colored_inkt import surface
from colored_inkt.cli import main
from colored_inkt.server import add_segment
from colored_inkt.store import Store
from colored_inkt.volume import open_volume
from tests.test_visualizers import synthetic_sheet


# --- Geometry and reading -----------------------------------------------------------------------------------

def test_geometry_scales_and_checks():
    """xy_scale is the binned pixel in 2.4 µm units (the filters' length unit); a bad side or size must fail
    rather than silently render the wrong face or scale."""
    assert surface.Geometry(layer_um=2.4, pixel_um=7.2, bin=2).xy_scale == pytest.approx(6.0)
    assert not surface.Geometry(layer_um=2.2, pixel_um=2.2).resamples  # open data: within 10 %
    assert surface.Geometry(layer_um=7.91, pixel_um=7.91).resamples
    for bad in ({"layer_um": 0, "pixel_um": 1}, {"layer_um": 1, "pixel_um": 1, "side": 0},
                {"layer_um": 1, "pixel_um": 1, "bin": 0}):
        with pytest.raises(ValueError):
            surface.Geometry(**bad)


def test_resampling_puts_planes_at_2_4_um_around_the_middle_layer():
    """A coarse scan is turned into 2.4 µm planes centred on the middle layer (where the segment lies), so the
    filters' plane offsets keep their physical meaning. A stack linear in depth must stay exactly linear."""
    geometry = surface.Geometry(layer_um=7.2, pixel_um=7.2)
    stack = (np.arange(11, dtype=np.float32) * 10)[:, None, None] * np.ones((1, 2, 2), np.float32)
    planes = geometry.planes(11)
    assert planes[len(planes) // 2] == pytest.approx(5.0)
    np.testing.assert_allclose(np.diff(planes), 1 / 3)
    np.testing.assert_allclose(surface.resample_layers(stack, geometry)[:, 0, 0], planes * 10, rtol=1e-6)


def test_binning_ignores_empty_pixels():
    """Pixels outside the traced segment are 0 in every layer; averaging them in would darken its border."""
    raw = np.full((3, 4, 4), 100, np.uint8)
    raw[:, 0, 0] = 0
    values, valid = surface.bin_block(raw, 2)
    np.testing.assert_array_equal(values[:, 0, 0], 100)
    assert valid.all()
    raw[:, :2, :2] = 0
    assert not surface.bin_block(raw, 2)[1][0, 0]


def test_area_reads_zero_outside_and_scales_16_bit():
    """Reads past the edge give zeros (marked invalid), and 16-bit data is brought to the 0–255 range the
    filters were tuned on."""
    array = np.full((5, 8, 8), 257 * 200, np.uint16)
    area = surface.Area(open_volume_from(array), (0, 0, 8, 8), surface.Geometry(2.4, 2.4))
    planes, valid = area.read(-2, -2, 6, 6)
    assert planes.dtype == np.float32 and planes[:, 2:, 2:].max() == pytest.approx(200)
    assert not valid[:2].any() and valid[2:, 2:].all()


def open_volume_from(array):
    return surface.Volume(array.shape, array.dtype, lambda y0, y1, x0, x1: array[:, y0:y1, x0:x1], "test")


def test_volume_readers_agree(tmp_path):
    """A layer folder (natural order: 2 before 10), a multi-page TIFF and a .npy give the same stack."""
    stack = np.random.default_rng(0).integers(1, 255, (12, 20, 30), dtype=np.uint8)
    folder = tmp_path / "layers"
    folder.mkdir()
    for i, layer in enumerate(stack):
        tifffile.imwrite(folder / f"{i}.tif", layer)
    tifffile.imwrite(tmp_path / "stack.tif", stack)
    np.save(tmp_path / "stack.npy", stack)
    for source in (folder, tmp_path / "stack.tif", tmp_path / "stack.npy"):
        volume = open_volume(source)
        assert volume.shape == stack.shape
        np.testing.assert_array_equal(volume.read(0, 0, 20, 30), stack)
        np.testing.assert_array_equal(volume.read(15, 25, 10, 10)[:, :5, :5], stack[:, 15:, 25:])


def test_volume_errors(tmp_path):
    """Layers of different sizes and missing paths fail with a message, not a crash later on."""
    with pytest.raises(FileNotFoundError):
        open_volume(tmp_path / "nope")
    tifffile.imwrite(tmp_path / "0.tif", np.zeros((4, 4), np.uint8))
    tifffile.imwrite(tmp_path / "1.tif", np.zeros((4, 5), np.uint8))
    with pytest.raises(ValueError, match="differ in size"):
        open_volume(tmp_path)


# --- Filter names and tiled rendering -----------------------------------------------------------------------

def test_short_names_resolve():
    """The command line and URLs use short names; each must point to one filter."""
    assert rendering.resolve("two_hue").name == "gen_two_hue"
    assert rendering.resolve("res_lbp").name == "res_lbp"
    assert rendering.prefilters(rendering.resolve("two_hue")) == (0, 50)  # despeckled at 50 µm, always
    with pytest.raises(KeyError):
        rendering.resolve("no_such_filter")
    for name in rendering.REGISTRY:
        assert rendering.resolve(name).name == name


def test_feathers_add_up_to_one_over_an_overlap():
    """Two neighbouring tiles must blend to full weight, or a seam appears along every tile edge."""
    left = rendering.feather(rendering.CORE, True, False)[-rendering.OVERLAP:]
    right = rendering.feather(rendering.CORE, False, True)[: rendering.OVERLAP]
    np.testing.assert_allclose(left + right, 1, atol=1e-6)
    assert (rendering.feather(rendering.CORE, False, False) > 0).all()


def sheet_products(size=160, seed=0, empty_corner=False):
    ct, _ = synthetic_sheet(size=size, seed=seed)
    if empty_corner:
        ct[:, :40, :40] = 0
    geometry = surface.Geometry(layer_um=2.4, pixel_um=2.4)
    area = surface.Area(open_volume_from(ct), (0, 0, size, size), geometry)
    return surface.compute_products(area), geometry


def test_tiled_rendering_matches_one_piece(monkeypatch, tmp_path):
    """Large areas are rendered in feathered tiles. For a filter that works pixel by pixel the tiles must give
    the same image as one piece, so any seam would be a tiling bug. Pixels without data are black."""
    products, geometry = sheet_products(empty_corner=True)
    whole = dict(rendering.render_area(products, geometry, ["mid_slice"], log=lambda m: None))["mid_slice"]
    monkeypatch.setattr(rendering, "calibration_windows", lambda h, w: [(slice(0, h), slice(0, w))])
    monkeypatch.setattr(rendering, "ONE_PIECE", 64)
    monkeypatch.setattr(rendering, "CORE", 80)
    monkeypatch.setattr(rendering, "CONTEXT", 16)
    monkeypatch.setattr(rendering, "OVERLAP", 16)
    tiled = dict(rendering.render_area(products, geometry, ["mid_slice"], scratch=tmp_path,
                                       log=lambda m: None))["mid_slice"]
    assert np.abs(tiled.astype(int) - whole.astype(int)).max() <= 1
    assert not whole[:40, :40].any() and whole[60:, 60:].any()


def test_blocks_join_without_seams(monkeypatch):
    """The face is found block by block with a context ring; with a ring wider than the search's reach, small
    blocks must find the same face as one block."""
    one, _ = sheet_products(size=128)
    monkeypatch.setattr(surface, "BLOCK", 64)
    blocks, _ = sheet_products(size=128)
    inner = (slice(8, -8), slice(8, -8))
    assert np.median(np.abs(blocks["height"] - one["height"])[inner]) < 0.05
    np.testing.assert_array_equal(blocks["valid"], one["valid"])


# --- The viewer's store and the server's checks -------------------------------------------------------------

def wait_ready(store, seg_id, timeout=300):
    start = time.time()
    while not store.ready(seg_id):
        assert not store.status.get(seg_id, "").startswith("failed"), store.status[seg_id]
        assert time.time() - start < timeout, "preparation did not finish"
        time.sleep(0.2)


def test_store_adds_prepares_renders_and_deletes(tmp_path):
    """An added segment at 7.2 µm is cropped, resampled and prepared in the background; its render is black where
    the scan is empty; deleting it removes its folder. The ten examples are listed but not downloaded."""
    ct, _ = synthetic_sheet(size=128, depth=40)
    ct[:, :, :20] = 0
    label = np.zeros(ct.shape[1:], np.uint8)
    label[50:70, 50:70] = 255
    store = Store(tmp_path)
    assert sum(seg.kind == "example" for seg in store.segments.values()) == 10
    seg = store.add(title="my sheet", source=open_volume_from(ct), layer_um=7.2, pixel_um=7.2, crop=(0, 0, 128, 101),
                    label=label)
    assert (seg.h, seg.w) == (128, 100)  # even sides
    wait_ready(store, seg.id)
    assert (store.folder(seg.id) / "label.png").exists()
    payload, kind = store.render(seg.id, "mid_slice")
    rgb = np.asarray(Image.open(io.BytesIO(payload)))
    assert kind == "image/png" and rgb.shape[:2] == (128, 100)
    assert not rgb[:, :20].any() and rgb[:, 30:].any()
    store.delete(seg.id)
    assert seg.id not in store.segments and not store.folder(seg.id).exists()


def test_add_segment_checks_its_input(tmp_path):
    """The server asks for the voxel size when the data does not give it, and rejects a label of another size."""
    np.save(tmp_path / "stack.npy", np.ones((8, 80, 80), np.uint8))
    Image.fromarray(np.zeros((50, 50), np.uint8)).save(tmp_path / "label.png")
    store = Store(tmp_path / "data")
    with pytest.raises(ValueError, match="layer spacing"):
        add_segment(store, {"path": str(tmp_path / "stack.npy")})
    with pytest.raises(ValueError, match="label"):
        add_segment(store, {"path": str(tmp_path / "stack.npy"), "layer_um": 2.4,
                            "label_path": str(tmp_path / "label.png")})
    with pytest.raises(FileNotFoundError):
        add_segment(store, {"path": str(tmp_path / "missing"), "layer_um": 2.4})


# --- Command line -------------------------------------------------------------------------------------------

def test_render_command_writes_images_and_record(tmp_path):
    """``colored-inkt render`` writes one PNG per filter, a record of the settings, and removes its work files."""
    ct, _ = synthetic_sheet(size=96)
    np.save(tmp_path / "stack.npy", ct)
    out = tmp_path / "out"
    main(["render", str(tmp_path / "stack.npy"), "-o", str(out), "--voxel-um", "2.4", "-f", "mid_slice", "lbp",
          "--preview"])
    assert sorted(p.name for p in out.iterdir()) == ["lbp.png", "lbp_quarter.jpg", "mid_slice.png",
                                                     "mid_slice_quarter.jpg", "render.json"]
    assert Image.open(out / "lbp.png").size == (96, 96)
    record = json.loads((out / "render.json").read_text())
    assert record["filters"] == ["mid_slice", "res_lbp"] and record["layer_um"] == 2.4


def test_render_command_needs_a_voxel_size(tmp_path):
    np.save(tmp_path / "stack.npy", np.ones((8, 80, 80), np.uint8))
    with pytest.raises(SystemExit, match="voxel size"):
        main(["render", str(tmp_path / "stack.npy"), "-o", str(tmp_path / "out")])
