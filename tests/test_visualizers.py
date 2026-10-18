import numpy as np
import pytest

from colored_inkt import visualizers as V


def synthetic_sheet(size=96, depth=64, seed=0):
    """Papyrus below a wavy surface (ink face towards +z), fiber stripes, a raised 'stroke'."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    surface = depth / 2 + 3 * np.sin(x / 12) + 2 * np.cos(y / 15)
    stroke = ((x - 48) ** 2 + (y - 48) ** 2) < 15**2
    surface = surface + 1.5 * stroke
    z = np.arange(depth, dtype=np.float32)[:, None, None]
    papyrus = 1 / (1 + np.exp((z - surface[None]) / 0.8))  # smooth fall at the surface
    fibers = 8 * np.sin(2 * np.pi * x / 6) + 6 * np.sin(2 * np.pi * y / 9)
    ct = 40 + papyrus * (90 + fibers[None]) + rng.normal(0, 3, (depth, size, size))
    return np.clip(ct, 0, 255).astype(np.uint8), surface


def test_sample_depth_is_linear_interpolation():
    volume = np.arange(10, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), np.float32) * 3
    z = np.full((4, 4), 2.25, np.float32)
    np.testing.assert_allclose(V.sample_depth(volume, z), 6.75)


def test_surface_recovers_wavy_face():
    ct, surface = synthetic_sheet()
    tile = V.Tile(ct, side=1)
    inner = (slice(8, -8), slice(8, -8))
    error = np.abs(tile.height - surface)[inner]
    assert np.median(error) < 0.5
    assert np.percentile(error, 95) < 1.5


def test_flipped_side_finds_mirrored_face():
    ct, surface = synthetic_sheet()
    tile = V.Tile(ct[::-1].copy(), side=-1)
    mirrored = ct.shape[0] - 1 - surface
    assert np.median(np.abs(tile.height - mirrored)[8:-8, 8:-8]) < 0.6


def test_flat_plane_zero_is_the_surface_value():
    ct, _ = synthetic_sheet()
    tile = V.Tile(ct)
    i0 = int(-V.FLAT_D[0])
    np.testing.assert_allclose(tile.flat[i0], V.sample_depth(tile.volume, tile.height), rtol=1e-6)
    # inside the sheet is denser than in front of it
    assert tile.band(2, 6).mean() > tile.band(-6, -2).mean() + 40


def test_dominant_orientations_find_both_fiber_families():
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    for angle in (0.0, 0.5):  # radians: axis-aligned and rotated fiber families
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)
        image = np.sin(2 * np.pi * u / 9) + np.sin(2 * np.pi * v / 6)
        found = V.dominant_orientations(image)
        expected = np.array([angle, angle + np.pi / 2])
        error = np.rad2deg(V.angular_distance(found, expected))
        assert (error < 3).all(), (np.rad2deg(found), np.rad2deg(expected))


def test_fiber_notch_removes_fiber_stripes_but_keeps_a_blob():
    y, x = np.mgrid[0:128, 0:128].astype(np.float32)
    stripes = np.sin(2 * np.pi * x / 7) + np.sin(2 * np.pi * y / 7)
    blob = 2 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 6**2))

    class Fake:
        skin = (stripes + blob).astype(np.float32)
        fiber_orientations = V.dominant_orientations(stripes)
        xy = staticmethod(lambda length: length)
        valid = None
        valid_mean = staticmethod(lambda image: float(image.mean()))

    residual = V._notch_channels(Fake, {})["v"]
    centre = residual[48:80, 48:80]
    assert np.std(residual[:32, :32]) < 0.25 * np.std(stripes)
    assert centre.max() - np.median(residual) > 1.0


def test_angular_distance_wraps_at_pi():
    d = V.angular_distance(np.array([np.pi - 0.01]), np.array([0.0, np.pi / 2]))
    assert d[0] == pytest.approx(0.01, abs=1e-6)


@pytest.mark.parametrize("name", list(V.REGISTRY))
def test_every_visualizer_renders(name):
    vis = V.REGISTRY[name]
    tiles = [V.Tile(synthetic_sheet(seed=s)[0]) for s in range(2)]
    calibration = V.calibrate(vis, tiles)
    out = V.render(vis, tiles[0], calibration)
    frames = out if isinstance(out, list) else [out]
    assert vis.animated == isinstance(out, list)
    for frame in frames:
        assert frame.shape == (96, 96, 3)
        assert np.isfinite(frame).all()
        assert frame.min() >= 0 and frame.max() <= 1
    assert V.to_uint8(frames[0]).dtype == np.uint8


def test_diverging_range_is_symmetric_about_the_median():
    vis = V.REGISTRY["skin_minus_body"]
    tiles = [V.Tile(synthetic_sheet(seed=s)[0]) for s in range(2)]
    lo, hi = V.channel_ranges(vis, tiles, {})["v"]
    values = np.concatenate([(t.skin - t.body)[::2, ::2].ravel() for t in tiles])
    assert (lo + hi) / 2 == pytest.approx(float(np.median(values)), abs=1e-4)


def test_colormap_endpoints():
    np.testing.assert_allclose(V.colormap(np.array([0.0]), "inferno")[0], np.array([0, 0, 4]) / 255, atol=1e-6)
    np.testing.assert_allclose(V.colormap(np.array([1.0]), "inferno")[0], np.array([252, 255, 164]) / 255, atol=1e-6)


def test_surface_found_at_level_two_scale():
    """Pyramid level 2 shrinks only in-plane; with xy_scale=4 the same face is found."""
    ct, surface = synthetic_sheet(size=192)
    small = ct.reshape(ct.shape[0], 48, 4, 48, 4).mean(axis=(2, 4)).astype(np.uint8)
    truth = surface.reshape(48, 4, 48, 4).mean(axis=(1, 3))
    tile = V.Tile(small, xy_scale=4)
    assert np.median(np.abs(tile.height - truth)[4:-4, 4:-4]) < 0.6


def test_scale_one_lengths_are_unchanged():
    tile = V.Tile(np.zeros((8, 8, 8), np.uint8))
    assert tile.xy(6) == 6 and tile.xy_size(31) == 31 and tile.xy_size(5) == 5
    quarter = V.Tile(np.zeros((8, 8, 8), np.uint8), xy_scale=4)
    assert quarter.xy(6) == 1.5 and quarter.xy_size(31) == 9 and quarter.xy_size(5) == 3


@pytest.mark.parametrize("name", list(V.REGISTRY))
def test_tile_from_products_renders_the_same(name):
    vis = V.REGISTRY[name]
    full = V.Tile(synthetic_sheet(seed=4)[0])
    compact = V.Tile.from_products(full.products())
    calibration = V.calibrate(vis, [full])
    a, b = V.render(vis, full, calibration), V.render(vis, compact, calibration)
    for fa, fb in zip(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]):
        # products are stored as float32 (heights are float64 in a full tile): ~1e-5 differences
        np.testing.assert_allclose(fa, fb, atol=1e-4)


def test_notched_tile_removes_fibers_and_keeps_the_mean():
    ct, _ = synthetic_sheet(size=128)
    tile = V.Tile(ct)
    notched = tile.notched(7.0)
    i0 = int(-V.FLAT_D[0])
    before, after = tile.flat[i0 + 2], notched.flat[i0 + 2]
    assert abs(after.mean() - before.mean()) < 1e-3 * abs(before.mean())
    assert np.std(after - after.mean()) < 0.6 * np.std(before - before.mean())
    np.testing.assert_array_equal(notched.coarse_height, tile.coarse_height)
    # every visualizer still renders on a notched tile
    out = V.render(V.REGISTRY["surface_skin"], notched, V.calibrate(V.REGISTRY["surface_skin"], [notched]))
    assert np.isfinite(out).all()


def skewed_mask(size=128, angle=0.5, half=0.3):
    """A square rotated by ``angle`` radians: the part of a block a skewed segment covers."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) - size / 2
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    return (np.abs(u) < half * size) & (np.abs(v) < half * size)


def test_fiber_notch_ignores_the_empty_part_of_a_block():
    """A skewed segment is rendered into square blocks with empty (0) corners. The notch kept the block's
    offset from those zeros, so every block came out as bright as it was empty (banding along the block
    grid); with the mask, the residual is centred on the data and the fiber orientations are those of the
    papyrus, not of the edge of the data."""
    ct, _ = synthetic_sheet(size=128)
    mask = skewed_mask()
    empty = ct.copy()
    empty[:, ~mask] = 0
    full = V.Tile(ct)
    masked = V.Tile(empty, valid=mask)
    spread = np.std(V._notch_channels(full, {})["v"])
    unmasked = V._notch_channels(V.Tile(empty), {})["v"][mask].mean()
    residual = V._notch_channels(masked, {})["v"][mask].mean()
    assert abs(unmasked) > spread  # the bug: an offset larger than the texture itself
    assert abs(residual) < 0.1 * spread
    error = np.rad2deg(V.angular_distance(masked.fiber_orientations, full.fiber_orientations))
    assert (error < 3).all(), (np.rad2deg(masked.fiber_orientations), np.rad2deg(full.fiber_orientations))
    notched = masked.notched(7.0)
    np.testing.assert_array_equal(notched.valid, mask)
    # empty pixels are the nearest filtered pixel again, so the next filter sees no step either
    np.testing.assert_array_equal(notched.flat, V.nearest_fill(notched.flat, mask))


def test_an_all_valid_mask_changes_nothing():
    """colored_inkt's own tiles have no mask; a mask that covers the whole tile must render byte for byte the
    same, so the A/B test renders and caches stay valid."""
    ct, _ = synthetic_sheet(size=96)
    plain, masked = V.Tile(ct), V.Tile(ct, valid=np.ones(ct.shape[1:], bool))
    np.testing.assert_array_equal(plain.fiber_orientations, masked.fiber_orientations)
    np.testing.assert_array_equal(V._notch_channels(plain, {})["v"], V._notch_channels(masked, {})["v"])
    np.testing.assert_array_equal(plain.notched(7.0).flat, masked.notched(7.0).flat)
    with pytest.raises(ValueError):
        V.Tile(ct, valid=np.ones((5, 5), bool))


def test_no_filter_depends_on_the_empty_part_of_a_block():
    """Every filter, on a masked block, renders the pixels with data independently of the empty part: of what
    the empty pixels hold (exactly) and of how much of the block is empty (the same patch in a larger block,
    each calibrated on its own). Otherwise a skewed segment's partly empty edge blocks come out brighter or
    darker than its full ones (banding along the block grid), or the edge of the data shows as a rim. Before
    the mask, 37 of the 76 filters changed by more than 2 % of their display range here."""
    from colored_inkt import generalise, research  # noqa: F401  (register their methods)

    ct, _ = synthetic_sheet(size=160, seed=3)
    patch = skewed_mask(size=160, half=34 / 160)
    small = (slice(32, 128), slice(32, 128))  # a multiple of 4: the strided samples are the same pixels
    zeros, junk = ct.copy(), ct.copy()
    zeros[:, ~patch] = 0
    junk[:, ~patch] = np.random.default_rng(1).integers(0, 255, junk[:, ~patch].shape)
    tiles = {"small": V.Tile(zeros[(slice(None), *small)], xy_scale=4, valid=patch[small]),
             "junk": V.Tile(junk[(slice(None), *small)], xy_scale=4, valid=patch[small]),
             "large": V.Tile(zeros, xy_scale=4, valid=patch)}
    data = patch[small]
    errors = {}
    for name, vis in V.REGISTRY.items():
        if vis.animated:
            continue
        out = {k: V.render(vis, t, V.calibrate(vis, [t])) for k, t in tiles.items()}
        np.testing.assert_array_equal(out["small"], out["junk"], err_msg=name)
        errors[name] = float(np.median(np.abs(out["small"][data] - out["large"][small][data])))
    # the notch is a Fourier filter of the whole block: which low frequencies it keeps depends on the block size
    tolerance = {"fiber_notch": 0.05}
    assert len(errors) >= 76
    assert not {k: v for k, v in errors.items() if v > tolerance.get(k, 0.02)}, errors


def test_variants_point_at_registered_parents():
    for vis in V.REGISTRY.values():
        if vis.variant_of:
            assert vis.variant_of in V.REGISTRY and vis.family == "variant"


def test_despeckle_removes_specks_but_keeps_fibers_and_patches():
    rng = np.random.default_rng(1)
    image = 100 + rng.normal(0, 2, (96, 96)).astype(np.float32)
    image[20, :] += 60  # a fiber: thin but long
    image[60:80, 60:80] += 60  # a patch larger than the speck size
    image[40:42, 40:42] += 120  # a speck
    out = V.despeckle(image, 5, threshold=30)
    assert out[40:42, 40:42].max() < 110
    np.testing.assert_array_equal(out[20, 10:90], image[20, 10:90])
    np.testing.assert_array_equal(out[62:78, 62:78], image[62:78, 62:78])
    assert (out != image).mean() < 0.01


def test_despeckled_tile_keeps_heights_and_renders():
    ct, _ = synthetic_sheet(size=128)
    ct[40:43, 60:62, 60:62] = 255  # a dense grain at the face
    tile = V.Tile(ct)
    clean = tile.despeckled(30)
    np.testing.assert_allclose(clean.height, tile.height, atol=1e-4)  # heights pass through (as float32)
    i0 = int(-V.FLAT_D[0])
    assert clean.flat[i0 - 4 : i0 + 8].max() < tile.flat[i0 - 4 : i0 + 8].max()
    vis = V.REGISTRY["dense_tail_volume"]
    assert np.isfinite(V.render(vis, clean, V.calibrate(vis, [clean]))).all()


def test_recolourable_render_swaps_only_the_colour_map():
    vis = V.REGISTRY["dense_tail_volume"]
    assert vis.display == {"kind": "image", "recolor": "inferno"}
    tile = V.Tile(synthetic_sheet(seed=2)[0])
    calibration = V.calibrate(vis, [tile])
    default = V.render(vis, tile, calibration)
    same = V.render(vis, tile, {**calibration, "params": {**calibration["params"], "cmap": "inferno"}})
    other = V.render(vis, tile, {**calibration, "params": {**calibration["params"], "cmap": "viridis"}})
    np.testing.assert_array_equal(default, same)
    assert np.abs(other - default).max() > 0.1
    for name in V.COLORMAPS:
        assert V.colormap(np.array([0.0, 0.5, 1.0]), name).shape == (3, 3)


def _scipy_gauss(a, sigma):
    from scipy import ndimage

    return ndimage.gaussian_filter(a, sigma, mode="nearest")


@pytest.mark.parametrize("shape, sigma", [((300, 280), 20.0), ((40, 300, 64), (0, 12.0, 1.5)),
                                          ((50, 60), 30.0), ((128, 96), (9.0, 0.0))])
def test_wide_gaussians_through_the_fft_equal_scipy(shape, sigma):
    """``gauss`` sends axes with σ ≥ FFT_SIGMA_PX through the FFT (generalise's 1.5 mm local statistics took
    ~0.4 s per block directly). It must stay scipy's filter: same kernel, same "nearest" edge (also when the
    kernel is wider than the image), every axis rounded like scipy's, so every method renders as before up to
    float round-off (VERSION 5)."""
    rng = np.random.default_rng(0)
    a = (100 + 30 * rng.standard_normal(shape)).astype(np.float32)
    a[..., : shape[-1] // 3] += 40  # a step towards one edge: the edge handling matters
    ref, out = _scipy_gauss(a, sigma), V.gauss(a, sigma)
    assert out.dtype == ref.dtype == np.float32 and out.shape == ref.shape
    ulp = np.spacing(np.abs(ref).astype(np.float32))
    assert np.max(np.abs(out - ref) / ulp) <= 2, np.max(np.abs(out - ref))
    double = a.astype(np.float64)
    np.testing.assert_allclose(V.gauss(double, sigma), _scipy_gauss(double, sigma), rtol=1e-12, atol=1e-10)


def test_narrow_and_integer_gaussians_are_scipy_itself():
    """Below the threshold (every in-plane filter but the local statistics) nothing changes, bit for bit."""
    rng = np.random.default_rng(1)
    a = rng.random((64, 64)).astype(np.float32)
    np.testing.assert_array_equal(V.gauss(a, 3.0), _scipy_gauss(a, 3.0))
    u8 = (a * 255).astype(np.uint8)
    np.testing.assert_array_equal(V.gauss(u8, 20.0), _scipy_gauss(u8, 20.0))


def test_wide_gaussian_of_a_nonnegative_map_has_no_negative_values():
    """Far from the data a masked map is exactly 0 and its direct Gaussian too; the FFT's round-off can make it
    -1e-17, and ``generalise.local_z`` takes the square root of such a Gaussian (NaN would then spread into the
    pixels with data). The FFT path clips a non-negative input's Gaussian at 0."""
    a = np.zeros((96, 2000), np.float32)
    a[:, :20] = 50.0
    out = V.gauss(a, (0, 160.0))
    assert out.min() >= 0
    assert np.isfinite(np.sqrt(out)).all()


def _eager(tile, transform, keep):
    """The pre-VERSION-5 ``notched`` / ``despeckled``: every product filtered at once (``from_products``)."""
    out = {}
    for name, values in tile.products().items():
        if name in keep:
            out[name] = values
        elif values.ndim == 3:
            out[name] = np.stack([transform(plane) for plane in values])
        else:
            out[name] = transform(values)
    return V.Tile.from_products(out, side=tile.side, xy_scale=tile.xy_scale, valid=tile.valid)


@pytest.mark.parametrize("masked", [False, True])
def test_derived_tiles_equal_the_eagerly_filtered_ones(masked):
    """``notched`` and ``despeckled`` now filter a product when a method first reads it (gen_two_hue reads 11 of
    the 25 planes); every product must be the one the eager version made, bit for bit, also on a partly empty
    block (the empty pixels nearest-filled from the filtered data) and when despeckled then notched (the brute
    force order)."""
    ct, _ = synthetic_sheet(size=128, seed=2)
    mask = skewed_mask() if masked else None
    if masked:
        ct = ct.copy()
        ct[:, ~mask] = 0
    tile = V.Tile(ct, xy_scale=4, valid=mask)
    length = tile.xy_size(50 / 2.4)
    face = tile.pixels(tile.flat[6:15], 2)
    threshold = 0.5 * float(np.percentile(face, 99) - np.percentile(face, 50))
    eager_speck = _eager(tile, lambda im: V.despeckle(im, length, threshold), ("coarse_height", "height"))
    orientations = eager_speck.fiber_orientations = tile.fiber_orientations

    def notch(image):
        mean = eager_speck.valid_mean(image)
        return V.notch_filter(image - mean, orientations, half_width_deg=7.0, valid=eager_speck.valid) + mean

    eager_both = _eager(eager_speck, notch, ("coarse_height",))
    lazy_speck = V.Tile(ct, xy_scale=4, valid=mask).despeckled(50.0)
    lazy_both = lazy_speck.notched(7.0)
    for eager, lazy in ((eager_speck, lazy_speck), (eager_both, lazy_both)):
        np.testing.assert_array_equal(lazy.band(0, 4), eager.band(0, 4))  # a band first: computed plane by plane
        for name in V.PRODUCTS:
            np.testing.assert_array_equal(getattr(lazy, name), getattr(eager, name), err_msg=name)


def test_a_band_computes_only_its_planes():
    """The point of the lazy planes: a method that reads the skin never computes (or filters) the rest."""
    ct, _ = synthetic_sheet(size=64)
    tile = V.Tile(ct).despeckled(50.0)
    skin = tile.skin
    assert "flat" not in tile.__dict__ and sorted(tile.__dict__["_flat_planes"]) == list(range(8, 13))
    np.testing.assert_array_equal(skin, tile.flat[8:13].mean(axis=0))  # and the whole stack agrees
    np.testing.assert_array_equal(tile.planes(-8, 16), tile.flat)
