"""Tests for the Looking Glass quilt renderer (waverider.lfd)."""

import math

import numpy as np
import pytest

from waverider.lfd import (
    QUILT_PRESETS,
    QuiltSpec,
    _encode_args,
    save_quilt,
    view_offsets,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def portrait() -> QuiltSpec:
    """Standard Looking Glass Portrait quilt spec (48 views, 8x6)."""
    return QUILT_PRESETS["portrait"]


@pytest.fixture
def tiny_spec() -> QuiltSpec:
    """Minimal 2x2 quilt for fast rendering tests."""
    return QuiltSpec(columns=2, rows=2, quilt_width=128, quilt_height=128, aspect=1.0)


# ---------------------------------------------------------------------------
# QuiltSpec geometry
# ---------------------------------------------------------------------------


class TestQuiltSpec:
    def test_portrait_preset(self, portrait):
        assert portrait.n_views == 48
        assert (portrait.columns, portrait.rows) == (8, 6)
        assert (portrait.quilt_width, portrait.quilt_height) == (3360, 3360)
        assert portrait.aspect == 0.75

    def test_portrait_tile_size(self, portrait):
        # 3360 / 8 = 420 wide, 3360 / 6 = 560 tall -> 0.75 view aspect
        assert (portrait.tile_width, portrait.tile_height) == (420, 560)
        assert portrait.tile_width / portrait.tile_height == pytest.approx(portrait.aspect)

    def test_presets_tiles_fit(self):
        # Tile grids must fit inside the quilt; official sizes are not always
        # exact multiples (65" is 8192 px over 9 rows), so allow a sub-tile
        # remainder but never overflow.
        for name, spec in QUILT_PRESETS.items():
            assert spec.tile_width > 0 and spec.tile_height > 0, name
            assert spec.tile_width * spec.columns <= spec.quilt_width, name
            assert spec.tile_height * spec.rows <= spec.quilt_height, name

    def test_official_preset_table(self):
        # Spot-check against the official quilt settings table:
        # https://lfdocs.lookingglassfactory.com/keyconcepts/quilts
        go = QUILT_PRESETS["go"]
        assert (go.columns, go.rows, go.n_views) == (11, 6, 66)
        assert (go.quilt_width, go.quilt_height) == (4092, 4092)
        assert go.aspect == 0.5625
        l65 = QUILT_PRESETS["65"]
        assert (l65.columns, l65.rows, l65.n_views) == (8, 9, 72)
        l27 = QUILT_PRESETS["27-landscape"]
        assert (l27.quilt_width, l27.quilt_height) == (7680, 4320)
        assert l27.aspect == 1.777

    def test_view_zero_is_bottom_left(self, portrait):
        x, y = portrait.tile_origin(0)
        assert x == 0
        assert y == portrait.quilt_height - portrait.tile_height

    def test_last_view_is_top_right(self, portrait):
        x, y = portrait.tile_origin(portrait.n_views - 1)
        assert x == portrait.quilt_width - portrait.tile_width
        assert y == 0

    def test_views_advance_left_to_right(self, portrait):
        x0, y0 = portrait.tile_origin(0)
        x1, y1 = portrait.tile_origin(1)
        assert x1 == x0 + portrait.tile_width
        assert y1 == y0

    def test_rows_advance_bottom_to_top(self, portrait):
        _, y0 = portrait.tile_origin(0)
        _, y1 = portrait.tile_origin(portrait.columns)  # first view of row 1
        assert y1 == y0 - portrait.tile_height

    def test_tile_origins_unique_and_in_bounds(self, portrait):
        origins = {portrait.tile_origin(i) for i in range(portrait.n_views)}
        assert len(origins) == portrait.n_views
        for x, y in origins:
            assert 0 <= x <= portrait.quilt_width - portrait.tile_width
            assert 0 <= y <= portrait.quilt_height - portrait.tile_height

    def test_tile_origin_out_of_range(self, portrait):
        with pytest.raises(ValueError):
            portrait.tile_origin(portrait.n_views)
        with pytest.raises(ValueError):
            portrait.tile_origin(-1)

    def test_filename_convention(self, portrait):
        # Looking Glass software parses _qs<cols>x<rows>a<aspect> suffixes.
        assert portrait.filename("helix") == "helix_qs8x6a0.75.png"

    def test_filename_go(self):
        assert QUILT_PRESETS["go"].filename("x", ext="jpg") == "x_qs11x6a0.5625.jpg"

    def test_with_grid_keeps_quilt_size(self, portrait):
        dense = portrait.with_grid(11, 6)
        assert dense.n_views == 66
        assert (dense.quilt_width, dense.quilt_height) == (3360, 3360)
        assert dense.aspect == portrait.aspect
        assert dense.tile_width < portrait.tile_width  # views cost resolution
        assert dense.filename("x") == "x_qs11x6a0.75.png"


# ---------------------------------------------------------------------------
# Video encoding arguments
# ---------------------------------------------------------------------------


class TestEncodeArgs:
    def test_portrait_uses_h264_yuv420p(self, portrait):
        args = _encode_args(portrait, crf=18)
        assert "libx264" in args
        assert "yuv420p" in args
        assert "-vf" not in args  # 3360 is even, no padding needed

    def test_8k_quilts_use_hevc(self):
        args = _encode_args(QUILT_PRESETS["65"], crf=18)
        assert "libx265" in args

    def test_odd_sizes_padded(self):
        # yuv420p needs even dimensions.  Built explicitly rather than taken
        # from QUILT_PRESETS: preset sizes track real hardware and change
        # when a device's spec is corrected, which is not what this asserts.
        odd = QuiltSpec(columns=7, rows=7, quilt_width=5999, quilt_height=5999, aspect=1.777)
        args = _encode_args(odd, crf=18)
        assert "-vf" in args
        assert "pad=" in args[args.index("-vf") + 1]


# ---------------------------------------------------------------------------
# Camera sweep
# ---------------------------------------------------------------------------


class TestViewOffsets:
    def test_count_and_order(self, portrait):
        offs = view_offsets(portrait, distance=10.0)
        assert offs.shape == (48,)
        assert np.all(np.diff(offs) > 0)  # strictly left -> right

    def test_symmetric_about_centre(self, portrait):
        offs = view_offsets(portrait, distance=10.0)
        np.testing.assert_allclose(offs, -offs[::-1], atol=1e-12)

    def test_cone_extent(self, portrait):
        # Extreme views sit at +/- half the view cone.
        offs = view_offsets(portrait, distance=10.0)
        expected = 10.0 * math.tan(math.radians(portrait.view_cone) / 2.0)
        assert offs[-1] == pytest.approx(expected)
        assert offs[0] == pytest.approx(-expected)

    def test_default_cone_is_35_degrees(self, portrait):
        assert portrait.view_cone == 35.0

    def test_single_view(self):
        spec = QuiltSpec(columns=1, rows=1, quilt_width=64, quilt_height=64, aspect=1.0)
        np.testing.assert_array_equal(view_offsets(spec, 5.0), [0.0])

    def test_scales_with_distance(self, portrait):
        np.testing.assert_allclose(view_offsets(portrait, 20.0), 2.0 * view_offsets(portrait, 10.0))


# ---------------------------------------------------------------------------
# Quilt assembly + I/O
# ---------------------------------------------------------------------------


class TestSaveQuilt:
    def test_writes_convention_filename(self, tmp_path, tiny_spec):
        pytest.importorskip("PIL")
        quilt = np.zeros((128, 128, 3), dtype=np.uint8)
        out = save_quilt(quilt, tmp_path / "scene", tiny_spec)
        assert out.name == "scene_qs2x2a1.png"
        assert out.exists()

    def test_strips_png_extension(self, tmp_path, tiny_spec):
        pytest.importorskip("PIL")
        quilt = np.zeros((128, 128, 3), dtype=np.uint8)
        out = save_quilt(quilt, tmp_path / "scene.png", tiny_spec)
        assert out.name == "scene_qs2x2a1.png"

    def test_roundtrip_pixels(self, tmp_path, tiny_spec):
        Image = pytest.importorskip("PIL.Image")
        rng = np.random.default_rng(0)
        quilt = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
        out = save_quilt(quilt, tmp_path / "rt", tiny_spec)
        back = np.asarray(Image.open(out))
        np.testing.assert_array_equal(back[..., :3], quilt)


# ---------------------------------------------------------------------------
# Off-axis rendering (requires pyvista + a working render window)
# ---------------------------------------------------------------------------


def _can_render() -> bool:
    """True if pyvista can produce an off-screen render in this environment."""
    try:
        import pyvista as pv

        p = pv.Plotter(off_screen=True, window_size=(32, 32))
        p.add_mesh(pv.Sphere())
        p.screenshot(None, return_img=True)
        p.close()
        return True
    except Exception:
        return False


requires_render = pytest.mark.skipif(
    not _can_render(), reason="pyvista off-screen rendering unavailable"
)


@requires_render
class TestRenderQuilt:
    def test_shape_and_views_differ(self, tiny_spec):
        import pyvista as pv

        from waverider.lfd import render_quilt

        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Cube(), color="red")
        p.add_mesh(pv.Sphere(center=(0, 0, 1.2), radius=0.3), color="blue")
        quilt = render_quilt(p, tiny_spec)
        p.close()

        assert quilt.shape == (128, 128, 3)
        assert quilt.dtype == np.uint8
        # Parallax: leftmost and rightmost views must differ.
        x0, y0 = tiny_spec.tile_origin(0)
        x3, y3 = tiny_spec.tile_origin(3)
        v0 = quilt[y0 : y0 + 64, x0 : x0 + 64]
        v3 = quilt[y3 : y3 + 64, x3 : x3 + 64]
        assert not np.array_equal(v0, v3)

    def test_focal_point_stays_centred(self, tiny_spec):
        """Off-axis shear must pin the focal plane: an object at the focal
        point should occupy the centre pixel of *every* view."""
        import pyvista as pv

        from waverider.lfd import render_quilt

        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Sphere(radius=0.2), color="white")
        p.set_background("black")
        p.camera_position = [(0, -10, 0), (0, 0, 0), (0, 0, 1)]
        quilt = render_quilt(p, tiny_spec, view_cone=35.0)
        p.close()

        th, tw = tiny_spec.tile_height, tiny_spec.tile_width
        for i in range(tiny_spec.n_views):
            x, y = tiny_spec.tile_origin(i)
            centre = quilt[y + th // 2, x + tw // 2]
            assert centre.sum() > 300, f"view {i}: focal object missing at centre {centre}"

    def test_camera_restored_after_render(self, tiny_spec):
        import pyvista as pv

        from waverider.lfd import render_quilt

        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Sphere())
        p.camera_position = [(0, -8, 0), (0, 0, 0), (0, 0, 1)]
        render_quilt(p, tiny_spec, fov=None)
        pos = np.asarray(p.camera.position)
        p.close()
        np.testing.assert_allclose(pos, [0, -8, 0], atol=1e-6)
        # WindowCenter reset so subsequent screenshots are on-axis.


def _have_ffmpeg() -> bool:
    try:
        from waverider.lfd import _find_ffmpeg

        _find_ffmpeg()
        return True
    except RuntimeError:
        return False


@requires_render
@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg unavailable")
class TestRenderQuiltVideo:
    def test_turntable_mp4(self, tmp_path, tiny_spec):
        import pyvista as pv

        from waverider.lfd import render_quilt_video

        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Cube(), color="red")
        out = render_quilt_video(
            p,
            tiny_spec,
            tmp_path / "spin",
            n_frames=4,
            fps=4,
            progress=False,
        )
        p.close()
        assert out.name == "spin_qs2x2a1.mp4"
        assert out.stat().st_size > 0

    def test_on_frame_callback_runs(self, tmp_path, tiny_spec):
        import pyvista as pv

        from waverider.lfd import render_quilt_video

        seen = []
        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Sphere())
        render_quilt_video(
            p,
            tiny_spec,
            tmp_path / "cb",
            n_frames=3,
            fps=3,
            orbit_degrees=0.0,
            on_frame=seen.append,
            progress=False,
        )
        p.close()
        assert seen == [0, 1, 2]
