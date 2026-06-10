"""Tests for the Hololuminescent Display renderer (waverider.hld)."""

import numpy as np
import pytest

from waverider.hld import (
    HLD_RESOLUTION,
    HLD_SAFE_MARGINS,
    _hld_encode_args,
    hld_orbit_speed,
)

# ---------------------------------------------------------------------------
# Spec constants + encoding
# ---------------------------------------------------------------------------


class TestHldSpec:
    def test_master_resolution_is_9_16_4k(self):
        w, h = HLD_RESOLUTION
        assert (w, h) == (2160, 3840)
        assert w / h == pytest.approx(9 / 16)

    def test_safe_margins(self):
        top, bottom, left, right = HLD_SAFE_MARGINS
        assert top > bottom  # alcove roof needs the bigger margin
        assert (top, bottom, left, right) == (0.09, 0.03, 0.03, 0.03)

    def test_encode_args_hevc_bt709(self):
        args = _hld_encode_args(fps=30, crf=18)
        assert "libx265" in args
        assert "yuv420p" in args
        assert args.count("bt709") == 3  # primaries, trc, colorspace

    def test_encode_args_rejects_off_spec_fps(self):
        with pytest.raises(ValueError):
            _hld_encode_args(fps=24, crf=18)
        _hld_encode_args(fps=60, crf=18)  # 60 allowed

    def test_orbit_speed(self):
        assert hld_orbit_speed(n_frames=300, fps=30) == pytest.approx(36.0)
        assert hld_orbit_speed(n_frames=0, fps=30) == 0.0


# ---------------------------------------------------------------------------
# Rendering (requires pyvista + a working render window)
# ---------------------------------------------------------------------------


def _can_render() -> bool:
    try:
        import pyvista as pv

        p = pv.Plotter(off_screen=True, window_size=(32, 32))
        p.add_mesh(pv.Sphere())
        p.screenshot(None, return_img=True)
        p.close()
        return True
    except Exception:
        return False


def _have_ffmpeg() -> bool:
    try:
        from waverider.looking_glass import _find_ffmpeg

        _find_ffmpeg()
        return True
    except RuntimeError:
        return False


requires_render = pytest.mark.skipif(
    not _can_render(), reason="pyvista off-screen rendering unavailable"
)


@requires_render
class TestStyleAndSafeArea:
    def test_subject_inside_safe_area(self):
        """After styling, all non-white pixels must respect the margins."""
        import pyvista as pv

        from waverider.hld import style_plotter_for_hld

        p = pv.Plotter(off_screen=True, window_size=(216, 384))
        p.add_mesh(pv.Sphere(), color="navy")
        style_plotter_for_hld(p)
        img = p.screenshot(None, return_img=True)
        p.close()

        h, w = img.shape[:2]
        nonwhite = np.argwhere((img[..., :3] < 250).any(axis=-1))
        assert len(nonwhite) > 0, "subject vanished"
        ys, xs = nonwhite[:, 0], nonwhite[:, 1]
        top, bottom, left, right = (0.09, 0.03, 0.03, 0.03)
        # Allow ~2% slack: reset_camera's fit is approximate.
        slack = 0.02
        assert ys.min() >= (top - slack) * h, f"breaches top margin ({ys.min()}/{h})"
        assert ys.max() <= (1 - bottom + slack) * h, f"breaches bottom ({ys.max()}/{h})"
        assert xs.min() >= (left - slack) * w, f"breaches left ({xs.min()}/{w})"
        assert xs.max() <= (1 - right + slack) * w, f"breaches right ({xs.max()}/{w})"

    def test_background_is_pure_white(self):
        import pyvista as pv

        from waverider.hld import style_plotter_for_hld

        p = pv.Plotter(off_screen=True, window_size=(216, 384))
        p.add_mesh(pv.Sphere(), color="navy")
        style_plotter_for_hld(p)
        img = p.screenshot(None, return_img=True)
        p.close()
        # Corners are background; pure white = invisible on the device.
        for corner in (img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]):
            np.testing.assert_array_equal(corner[:3], [255, 255, 255])

    def test_floor_shadow_adds_grey_below_subject(self):
        import pyvista as pv

        from waverider.hld import add_floor_shadow, style_plotter_for_hld

        p = pv.Plotter(off_screen=True, window_size=(216, 384))
        sphere = pv.Sphere(radius=0.5)
        p.add_mesh(sphere, color="navy")
        add_floor_shadow(p, sphere.bounds)
        style_plotter_for_hld(p)
        p.camera.Elevation(20)  # raise the camera so the floor disc shows
        img = p.screenshot(None, return_img=True)
        p.close()
        px = img[..., :3].astype(int)
        grey = (px.max(-1) - px.min(-1) < 12) & (px.mean(-1) < 245) & (px.mean(-1) > 80)
        assert grey.sum() > 20, "no grey shadow pixels found"


@requires_render
@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg unavailable")
class TestRenderHldVideo:
    def test_turntable_master(self, tmp_path):
        import pyvista as pv

        from waverider.hld import render_hld_video, style_plotter_for_hld

        p = pv.Plotter(off_screen=True)
        p.add_mesh(pv.Cube(), color="firebrick")
        style_plotter_for_hld(p)
        out = render_hld_video(
            p,
            tmp_path / "spin",
            n_frames=4,
            fps=30,
            resolution=(216, 384),
            progress=False,
        )
        p.close()
        assert out.name == "spin_hld.mp4"
        assert out.stat().st_size > 0
