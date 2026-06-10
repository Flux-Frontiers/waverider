"""
Hololuminescent Display (HLD) Renderer
======================================

Renders PyVista scenes as videos for Looking Glass *Hololuminescent
Displays* — the HLD product line (16" / 27" / 86" Portrait).

HLDs are a different technology from the classic light-field Looking Glass
devices (which consume multi-view quilts; see
:mod:`waverider.looking_glass`).  An HLD is an LCD with a fixed holographic
"alcove" volume embedded in its optical stack; ordinary **flat 2-D video**
is multiply-blended into that volume.  The consequences for rendering:

* **Pure white pixels are invisible** — they show only the holographic
  alcove.  Subjects must sit on a white background.
* The subject should be **centred inside safe-area margins** (~9% top,
  3% bottom/left/right) so it stays within the alcove.
* A **slow turntable orbit** with an otherwise static camera reads best.
* The master format is a standard **2160x3840 (9:16) MP4**, HEVC,
  30 or 60 fps, bt709 — no depth, layers, alpha, or quilt tiling.

Spec: https://hlddocs.lookingglassfactory.com/resources/media-specs-and-encoding

Delivery: run the rendered master through Looking Glass's free **HLD
Author** app (which validates, rotates 90° CCW for the bundled Raspberry
Pi player, and re-encodes), then copy to the player's USB drive.  For
signage players (BrightSign/Yodeck) or direct HDMI, use the master as-is.

Typical usage::

    import pyvista as pv
    from waverider.hld import render_hld_video, style_plotter_for_hld

    p = pv.Plotter(off_screen=True)
    p.add_mesh(pv.ParametricTorus(), color="teal")
    style_plotter_for_hld(p)                  # white bg, safe-area framing
    render_hld_video(p, "torus")              # -> torus_hld.mp4 (10s orbit)
    p.close()

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from waverider.looking_glass import _find_ffmpeg

try:
    import pyvista as pv  # noqa: F401

    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False


def _require_pyvista(fn_name: str) -> None:
    """Raise a clear ImportError if pyvista is not installed."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError(
            f"{fn_name}() requires pyvista.\nInstall with:  poetry install --with viz"
        )


#: Master resolution from the official HLD media spec (9:16 portrait).
#: One 4K master serves all HLD sizes; players downscale for the 16".
HLD_RESOLUTION: tuple[int, int] = (2160, 3840)

#: Safe-area margins from the official content guidelines, as fractions of
#: frame size: (top, bottom, left, right).  The larger top margin keeps the
#: subject clear of the alcove roof.
HLD_SAFE_MARGINS: tuple[float, float, float, float] = (0.09, 0.03, 0.03, 0.03)


def apply_safe_area(camera, margins: tuple[float, float, float, float] = HLD_SAFE_MARGINS) -> None:
    """Frame the current camera view inside the HLD safe area.

    Assumes the camera currently frames the subject to the full viewport
    (e.g. after ``reset_camera()``).  Zooms out so the subject fits the
    safe-area box and shifts the projection window so the box's centre
    (slightly below frame centre, because the top margin is larger) holds
    the subject.

    :param camera: ``pv.Camera`` / vtkCamera to mutate.
    :param margins: ``(top, bottom, left, right)`` fractions of frame size.
    """
    top, bottom, left, right = margins
    fit = min(1.0 - left - right, 1.0 - top - bottom)
    camera.Zoom(fit)
    # WindowCenter shifts the frustum in NDC (half-extent = 1): moving the
    # frustum up by (top - bottom) moves the rendered subject down to the
    # safe-area centre.
    wcx = camera.GetWindowCenter()[0] + (left - right)
    wcy = camera.GetWindowCenter()[1] + (top - bottom)
    camera.SetWindowCenter(wcx, wcy)


def style_plotter_for_hld(
    plotter,
    *,
    safe_area: bool = True,
    resolution: tuple[int, int] = HLD_RESOLUTION,
) -> None:
    """Apply HLD content rules to a composed plotter.

    Sets the pure-white background (white = transparent on the device),
    switches the window to the 9:16 master resolution, refits the camera
    distance to the scene **at that aspect** (framing computed at the wrong
    aspect overflows the narrow portrait frame), and frames the subject
    inside the safe-area margins.  A camera position set beforehand keeps
    its view *direction*; only the distance is refitted.

    :param plotter: ``pv.Plotter`` with the scene composed.
    :param safe_area: Apply :func:`apply_safe_area` framing.
    :param resolution: Target ``(width, height)``; must match the
        resolution later passed to :func:`render_hld_video`.
    """
    _require_pyvista("style_plotter_for_hld")
    plotter.set_background("white")
    plotter.window_size = resolution
    plotter.render()  # apply window size so reset_camera sees 9:16
    if not plotter.camera.is_set:
        plotter.camera_position = plotter.renderer.get_default_cam_pos()
    plotter.reset_camera()
    if safe_area:
        apply_safe_area(plotter.camera)


def _hld_encode_args(fps: int, crf: int) -> list[str]:
    """ffmpeg output arguments per the official HLD media spec.

    HEVC in MP4, yuv420p, bt709 tagging.  *fps* is constrained to the
    spec's 30/60.
    """
    if fps not in (30, 60):
        raise ValueError(f"HLD spec requires 30 or 60 fps, got {fps}")
    return [
        "-vcodec",
        "libx265",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-tag:v",
        "hvc1",
    ]


def render_hld_video(
    plotter,
    out_stem: str | Path,
    *,
    n_frames: int = 300,
    fps: int = 30,
    orbit_degrees: float = 360.0,
    resolution: tuple[int, int] = HLD_RESOLUTION,
    crf: int = 18,
    rotate_for_player: bool = False,
    on_frame=None,
    progress: bool = True,
) -> Path:
    """Render a turntable HLD master video of the plotter's scene.

    One ordinary 2-D render per frame (no multi-view sweep), camera
    orbiting the focal point, encoded to the official HLD master spec.
    Style the scene first with :func:`style_plotter_for_hld` (white
    background is what makes the hologram read on the device).

    :param plotter: An *off-screen* ``pv.Plotter`` with the scene composed.
    :param out_stem: Output path; ``_hld.mp4`` is appended.
    :param n_frames: Frame count (default 300 @ 30 fps = 10 s loop).
    :param fps: 30 or 60 per the HLD spec.
    :param orbit_degrees: Total orbit over the clip; 360 loops seamlessly.
        Pass 0 to disable the turntable (use *on_frame*).
    :param resolution: Master ``(width, height)``; default 2160x3840.
    :param crf: x265 quality (lower = better; 15-20 sensible).
    :param rotate_for_player: Also rotate 90° counter-clockwise, as the
        bundled Raspberry Pi player expects.  Leave ``False`` and use
        Looking Glass's HLD Author app for that step (recommended), or for
        signage/HDMI delivery which takes the un-rotated master.
    :param on_frame: Optional ``callback(frame_index)`` before each frame.
    :param progress: Print a progress line while rendering.
    :return: Path of the MP4 written.
    """
    _require_pyvista("render_hld_video")
    ffmpeg = _find_ffmpeg()

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "render_hld_video() requires pillow.\nInstall with:  poetry install --with viz"
        ) from exc

    out_stem = Path(out_stem)
    if out_stem.suffix.lower() == ".mp4":
        out_stem = out_stem.with_suffix("")
    out_path = out_stem.parent / f"{out_stem.name}_hld.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter.window_size = resolution
    if not plotter.camera.is_set:
        plotter.camera_position = plotter.renderer.get_default_cam_pos()
        plotter.reset_camera()

    step = orbit_degrees / n_frames if n_frames else 0.0
    with tempfile.TemporaryDirectory(prefix="hld_frames_") as tmp:
        for i in range(n_frames):
            if on_frame is not None:
                on_frame(i)
            plotter.renderer.reset_camera_clipping_range()
            plotter.render()
            img = plotter.screenshot(None, return_img=True)[..., :3]
            Image.fromarray(img).save(f"{tmp}/frame{i:05d}.png")
            plotter.camera.Azimuth(step)
            if progress:
                print(f"\r  HLD frame {i + 1}/{n_frames}", end="", flush=True)
        if progress:
            print()

        args = _hld_encode_args(fps, crf)
        if rotate_for_player:
            args += ["-vf", "transpose=2"]  # 90° counter-clockwise
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{tmp}/frame%05d.png",
            *args,
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr[-2000:]}")
    return out_path


def add_floor_shadow(
    plotter,
    subject_bounds: tuple[float, float, float, float, float, float],
    *,
    opacity: float = 0.25,
    scale: float = 1.4,
) -> None:
    """Add a soft fake contact shadow under the subject.

    The HLD guidelines call contact shadows on the alcove floor "crucial
    for the 3D effect".  PyVista's ray-traced shadows are unreliable with
    translucent voxel clouds, so this paints a flattened grey disc just
    below the subject's bounding box — reading as a soft contact shadow
    once multiply-blended into the alcove floor.

    :param plotter: Active ``pv.Plotter``.
    :param subject_bounds: ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the
        subject (e.g. ``grid.bounds``).
    :param opacity: Shadow darkness (0 = none).
    :param scale: Shadow radius as a fraction of the subject's half-extent;
        keep it > 1 so the shadow spreads past the footprint and stays
        visible from elevated camera angles.
    """
    _require_pyvista("add_floor_shadow")
    xmin, xmax, ymin, ymax, zmin, zmax = subject_bounds
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    radius = scale * max(xmax - xmin, ymax - ymin) / 2.0
    drop = 0.02 * (zmax - zmin)
    disc = pv.Disc(
        center=(cx, cy, zmin - drop),
        inner=0.0,
        outer=radius,
        normal=(0, 0, 1),
        c_res=64,
    )
    # Radial falloff: darkest at centre, fading to white at the rim.  White
    # blends into the white background on screen and is invisible on the
    # device, so no transparency is needed.
    pts = disc.points
    r = np.linalg.norm(pts[:, :2] - np.array([cx, cy]), axis=1) / max(radius, 1e-12)
    disc.point_data["shadow"] = (1.0 - np.clip(r, 0.0, 1.0)) ** 2
    plotter.add_mesh(
        disc,
        scalars="shadow",
        cmap="Greys",
        # Stretch the ramp so the darkest point is only `opacity` grey.
        clim=(0.0, 1.0 / max(opacity, 1e-6)),
        show_scalar_bar=False,
        lighting=False,
    )


def hld_orbit_speed(n_frames: int, fps: int) -> float:
    """Degrees of rotation per second for a given clip configuration."""
    return 360.0 * fps / n_frames if n_frames else 0.0
