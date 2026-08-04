"""
Looking Glass Quilt Renderer
============================

Renders any PyVista scene into a *quilt* — the tiled multi-view image format
used by Looking Glass holographic light-field displays.

A quilt packs N renders of the same scene, captured from camera positions
swept horizontally across a viewing cone, into a single image.  Views are
tiled left-to-right, bottom-to-top: view 0 (leftmost camera) sits at the
bottom-left tile and view N-1 (rightmost camera) at the top-right.  Looking
Glass software (Bridge, Studio) detects quilt settings from the filename
suffix ``_qs<cols>x<rows>a<aspect>.png``, so files saved through
:func:`save_quilt` are recognised automatically.

Each view uses an *off-axis* (asymmetric-frustum) projection rather than a
"toe-in" rotation: the camera translates along its horizontal axis while the
frustum is sheared back toward the focal plane.  This keeps the focal plane
identical across views — the geometric requirement for the display's lenticular
optics to fuse the views into a stable hologram.  Content at the focal plane
appears at the physical screen surface; content nearer/farther floats in
front of / behind the glass.

**Optional dependencies** — install the ``viz`` extras group::

    poetry install --with viz   # pyvista, pillow, scipy, ...

Typical usage::

    import pyvista as pv
    from waverider.lfd import QUILT_PRESETS, render_quilt, save_quilt

    p = pv.Plotter(off_screen=True)
    p.add_mesh(pv.ParametricTorus())
    spec = QUILT_PRESETS["portrait"]
    quilt = render_quilt(p, spec)
    save_quilt(quilt, "torus", spec)        # -> torus_qs8x6a0.75.png
    p.close()

The saved quilt can be displayed on the device by dragging it into Looking
Glass Studio, or cast directly from Python via :func:`cast_quilt` if Looking
Glass Bridge is running on the machine driving the display.

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

# pylint: disable=import-outside-toplevel  # viz deps (pillow, imageio-ffmpeg) are optional/heavy; lazy-loaded only when needed
import json
import math
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

try:
    import pyvista as pv  # noqa: F401  (re-exported pattern matches voxel_viz)

    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False


def _require_pyvista(fn_name: str) -> None:
    """Raise a clear ImportError if pyvista is not installed."""
    if not _PYVISTA_AVAILABLE:
        raise ImportError(
            f"{fn_name}() requires pyvista.\nInstall with:  poetry install --with viz"
        )


# ---------------------------------------------------------------------------
# Quilt specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuiltSpec:
    """Geometry of a quilt: tiling grid, total pixel size, and view cone.

    :param columns: Number of view tiles per quilt row.
    :param rows: Number of view tiles per quilt column.
    :param quilt_width: Total quilt width in pixels.
    :param quilt_height: Total quilt height in pixels.
    :param aspect: Aspect ratio (width / height) of a *single view*, which
        matches the target display's aspect.  Embedded in the quilt filename
        so Looking Glass software can configure playback correctly.
    :param view_cone: Total horizontal sweep of the camera in degrees.
        Looking Glass documents 35° as the standard rendering cone (the
        physical display cone is wider, ~40-58° depending on model;
        rendering slightly narrower adds apparent depth).
    """

    columns: int
    rows: int
    quilt_width: int
    quilt_height: int
    aspect: float
    view_cone: float = 35.0

    @property
    def n_views(self) -> int:
        """Total number of views in the quilt."""
        return self.columns * self.rows

    @property
    def tile_width(self) -> int:
        """Width of a single view tile in pixels."""
        return self.quilt_width // self.columns

    @property
    def tile_height(self) -> int:
        """Height of a single view tile in pixels."""
        return self.quilt_height // self.rows

    def tile_origin(self, view_index: int) -> tuple[int, int]:
        """Pixel ``(x, y)`` of a view's top-left corner within the quilt image.

        Quilt convention: view 0 at the *bottom-left*, advancing
        left-to-right then bottom-to-top.  The returned ``y`` is measured
        from the image top (numpy/PIL row order).

        :param view_index: View number in ``[0, n_views)``.
        :return: ``(x, y)`` pixel offsets of the tile.
        """
        if not 0 <= view_index < self.n_views:
            raise ValueError(f"view_index {view_index} outside [0, {self.n_views})")
        col = view_index % self.columns
        row = view_index // self.columns  # 0 = bottom row
        x = col * self.tile_width
        y = (self.rows - 1 - row) * self.tile_height
        return x, y

    def filename(self, stem: str, ext: str = "png") -> str:
        """Quilt filename with the metadata suffix Looking Glass software parses.

        :param stem: Base name without extension (e.g. ``"helix_density"``).
        :param ext: File extension without the dot.
        :return: e.g. ``"helix_density_qs8x6a0.75.png"``.
        """
        return f"{stem}_qs{self.columns}x{self.rows}a{self.aspect:g}.{ext}"

    def with_grid(self, columns: int, rows: int) -> QuiltSpec:
        """Same quilt at a different view-grid density.

        Total quilt pixels stay fixed, so more views means fewer pixels per
        view: the device's lenticular optics interpolate between views, so
        extra views give smoother look-around at the cost of per-view
        sharpness.  The official presets are the factory-calibrated balance.

        :param columns: New number of tile columns.
        :param rows: New number of tile rows.
        :return: A new :class:`QuiltSpec` with the requested grid.
        """
        return replace(self, columns=columns, rows=rows)


#: Standard ("ideal") quilt settings per Looking Glass device, from the
#: official docs: https://lfdocs.lookingglassfactory.com/keyconcepts/quilts
QUILT_PRESETS: dict[str, QuiltSpec] = {
    "portrait": QuiltSpec(columns=8, rows=6, quilt_width=3360, quilt_height=3360, aspect=0.75),
    "go": QuiltSpec(columns=11, rows=6, quilt_width=4092, quilt_height=4092, aspect=0.5625),
    # Gen3 16" Landscape (hardwareVersion "16_gen3_l"), verified against the
    # defaultQuilt Bridge reports for LKG-J00332.  Its native view cone is 50
    # degrees, wider than the 35-degree QuiltSpec default.
    "16-landscape": QuiltSpec(
        columns=8, rows=6, quilt_width=7680, quilt_height=4320, aspect=1.77778, view_cone=50.0
    ),
    "16-portrait": QuiltSpec(
        columns=11, rows=6, quilt_width=5995, quilt_height=6000, aspect=0.5625
    ),
    "27-landscape": QuiltSpec(columns=8, rows=6, quilt_width=7680, quilt_height=4320, aspect=1.777),
    "27-portrait": QuiltSpec(
        columns=12, rows=4, quilt_width=7680, quilt_height=4320, aspect=0.5625
    ),
    "32-landscape": QuiltSpec(columns=7, rows=7, quilt_width=8190, quilt_height=8190, aspect=1.777),
    "32-portrait": QuiltSpec(
        columns=11, rows=6, quilt_width=8184, quilt_height=8184, aspect=0.5625
    ),
    "65": QuiltSpec(columns=8, rows=9, quilt_width=8192, quilt_height=8192, aspect=1.777),
}


# ---------------------------------------------------------------------------
# Off-axis camera math
# ---------------------------------------------------------------------------


def view_offsets(spec: QuiltSpec, distance: float) -> np.ndarray:
    """Horizontal camera offsets (world units) for every view in the quilt.

    Cameras sweep a total angle of ``spec.view_cone`` centred on the base
    camera position, at constant distance from the focal plane.  Offsets are
    ordered to match quilt view order: view 0 is the leftmost camera.

    :param spec: Quilt specification (view count + cone angle).
    :param distance: Distance from camera to the focal plane.
    :return: Array of shape ``(n_views,)`` with signed offsets along the
        camera's right vector.
    """
    half_cone = math.radians(spec.view_cone) / 2.0
    n = spec.n_views
    if n == 1:
        return np.zeros(1)
    # Even angular spacing across the cone; tan() converts angle to lateral
    # shift so the focal plane is sampled like the physical display does.
    angles = np.linspace(-half_cone, half_cone, n)
    return distance * np.tan(angles)


def _camera_frame(camera) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Decompose a vtkCamera into position, focal point, right/up basis, distance."""
    pos = np.asarray(camera.position, dtype="d")
    focal = np.asarray(camera.focal_point, dtype="d")
    up = np.asarray(camera.up, dtype="d")
    forward = focal - pos
    distance = float(np.linalg.norm(forward))
    forward /= distance
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return pos, focal, right, true_up, distance


def _apply_off_axis_view(camera, base, offset: float, tile_aspect: float) -> None:
    """Position *camera* for one quilt view using an off-axis projection.

    The camera and focal point translate together by *offset* along the
    camera's right vector (no rotation — view direction stays parallel), then
    the projection window centre is sheared so the original focal point stays
    centred on screen.  VTK's ``WindowCenter`` shifts the frustum by
    ``wcx * half_width`` at every depth, so setting
    ``wcx = -offset / half_width_at_focal_plane`` re-centres the focal plane
    exactly — the standard Looking Glass off-axis recipe.

    :param camera: ``pv.Camera`` / vtkCamera to mutate.
    :param base: Tuple from :func:`_camera_frame` of the *original* camera.
    :param offset: Lateral world-space offset for this view.
    :param tile_aspect: Width/height of the render viewport.
    """
    pos, focal, right, true_up, distance = base
    camera.position = tuple(pos + right * offset)
    camera.focal_point = tuple(focal + right * offset)
    camera.up = tuple(true_up)
    half_height = distance * math.tan(math.radians(camera.view_angle) / 2.0)
    half_width = half_height * tile_aspect
    camera.SetWindowCenter(-offset / half_width, 0.0)


# ---------------------------------------------------------------------------
# Quilt rendering
# ---------------------------------------------------------------------------


def assemble_quilt(views: Iterable[np.ndarray], spec: QuiltSpec) -> np.ndarray:
    """Tile per-view images into a single quilt image.

    This is the renderer-agnostic half of quilt production: it takes views
    that some backend already rendered — VTK via :func:`render_quilt`, a
    ray-tracer via :mod:`waverider.povray` — and lays them out in quilt
    order.  Views are consumed lazily, so a backend can stream them without
    holding all ``n_views`` frames in memory at once.

    Views whose pixel size differs from the tile size are resampled, which
    is what makes anamorphic quilts (tile pixel aspect != view aspect, e.g.
    the 27" presets) come out correctly.

    :param views: Iterable of ``uint8`` RGB (or RGBA) arrays in view order —
        view 0 is the leftmost camera.  Must yield exactly ``spec.n_views``.
    :param spec: Quilt specification (grid, size, aspect).
    :return: ``uint8`` RGB array of shape ``(quilt_height, quilt_width, 3)``.
    :raises ValueError: If the number of views does not match ``spec``.
    """
    quilt = np.zeros((spec.quilt_height, spec.quilt_width, 3), dtype=np.uint8)
    n = 0
    for i, img in enumerate(views):
        if i >= spec.n_views:
            raise ValueError(f"got more than {spec.n_views} views for a {spec.columns}x{spec.rows} quilt")
        img = np.asarray(img)[..., :3]
        if img.shape[:2] != (spec.tile_height, spec.tile_width):
            img = _resize_view(img, spec.tile_width, spec.tile_height)
        x, y = spec.tile_origin(i)
        quilt[y : y + spec.tile_height, x : x + spec.tile_width] = img
        n = i + 1
    if n != spec.n_views:
        raise ValueError(f"expected {spec.n_views} views for a {spec.columns}x{spec.rows} quilt, got {n}")
    return quilt


def render_quilt(
    plotter,
    spec: QuiltSpec,
    *,
    view_cone: float | None = None,
    fov: float | None = 14.0,
    zoom: float | None = None,
) -> np.ndarray:
    """Render the plotter's scene into a quilt image.

    The plotter's current camera defines the centre view; its focal point
    becomes the holographic focal plane (the physical surface of the
    display).  Position the camera before calling — e.g. via
    ``plotter.camera_position`` or ``plotter.reset_camera()`` — exactly as
    you would for a normal screenshot.

    :param plotter: An *off-screen* ``pv.Plotter`` with the scene composed.
    :param spec: Quilt specification (grid, size, aspect, cone).
    :param view_cone: Override the spec's view cone in degrees.
    :param fov: Vertical field of view in degrees for the quilt cameras.
        Looking Glass recommends ~14° (matches real-world parallax at
        typical viewing distance); the camera is dollied back so the scene
        stays the same size in frame.  Pass ``None`` to keep the plotter's
        current FOV and distance.
    :param zoom: Optional camera zoom factor applied after framing, before
        the view sweep.  Values > 1 make the subject fill more of each tile,
        which is what drives perceived depth — parallax is proportional to
        on-screen size, so a subject occupying a third of the frame yields a
        third of the available look-around.
    :return: ``uint8`` RGB array of shape ``(quilt_height, quilt_width, 3)``.
    """
    _require_pyvista("render_quilt")
    if view_cone is not None:
        spec = replace(spec, view_cone=view_cone)

    # Views are *captured* at the declared view aspect (= display aspect) so
    # the frustum is undistorted, then resampled into the tile.  For most
    # devices these match; some ideal quilts (e.g. 27") store views
    # anamorphically, with tile pixel aspect != view aspect.
    render_h = spec.tile_height
    render_w = round(render_h * spec.aspect)
    plotter.window_size = (render_w, render_h)
    if not plotter.camera.is_set:
        # Mirror pyvista's first-render behaviour (it only runs on
        # show()/screenshot(), after we have already read the camera).
        plotter.camera_position = plotter.renderer.get_default_cam_pos()
        plotter.reset_camera()
    plotter.render()

    camera = plotter.camera
    if fov is not None:
        # Narrow the FOV and dolly back so the focal plane stays the same
        # size in frame: new_distance = half_height / tan(fov/2).
        pos, focal, _, _, distance = _camera_frame(camera)
        half_height = distance * math.tan(math.radians(camera.view_angle) / 2.0)
        new_distance = half_height / math.tan(math.radians(fov) / 2.0)
        forward = (focal - pos) / distance
        camera.position = tuple(focal - forward * new_distance)
        camera.view_angle = fov
    if zoom is not None and zoom != 1.0:
        # Dolly rather than narrow the view angle: pulling the camera in
        # magnifies the subject while preserving the FOV the parallax
        # geometry was built around, and the focal plane stays on the
        # display surface.  view_offsets() rescales with the new distance,
        # so the angular look-around is unchanged.
        camera.Dolly(zoom)
    base = _camera_frame(camera)
    distance = base[4]
    offsets = view_offsets(spec, distance)
    render_aspect = render_w / render_h

    def views():
        for offset in offsets:
            _apply_off_axis_view(camera, base, float(offset), render_aspect)
            # The dolly + lateral sweep move the camera well outside the range
            # VTK computed for the original position; re-fit it to the scene.
            plotter.renderer.reset_camera_clipping_range()
            # screenshot() alone returns the previous framebuffer; force a
            # render so each view reflects this view's camera.
            plotter.render()
            yield plotter.screenshot(None, return_img=True)

    quilt = assemble_quilt(views(), spec)

    # Restore the centre view so the plotter is reusable afterwards.
    _apply_off_axis_view(camera, base, 0.0, render_aspect)
    camera.SetWindowCenter(0.0, 0.0)
    plotter.renderer.reset_camera_clipping_range()
    return quilt


def _resize_view(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a rendered view to the quilt tile size (anamorphic storage)."""
    from PIL import Image

    resized = Image.fromarray(img).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized)


def save_quilt(quilt: np.ndarray, stem: str | Path, spec: QuiltSpec) -> Path:
    """Write a quilt to PNG using the Looking Glass filename convention.

    :param quilt: RGB array from :func:`render_quilt`.
    :param stem: Output path *without* the quilt suffix or extension.
        Any ``.png`` extension is stripped first.
    :param spec: Quilt specification (encodes the suffix metadata).
    :return: The path written, e.g. ``out/helix_qs8x6a0.75.png``.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "save_quilt() requires pillow.\nInstall with:  poetry install --with viz"
        ) from exc

    stem = Path(stem)
    if stem.suffix.lower() == ".png":
        stem = stem.with_suffix("")
    out_path = stem.parent / spec.filename(stem.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(quilt).save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Quilt video (turntable / animated holograms)
# ---------------------------------------------------------------------------


def _find_ffmpeg() -> str:
    """Locate an ffmpeg binary: system PATH first, then imageio-ffmpeg's."""
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "Quilt video encoding requires ffmpeg.\n"
            "Install it system-wide, or:  pip install imageio-ffmpeg"
        ) from exc


def _encode_args(spec: QuiltSpec, crf: int) -> list[str]:
    """ffmpeg output arguments for the official quilt-video requirements.

    MP4 with ``yuv420p`` pixel format is required by Looking Glass players.
    H.264 is fine for quilts up to 6000 px on their longest side (Portrait,
    Go, 16" portrait); anything larger uses HEVC.  yuv420p also needs even
    dimensions, so odd quilt sizes are padded by one pixel.
    """
    codec = "libx265" if max(spec.quilt_width, spec.quilt_height) > 6000 else "libx264"
    args = ["-vcodec", codec, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if spec.quilt_width % 2 or spec.quilt_height % 2:
        args += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    return args


def render_quilt_video(
    plotter,
    spec: QuiltSpec,
    out_stem: str | Path,
    *,
    n_frames: int = 180,
    fps: int = 24,
    orbit_degrees: float = 360.0,
    view_cone: float | None = None,
    fov: float | None = 14.0,
    zoom: float | None = None,
    crf: int = 18,
    on_frame=None,
    progress: bool = True,
) -> Path:
    """Render an animated quilt video (default: a full turntable orbit).

    Renders one quilt per frame, rotating the camera about the focal point
    between frames, then encodes the sequence to MP4 per the Looking Glass
    quilt-video spec (``yuv420p``; H.264, or HEVC for 8K quilts).  The
    filename carries the ``_qs<cols>x<rows>a<aspect>`` suffix so Studio /
    Bridge auto-detect playback settings.

    Note the cost: a Portrait video renders ``n_frames x 48`` views.

    :param plotter: An *off-screen* ``pv.Plotter`` with the scene composed.
    :param spec: Quilt specification (grid, size, aspect, cone).
    :param out_stem: Output path; quilt suffix + ``.mp4`` are appended.
    :param n_frames: Number of video frames (with *fps* sets loop duration).
    :param orbit_degrees: Total camera orbit over the clip; 360 loops
        seamlessly.  Pass 0 to disable the turntable (use *on_frame*).
    :param view_cone: Override the spec's view cone in degrees.
    :param fov: Per-view vertical FOV; see :func:`render_quilt`.
    :param zoom: Camera dolly factor; see :func:`render_quilt`.
    :param crf: x264/x265 quality (lower = better; 15-20 sensible).
    :param on_frame: Optional ``callback(frame_index)`` invoked before each
        frame renders — mutate the scene here for custom animation.
    :param progress: Print a progress line while rendering.
    :return: Path of the quilt MP4 written.
    """
    import subprocess
    import tempfile

    _require_pyvista("render_quilt_video")
    ffmpeg = _find_ffmpeg()

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "render_quilt_video() requires pillow.\nInstall with:  poetry install --with viz"
        ) from exc

    out_stem = Path(out_stem)
    if out_stem.suffix.lower() == ".mp4":
        out_stem = out_stem.with_suffix("")
    out_path = out_stem.parent / spec.filename(out_stem.name, ext="mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = orbit_degrees / n_frames if n_frames else 0.0
    with tempfile.TemporaryDirectory(prefix="quilt_frames_") as tmp:
        for i in range(n_frames):
            if on_frame is not None:
                on_frame(i)
            # Zoom only on the first frame: render_quilt leaves the camera at
            # the dollied distance, and Azimuth() preserves it, so re-applying
            # it every frame would compound into a creeping zoom-in.
            quilt = render_quilt(
                plotter, spec, view_cone=view_cone, fov=fov, zoom=zoom if i == 0 else None
            )
            Image.fromarray(quilt).save(f"{tmp}/frame{i:05d}.png")
            plotter.camera.Azimuth(step)
            if progress:
                print(f"\r  quilt frame {i + 1}/{n_frames}", end="", flush=True)
        if progress:
            print()

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            f"{tmp}/frame%05d.png",
            *_encode_args(spec, crf),
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{result.stderr[-2000:]}")
    return out_path


# ---------------------------------------------------------------------------
# Casting to the display via Looking Glass Bridge
# ---------------------------------------------------------------------------

#: Default address of the Looking Glass Bridge HTTP API (Bridge >= 2.2).
BRIDGE_URL = "http://localhost:33334"


def _bridge_post(bridge_url: str, endpoint: str, payload: dict, timeout: float) -> dict:
    """Send a JSON payload to a Bridge endpoint and decode the response.

    Bridge's HTTP API expects ``PUT``.  It answers ``POST`` with ``200 OK``
    and an *empty* body, so using the wrong verb fails silently: the caller
    reads back no orchestration token and every later call is a no-op.
    """
    req = urllib.request.Request(
        f"{bridge_url}/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}


def _enter_orchestration(bridge_url: str, timeout: float) -> str:
    """Enter (or rejoin) the default Bridge orchestration session.

    Bridge scopes all playback control to an orchestration session; calling
    ``enter_orchestration`` again while one is already active returns the
    *same* token rather than starting a new session, so every helper in this
    module can call this independently without stepping on the others.

    :return: The orchestration token required by every other Bridge call.
    """
    resp = _bridge_post(bridge_url, "enter_orchestration", {"name": "default"}, timeout)
    token = resp.get("payload", {}).get("value", "")
    if not token:
        raise RuntimeError(
            f"Looking Glass Bridge at {bridge_url} returned no orchestration token "
            f"(response: {resp!r}).  Is Bridge running and >= 2.2?"
        )
    return token


def cast_quilt(
    quilt_path: str | Path,
    spec: QuiltSpec,
    *,
    bridge_url: str = BRIDGE_URL,
    playlist: str = "waverider",
    timeout: float = 10.0,
) -> dict:
    """Show a saved quilt on the connected Looking Glass via Bridge.

    Requires `Looking Glass Bridge <https://lookingglassfactory.com/software/looking-glass-bridge>`_
    (>= 2.2) running on the machine the display is plugged into.  Follows
    Bridge's orchestration sequence: enter orchestration, show the display
    window, create a playlist holding the quilt, and play it.

    :param quilt_path: Path to a quilt PNG on the *Bridge host's* filesystem.
    :param spec: Quilt specification (tiling + aspect sent to Bridge).
    :param bridge_url: Base URL of the Bridge HTTP API.
    :param playlist: Name of the Bridge playlist to (re)create.
    :param timeout: HTTP timeout in seconds per request.
    :return: Decoded JSON response of the final ``play_playlist`` call.
    """
    token = _enter_orchestration(bridge_url, timeout)

    _bridge_post(
        bridge_url,
        "show_window",
        {"orchestration": token, "show_window": True, "head_index": -1},
        timeout,
    )
    _bridge_post(
        bridge_url,
        "instance_playlist",
        {"orchestration": token, "name": playlist, "loop": True},
        timeout,
    )
    _bridge_post(
        bridge_url,
        "insert_playlist_entry",
        {
            "orchestration": token,
            "name": playlist,
            "index": 0,
            "uri": str(Path(quilt_path).resolve()),
            "rows": spec.rows,
            "cols": spec.columns,
            "aspect": spec.aspect,
            "view_count": spec.n_views,
            "durationMS": 20000,
            "isRGBD": 0,
        },
        timeout,
    )
    return _bridge_post(
        bridge_url,
        "play_playlist",
        {"orchestration": token, "name": playlist, "head_index": -1},
        timeout,
    )


def pause_quilt(*, bridge_url: str = BRIDGE_URL, timeout: float = 10.0) -> dict:
    """Pause playback on the connected Looking Glass.

    Freezes the current frame; the playlist and its position are retained,
    so :func:`resume_quilt` continues from where it left off. This is
    Bridge's *transport control* group — there is no ``stop_playlist`` or
    ``pause_playlist`` endpoint (a guessed endpoint name doesn't 404: Bridge
    answers with ``200 OK`` and an empty body, indistinguishable from a slow
    success unless you check that the response has no ``status`` field).
    Confirmed against the endpoint list in the official
    `bridge.js <https://github.com/Looking-Glass/bridge.js>`_ SDK source.

    :param bridge_url: Base URL of the Bridge HTTP API.
    :param timeout: HTTP timeout in seconds.
    :return: Decoded JSON response of the ``transport_control_pause`` call.
    """
    token = _enter_orchestration(bridge_url, timeout)
    return _bridge_post(bridge_url, "transport_control_pause", {"orchestration": token}, timeout)


def resume_quilt(*, bridge_url: str = BRIDGE_URL, timeout: float = 10.0) -> dict:
    """Resume playback after :func:`pause_quilt`.

    :param bridge_url: Base URL of the Bridge HTTP API.
    :param timeout: HTTP timeout in seconds.
    :return: Decoded JSON response of the ``transport_control_play`` call.
    """
    token = _enter_orchestration(bridge_url, timeout)
    return _bridge_post(bridge_url, "transport_control_play", {"orchestration": token}, timeout)


def stop_quilt(*, bridge_url: str = BRIDGE_URL, timeout: float = 10.0) -> dict:
    """Stop playback: pause the current frame and hide the display window.

    Bridge's own `bridge.js <https://github.com/Looking-Glass/bridge.js>`_
    SDK documents ``delete_playlist`` as *the* way to stop a playlist, and
    an earlier version of this function called it. In testing it reliably
    left Bridge unresponsive to every further HTTP call — reproduced twice,
    once mid-video and once on a single still image, so it isn't a
    large-file decode race. This function deliberately avoids
    ``delete_playlist`` and reaches the same end state (nothing visible,
    playback halted) through calls already proven safe: the playlist from
    :func:`cast_quilt` is left instantiated but paused and hidden, rather
    than deleted, so :func:`cast_quilt` can safely replace it later.

    :param bridge_url: Base URL of the Bridge HTTP API.
    :param timeout: HTTP timeout in seconds.
    :return: Decoded JSON response of the final ``show_window`` call.
    """
    token = _enter_orchestration(bridge_url, timeout)
    _bridge_post(bridge_url, "transport_control_pause", {"orchestration": token}, timeout)
    return _bridge_post(
        bridge_url,
        "show_window",
        {"orchestration": token, "show_window": False, "head_index": -1},
        timeout,
    )
