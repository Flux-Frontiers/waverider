# Looking Glass Holographic Output

**Module**: `waverider.looking_glass`
**CLI**: `waverider-voxel-viz --quilt <device>`
**Source**: `src/waverider/looking_glass.py`

> *"A manifold you can slice with a mouse is good. A manifold floating
> behind glass is better."*

> **Which Looking Glass do you have?** This page covers the **light-field
> line** (Portrait, Go, 16″/27″/32″/65″ LFD), which consumes multi-view
> quilts via Bridge/Studio. The **Hololuminescent Displays** (16″/27″/86″
> HLD) are a different technology that plays ordinary 2-D video — see
> [hld.md](hld.md).

---

## Concept

Looking Glass displays are lenticular light-field screens: they emit dozens of
slightly different views of a scene across a horizontal cone, and your two
eyes (plus head movement) pick up different views — true glasses-free 3-D.

The content format is a **quilt**: a single PNG that tiles N renders of the
scene in a grid, view 0 at the *bottom-left* (leftmost camera) through the
top-right (rightmost camera). The filename suffix
`_qs<cols>x<rows>a<aspect>` carries the layout metadata, so Looking Glass
Studio and Bridge configure playback automatically.

`waverider.looking_glass` turns **any PyVista scene** into a quilt:

1. The plotter's camera defines the centre view; its **focal point becomes
   the plane of the physical glass**.
2. The camera sweeps a 35° cone (the documented rendering standard) in
   N steps — *translating*, never rotating ("toe-in" breaks the optics).
3. Each view uses an **off-axis (asymmetric-frustum) projection** via VTK's
   window-centre shear, so the focal plane is pixel-identical across all
   views — the geometric requirement for the display to fuse them.
4. Views are captured at ~14° FOV (camera dollied back to compensate),
   matching real-world parallax at typical viewing distance.
5. Tiles are assembled and saved with the quilt filename convention.

---

## Quick start

### From the CLI

```bash
# Render the iris manifold as a Looking Glass Portrait quilt
waverider-voxel-viz --dataset iris --quilt portrait --out iris

# -> iris_qs8x6a0.75.png   (48 views, 8x6, 3360x3360)

# Render and immediately show it on the display (Bridge must be running)
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast

# Rotating manifold: looping turntable quilt video
waverider-voxel-viz --dataset iris --quilt portrait --quilt-video \
    --frames 180 --fps 24 --out iris_spin --cast
# -> iris_spin_qs8x6a0.75.mp4   (7.5 s seamless 360-degree loop)

# More views (smoother look-around, less per-view resolution)
waverider-voxel-viz --dataset iris --quilt portrait --quilt-grid 11x6 --out iris

# Wider/narrower apparent depth
waverider-voxel-viz --dataset digits --quilt go --view-cone 40 --out digits
```

### From Python — any PyVista scene

```python
import pyvista as pv
from waverider import QUILT_PRESETS, render_quilt, save_quilt, cast_quilt

p = pv.Plotter(off_screen=True)
p.add_mesh(pv.ParametricTorus(), color="teal")

spec = QUILT_PRESETS["portrait"]
quilt = render_quilt(p, spec)              # (3360, 3360, 3) uint8
path = save_quilt(quilt, "torus", spec)    # torus_qs8x6a0.75.png
cast_quilt(path, spec)                     # show it on the device
p.close()
```

### From the manifold pipeline

```python
from waverider import build_grid, fit_and_observe, render_quilt_single, voxelize

_, _, pf, pca_info = fit_and_observe(X, y)
grid = build_grid(voxelize(pf))
render_quilt_single(grid, pf, scalar="curvature", out_path="curvature",
                    device="portrait", pca_info=pca_info, cast=True)

# Rotating version (MP4 turntable)
render_quilt_single(grid, pf, scalar="curvature", out_path="curvature_spin",
                    device="portrait", pca_info=pca_info,
                    video=True, n_frames=180, fps=24)
```

### Animated holograms beyond turntables

`render_quilt_video()` accepts an `on_frame(i)` callback that runs before
each frame — mutate the scene (advance a time step, move a slice plane,
update scalars) for arbitrary animation:

```python
from waverider import QUILT_PRESETS, render_quilt_video

render_quilt_video(plotter, QUILT_PRESETS["portrait"], "evolving",
                   n_frames=240, fps=24, orbit_degrees=0.0,
                   on_frame=lambda i: advance_simulation(plotter, i))
```

Video encoding needs ffmpeg on the PATH, or `pip install imageio-ffmpeg`
for a bundled binary. Encoding follows the official quilt-video spec:
MP4 + `yuv420p`, H.264 for Portrait/Go/16" sizes, HEVC for the 8K-quilt
devices, and the same `_qs…a…` filename convention.

---

## Device presets

`QUILT_PRESETS` follows the official ideal-quilt table
(<https://lfdocs.lookingglassfactory.com/keyconcepts/quilts>):

| Key | Device | Quilt | Grid | Views | View aspect |
|---|---|---|---|---|---|
| `portrait` | Portrait | 3360×3360 | 8×6 | 48 | 0.75 |
| `go` | Go | 4092×4092 | 11×6 | 66 | 0.5625 |
| `16-landscape` | 16″ landscape | 5999×5999 | 7×7 | 49 | 1.777 |
| `16-portrait` | 16″ portrait | 5995×6000 | 11×6 | 66 | 0.5625 |
| `27-landscape` | 27″ landscape | 7680×4320 | 8×6 | 48 | 1.777 |
| `27-portrait` | 27″ portrait | 7680×4320 | 12×4 | 48 | 0.5625 |
| `32-landscape` | 32″ landscape | 8190×8190 | 7×7 | 49 | 1.777 |
| `32-portrait` | 32″ portrait | 8184×8184 | 11×6 | 66 | 0.5625 |
| `65` | 65″ | 8192×8192 | 8×9 | 72 | 1.777 |

Some presets store views anamorphically (tile pixel aspect ≠ view aspect);
the renderer captures each view at the declared aspect and resamples into
the tile, so geometry is never distorted.

Custom layouts are just a dataclass away:

```python
from waverider import QuiltSpec
spec = QuiltSpec(columns=8, rows=6, quilt_width=3360, quilt_height=3360,
                 aspect=0.75, view_cone=35.0)
```

---

## Setting up the display (Looking Glass Bridge)

**Looking Glass Bridge** is the driver/daemon ("glass server") that talks
to the hardware. One-time setup:

1. Download Bridge from
   <https://lookingglassfactory.com/software/looking-glass-bridge>
   (macOS / Windows / Linux) and install it on the computer the display
   is plugged into. It runs in the background (menu-bar/tray icon).
2. Connect the display — the Portrait needs **both** cables: USB-C (data /
   calibration) and HDMI (video).
3. Verify Bridge sees the device: the tray icon lists it, or
   `curl http://localhost:33334/` answers (the HTTP API, Bridge ≥ 2.2).
4. Optionally install **Looking Glass Studio** (same downloads page) — a
   quilt player/library app: drag any `*_qs*.png` / `*_qs*.mp4` in and
   settings are auto-detected from the filename.

`cast_quilt()` / `--cast` then displays renders directly from Python via
Bridge's HTTP orchestration flow (`enter_orchestration → show_window →
instance_playlist → insert_playlist_entry → play_playlist`). Run it on
the machine the display is plugged into.

## How many views?

The presets follow each device's factory-calibrated ideal (48 for the
Portrait). The lenticular driver *interpolates* between quilt views, so:

- **More views** (`--quilt-grid 11x6` = 66) → smoother look-around and
  less ghosting at the cone edges, but each view gets fewer pixels (the
  quilt's total pixel budget is fixed per device).
- **Fewer views** → sharper individual views, more visible "jumping"
  as you move your head.

48 views is genuinely the sweet spot for the Portrait's optics; go denser
only if you notice stepping artifacts in deep scenes.

## Composition tips

- Centre the most important structure at the camera's **focal point** — it
  sits at the glass surface and stays sharpest.
- Depth budget is asymmetric: content can recede far behind the glass, but
  pop-out in front degrades quickly.
- 2-D overlays (scalar bars, titles) render identically in every view, so
  they read as labels printed on the glass — use sparingly.
- For turntable videos, 360° over 6–10 s (`--frames 180 --fps 24`) reads
  well; faster spins fight the depth effect.
