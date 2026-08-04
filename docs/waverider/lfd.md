# Light-Field Display (LFD) Output

**Module**: `waverider.lfd`
**CLI**: `waverider-voxel-viz --quilt <device>`
**Source**: `src/waverider/lfd.py`

> *"A manifold you can slice with a mouse is good. A manifold floating
> behind glass is better."*

> **Which Looking Glass do you have?** This page covers the **light-field
> line** (Portrait, Go, 16″/27″/32″/65″ LFD), which consumes multi-view
> quilts via Bridge/Studio. The **Hololuminescent Displays** (16″/27″/86″
> HLD) are a different technology that plays ordinary 2-D video — see
> [hld.md](hld.md).

---

## Concept

Looking Glass light-field displays are lenticular screens: they emit dozens
of slightly different views of a scene across a horizontal cone, and your
two eyes (plus head movement) pick up different views — true glasses-free
3-D.

The content format is a **quilt**: a single PNG that tiles N renders of the
scene in a grid, view 0 at the *bottom-left* (leftmost camera) through the
top-right (rightmost camera). The filename suffix
`_qs<cols>x<rows>a<aspect>` carries the layout metadata, so Looking Glass
Studio and Bridge configure playback automatically.

`waverider.lfd` turns **any PyVista scene** into a quilt:

1. The plotter's camera defines the centre view; its **focal point becomes
   the plane of the physical glass**.
2. The camera sweeps the device's view cone (35° default; the Gen3 16″
   landscape reports 50°) in N steps — *translating*, never rotating
   ("toe-in" breaks the optics).
3. Each view uses an **off-axis (asymmetric-frustum) projection** via VTK's
   window-centre shear, so the focal plane is pixel-identical across all
   views — the geometric requirement for the display to fuse them.
4. Views are captured at ~14° FOV (camera dollied back to compensate),
   matching real-world parallax at typical viewing distance.
5. Tiles are assembled and saved with the quilt filename convention.

Step 2 is the one people get wrong. The intuitive approach — orbit the
camera slightly per view, aiming each at the subject — is "toe-in," and it
shears the focal plane differently in every view. The display cannot fuse
the result, and you get ghosting instead of depth.

---

## 1. Set up Bridge and the display

**Looking Glass Bridge** is the driver/daemon that talks to the hardware.
One-time setup:

1. Download Bridge from
   <https://lookingglassfactory.com/software/looking-glass-bridge>
   (macOS / Windows / Linux) and install it on the computer the display is
   plugged into. It runs in the background (menu-bar/tray icon).
2. Connect the display. LFD panels need **both** cables: USB-C (data and
   calibration) and HDMI/DisplayPort (video). The panel also appears to the
   OS as an ordinary external monitor.
3. Verify Bridge sees the device — see the next section.
4. Optionally install **Looking Glass Studio** (same downloads page), a
   quilt player/library app: drag any `*_qs*.png` / `*_qs*.mp4` in and the
   settings are auto-detected from the filename.

`cast_quilt()` / `--cast` then displays renders directly from Python via
Bridge's HTTP orchestration flow (`enter_orchestration → show_window →
instance_playlist → insert_playlist_entry → play_playlist`). Run it on the
machine the display is plugged into.

## 2. Ask the panel what it wants

**Do this before your first render.** Quilt specs differ between
generations of the same nominal size, and the published tables lag the
hardware — the Gen3 16″ landscape wants 8×6 @ 7680×4320, where older docs
list 7×7 @ 5999×5999. Bridge reports the truth:

```bash
TOK=$(curl -s -X PUT -H 'Content-Type: application/json' -d '{"name":"probe"}' \
      http://localhost:33334/enter_orchestration | python3 -c \
      'import sys,json; print(json.load(sys.stdin)["payload"]["value"])')
curl -s -X PUT -H 'Content-Type: application/json' -d "{\"orchestration\":\"$TOK\"}" \
     http://localhost:33334/available_output_devices | python3 -m json.tool
```

On the entry whose `hwid` matches your serial, read:

| Field | Use |
|---|---|
| `hardwareVersion` | Device + generation, e.g. `16_gen3_l` |
| `defaultQuilt` | `quiltX`/`quiltY` (pixels) and `tileX`/`tileY` (grid) |
| `calibration` → `viewCone` | View cone in degrees — pass to `--view-cone` |
| `calibration` → `screenW`/`screenH` | Native panel resolution |

Match those against the preset table below. If they disagree, trust the
device and override with `--quilt-grid` / `--view-cone`, or build a
`QuiltSpec` directly.

> **Bridge's HTTP API requires `PUT`.** It answers `POST` with `200 OK` and
> an empty body, so a `POST` client silently reads back no orchestration
> token and every subsequent call quietly does nothing. If a cast "succeeds"
> but the glass never changes, check the verb first.

Note that non-Looking-Glass monitors may also be listed (with
`hardwareVersion` `thirdparty` and an empty `calibration`). `head_index: -1`
lets Bridge pick the real device.

## 3. Render a quilt

```bash
# Gen3 16" landscape — render and show it on the glass
waverider-voxel-viz --dataset iris --quilt 16-landscape --out iris --cast
# -> iris_qs8x6a1.77778.png   (48 views, 8x6, 7680x4320)

# Looking Glass Portrait
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast
# -> iris_qs8x6a0.75.png      (48 views, 8x6, 3360x3360)

# Rotating manifold: looping turntable quilt video
waverider-voxel-viz --dataset iris --quilt 16-landscape --quilt-video \
    --frames 180 --fps 24 --out iris_spin
# -> iris_spin_qs8x6a1.77778.mp4   (7.5 s seamless 360-degree loop)

# More views: smoother look-around, less per-view resolution
waverider-voxel-viz --dataset iris --quilt 16-landscape --quilt-grid 11x6 --out iris

# Override the view cone (use what the device reported)
waverider-voxel-viz --dataset digits --quilt 16-landscape --view-cone 50 --out digits

# CT/MRI demo mode: brain MRI instead of a manifold fit
waverider-voxel-viz --ct-demo --ct-dataset brain --quilt portrait --out brain --cast
```

Drop `--cast` to just write the file, then drag it into Looking Glass
Studio — the `_qs…a…` suffix configures playback automatically.

### From Python — any PyVista scene

```python
import pyvista as pv
from waverider import QUILT_PRESETS, render_quilt, save_quilt, cast_quilt

p = pv.Plotter(off_screen=True)
p.add_mesh(pv.ParametricTorus(), color="teal")

spec = QUILT_PRESETS["16-landscape"]
quilt = render_quilt(p, spec, zoom=1.6)          # (4320, 7680, 3) uint8
path = save_quilt(quilt, "torus", spec)          # torus_qs8x6a1.77778.png
cast_quilt(path, spec)                           # show it on the device
p.close()
```

### From the manifold pipeline

```python
from waverider import build_grid, fit_and_observe, render_quilt_single, voxelize

_, _, pf, pca_info = fit_and_observe(X, y)
grid = build_grid(voxelize(pf))
render_quilt_single(grid, pf, scalar="curvature", out_path="curvature",
                    device="16-landscape", pca_info=pca_info, cast=True)

# Rotating version (MP4 turntable)
render_quilt_single(grid, pf, scalar="curvature", out_path="curvature_spin",
                    device="16-landscape", pca_info=pca_info,
                    video=True, n_frames=180, fps=24)
```

### From the CT/MRI demo mode

```python
from waverider.voxel_viz import render_ct_quilt

render_ct_quilt(ct_dataset="brain", out_path="brain",
                device="16-landscape", cast=True)

# Rotating version (MP4 turntable)
render_ct_quilt(ct_dataset="full_head", out_path="head_spin",
                device="16-landscape", video=True, n_frames=180, fps=24)
```

### Animated holograms beyond turntables

`render_quilt_video()` accepts an `on_frame(i)` callback that runs before
each frame — mutate the scene (advance a time step, move a slice plane,
update scalars) for arbitrary animation:

```python
from waverider import QUILT_PRESETS, render_quilt_video

render_quilt_video(plotter, QUILT_PRESETS["16-landscape"], "evolving",
                   n_frames=240, fps=24, orbit_degrees=0.0,
                   on_frame=lambda i: advance_simulation(plotter, i))
```

Video encoding needs ffmpeg on the PATH, or `pip install imageio-ffmpeg`
for a bundled binary. Encoding follows the official quilt-video spec:
MP4 + `yuv420p`, H.264 for quilts up to 6000 px on the longest side and
HEVC above that (so the 16″ landscape's 7680 px quilt encodes as HEVC), and
the same `_qs…a…` filename convention.

Note that a quilt video renders `n_frames × n_views` images — a 180-frame
turntable on a 48-view device is 8 640 renders. Preview with `--frames 30`
before committing to a long clip.

---

## Device presets

`QUILT_PRESETS` follows the official ideal-quilt table
(<https://lfdocs.lookingglassfactory.com/keyconcepts/quilts>), with
`16-landscape` corrected against a physical Gen3 panel:

| Key | Device | Quilt | Grid | Views | View aspect | Cone |
|---|---|---|---|---|---|---|
| `portrait` | Portrait | 3360×3360 | 8×6 | 48 | 0.75 | 35° |
| `go` | Go | 4092×4092 | 11×6 | 66 | 0.5625 | 35° |
| `16-landscape` | 16″ landscape (Gen3) | 7680×4320 | 8×6 | 48 | 1.77778 | **50°** |
| `16-portrait` | 16″ portrait | 5995×6000 | 11×6 | 66 | 0.5625 | 35° |
| `27-landscape` | 27″ landscape | 7680×4320 | 8×6 | 48 | 1.777 | 35° |
| `27-portrait` | 27″ portrait | 7680×4320 | 12×4 | 48 | 0.5625 | 35° |
| `32-landscape` | 32″ landscape | 8190×8190 | 7×7 | 49 | 1.777 | 35° |
| `32-portrait` | 32″ portrait | 8184×8184 | 11×6 | 66 | 0.5625 | 35° |
| `65` | 65″ | 8192×8192 | 8×9 | 72 | 1.777 | 35° |

Only `16-landscape` has been verified against physical hardware; the rest
come from the published table. Confirm yours with the probe above before
trusting a row.

Some presets store views anamorphically (tile pixel aspect ≠ view aspect).
The 16″ landscape is one: 7680×4320 in an 8×6 grid gives 960×720 tiles
(4:3) holding 16:9 views. The renderer captures each view at the declared
aspect and resamples into the tile, so geometry is never distorted.

Custom layouts are just a dataclass away:

```python
from waverider import QuiltSpec
spec = QuiltSpec(columns=8, rows=6, quilt_width=7680, quilt_height=4320,
                 aspect=1.77778, view_cone=50.0)
```

---

## Framing is a depth budget

Perceived depth scales with how much of each view the subject fills. A
volume occupying a third of the frame delivers roughly a third of the
parallax the panel can show, and wastes most of the per-view resolution —
each view is only a fraction of the quilt.

PyVista's default framing leaves a lot of empty space, so `--quilt-zoom`
defaults to **1.6**. On the iris manifold, going from PyVista's framing to
the default zoom moved subject coverage from 35% to 85% of frame width and
raised the mean difference between the extreme views from 15.1 to 51.0.

This **dollies** the camera rather than narrowing the view angle. That
matters: pulling in preserves the ~14° FOV the parallax geometry assumes
and keeps the focal plane on the glass, whereas a zoom that shrinks the
view angle changes the frustum the whole cone was built around.

```bash
# Default framing (zoom 1.6)
waverider-voxel-viz --dataset iris --quilt 16-landscape --out iris --cast

# PyVista's own framing, with the scale bar restored
waverider-voxel-viz --dataset iris --quilt 16-landscape --out iris \
    --quilt-zoom 1.0 --quilt-scalar-bar
```

The **colour scale bar is off by default** for quilts. Any 2-D overlay
renders identically in every view, which pins it to the focal plane — it
reads as a flat pane cutting through the hologram rather than a label. It
stays legible, so `--quilt-scalar-bar` is there when you need the values
more than the depth.

## How many views?

The presets follow each device's factory-calibrated ideal. The lenticular
driver *interpolates* between quilt views, so:

- **More views** (`--quilt-grid 11x6` = 66) → smoother look-around and less
  ghosting at the cone edges, but each view gets fewer pixels (the quilt's
  total pixel budget is fixed per device).
- **Fewer views** → sharper individual views, more visible "jumping" as you
  move your head.

The factory default is the sweet spot for the panel's optics; go denser
only if you notice stepping artifacts in deep scenes.

## Composition tips

- Centre the most important structure at the camera's **focal point** — it
  sits at the glass surface and stays sharpest.
- Depth budget is asymmetric: content can recede far behind the glass, but
  pop-out in front degrades quickly.
- Keep 2-D overlays (scalar bars, titles) to a minimum — see above.
- For turntable videos, 360° over 6–10 s (`--frames 180 --fps 24`) reads
  well; faster spins fight the depth effect.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `--cast` reports success, glass unchanged | Bridge not running, or a client using `POST` instead of `PUT`. `cast_quilt()` now raises `RuntimeError` on an empty orchestration token rather than failing silently. |
| `RuntimeError: … returned no orchestration token` | Bridge is reachable but rejected the handshake. Check it's ≥ 2.2 and that the display is connected. |
| Connection refused on :33334 | Bridge isn't running, or you're not on the machine the display is plugged into. `cast_quilt()` takes a `bridge_url` if it lives elsewhere. |
| Ghosting instead of depth | Quilt tiling doesn't match what the device expects. Re-check `defaultQuilt` against your preset — a quilt read with the wrong grid mixes views. |
| Shallow or absent parallax | Subject too small in frame (raise `--quilt-zoom`), or view cone set below the panel's (`--view-cone`). |
| Depth reads but the scene looks flat and papery | 2-D overlays sitting on the focal plane; drop `--quilt-scalar-bar`. |

To inspect a quilt without the hardware, open the PNG and check the tile
count, that no tile is blank, and that the first and last tiles differ —
if extreme views are identical, the camera sweep didn't happen.
