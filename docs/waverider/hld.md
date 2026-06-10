# Hololuminescent Display (HLD) Output

**Module**: `waverider.hld`
**CLI**: `waverider-voxel-viz --hld`
**Source**: `src/waverider/hld.py`

> *"On an HLD, white is the new alpha."*

---

## HLD vs. light-field: two different devices

Looking Glass sells two unrelated display technologies — know which one
you're rendering for:

| | **Hololuminescent (HLD)** | **Light-field (LFD)** |
|---|---|---|
| Models | 16″ / 27″ / 86″ Portrait HLD | Portrait, Go, 16″/27″/32″/65″ LFD |
| Content | **ordinary 2-D video** | multi-view **quilts** |
| True 3-D parallax | no — fixed holographic "alcove" effect | yes — ~48–100 views across a cone |
| Software | HLD Author + USB media player | Bridge / Studio |
| WaveRider module | `waverider.hld` (this page) | `waverider.looking_glass` |

The HLD is an LCD with a fixed holographic volume — a stage-like "alcove" —
embedded in its optical stack. Your flat video is multiply-blended into it:
**pure white pixels become invisible**, showing only the glowing alcove;
darker pixels appear as a subject standing inside it.

## What the device wants

From the official media spec and content guidelines
(<https://hlddocs.lookingglassfactory.com/>):

- **Master**: 2160×3840 (9:16 portrait) MP4, HEVC, 30 or 60 fps, bt709.
  One 4K master serves all sizes.
- **White background** — white is transparent on the device. Busy or dark
  backgrounds kill the effect.
- **Centered subject** inside safe margins (~9% top, 3% bottom/sides) so it
  stays within the alcove.
- **Contact shadow** on the floor ("crucial for the 3D effect").
- **Slow turntable orbit**, otherwise static camera.

`render_hld_video()` / `--hld` handles all of this: white background,
safe-area framing computed at the 9:16 aspect, a soft procedural contact
shadow, a seamless 360° orbit, and spec-exact encoding.

## Usage

```bash
# 10-second rotating manifold for the HLD (300 frames @ 30 fps)
waverider-voxel-viz --dataset iris --hld --out iris --volume
# -> iris_hld.mp4   (2160x3840, HEVC, bt709)

# Faster render for previews
waverider-voxel-viz --dataset iris --hld --frames 90 --out preview
```

From Python, for any PyVista scene:

```python
import pyvista as pv
from waverider import render_hld_video, style_plotter_for_hld

p = pv.Plotter(off_screen=True)
p.add_mesh(pv.ParametricTorus(), color="teal")
style_plotter_for_hld(p)          # white bg + 9:16 + safe-area framing
render_hld_video(p, "torus")      # torus_hld.mp4
p.close()
```

`render_hld_video()` also takes `on_frame(i)` for custom animation and
`orbit_degrees=0` to disable the turntable.

## Getting it on the display

1. Install **HLD Author** (free, Win/Mac, from the HLD docs site). Open the
   rendered `*_hld.mp4` — it previews the video inside a virtual HLD,
   validates the spec, rotates it for the player, and exports
   `*_HLD_encoded.mp4`.
2. Copy that file onto the USB drive that shipped with the display and plug
   it into the bundled Raspberry Pi player. Videos loop in alphanumeric
   filename order; hot-swap works.
3. Alternatively: drive the HLD directly over HDMI/DisplayPort as a normal
   portrait monitor (1080×1920 for the 16″, 2160×3840 for 27″/86″) and play
   the un-rotated master full-screen.

Note: Looking Glass **Bridge and quilts do not apply to HLDs** — those are
for the light-field line (see
[looking_glass.md](looking_glass.md)).

## Composition tips

- The subject *is* whatever isn't white. Saturated, mid-to-dark colors pop;
  near-white detail washes out.
- Don't let the subject touch the frame edges or it visibly clips against
  the alcove walls.
- The default scene shadow (`--no-shadow` to disable) is a soft procedural
  disc; it sells the "standing on the alcove floor" illusion.
- 8–12 s per loop reads well in ambient/signage settings; the player loops
  seamlessly with a 360° orbit.
