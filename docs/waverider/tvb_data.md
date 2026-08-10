# TVB Brain Datasets

*Eric G. Suchanek, PhD — Flux-Frontiers*

Real human, macaque and mouse brain geometry from
[The Virtual Brain](https://www.thevirtualbrain.org/) (TVB), rendered to
holographic hardware through the same
[quiltwright](https://github.com/suchanek/quiltwright) back end that
WaveRider's manifold and CT/MRI scenes use.

**The loaders live in quiltwright**, as `quiltwright.tvb_data` — brain
geometry is a scene source, like POV-Ray scenes and PyVista's example
datasets, not manifold science. See
[quiltwright's docs/tvb-data.md](https://github.com/suchanek/quiltwright/blob/main/docs/tvb-data.md)
for the download, cache, licensing and dataset reference.

**This page covers the WaveRider side**: the scene presets and the
`--tvb-demo` CLI built on top of that loader.

---

## Getting the data

Nothing to do — the ~337 MB `tvb-data` archive is downloaded from Zenodo on
first use and cached by quiltwright. It is GPL-3.0 and is never vendored
into either package.

Override the cache location with `$QUILTWRIGHT_TVB_CACHE`; the default is
your platform's native per-user cache directory
(`~/Library/Caches/quiltwright/tvb` on macOS).

```bash
waverider-voxel-viz --tvb-clear-cache      # drop the cached archive
```

Needs the viz extras (`poetry install --with viz`). Stills and quilts need
nothing else; only video output reaches for ffmpeg.

---

## Scene presets

```bash
waverider-voxel-viz --tvb-demo --tvb-dataset <name>
```

| Preset | Kind | What it is |
|---|---|---|
| `cortex` | surface | 16,384-vertex cortex, coloured by the 76-region parcellation |
| `cortex_80k` | surface | 81,924-vertex cortex, coloured by the 80-region parcellation |
| `cortex_hires` | surface | 283,380-vertex two-hemisphere cortex, decimated 50% by default |
| `connectome` | connectome | 76-region structural connectome inside a translucent cortex |
| `connectome_998` | connectome | 998-region connectome, top 1% of tracts |
| `head_layers` | layers | Nested shells — cortex, inner/outer skull, scalp |
| `macaque` | surface | 147,460-vertex macaque cortex, 84-region parcellation |
| `macaque_connectome` | connectome | 84-region macaque connectome in a translucent macaque cortex |

`connectome` scenes draw region centres as spheres scaled by weighted degree
and the surviving tracts as weight-coloured tubes. A full connectome is far
too dense to fuse as a hologram, so only the strongest edges are drawn —
`--tvb-percentile` controls the cut.

As with the CT presets, no preset colour is pure white: white renders as
transparent on an HLD.

---

## CLI

```bash
# Interactive viewer (default preset: cortex)
waverider-voxel-viz --tvb-demo

# Connectome, with more tracts surviving the weight threshold
waverider-voxel-viz --tvb-demo --tvb-dataset connectome --tvb-percentile 80

# Headless PNG
waverider-voxel-viz --tvb-demo --tvb-dataset head_layers --out head.png

# HLD turntable video (10 s loop, 3840x2160 HEVC bt709)
waverider-voxel-viz --tvb-demo --tvb-dataset cortex --hld --out cortex

# HLD still — no ffmpeg needed
waverider-voxel-viz --tvb-demo --tvb-dataset cortex --hld --still --out cortex

# Looking Glass quilt, cast live to the display via Bridge
waverider-voxel-viz --tvb-demo --tvb-dataset connectome \
    --quilt portrait --out connectome --cast

# Dense cortex, decimated hard for a 48-view sweep
waverider-voxel-viz --tvb-demo --tvb-dataset cortex_hires \
    --tvb-decimate 0.7 --quilt portrait --out cortex_hires
```

### TVB-specific flags

| Flag | Meaning |
|---|---|
| `--tvb-demo` | Enter TVB mode (bypasses `ManifoldModel` entirely) |
| `--tvb-dataset` | Which preset to render (default `cortex`) |
| `--tvb-percentile` | Connectome edge threshold, as a percentile of non-zero weights |
| `--tvb-decimate` | Fraction of triangles to remove, 0.0–1.0 |
| `--tvb-smooth` | Laplacian smoothing iterations, overriding the preset |
| `--tvb-clear-cache` | Delete the cached archive and exit |

The shared output flags behave exactly as they do in `--ct-demo` mode:
`--hld`, `--still`, `--quilt`, `--quilt-video`, `--quilt-grid`, `--cast`,
`--frames`, `--fps`, `--orbit`, `--zoom`, `--view-cone`, `--out`.

### Choosing a triangle budget

A quilt renders the whole scene **once per view** — 48 times on a Portrait.
`cortex` (32,760 triangles) sweeps comfortably; `cortex_hires` at full
density (566,752 triangles) does not. Raise `--tvb-decimate` for quilts and
lower it for HLD video, which is one ordinary render per frame.

Decimating a parcellated surface is safe: the region labels are re-assigned
to the decimated vertices by nearest neighbour.

---

## Python API

The renderers are importable directly:

```python
from waverider import render_tvb_quilt, render_tvb_hld, render_tvb_still, TVB_PRESETS

render_tvb_quilt("connectome", out_path="connectome", device="portrait")
render_tvb_hld("cortex", out_path="cortex", n_frames=300, fps=30)
```

For the underlying geometry — surfaces, connectomes, parcellations, sensors
— import from quiltwright:

```python
from quiltwright.tvb_data import load_surface, load_connectivity, surface_polydata
```

---

*Part of WaveRider — https://github.com/Flux-Frontiers/waverider*
