# TVB Brain Datasets

*Eric G. Suchanek, PhD — Flux-Frontiers*

Real human, macaque and mouse brain geometry from
[The Virtual Brain](https://www.thevirtualbrain.org/) (TVB), fetched on
demand and rendered to holographic hardware through the same
[quiltwright](https://github.com/suchanek/quiltwright) back end that
WaveRider's manifold and CT/MRI scenes use.

- Loader API — `waverider.tvb_data`
- Scene presets and CLI — `waverider.voxel_viz`, `--tvb-demo`

---

## Where the data comes from

The [tvb-root](https://github.com/the-virtual-brain/tvb-root) source tree
contains **no data**. Every demonstration dataset ships separately as
`tvb-data`, published on Zenodo as a single ~337 MB archive:

| | |
|---|---|
| DOI | [10.5281/zenodo.10128131](https://doi.org/10.5281/zenodo.10128131) |
| Version | 2.8.1 |
| Archive | `tvb_data.zip`, 337,115,643 bytes |
| MD5 | `08ae19833ba8ac158c91fbcb988b9bf0` |
| License | GPL-3.0 |

The old `tvb-data` GitHub repository is deprecated, and the PyPI package
carries a reduced file set because of size limits, so Zenodo is the
canonical source and the only one this module reads.

### Licensing

`tvb-data` is GPL-3.0; WaveRider is Elastic-2.0. Nothing is vendored — the
archive is downloaded at runtime and cached outside the source tree, the
same pattern `pyvista.examples` uses for its own downloads. That keeps the
GPL data out of this repository and out of any WaveRider distribution.

If you publish work using these datasets, TVB asks that you cite the
platform; the requested citation is available as
`waverider.tvb_data.TVB_CITATION`.

### Caching

The archive is downloaded once, to `$WAVERIDER_TVB_CACHE` if set, otherwise
to the platform's native per-user cache directory:

| Platform | Location |
|---|---|
| macOS | `~/Library/Caches/waverider/tvb` |
| Linux | `$XDG_CACHE_HOME/waverider/tvb`, default `~/.cache/waverider/tvb` |
| Windows | `%LOCALAPPDATA%\waverider\Cache\tvb` |

This is resolved with [`platformdirs`](https://pypi.org/project/platformdirs/),
matching PyVista — which caches its own downloads via `pooch.os_cache` and so
puts them in `~/Library/Caches/pyvista_3` on macOS. Hard-coding `~/.cache`
would drop a 337 MB file somewhere non-native on two of the three platforms.

Set `$WAVERIDER_TVB_CACHE` to put the archive on a different volume, share
one copy between checkouts, or pin it for CI.

Downloads stream to a temporary file and are moved into place only after the
MD5 check passes, so an interrupted transfer can never leave a truncated
archive behind. Individual files are read straight out of the zip — the
337 MB is never expanded on disk.

```bash
waverider-voxel-viz --tvb-clear-cache      # drop the cached archive
```

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

# HLD turntable video (10 s loop, 3840x2160 HEVC)
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
to the decimated vertices by nearest neighbour, because interpolating
between region 3 and region 70 would be meaningless.

---

## Python API

```python
from waverider.tvb_data import load_surface, load_connectivity

vertices, triangles, normals = load_surface("cortex_16384")   # downloads once
conn = load_connectivity("connectivity_76")

conn.weights.shape        # (76, 76) structural connection strengths
conn.tract_lengths.shape  # (76, 76) fibre lengths in mm
conn.centres.shape        # (76, 3)  region centres in mm
conn.labels[:3]           # ['rA1', 'rA2', 'rAMYG']
conn.degree               # weighted degree per region
```

Straight to PyVista (needs the `viz` extras):

```python
from waverider.tvb_data import surface_polydata, connectome_polydata

cortex = surface_polydata("cortex_16384", region_mapping="regionMapping_16k_76")
nodes, edges = connectome_polydata("connectivity_76", percentile=90.0)
```

Both return ordinary `pyvista` meshes, so a TVB scene composes with anything
else in a plotter and can be handed directly to
`quiltwright.lfd.render_quilt` or `quiltwright.hld.render_hld_video`.

### Available datasets

`SURFACES`, `CONNECTIVITIES`, `REGION_MAPPINGS` and `SENSORS` are plain
dicts mapping a short name to its path inside the archive:

```python
from waverider.tvb_data import SURFACES, CONNECTIVITIES
sorted(SURFACES)        # cortex_16384, cortex_80k, cortex_2x120k, skull/scalp shells, macaque_147k
sorted(CONNECTIVITIES)  # connectivity_66 … connectivity_998, macaque_84
```

Region mappings pair with a surface of matching vertex count:

| Mapping | Vertices | Pairs with |
|---|---|---|
| `regionMapping_16k_76` | 16,384 | `cortex_16384` |
| `regionMapping_80k_80` | 81,924 | `cortex_80k` |
| `regionMapping_147k_84` | 147,460 | `macaque_147k` |
| `regionMapping_16k_192` | 16,500 | *no surface in the archive matches* |

`surface_polydata` validates the pairing and raises rather than producing a
mis-coloured mesh.

---

## Archive quirks the loader absorbs

The TVB archive is not uniformly formatted. These are handled transparently,
and each is covered by a test:

| Quirk | Where | Handling |
|---|---|---|
| **1-based triangle indices** | `cortex_2x120k` | Detected and rebased. Loading as-is yields an index one past the last vertex — a silently corrupt mesh, not an error. |
| **Split hemispheres** | `cortex_2x120k` | `verticesl`/`verticesr` concatenated, right-hemisphere indices offset by the left vertex count, after each side is rebased independently. |
| **Folder-nested members** | `macaque_147k` | Members matched by basename. |
| **Float-encoded indices** | `macaque_147k` | `1.0000000e+00` parsed and checked for integrality. |
| **bz2-compressed members** | `connectivity_68`, some sensors | Decompressed transparently. |
| **Empty normals stub** | `face_8614` | A 1-byte `vertex_normals.txt` reads as "no normals"; PyVista computes its own. |

---

## Not yet wired up

The archive holds more than the presets expose. Left for later, in rough
order of value:

- **Simulated time series** — `nifti/time_series_152.nii.gz` and
  `gifti/sample.time_series.gii`. These would drive a per-vertex scalar over
  time, turning the HLD turntable into an activity animation rather than a
  static orbit. Needs a NIfTI/GIFTI reader (`nibabel`), which is why it is
  not in this pass.
- **Mouse brains** — `mouse/allen_2mm` and `mouse/calabrese`, stored as HDF5
  and NIfTI volumes rather than the plain-text surfaces used here.
- **EEG/MEG/sEEG sensors** — already loadable via `load_sensors`, but not yet
  given a scene preset that places electrodes over the scalp shell.
- **Projection matrices** and **local connectivity** — present in the
  archive, no obvious holographic use yet.

---

*Part of WaveRider — https://github.com/Flux-Frontiers/waverider*
