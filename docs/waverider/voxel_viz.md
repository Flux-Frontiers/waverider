# Voxel Visualizer

**Module**: `waverider.voxel_viz`
**CLI command**: `waverider-voxel-viz`
**Source**: `src/waverider/voxel_viz.py`

> *"Since we can move freely in embedding spaces and perform arbitrary subspace
> projections — why not materialize those projections as voxels and slice them?"*

---

## Overview

`waverider.voxel_viz` ships two independent rendering modes under one CLI command:

| Mode | Flag | What it does |
|---|---|---|
| **Manifold mode** | *(default)* | Fits `ManifoldModel` + `ManifoldObserver`, rasterizes geometric fields into a voxel grid, opens an interactive PyVista slice viewer |
| **CT / MRI demo mode** | `--ct-demo` | Loads a real biomedical CT or MRI volume from PyVista's built-in library, extracts layered isosurfaces (skin / tissue / bone), opens an interactive viewer or encodes an HLD turntable video |

Both modes share the same PyVista renderer and the same Looking Glass / HLD output pipeline.

**Dependencies** (``viz`` extras group):

```bash
poetry install --with viz   # pyvista, scipy, pillow
```

---

## Manifold Mode

### Concept

High-dimensional manifolds are invisible. WaveRider's `ManifoldObserver` computes
curvature, height above the tangent plane, and local intrinsic dimensionality at
every training node — but those fields live in N-dimensional space. The voxel
visualizer solves this by:

1. **Projecting** the manifold into a 3-D PCA subspace
2. **Rasterizing** the observer's geometric fields onto a uniform voxel grid
3. **Rendering** the grid in PyVista with draggable orthogonal slice planes

The result is a live cross-sectional anatomy of the manifold: slice through density
concentrations, follow curvature ridges, see where class boundaries cut the embedding.

### Pipeline

```
Data (X, y)
    │
    ├─ [optional] Pre-PCA: d → pre_pca dims     (recommended for MNIST / CIFAR)
    │
    ▼
ManifoldModel.fit(X, y)
    │  KNN graph, local PCA at each node
    │  stores: basis, eigenvalues, intrinsic_dim, label
    ▼
ManifoldObserver.observe()
    │  lifts each node to (N+1)-space via normal extension
    │  computes: curvature, height, tangent_spread per node
    ▼
PCA projection: N-D → 3-D  (visualization subspace)
    │
    ▼
Voxelization: 3-D points → uniform ImageData grid
    │  fields: density, curvature, height, intrinsic_dim, class_vote
    │  density smoothed with Gaussian blur (sigma=1 voxel)
    ▼
PyVista Plotter
    └─ add_mesh_slice_orthogonal()   ← drag planes interactively
    └─ scatter overlay of raw points (coloured by class)
```

### Scalar fields

Every voxel carries five scalar fields, each the **mean of all training points**
that fall into it (majority vote for `class_vote`).

| Field | Colour map | What it encodes |
|---|---|---|
| `density` | `plasma` | Training-point concentration (Gaussian-smoothed). Shows thick/thin manifold regions. |
| `curvature` | `coolwarm` | Mean ManifoldObserver curvature — rate of tangent-plane rotation between adjacent nodes. High = bent, folded, or boundary regions. |
| `height` | `viridis` | Mean reconstruction error above the tangent plane. Low = locally flat manifold. |
| `intrinsic_dim` | `tab10` | Local d* — PCA components needed to explain τ of local variance. Reveals where the manifold is higher or lower-dimensional. |
| `class_vote` | `Set1` | Majority class label per voxel — a direct cross-section of the decision surface. |

### Datasets

#### Synthetic (no extra dependencies)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `helix` | n × 5 | 2 | 1-manifold helix in 3-D, embedded in 5-D with Gaussian noise. **Default.** |
| `swiss_roll` | n × 3 | 2 | Classic 2-manifold Swiss roll. |
| `torus` | n × 4 | 4 | Flat torus (R=2, r=0.6) embedded in 4-D, four quadrant labels. |

#### Real — sklearn (always available)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `iris` | 150 × 4 | 3 | Fisher's Iris — setosa / versicolor / virginica. Fast; good for a first run. |
| `wine` | 178 × 13 | 3 | UCI Wine recognition — 3 cultivar classes. |
| `breast_cancer` | 569 × 30 | 2 | Wisconsin breast cancer — malignant / benign. |
| `digits` | 1797 × 64 | 10 | sklearn 8×8 handwritten digits. Rich enough to show class structure. |

All sklearn datasets are StandardScaler-normalised before fitting.

#### Real — large (needs `tensorflow`)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `mnist` | 70 000 × 784 | 10 | Handwritten digits. Auto-subsampled; auto pre-PCA 50-D. |
| `cifar10` | 60 000 × 3072 | 10 | Colour images. Auto-subsampled; auto pre-PCA 50-D. |
| `cifar100` | 60 000 × 3072 | 100 | Fine-grained colour images. Auto-subsampled; auto pre-PCA 50-D. |

Install TensorFlow with: `poetry install --with neural`

#### Custom

```bash
waverider-voxel-viz --dataset load --X-file embeddings.npy --y-file labels.npy
```

`X` must be float `(n, d)`; `y` must be integer `(n,)`. Omit `--y-file` to treat
all points as class 0.

### CLI reference — manifold mode

#### Dataset arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `helix` | Dataset name (see tables above). |
| `--X-file` | `X.npy` | Embedding array path when `--dataset load`. |
| `--y-file` | *(none)* | Label array path when `--dataset load`. |
| `--n-points` | `800` | Maximum points; large datasets are stratified-subsampled. |
| `--seed` | `42` | RNG seed for subsampling and synthetic datasets. |

#### ManifoldModel arguments

| Argument | Default | Description |
|---|---|---|
| `--k-graph` | `10` | KNN graph degree. |
| `--k-pca` | `20` | Neighbours for local PCA at each node. |
| `--k-vote` | `7` | Neighbours for classification voting. |
| `--tau` | `0.90` | Variance threshold for intrinsic dimensionality selection. |
| `--pre-pca` | `0` | Pre-reduce to this many dims before ManifoldModel (0 = off). Auto-set to 50 for `mnist`/`cifar10`/`cifar100`. |
| `--pca-components` | `1,2,3` | Which 3 principal components to visualize (1-based). Use `4,5,6` to explore deeper subspaces. |

#### Voxelization arguments

| Argument | Default | Description |
|---|---|---|
| `--resolution` | `32` | Voxels per axis (total grid = N³). 48–64 for finer detail. |
| `--padding` | `0.05` | Fractional bounding-box padding on each side. |

#### Rendering arguments

| Argument | Default | Description |
|---|---|---|
| `--scalar` | `density` | Field to display in single-panel mode. |
| `--multi-scalar` | off | Show all four fields in a 2×2 panel. |
| `--no-points` | off | Suppress the scatter overlay of raw training points. |
| `--volume` | off | Render the full voxel cloud behind the slice planes. |
| `--vol-opacity` | `0.12` | Voxel cloud opacity (0–1). |
| `--vol-threshold` | `0.04` | Density threshold as fraction of max — filters near-empty voxels. |
| `--off-screen` | off | Render headless (PNG export). |
| `--out` | *(none)* | Output PNG path; implies `--off-screen`. |
| `--zoom` | `1.0` | Camera zoom after safe-area framing for `--hld` output. |

### CLI examples — manifold mode

```bash
# Synthetic helix — default, interactive viewer
waverider-voxel-viz

# Iris — 3 classes, instant
waverider-voxel-viz --dataset iris

# Iris — all four fields in a 2×2 panel
waverider-voxel-viz --dataset iris --multi-scalar

# Iris — slices + semi-transparent voxel cloud ghost
waverider-voxel-viz --dataset iris --volume --vol-opacity 0.08

# sklearn Digits — curvature field
waverider-voxel-viz --dataset digits --scalar curvature

# MNIST — 1 500 pts, pre-reduce to 50-D
waverider-voxel-viz --dataset mnist --n-points 1500 --pre-pca 50

# CIFAR-10 — 1 000 pts, pre-reduce to 40-D
waverider-voxel-viz --dataset cifar10 --n-points 1000 --pre-pca 40

# Headless PNG export
waverider-voxel-viz --dataset iris --multi-scalar --out iris_voxels.png

# Looking Glass quilt — Portrait device, cast to display
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast

# HLD turntable video (10-second, 3840×2160 HEVC)
waverider-voxel-viz --dataset iris --hld --out iris_hld
```

### Programmatic API — manifold mode

All pipeline stages are importable and composable:

```python
from waverider.voxel_viz import (
    fit_and_observe,   # fit ManifoldModel + ManifoldObserver, project to 3-D
    voxelize,          # rasterize PointField → voxel dict
    build_grid,        # voxel dict → pv.ImageData
    render_single,     # single-scalar interactive viewer / PNG
    render_multi,      # 2×2 panel viewer / PNG
    render_hld_single, # HLD turntable video — manifold scene
    PointField,        # NamedTuple: X3, density_w, curvature, height, intrinsic_dim, labels
    PCAInfo,           # NamedTuple: explained_variance_ratio, total_explained, ambient_dim, components
    CMAP_MAP,          # default colour maps per scalar field
)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

bunch = load_iris()
X = StandardScaler().fit_transform(bunch.data).astype("d")
y = bunch.target.astype(int)

subject, observer, pf, pca_info = fit_and_observe(
    X, y, k_graph=10, k_pca=20, k_vote=7, tau=0.90
)
vox = voxelize(pf, resolution=32)
grid = build_grid(vox)

# Interactive viewer
render_single(grid, pf, scalar="density", pca_info=pca_info)

# Headless PNG
render_single(grid, pf, scalar="density", off_screen=True,
              out_path="iris_density.png", pca_info=pca_info)

# HLD video
render_hld_single(grid, pf, scalar="density",
                  out_path="iris_hld", pca_info=pca_info)
```

`fit_and_observe`, `voxelize`, and `build_grid` have no PyVista dependency and
can be used for analysis without a display. Only the `render_*` functions require
PyVista.

---

## CT / MRI Demo Mode

### Concept

`--ct-demo` bypasses the ManifoldModel pipeline entirely. It loads one of
PyVista's built-in biomedical volumes directly from the PyVista example library
(auto-cached on first use), extracts layered isosurfaces for skin, soft tissue,
and bone, and renders the result interactively or as an HLD master video.

The volumes are real clinical data — CT scans and T1 MRI — making this mode
useful for:

- Demonstrating the HLD display with immediately recognizable content
- Validating the holographic rendering pipeline before producing manifold videos
- Standalone biomedical visualization without any machine learning overhead

### Datasets

All datasets are downloaded and cached automatically by PyVista on first use.

| Key | Dims | Modality | Scalar | Isosurface layers |
|---|---|---|---|---|
| `brain` | 181 × 217 × 181 | T1 MRI (1 mm isotropic) | `image_data` | gray matter / white matter / bright structures |
| `full_head` | 256 × 256 × 94 | CT head (12-bit) | `MetaImage` | skin / bone |
| `head_2` | 256 × 256 × 94 | CT head alternate (12-bit) | `Scalars_` | skin / bone |
| `whole_body_ct_male` | 160 × 160 × 273 | Whole-body CT male (HU) | `NIFTI` | skin / muscle / bone |
| `whole_body_ct_female` | 160 × 160 × 271 | Whole-body CT female (HU) | `NIFTI` | skin / muscle / bone |

### Isosurface presets

Each dataset ships with per-layer isovalue thresholds, colours, and opacities
tuned to its intensity distribution (see `CT_PRESETS` in the source). All colours
avoid pure white so they remain visible on the HLD (white = transparent on-device).

| Dataset | Layer | Isovalue | Colour | Opacity |
|---|---|---|---|---|
| `brain` | gray matter | 40 | `#e8c4a0` (peach) | 0.25 |
| `brain` | white matter | 100 | `#e07030` (amber) | 0.60 |
| `brain` | bright structures | 180 | `#c02020` (red) | 0.90 |
| `full_head` / `head_2` | skin | 400 | `#f4c8a0` (skin) | 0.20 |
| `full_head` / `head_2` | bone | 1500 | `#f0e8c0` (ivory) | 0.85 |
| `whole_body_ct_*` | skin | −100 | `#f0c080` (peach) | 0.15 |
| `whole_body_ct_*` | muscle | 300 | `#d08040` (tan) | 0.45 |
| `whole_body_ct_*` | bone | 700 | `#f0e8c0` (ivory) | 0.85 |

Override any preset with `--ct-isovalues V1,V2,...`.

### CLI reference — CT / MRI demo mode

| Argument | Default | Description |
|---|---|---|
| `--ct-demo` | off | Enable CT/MRI demo mode (skips ManifoldModel). |
| `--ct-dataset` | `brain` | Dataset key (see table above). |
| `--ct-isovalues` | *(preset)* | Comma-separated isovalue overrides, e.g. `400,1500`. |
| `--no-shadow` | off | Suppress the contact shadow disc under the volume. |
| `--hld` | off | Render HLD turntable video instead of opening the viewer. |
| `--frames` | `300` | Video frame count (with `--hld`). |
| `--fps` | `30` | Video frame rate — 30 or 60 per HLD spec (with `--hld`). |
| `--orbit` | `360` | Total turntable rotation in degrees. |
| `--zoom` | `1.8` (CT), `1.0` (manifold) | Camera zoom applied after safe-area framing. Values > 1 enlarge the subject in the 16:9 frame; useful for portrait-shaped subjects (brains, bodies). |
| `--still` | off | With `--hld`, export a single 3840×2160 PNG instead of a video. No ffmpeg required. |
| `--out` | *(auto)* | Output stem for PNG or video; suffix is appended automatically. |

### CLI examples — CT / MRI demo mode

```bash
# Interactive viewer — T1 MRI brain (default)
waverider-voxel-viz --ct-demo

# Interactive viewer — CT head
waverider-voxel-viz --ct-demo --ct-dataset full_head

# Interactive viewer — whole-body CT male
waverider-voxel-viz --ct-demo --ct-dataset whole_body_ct_male

# Headless PNG screenshot
waverider-voxel-viz --ct-demo --ct-dataset brain --out brain_preview.png

# HLD turntable video — brain MRI, 10-second loop
waverider-voxel-viz --ct-demo --ct-dataset brain --hld --out brain_hld

# HLD video — CT head with custom isovalue thresholds, 20-second loop
waverider-voxel-viz --ct-demo --ct-dataset full_head \
    --ct-isovalues 300,1200 --frames 600 --fps 30 --out head_hld

# HLD video — whole-body CT, no floor shadow
waverider-voxel-viz --ct-demo --ct-dataset whole_body_ct_male \
    --hld --no-shadow --out body_hld
```

### Programmatic API — CT / MRI demo mode

```python
from waverider.voxel_viz import CT_PRESETS, render_ct_viewer, render_ct_hld

# List available datasets and their presets
for name, preset in CT_PRESETS.items():
    print(name, preset["description"], preset["isovalues"])

# Interactive viewer — dark background, contact shadow
render_ct_viewer(ct_dataset="brain")

# Headless PNG
render_ct_viewer(ct_dataset="full_head", off_screen=True, out_path="head.png")

# HLD turntable video
render_ct_hld(
    ct_dataset="brain",
    out_path="brain_hld",     # _hld.mp4 is appended
    n_frames=300,
    fps=30,
    orbit=360.0,
    shadow=True,
)

# Override isovalues
render_ct_hld(
    ct_dataset="full_head",
    out_path="head_custom",
    isovalues=[300, 1200],
)
```

---

## Looking Glass Holographic Output

Both modes support the full Looking Glass output pipeline.

### Light-field quilts (LFD line)

Generates a multi-view quilt PNG that Looking Glass Studio / Bridge auto-detects:

```bash
# Manifold mode — Portrait device
waverider-voxel-viz --dataset iris --quilt portrait --out iris

# Manifold mode — cast to connected device after render
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast

# Quilt video (looping turntable MP4)
waverider-voxel-viz --dataset iris --quilt portrait --quilt-video --out iris_loop
```

Available devices: `portrait`, `go`, `16-landscape`, and others (see
`waverider.lfd.QUILT_PRESETS`, or the table in
[lfd.md](lfd.md)).

Framing matters more here than for a flat render: perceived depth scales
with how much of each view the subject fills, so the default `--quilt-zoom`
of 1.6 dollies the camera in until the volume fills the frame. The colour
scale bar is off by default for quilts — a 2-D overlay is pinned to the
focal plane and reads as a flat pane cutting through the hologram.

```bash
# Gen3 16" landscape, cast to the connected device
waverider-voxel-viz --dataset iris --quilt 16-landscape --out iris --cast

# Keep the scale bar; use PyVista's own framing instead of the zoom default
waverider-voxel-viz --dataset iris --quilt 16-landscape --out iris \
    --quilt-zoom 1.0 --quilt-scalar-bar
```

### Hololuminescent Display (HLD)

HLDs (16"/27"/86" HLD line) play ordinary 2-D video — no quilts. White pixels
are invisible on-device; subjects must sit on a white background. The HLD output
pipeline renders a turntable orbit to the official master spec:

- **Resolution**: 3840 × 2160 (16:9 landscape, as required by HLD Author)
- **Codec**: HEVC (H.265), CRF 18, yuv420p, bt709
- **Frame rate**: 30 or 60 fps (per HLD spec)

After rendering, open the master in Looking Glass's free **HLD Author** app to
encode for the device's built-in USB player. For HDMI or signage delivery, use
the master as-is.

```bash
# Manifold mode — HLD video
waverider-voxel-viz --dataset iris --hld --out iris_hld

# CT demo mode — HLD video
waverider-voxel-viz --ct-demo --ct-dataset brain --hld --out brain_hld

# 60 fps version
waverider-voxel-viz --ct-demo --ct-dataset brain \
    --hld --fps 60 --frames 600 --out brain_60fps
```

---

## Interactive Controls (PyVista)

Applies to both modes when no `--hld`, `--quilt`, or `--out` flag is given.

| Action | Control |
|---|---|
| Rotate view | Left-click + drag |
| Zoom | Scroll wheel or right-click + drag |
| Pan | Middle-click + drag |
| Move a slice plane | Left-click + drag the plane's handle |
| Reset camera | `r` |
| Screenshot | `s` (saves `screenshot.png` in CWD) |
| Quit | `q` or close window |

---

## Design Notes

### Shadow disc on light vs dark backgrounds

The contact shadow under the volume is a flat disc with a radial falloff painted
using `cmap="Greys"` (light bg) or `cmap="Greys_r"` (dark bg). On the HLD's
white background the rim fades to white and disappears; on the viewer's black
background the rim fades to black and disappears. The centre is a muted grey in
both cases. Suppress with `--no-shadow`.

### Why PCA for the 3-D projection?

PCA maximizes variance in three dimensions, so the first three components roughly
follow the manifold's principal curvature directions. The explained-variance
percentage is printed at runtime. Use `--pca-components 4,5,6` to explore deeper
subspaces that the top three components miss.

### Voxel cloud + slices

`--volume` renders a semi-transparent voxel cloud behind the slice planes (manifold
mode only). The cloud gives the global silhouette; slices probe the interior.
`--vol-opacity 0.08–0.20` and `--vol-threshold 0.02–0.08` are the useful ranges.

### Pre-PCA for high-dimensional data

`ManifoldModel` builds a KNN graph with O(n² · d) distance computations. Pre-
reducing to 40–50 dimensions with global PCA (≥90% variance retained in practice)
speeds graph construction by 15–75× without meaningfully changing the discovered
geometry. The `--pre-pca` step feeds ManifoldModel; the final 3-D visualization
PCA is separate.

### Voxel occupancy

The script prints occupied-voxel percentage at runtime. For a 32³ grid with 800
points expect 5–20% occupancy. If occupancy is very low, reduce `--resolution`
or increase `--n-points`; if above 50%, increase `--resolution` for finer detail.

---

## Related Components

| Component | Location |
|---|---|
| `ManifoldModel` | `src/waverider/manifold_model.py` |
| `ManifoldObserver` | `src/waverider/manifold_observer.py` |
| `TurtleND` | `src/waverider/turtleND.py` |
| `lfd.py` | `src/waverider/lfd.py` |
| `hld.py` | `src/waverider/hld.py` |
| voxel_viz source | `src/waverider/voxel_viz.py` |
| Looking Glass reference | [docs/waverider/lfd.md](lfd.md) |
| HLD reference | [docs/waverider/hld.md](hld.md) |
| Helix benchmark | `benchmarks/canonical_tests/helix_manifold_observer.py` |
| Dimensionality probe | `benchmarks/canonical_tests/manifold_dim_probe.py` |
