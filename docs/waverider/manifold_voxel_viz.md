# Manifold Voxel Visualizer

**Module**: `waverider.voxel_viz`
**CLI command**: `waverider-voxel-viz`
**Source**: `src/waverider/voxel_viz.py`

> *"Since we can move freely in embedding spaces and perform arbitrary subspace
> projections — why not materialize those projections as voxels and slice them?"*

---

## Concept

High-dimensional manifolds are invisible. WaveRider's `ManifoldObserver` can
*see* them — it computes curvature, height above the tangent plane, and local
intrinsic dimensionality at every training node. But these geometric fields live
in N-dimensional space; there is no natural way to look at them.

The voxel visualizer solves this by:

1. **Projecting** the manifold into a 3-D PCA subspace
2. **Rasterizing** the observer's geometric fields onto a uniform voxel grid
3. **Rendering** the grid in PyVista — a VTK-backed interactive viewer where
   you drag orthogonal slice planes through the volume in real time

The result is a **live cross-sectional anatomy of the manifold**: you can slice
through density concentrations, follow curvature ridges, and see where class
boundaries cut through the embedding space.

---

## Pipeline

```
Data (X, y)
    │
    ├─ [optional] Pre-PCA: d → pre_pca dims     (for MNIST / CIFAR-10)
    │
    ▼
ManifoldModel.fit(X, y)
    │  builds KNN graph, local PCA at each node
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

---

## Scalar Fields

Every voxel in the grid carries five scalar fields. Each is the **mean of all
training points** that fall into that voxel (majority vote for `class_vote`).

| Field | Colour map | What it encodes |
|---|---|---|
| `density` | `plasma` | How many training points land here (Gaussian-smoothed). Shows manifold concentration and thin/thick regions. |
| `curvature` | `coolwarm` | Mean ManifoldObserver curvature — rate at which the tangent plane rotates between adjacent nodes. High curvature = bent, folded, or boundary regions. |
| `height` | `viridis` | Mean reconstruction error: how far each node sits above its own tangent plane. Low height = the manifold is locally flat. |
| `intrinsic_dim` | `tab10` | Local d* — how many PCA components are needed to explain τ of local variance. Reveals where the manifold is higher- or lower-dimensional. |
| `class_vote` | `Set1` | Majority class label in each voxel. A direct cross-section of the decision surface. |

---

## Supported Datasets

### Synthetic (no extra dependencies)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `helix` | n × 5 | 2 | 1-manifold helix in 3-D, embedded in 5-D with Gaussian noise. Default. |
| `swiss_roll` | n × 3 | 2 | Classic 2-manifold Swiss roll. |
| `torus` | n × 4 | 4 | Flat torus (R=2, r=0.6) embedded in 4-D, four quadrant labels. |

### Real — sklearn (always available)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `iris` | 150 × 4 | 3 | Fisher's Iris — setosa / versicolor / virginica. Fast; good for first run. |
| `wine` | 178 × 13 | 3 | UCI Wine recognition — 3 cultivar classes. |
| `breast_cancer` | 569 × 30 | 2 | Wisconsin breast cancer — malignant / benign. |
| `digits` | 1797 × 64 | 10 | sklearn 8×8 handwritten digits. Rich enough to see class structure. |

All sklearn datasets are StandardScaler-normalised before fitting.

### Real — large (needs `tensorflow`)

| Name | Shape | Classes | Notes |
|---|---|---|---|
| `mnist` | 70 000 × 784 | 10 | Handwritten digits. Auto-subsampled; auto pre-PCA 50-D. |
| `cifar10` | 60 000 × 3072 | 10 | Colour images. Auto-subsampled; auto pre-PCA 50-D. |
| `cifar100` | 60 000 × 3072 | 100 | Fine-grained colour images. Auto-subsampled; auto pre-PCA 50-D. |

Install TensorFlow with: `poetry install --with neural`

### Custom

```bash
waverider-voxel-viz --dataset load --X-file embeddings.npy --y-file labels.npy
```

X must be a float array of shape `(n, d)`; y must be an integer array of shape
`(n,)`. If `--y-file` is omitted, all labels are treated as class 0.

---

## Installation

`pyvista` and `scipy` are in the `viz` dependency group:

```bash
poetry install --with viz
```

For benchmarks (matplotlib, openpyxl, etc.) additionally:

```bash
poetry install --with benchmarks,viz
```

For MNIST / CIFAR-10 / CIFAR-100 additionally:

```bash
poetry install --with viz,neural
```

---

## Usage

### Quick start

```bash
# Synthetic helix — opens interactive PyVista window
waverider-voxel-viz

# Iris — 3 classes, 4-D, instant
waverider-voxel-viz --dataset iris

# Iris — slices + full voxel cloud ghost behind them
waverider-voxel-viz --dataset iris --volume

# Tune cloud density: more transparent, lower threshold (show sparse regions too)
waverider-voxel-viz --dataset iris --volume \
    --vol-opacity 0.08 --vol-threshold 0.02

# Iris — 2×2 panel showing all four fields at once
waverider-voxel-viz --dataset iris --multi-scalar

# sklearn Digits — curvature field, 500 pts
waverider-voxel-viz \
    --dataset digits --scalar curvature --n-points 500

# MNIST — 1 500 points, pre-reduce to 50-D before ManifoldModel
waverider-voxel-viz \
    --dataset mnist --n-points 1500 --pre-pca 50

# CIFAR-10 — 1 000 points, pre-reduce to 40-D
waverider-voxel-viz \
    --dataset cifar10 --n-points 1000 --pre-pca 40

# Headless PNG export — no window
waverider-voxel-viz \
    --dataset iris --multi-scalar --out iris_voxels.png
```

### Full argument reference

#### Dataset

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `helix` | Dataset name (see table above). |
| `--X-file` | `X.npy` | Path to embedding array when `--dataset load`. |
| `--y-file` | *(none)* | Path to label array when `--dataset load`. |
| `--n-points` | `800` | Maximum points; large datasets are stratified-subsampled. |
| `--seed` | `42` | RNG seed for subsampling and synthetic datasets. |

#### ManifoldModel

| Argument | Default | Description |
|---|---|---|
| `--k-graph` | `10` | KNN graph degree. |
| `--k-pca` | `20` | Neighbours used for local PCA at each node. |
| `--k-vote` | `7` | Neighbours used for classification voting. |
| `--tau` | `0.90` | Variance threshold for intrinsic dimensionality. |
| `--pre-pca` | `0` | Pre-reduce to this many dims before ManifoldModel (0 = off). Auto-set to 50 for `mnist`/`cifar10`. |

#### Voxelization

| Argument | Default | Description |
|---|---|---|
| `--resolution` | `32` | Voxels per axis (total grid = N³). Increase for finer detail; 48–64 is a good maximum. |
| `--padding` | `0.05` | Fractional bounding-box padding on each side. |

#### Rendering

| Argument | Default | Description |
|---|---|---|
| `--scalar` | `density` | Field to display in single-panel mode. |
| `--multi-scalar` | *(off)* | Show all four fields in a 2×2 panel layout. |
| `--no-points` | *(off)* | Suppress the scatter overlay of raw training points. |
| `--volume` | *(off)* | Render the full voxel cloud behind the slice planes. |
| `--vol-opacity` | `0.12` | Opacity of the voxel cloud (0 = invisible, 1 = solid). |
| `--vol-threshold` | `0.04` | Density threshold as fraction of max — filters near-empty voxels from the cloud. |
| `--off-screen` | *(off)* | Render headless (implies `--out`). |
| `--out` | *(none)* | Output PNG path; implies `--off-screen`. |

---

## Interactive Controls (PyVista)

When the viewer opens you get three orthogonal slice planes — XY, XZ, and YZ —
each with a draggable handle.

| Action | How |
|---|---|
| Rotate view | Left-click + drag |
| Zoom | Scroll wheel or right-click + drag |
| Pan | Middle-click + drag |
| Move a slice plane | Left-click + drag the plane's coloured handle |
| Reset camera | Press `r` |
| Screenshot | Press `s` (saves `screenshot.png` in CWD) |
| Quit | Press `q` or close the window |

---

## Design Notes

### Why PCA for the 3-D projection?

PCA captures maximum variance in three dimensions. For manifold data this means
the first three components roughly follow the manifold's principal curvature
directions — which is exactly what you want to see in a voxel slice. The
explained-variance percentage is printed at runtime so you know how much of the
structure survives the projection.

### Voxel cloud + slices together

`--volume` adds a semi-transparent voxel cloud (`grid.threshold()` on the
density field) rendered at low opacity *behind* the slice planes. The cloud
gives you the global silhouette of the manifold — where it lives, how it curves,
whether it is one blob or several — while the slice planes let you probe the
interior. `--vol-opacity` controls transparency (0.08–0.20 is a useful range)
and `--vol-threshold` filters out near-empty voxels so the cloud shows structure
rather than noise.

### Why voxels instead of point clouds?

Point clouds are sparse and hard to slice. A voxel grid lets PyVista interpolate
smoothly between points and gives you a continuous volume you can cut at any
plane. The Gaussian-smoothed density field in particular reads like a topographic
map of the manifold — thick, populated regions glow brightly; thin or empty
regions are dark.

### Pre-PCA for high-dimensional data

`ManifoldModel` builds a KNN graph with O(n² · d) distance computations. At
d = 784 (MNIST) with n = 2 000 this is ~3 billion operations — slow but
feasible. At d = 3072 (CIFAR-10) with n = 2 000 it tips into ~12 billion. Pre-
reducing to 40–50 dimensions with global PCA (which captures ≥ 90% of variance
in practice) speeds up graph construction by 15–75× without meaningfully
changing the manifold geometry that ManifoldModel discovers.

The `--pre-pca` step is separate from the final 3-D visualization PCA — the
former feeds ManifoldModel; the latter projects the already-fitted manifold into
the voxel grid.

### Voxel occupancy

The script prints occupied-voxel percentage at runtime. For a 32³ grid with
800 points expect 5–20% occupancy — the manifold is a low-dimensional surface
inside a 3-D box. If occupancy is very low, reduce `--resolution` or increase
`--n-points`. If it is very high (> 50%), increase `--resolution` for finer
detail.

---

## Example Outputs

### Iris — density field

The three Iris classes form three distinct density blobs in PCA space, separated
most cleanly along PC1. The `class_vote` slice shows a clean 3-region partition;
the `curvature` slice shows elevated curvature at the versicolor/virginica
boundary where the two manifolds approach each other.

### Breast Cancer — density field

The Wisconsin breast cancer dataset (569 × 30) separates into two sharply
defined density lobes in PCA space — malignant and benign tumours occupy
distinct regions with a narrow, high-curvature boundary between them. The
`density` slice reveals that benign cases cluster tightly into a single
high-density peak while malignant cases spread across a broader, lower-density
region, reflecting their greater morphological heterogeneity. The `class_vote`
cross-section exposes the decision boundary as a thin curved sheet cutting
between the two lobes — a compelling illustration of how geometric separability
maps directly onto clinical outcome.

### MNIST — class_vote cross-section

At 1 500 points and 50-D pre-PCA, the 10 digit classes organize into overlapping
clusters. Cross-sectioning through the `class_vote` volume reveals which digits
share embedding territory and where the manifold decision boundaries lie.

---

## Programmatic API

All pipeline stages are importable and composable without the CLI:

```python
from waverider import (
    fit_and_observe,   # fit ManifoldModel + ManifoldObserver, project to 3-D
    voxelize,          # rasterize PointField → voxel dict
    build_grid,        # voxel dict → pv.ImageData
    render_single,     # single-scalar interactive viewer
    render_multi,      # 2×2 panel viewer
    PointField,        # NamedTuple: X3, density_w, curvature, height, intrinsic_dim, labels
    PCAInfo,           # NamedTuple: explained_variance_ratio, total_explained, ambient_dim, components
    CMAP_MAP,          # default colour maps per scalar field
)

# Example: Iris, headless PNG export
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
render_single(grid, pf, scalar="density", off_screen=True,
              out_path="iris_density.png", pca_info=pca_info)
```

`fit_and_observe`, `voxelize`, and `build_grid` have no PyVista dependency —
they can be used for analysis without a display.  Only `render_single` and
`render_multi` require PyVista (and will raise `ImportError` with an install
hint if it is missing).

---

## Related Components

| Component | Where |
|---|---|
| `ManifoldModel` | `src/waverider/manifold_model.py` |
| `ManifoldObserver` | `src/waverider/manifold_observer.py` |
| `TurtleND` | `src/waverider/turtleND.py` |
| `voxel_viz` module | `src/waverider/voxel_viz.py` |
| Helix benchmark | `benchmarks/canonical_tests/helix_manifold_observer.py` |
| Dimensionality probe | `benchmarks/canonical_tests/manifold_dim_probe.py` |
