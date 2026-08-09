"""
Manifold Voxel Visualizer
=========================

Two rendering modes share this module:

**Manifold mode** — the primary pipeline.  Projects a fitted
:class:`~waverider.ManifoldModel` + :class:`~waverider.ManifoldObserver` into
a 3-D PCA subspace, rasterizes the observer's geometric fields (density,
curvature, height, intrinsic dimensionality, class vote) into a uniform voxel
grid, and opens an interactive PyVista viewer where you can slice, clip, and
inspect the manifold from any angle.

**CT / MRI demo mode** (``--ct-demo``) — bypasses the ManifoldModel entirely.
Loads one of PyVista's built-in biomedical volumes, extracts layered
isosurfaces (skin / tissue / bone), and opens an interactive viewer, renders
a Looking Glass quilt (``--quilt``, LFD devices), or renders a turntable
video to the Looking Glass Hololuminescent Display (``--hld``, HLD master
spec).

**Optional dependencies** — install the ``viz`` extras group::

    poetry install --with viz   # pyvista, scipy, pillow

Manifold pipeline
-----------------
1. Load or generate embeddings + integer labels.
2. Optional: pre-reduce with PCA to ``pre_pca`` dims before
   :class:`~waverider.ManifoldModel` (recommended for MNIST / CIFAR).
3. Discover intrinsic dimensionality via local PCA (thin SVD) at multiple
   variance thresholds, with per-class analysis.
4. Fit :class:`~waverider.ManifoldModel`, wrap in
   :class:`~waverider.ManifoldObserver`, call ``observe()`` to populate the
   geometric field.
5. Project training points to 3-D via PCA, annotating axes with
   explained-variance ratios and rendering direction arrows.
6. Rasterize each scalar field (density / curvature / height / d* / class)
   onto a uniform ``pv.ImageData`` voxel grid via ``numpy.bincount``
   accumulation and optional Gaussian smoothing.
7. Launch a PyVista plotter with an orthogonal-slice widget and optional
   scatter overlay of the raw training points.

CT / MRI demo pipeline
----------------------
1. Download a PyVista built-in biomedical ``ImageData`` volume (auto-cached).
2. Extract per-preset isosurfaces with ``contour()`` — typically 2–3 layers
   for skin / soft tissue / bone.
3. Laplacian-smooth each surface to reduce marching-cubes faceting.
4. Compose scene in a PyVista plotter: black background for the interactive
   viewer, unmodified background for LFD quilts, white background for the
   HLD video (white = invisible on device).
5. Open the interactive viewer (``render_ct_viewer``), sweep the camera into
   a Looking Glass quilt still/video (``render_ct_quilt``), or encode a
   10-second HEVC turntable orbit to the HLD master spec (``render_ct_hld``).

Manifold datasets
-----------------
Synthetic:
    helix         1-manifold in 3-D embedded in 5-D (default)
    swiss_roll    2-manifold in 3-D
    torus         2-manifold in 4-D

Real (sklearn, always available):
    iris          150 × 4,  3 classes (flowers)
    wine          178 × 13, 3 classes (wines)
    breast_cancer 569 × 30, 2 classes (tumour)
    digits        1797 × 64, 10 classes (8×8 handwritten digits)

Real (large, needs keras/tensorflow):
    mnist         70 000 × 784, 10 classes — subsampled + pre-PCA
    cifar10       60 000 × 3072, 10 classes — subsampled + pre-PCA
    cifar100      60 000 × 3072, 100 classes — subsampled + pre-PCA

Custom:
    load          X_file=<X.npy>  [y_file=<y.npy>]

CT / MRI datasets (PyVista built-in, auto-downloaded on first use)
-----------------------------------------------------------------
brain               181 × 217 × 181, T1 MRI (1 mm isotropic)
full_head           256 × 256 × 94,  CT head (12-bit)
head_2              256 × 256 × 94,  CT head alternate (12-bit)
whole_body_ct_male  160 × 160 × 273, CT full body male (Hounsfield units)
whole_body_ct_female 160 × 160 × 271, CT full body female (Hounsfield units)

TVB brain datasets (The Virtual Brain, auto-downloaded on first use)
--------------------------------------------------------------------
cortex              16 384-vertex cortex, coloured by 76-region parcellation
cortex_80k          81 924-vertex cortex, coloured by 80-region parcellation
cortex_hires        283 380-vertex two-hemisphere cortex (decimated)
connectome          76-region structural connectome in a translucent cortex
connectome_998      998-region connectome, top 1% of tracts
head_layers         nested shells — cortex, inner/outer skull, scalp
macaque             147 460-vertex macaque cortex, 84-region parcellation
macaque_connectome  84-region macaque connectome in a translucent cortex

CLI usage — manifold mode
-------------------------
::

    # Synthetic helix (default, interactive viewer)
    waverider-voxel-viz

    # Real: Iris dataset
    waverider-voxel-viz --dataset iris

    # Real: sklearn Digits, curvature field, 2×2 panel
    waverider-voxel-viz --dataset digits --multi-scalar

    # Real: MNIST — subsample 1 500 pts, pre-reduce to 50 D
    waverider-voxel-viz --dataset mnist --n-points 1500 --pre-pca 50

    # Real: CIFAR-10 — subsample 1 000 pts, pre-reduce to 40 D
    waverider-voxel-viz --dataset cifar10 --n-points 1000 --pre-pca 40

    # Headless PNG export
    waverider-voxel-viz --dataset iris --off-screen --out iris_voxels.png

    # Looking Glass holographic quilt (Portrait device), cast to the display
    waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast

    # HLD turntable video (10-second loop)
    waverider-voxel-viz --dataset iris --hld --out iris_hld

CLI usage — CT / MRI demo mode
-------------------------------
::

    # Interactive viewer — T1 MRI brain (default)
    waverider-voxel-viz --ct-demo

    # Interactive viewer — CT head
    waverider-voxel-viz --ct-demo --ct-dataset full_head

    # Interactive viewer — whole-body CT male
    waverider-voxel-viz --ct-demo --ct-dataset whole_body_ct_male

    # Headless PNG screenshot
    waverider-voxel-viz --ct-demo --ct-dataset brain --out brain_preview.png

    # HLD turntable video (10-second, 30 fps)
    waverider-voxel-viz --ct-demo --ct-dataset brain --hld --out brain

    # HLD video with custom isovalue thresholds and longer clip
    waverider-voxel-viz --ct-demo --ct-dataset full_head \\
        --ct-isovalues 300,1200 --frames 600 --fps 30 --out head_hld

    # Looking Glass quilt still (Portrait device), cast to the display
    waverider-voxel-viz --ct-demo --ct-dataset brain \\
        --quilt portrait --out brain --cast

    # Looking Glass quilt turntable video (Go device)
    waverider-voxel-viz --ct-demo --ct-dataset full_head \\
        --quilt go --quilt-video --out head

CLI usage — TVB brain demo
--------------------------
Real brain geometry from `The Virtual Brain <https://www.thevirtualbrain.org/>`_.
The ~337 MB ``tvb-data`` archive is downloaded from Zenodo on first use and
cached; see :mod:`waverider.tvb_data`.
::

    # Interactive viewer — cortex coloured by 76-region parcellation
    waverider-voxel-viz --tvb-demo

    # Interactive viewer — structural connectome in a translucent cortex
    waverider-voxel-viz --tvb-demo --tvb-dataset connectome

    # Denser connectome (more tracts survive the weight threshold)
    waverider-voxel-viz --tvb-demo --tvb-dataset connectome --tvb-percentile 80

    # Headless PNG screenshot
    waverider-voxel-viz --tvb-demo --tvb-dataset head_layers --out head.png

    # HLD turntable video (10-second loop)
    waverider-voxel-viz --tvb-demo --tvb-dataset cortex --hld --out cortex

    # Looking Glass quilt still (Portrait device), cast to the display
    waverider-voxel-viz --tvb-demo --tvb-dataset connectome \\
        --quilt portrait --out connectome --cast

    # High-resolution cortex, decimated hard for a 48-view sweep
    waverider-voxel-viz --tvb-demo --tvb-dataset cortex_hires \\
        --tvb-decimate 0.7 --quilt portrait --out cortex_hires

    # Macaque cortex
    waverider-voxel-viz --tvb-demo --tvb-dataset macaque --hld --out macaque

    # Drop the cached archive
    waverider-voxel-viz --tvb-clear-cache

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""
# pylint: disable=import-outside-toplevel  # keras is optional/heavy; lazy-loaded only when needed

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from quiltwright.lfd import (
    QUILT_PRESETS,
    cast_quilt,
    render_quilt,
    render_quilt_video,
    save_quilt,
)
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_swiss_roll,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from waverider.dimensionality_discovery import (
    discover_dimensionality,
    discover_per_class_dimensionality,
)
from waverider.manifold_model import ManifoldModel
from waverider.manifold_observer import ManifoldObserver

try:
    import pyvista as pv

    _PYVISTA_AVAILABLE = True
except ImportError:
    _PYVISTA_AVAILABLE = False

try:
    import scipy.ndimage

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _require_viz(fn_name: str) -> None:
    """Raise a clear ImportError if the viz extras are not installed."""
    missing = []
    if not _PYVISTA_AVAILABLE:
        missing.append("pyvista")
    if not _SCIPY_AVAILABLE:
        missing.append("scipy")
    if missing:
        raise ImportError(
            f"{fn_name}() requires the 'viz' extras: {', '.join(missing)}.\n"
            "Install with:  poetry install --with viz"
        )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_REAL_DATASETS = {"iris", "wine", "breast_cancer", "digits", "mnist", "cifar10", "cifar100"}


def _make_helix(n: int = 600, noise: float = 0.02, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """1-manifold helix in 3-D, embedded into 5-D with Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0 * np.pi, n)
    X = np.column_stack(
        [
            np.cos(t),
            np.sin(t),
            t / 6.0,
            rng.normal(0.0, noise, n),
            rng.normal(0.0, noise, n),
        ]
    ).astype("d")
    y = (t >= 3.0 * np.pi).astype(int)
    return X, y


def _make_swiss_roll(
    n: int = 1500, noise: float = 0.1, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """2-manifold Swiss roll in 3-D."""
    X, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    y = (t > t.mean()).astype(int)
    return X.astype("d"), y


def _make_torus(
    n: int = 2000, noise: float = 0.05, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """2-manifold flat torus (R=2, r=0.6) embedded in 4-D."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    R, r = 2.0, 0.6
    X = np.column_stack(
        [
            (R + r * np.cos(phi)) * np.cos(theta),
            (R + r * np.cos(phi)) * np.sin(theta),
            r * np.sin(phi),
            r * np.sin(phi + theta),
        ]
    ).astype("d")
    X += rng.normal(0, noise, X.shape)
    y = (theta > np.pi).astype(int) * 2 + (phi > np.pi).astype(int)
    return X, y


def _subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to at most *n* points."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    classes, _ = np.unique(y, return_counts=True)
    per_class = max(1, n // len(classes))
    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        take = min(per_class, len(c_idx))
        idx.append(rng.choice(c_idx, size=take, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _load_sklearn(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load an sklearn toy dataset, scale, and subsample."""
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    bunch = loaders[name]()
    X = StandardScaler().fit_transform(bunch.data).astype("d")
    y = bunch.target.astype(int)
    return _subsample(X, y, n, seed)


def _load_mnist(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST via keras, flatten, scale, subsample."""
    try:
        from keras.datasets import mnist  # type: ignore
    except ImportError:
        print("ERROR: keras not installed — cannot load MNIST.")
        print("       Install with: pip install keras  or use --dataset digits")
        sys.exit(1)
    (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
    X = np.concatenate([X_tr, X_te], axis=0).reshape(-1, 784).astype("d") / 255.0
    y = np.concatenate([y_tr, y_te]).astype(int)
    return _subsample(X, y, n, seed)


def _load_cifar10(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 via keras, flatten, scale, subsample."""
    try:
        from keras.datasets import cifar10  # type: ignore
    except ImportError:
        print("ERROR: keras not installed — cannot load CIFAR-10.")
        print("       Install with: pip install keras  or use --dataset digits")
        sys.exit(1)
    (X_tr, y_tr), (X_te, y_te) = cifar10.load_data()
    X = np.concatenate([X_tr, X_te], axis=0).reshape(-1, 3072).astype("d") / 255.0
    y = np.concatenate([y_tr, y_te]).ravel().astype(int)
    return _subsample(X, y, n, seed)


def _load_cifar100(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-100 via keras, flatten, scale, subsample."""
    try:
        from keras.datasets import cifar100  # type: ignore
    except ImportError:
        print("ERROR: keras not installed — cannot load CIFAR-100.")
        print("       Install with: pip install keras  or use --dataset digits")
        sys.exit(1)
    (X_tr, y_tr), (X_te, y_te) = cifar100.load_data(label_mode="fine")
    X = np.concatenate([X_tr, X_te], axis=0).reshape(-1, 3072).astype("d") / 255.0
    y = np.concatenate([y_tr, y_te]).ravel().astype(int)
    return _subsample(X, y, n, seed)


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for the dataset specified in *args*.

    :param args: Parsed CLI namespace.  Relevant fields: ``dataset``,
        ``n_points``, ``seed``, ``X_file``, ``y_file``.
    :return: Tuple ``(X, y)`` with shapes ``(n, d)`` and ``(n,)``.
    """
    ds = args.dataset
    n = args.n_points
    seed = args.seed

    if ds == "helix":
        return _make_helix(n=n, seed=seed)
    if ds == "swiss_roll":
        return _make_swiss_roll(n=n, seed=seed)
    if ds == "torus":
        return _make_torus(n=n, seed=seed)
    if ds in {"iris", "wine", "breast_cancer", "digits"}:
        return _load_sklearn(ds, n=n, seed=seed)
    if ds == "mnist":
        return _load_mnist(n=n, seed=seed)
    if ds == "cifar10":
        return _load_cifar10(n=n, seed=seed)
    if ds == "cifar100":
        return _load_cifar100(n=n, seed=seed)
    if ds == "load":
        X = np.load(args.X_file).astype("d")
        y = np.load(args.y_file).astype(int) if args.y_file else np.zeros(len(X), dtype=int)
        return X, y
    raise ValueError(f"Unknown dataset: {ds}")


# ---------------------------------------------------------------------------
# Observer field extraction
# ---------------------------------------------------------------------------


class PointField(NamedTuple):
    """Per-point geometric scalars extracted from a ManifoldObserver field.

    :param X3: PCA-projected coordinates, shape ``(n, 3)``.
    :param density_w: Per-point weight for density accumulation (all ones).
    :param curvature: Scalar curvature at each point, shape ``(n,)``.
    :param height: Height above the tangent plane, shape ``(n,)``.
    :param intrinsic_dim: Local intrinsic dimensionality d*, shape ``(n,)``.
    :param labels: Integer class labels, shape ``(n,)``.
    """

    X3: np.ndarray
    density_w: np.ndarray
    curvature: np.ndarray
    height: np.ndarray
    intrinsic_dim: np.ndarray
    labels: np.ndarray


class PCAInfo(NamedTuple):
    """Metadata from the 3-D PCA projection used for axis annotation.

    :param explained_variance_ratio: Per-component explained variance,
        shape ``(3,)``.
    :param total_explained: Sum of the three ratios.
    :param ambient_dim: Dimensionality before projection.
    :param components: 1-based PC indices visualised, e.g. ``(1, 2, 3)``.
    """

    explained_variance_ratio: np.ndarray
    total_explained: float
    ambient_dim: int
    components: tuple[int, int, int]


def fit_and_observe(
    X: np.ndarray,
    y: np.ndarray,
    k_graph: int,
    k_pca: int,
    k_vote: int,
    tau: float,
    pre_pca: int = 0,
    pca_components: tuple[int, int, int] = (1, 2, 3),
) -> tuple[ManifoldModel, ManifoldObserver, PointField, PCAInfo | None]:
    """Fit :class:`~waverider.ManifoldModel`, run :class:`~waverider.ManifoldObserver`,
    project to 3-D PCA.

    :param X: Training embeddings, shape ``(n, d)``.
    :param y: Integer labels, shape ``(n,)``.
    :param k_graph: KNN graph degree.
    :param k_pca: Neighbours used for local PCA.
    :param k_vote: Neighbours used for classification vote.
    :param tau: Variance threshold for intrinsic dim selection.
    :param pre_pca: If > 0, reduce *X* to this many dims via PCA before
        fitting.  Recommended for high-dimensional data (MNIST, CIFAR).
    :param pca_components: Which 3 principal components to project onto,
        as 1-based indices.  Default ``(1, 2, 3)`` selects the three
        highest-variance axes.  Use e.g. ``(4, 5, 6)`` to explore deeper
        subspaces.
    :return: ``(subject, observer, point_field, pca_info)``
    """
    if pre_pca > 0 and X.shape[1] > pre_pca:
        print(f"  Pre-PCA: {X.shape[1]}D → {pre_pca}D ...", flush=True)
        reducer = PCA(n_components=pre_pca, random_state=42)
        X = reducer.fit_transform(X).astype("d")
        ev = reducer.explained_variance_ratio_.sum()
        print(f"    Explained variance: {ev:.1%}")

    print(f"  Fitting ManifoldModel  n={len(X)}  d={X.shape[1]} ...", flush=True)
    subject = ManifoldModel(k_graph=k_graph, k_pca=k_pca, k_vote=k_vote, variance_threshold=tau)
    subject.fit(X, y)

    print("  Running ManifoldObserver ...", flush=True)
    observer = ManifoldObserver(subject)
    observer.lift_data()
    field = observer.observe()

    curvatures = np.array([o.curvature for o in field], dtype="d")
    heights = np.array([o.height for o in field], dtype="d")
    idims = np.array([o.intrinsic_dim for o in field], dtype="d")
    node_labels = np.array([(subject._geometries[o.node_id].label or 0) for o in field], dtype=int)

    pca_info = None
    if X.shape[1] > 3:
        n_comp = min(max(pca_components), X.shape[1], len(X))
        pc_label = ",".join(str(c) for c in pca_components)
        print(f"  PCA projection to 3-D  (PC{pc_label}) ...", flush=True)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X).astype("d")
        sel = [c - 1 for c in pca_components]
        X3 = X_pca[:, sel]
        evr = pca.explained_variance_ratio_[sel]
        total = float(evr.sum())
        print(
            f"    Explained variance: {total:.1%}  "
            f"(PC{pca_components[0]}={evr[0]:.1%}  "
            f"PC{pca_components[1]}={evr[1]:.1%}  "
            f"PC{pca_components[2]}={evr[2]:.1%})"
        )
        pca_info = PCAInfo(
            explained_variance_ratio=evr,
            total_explained=total,
            ambient_dim=X.shape[1],
            components=pca_components,
        )
    else:
        X3 = X[:, :3].copy()

    return (
        subject,
        observer,
        PointField(
            X3=X3,
            density_w=np.ones(len(X3), dtype="d"),
            curvature=curvatures,
            height=heights,
            intrinsic_dim=idims,
            labels=node_labels,
        ),
        pca_info,
    )


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------


def voxelize(pf: PointField, resolution: int = 32, padding: float = 0.05) -> dict:
    """Rasterize per-point fields into a uniform 3-D voxel grid.

    Each scalar is accumulated per voxel cell and then averaged.  The
    resulting arrays are in Fortran order ready for ``pv.ImageData``.

    Requires ``scipy`` (``pip install scipy`` or ``poetry install --with viz``).

    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param resolution: Number of voxels along each axis.
    :param padding: Fractional padding beyond bounding box (e.g., 0.05 = 5 %).
    :return: Dict with keys ``origin``, ``spacing``, ``dims``, and one key
        per scalar field, each a ``(resolution**3,)`` float32 array.
    """
    _require_viz("voxelize")

    res = resolution
    X3 = pf.X3

    lo = X3.min(axis=0)
    hi = X3.max(axis=0)
    span = hi - lo
    lo -= span * padding
    hi += span * padding
    spacing = (hi - lo) / res

    idx = np.clip(
        np.floor((X3 - lo) / spacing).astype(int),
        0,
        res - 1,
    )
    flat_idx = idx[:, 0] + res * idx[:, 1] + res * res * idx[:, 2]

    total_cells = res**3
    count = np.bincount(flat_idx, minlength=total_cells).astype("f4")

    def _mean_field(values: np.ndarray) -> np.ndarray:
        s = np.bincount(flat_idx, weights=values.astype("d"), minlength=total_cells)
        return np.where(count > 0, s / np.maximum(count, 1), 0.0).astype("f4")

    def _majority_vote(labels: np.ndarray) -> np.ndarray:
        n_classes = int(labels.max()) + 1
        votes = np.zeros((total_cells, n_classes), dtype="f4")
        for c in range(n_classes):
            mask = labels == c
            if mask.any():
                votes[:, c] = np.bincount(flat_idx[mask], minlength=total_cells)
        return votes.argmax(axis=1).astype("f4")

    density_raw = count.reshape(res, res, res)
    density_smooth = scipy.ndimage.gaussian_filter(density_raw, sigma=1.0).astype("f4").ravel()

    return {
        "origin": lo,
        "spacing": spacing,
        "dims": (res, res, res),
        "density": density_smooth,
        "curvature": _mean_field(pf.curvature),
        "height": _mean_field(pf.height),
        "intrinsic_dim": _mean_field(pf.intrinsic_dim),
        "class_vote": _majority_vote(pf.labels),
    }


# ---------------------------------------------------------------------------
# PyVista grid + rendering
# ---------------------------------------------------------------------------

#: Default colour maps for each scalar field.
CMAP_MAP: dict[str, str] = {
    "density": "plasma",
    "curvature": "coolwarm",
    "height": "viridis",
    "intrinsic_dim": "tab10",
    "class_vote": "Set1",
}


def build_grid(vox: dict):
    """Construct a ``pv.ImageData`` from a voxelization dict.

    Requires ``pyvista`` (``pip install pyvista`` or
    ``poetry install --with viz``).

    :param vox: Dict returned by :func:`voxelize`.
    :return: ``pv.ImageData`` with all scalar fields attached as point data.
    """
    _require_viz("build_grid")
    res = vox["dims"][0]
    grid = pv.ImageData()
    grid.dimensions = (res, res, res)
    grid.origin = vox["origin"].tolist()
    grid.spacing = vox["spacing"].tolist()

    for key in ("density", "curvature", "height", "intrinsic_dim", "class_vote"):
        arr = vox[key]
        grid.point_data[key] = arr.reshape(res, res, res).ravel(order="F")

    return grid


def _add_pca_axes(p, pca_info: PCAInfo | None) -> None:
    """Add axes labeled with PCA explained-variance ratios."""
    if pca_info is not None:
        evr = pca_info.explained_variance_ratio
        c = pca_info.components
        p.add_axes(
            xlabel=f"PC{c[0]} ({evr[0]:.1%})",
            ylabel=f"PC{c[1]} ({evr[1]:.1%})",
            zlabel=f"PC{c[2]} ({evr[2]:.1%})",
        )
    else:
        p.add_axes()


def _add_pca_arrows(p, pf: PointField, pca_info: PCAInfo | None) -> None:
    """Render scaled arrows at the data centroid showing principal directions.

    Arrow length is proportional to explained variance of each component,
    giving a visual sense of which axis carries the most information.
    """
    if pca_info is None:
        return
    centroid = pf.X3.mean(axis=0).astype("f4")
    span = (pf.X3.max(axis=0) - pf.X3.min(axis=0)).astype("f4")
    evr = pca_info.explained_variance_ratio
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i in range(3):
        direction = np.zeros(3, dtype="f4")
        direction[i] = 1.0
        length = float(span[i]) * 0.35 * max(float(np.sqrt(evr[i] / evr[0])), 0.30)
        arrow = pv.Arrow(
            start=centroid,
            direction=direction,
            scale=length,
            tip_length=0.25,
            tip_radius=0.08,
            shaft_radius=0.025,
            tip_resolution=32,
            shaft_resolution=32,
        )
        p.add_mesh(arrow, color=colors[i], opacity=0.7, show_scalar_bar=False)


def _add_nav_help(p, *, corner_widget: bool = True, help_text: bool = True) -> None:
    """Add navigation aids: orientation cube widget and keyboard-shortcut overlay.

    :param p: Active ``pv.Plotter`` (or active subplot).
    :param corner_widget: If ``True``, embed an interactive orientation-cube
        widget in the lower-left corner.
    :param help_text: If ``True``, overlay a compact key-binding cheat-sheet
        in the upper-left corner.
    """
    if corner_widget:
        p.add_camera_orientation_widget()

    if help_text:
        lines = (
            "Navigation\n"
            "  Rotate      left-drag\n"
            "  Zoom        scroll / right-drag\n"
            "  Pan         middle-drag\n"
            "  Move slice  drag plane handle\n"
            "  Reset cam   r\n"
            "  Screenshot  s\n"
            "  Quit        q"
        )
        p.add_text(
            lines,
            position="upper_left",
            font_size=8,
            color="white",
            shadow=True,
            font="courier",
        )


def _add_voxel_cloud(
    p,
    grid,
    scalar: str,
    opacity: float = 0.12,
    threshold_frac: float = 0.04,
) -> None:
    """Overlay a semi-transparent voxel cloud showing the full volume extent.

    Thresholds the density field at ``threshold_frac × max_density``, then
    renders the surviving voxel cells coloured by *scalar* at low opacity.
    This sits behind the slice planes and gives a ghostly silhouette of the
    whole manifold shape.

    :param p: Active ``pv.Plotter``.
    :param grid: ``pv.ImageData`` from :func:`build_grid`.
    :param scalar: Scalar field to colour the cloud by.
    :param opacity: Alpha for the voxel cloud mesh (0 = invisible, 1 = solid).
    :param threshold_frac: Keep voxels whose density ≥ this fraction of max.
    """
    density = grid.point_data["density"]
    min_val = float(density.max()) * threshold_frac
    if min_val <= 0.0:
        return
    cloud = grid.threshold(min_val, scalars="density")
    if cloud.n_cells == 0:
        return
    p.add_mesh(
        cloud,
        scalars=scalar,
        cmap=CMAP_MAP.get(scalar, "plasma"),
        opacity=opacity,
        show_scalar_bar=False,
        show_edges=False,
    )


def _compose_single_scene(
    p,
    grid,
    pf: PointField,
    scalar: str,
    show_points: bool,
    show_volume: bool,
    vol_opacity: float,
    vol_threshold: float,
    pca_info: PCAInfo | None,
    *,
    sliceable: bool = True,
    scalar_bar: bool = True,
) -> None:
    """Add the single-scalar scene (slices/cloud/points/arrows) to a plotter.

    Shared by :func:`render_single` (interactive / PNG),
    :func:`render_quilt_single` (Looking Glass quilt export), and
    :func:`render_hld_single` (Hololuminescent Display video).

    :param p: Active ``pv.Plotter``.
    :param sliceable: If ``True``, use the interactive orthogonal-slice
        widget; if ``False``, add static orthogonal slices (widgets cannot
        be rendered repeatedly off-screen for quilt capture).
    :param scalar_bar: Show the colour scale bar.  Disabled for HLD output,
        where 2-D overlays breach the safe-area margins and clutter the
        hologram.
    """
    if show_volume:
        _add_voxel_cloud(p, grid, scalar, opacity=vol_opacity, threshold_frac=vol_threshold)

    cmap = CMAP_MAP.get(scalar, "plasma")
    if sliceable:
        p.add_mesh_slice_orthogonal(grid, scalars=scalar, cmap=cmap, show_scalar_bar=scalar_bar)
    else:
        p.add_mesh(grid.slice_orthogonal(), scalars=scalar, cmap=cmap, show_scalar_bar=scalar_bar)

    if show_points:
        cloud = pv.PolyData(pf.X3.astype("f4"))
        cloud.point_data["label"] = pf.labels.astype("f4")
        p.add_points(
            cloud, scalars="label", cmap="Set1", point_size=8, opacity=0.7, show_scalar_bar=False
        )

    _add_pca_arrows(p, pf, pca_info)


def render_single(
    grid,
    pf: PointField,
    scalar: str = "density",
    off_screen: bool = False,
    out_path: Path | None = None,
    show_points: bool = True,
    show_volume: bool = False,
    vol_opacity: float = 0.12,
    vol_threshold: float = 0.04,
    pca_info: PCAInfo | None = None,
) -> None:
    """Single-scalar orthogonal-slice viewer with optional voxel cloud.

    Requires ``pyvista`` (``poetry install --with viz``).

    :param grid: ``pv.ImageData`` from :func:`build_grid`.
    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param scalar: Which scalar field to display.
    :param off_screen: If ``True``, render headless (no window).
    :param out_path: PNG path for headless export (implies *off_screen*).
    :param show_points: Overlay scatter of raw training points.
    :param show_volume: Render the full voxel cloud behind the slices.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as a fraction of max.
    :param pca_info: If provided, label axes with PCA variance and add
        direction arrows.
    """
    _require_viz("render_single")
    p = pv.Plotter(off_screen=off_screen, title=f"Manifold Voxels — {scalar}")

    _compose_single_scene(
        p, grid, pf, scalar, show_points, show_volume, vol_opacity, vol_threshold, pca_info
    )
    _add_pca_axes(p, pca_info)
    if not off_screen:
        _add_nav_help(p)

    if pca_info is not None:
        title = (
            f"Manifold subspace — {scalar}   "
            f"[{pca_info.ambient_dim}D → 3D, "
            f"captured {pca_info.total_explained:.1%}]"
        )
    else:
        title = f"Manifold subspace — {scalar}"
    p.add_title(title, font_size=11)

    if off_screen and out_path:
        p.show(auto_close=False)
        p.screenshot(str(out_path))
        p.close()
        print(f"  Saved {out_path}")
    else:
        p.show()


def render_multi(
    grid,
    pf: PointField,
    off_screen: bool = False,
    out_path: Path | None = None,
    show_volume: bool = False,
    vol_opacity: float = 0.12,
    vol_threshold: float = 0.04,
    pca_info: PCAInfo | None = None,
    scalars: list[tuple[str, str, str]] | None = None,
) -> None:
    """2×2 panel: intrinsic_dim / curvature / height / class_vote (default).

    Requires ``pyvista`` (``poetry install --with viz``).

    :param grid: ``pv.ImageData`` from :func:`build_grid`.
    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param off_screen: If ``True``, render headless.
    :param out_path: PNG path for headless export.
    :param show_volume: Render the full voxel cloud in each panel.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as fraction of max.
    :param pca_info: If provided, label axes with PCA variance and add
        direction arrows.
    :param scalars: List of (field_name, colormap, title) tuples for the 2×2
        panels.  Defaults to intrinsic_dim / curvature / height / class_vote.
    """
    _require_viz("render_multi")
    if scalars is None:
        scalars = [
            ("intrinsic_dim", "tab10", "Intrinsic dim (d*)"),
            ("curvature", "coolwarm", "Mean curvature"),
            ("height", "viridis", "Height above tangent"),
            ("class_vote", "Set1", "Majority class vote"),
        ]

    if pca_info is not None:
        window_title = (
            f"Manifold Voxels \u2014 multi-scalar   "
            f"[{pca_info.ambient_dim}D \u2192 3D, "
            f"{pca_info.total_explained:.1%} var]"
        )
    else:
        window_title = "Manifold Voxels \u2014 multi-scalar"

    p = pv.Plotter(
        shape=(2, 2),
        off_screen=off_screen,
        title=window_title,
    )

    for i, (scalar, cmap, title) in enumerate(scalars):
        row, col = divmod(i, 2)
        p.subplot(row, col)

        if show_volume:
            _add_voxel_cloud(p, grid, scalar, opacity=vol_opacity, threshold_frac=vol_threshold)

        p.add_mesh_slice_orthogonal(
            grid,
            scalars=scalar,
            cmap=cmap,
            show_scalar_bar=True,
        )
        cloud = pv.PolyData(pf.X3.astype("f4"))
        cloud.point_data["label"] = pf.labels.astype("f4")
        p.add_points(
            cloud, scalars="label", cmap="Set1", point_size=6, opacity=0.6, show_scalar_bar=False
        )
        _add_pca_axes(p, pca_info)
        _add_pca_arrows(p, pf, pca_info)
        if not off_screen:
            _add_nav_help(p, help_text=(i == 0))
        p.add_title(title, font_size=9)

    if off_screen and out_path:
        p.show(auto_close=False)
        p.screenshot(str(out_path))
        p.close()
        print(f"  Saved {out_path}")
    else:
        p.show()


def render_quilt_single(
    grid,
    pf: PointField,
    scalar: str = "density",
    out_path: Path | str = "manifold",
    device: str = "portrait",
    view_cone: float | None = None,
    show_points: bool = True,
    show_volume: bool = False,
    vol_opacity: float = 0.12,
    vol_threshold: float = 0.04,
    pca_info: PCAInfo | None = None,
    cast: bool = False,
    quilt_grid: tuple[int, int] | None = None,
    video: bool = False,
    n_frames: int = 180,
    fps: int = 24,
    orbit: float = 360.0,
    zoom: float = 1.6,
    scalar_bar: bool = False,
) -> Path:
    """Render the single-scalar scene as a Looking Glass quilt (PNG or MP4).

    Composes the same scene as :func:`render_single` (static slices instead
    of the interactive widget), then sweeps the camera across the device's
    view cone with off-axis projections and tiles the views into a quilt.
    With ``video=True``, renders a turntable orbit of quilt frames and
    encodes them to MP4.  Output filenames carry the
    ``_qs<cols>x<rows>a<aspect>`` suffix so Looking Glass Studio / Bridge
    auto-detect the settings.

    Requires ``pyvista`` + ``pillow`` (``poetry install --with viz``);
    video additionally needs ffmpeg (or ``pip install imageio-ffmpeg``).

    :param grid: ``pv.ImageData`` from :func:`build_grid`.
    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param scalar: Which scalar field to display.
    :param out_path: Output stem; the quilt suffix + extension are appended.
    :param device: Key into :data:`quiltwright.lfd.QUILT_PRESETS`
        (e.g. ``"portrait"``, ``"go"``, ``"16-landscape"``).
    :param view_cone: Override the preset's view cone in degrees.
    :param show_points: Overlay scatter of raw training points.
    :param show_volume: Render the full voxel cloud behind the slices.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as a fraction of max.
    :param pca_info: If provided, add PCA direction arrows.
    :param cast: If ``True``, also send the quilt to a connected Looking
        Glass via the local Looking Glass Bridge service.
    :param quilt_grid: Optional ``(columns, rows)`` override of the preset's
        view grid — more views = smoother look-around, lower per-view
        resolution.
    :param video: Render a turntable quilt video instead of a still.
    :param n_frames: Video frame count (with *fps* sets loop duration).
    :param fps: Video frame rate.
    :param orbit: Total camera orbit in degrees over the clip (360 loops).
    :param zoom: Camera dolly factor.  PyVista's default framing leaves the
        volume filling roughly a third of the frame, which wastes both the
        per-view resolution and the parallax budget; > 1 pulls the camera in.
    :param scalar_bar: Show the colour scale bar.  Off by default for quilts:
        a 2-D overlay is pinned to the focal plane and reads as a flat pane
        cutting through the hologram.
    :return: Path of the quilt PNG/MP4 written.
    """
    _require_viz("render_quilt_single")
    spec = QUILT_PRESETS[device]
    if quilt_grid is not None:
        spec = spec.with_grid(*quilt_grid)

    p = pv.Plotter(off_screen=True)
    _compose_single_scene(
        p,
        grid,
        pf,
        scalar,
        show_points,
        show_volume,
        vol_opacity,
        vol_threshold,
        pca_info,
        sliceable=False,
        scalar_bar=scalar_bar,
    )

    if video:
        saved = render_quilt_video(
            p,
            spec,
            out_path,
            n_frames=n_frames,
            fps=fps,
            orbit_degrees=orbit,
            view_cone=view_cone,
            zoom=zoom,
        )
        p.close()
        print(
            f"  Saved quilt video {saved}  "
            f"({n_frames} frames x {spec.n_views} views, {n_frames / fps:.1f}s loop)"
        )
    else:
        quilt = render_quilt(p, spec, view_cone=view_cone, zoom=zoom)
        p.close()
        saved = save_quilt(quilt, out_path, spec)
        print(f"  Saved quilt {saved}  ({spec.n_views} views, {spec.columns}x{spec.rows})")

    if cast:
        cast_quilt(saved, spec)
        print("  Cast to Looking Glass via Bridge")
    return saved


def render_hld_single(
    grid,
    pf: PointField,
    scalar: str = "density",
    out_path: Path | str = "manifold",
    show_points: bool = True,
    show_volume: bool = False,
    vol_opacity: float = 0.12,
    vol_threshold: float = 0.04,
    pca_info: PCAInfo | None = None,
    n_frames: int = 300,
    fps: int = 30,
    orbit: float = 360.0,
    shadow: bool = True,
    zoom: float = 1.0,
) -> Path:
    """Render the single-scalar scene as a Hololuminescent Display video.

    HLDs (the 16"/27"/86" Hololuminescent line) play ordinary 2-D video —
    no quilts — with pure white rendered invisible by the display's optics.
    This composes the same scene as :func:`render_single` on a white
    background, adds a soft contact shadow under the voxel volume, frames
    it inside the HLD safe area, and renders a slow turntable orbit to the
    official HLD master spec (3840×2160 landscape HEVC bt709).

    Run the output through Looking Glass's HLD Author app, then copy it to
    the display's USB media player.  See :mod:`quiltwright.hld`.

    Requires ``pyvista`` + ``pillow`` + ffmpeg (``poetry install --with viz``).

    :param grid: ``pv.ImageData`` from :func:`build_grid`.
    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param scalar: Which scalar field to display.
    :param out_path: Output stem; ``_hld.mp4`` is appended.
    :param show_points: Overlay scatter of raw training points.
    :param show_volume: Render the full voxel cloud behind the slices.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as a fraction of max.
    :param pca_info: If provided, add PCA direction arrows.
    :param n_frames: Frame count (default 300 @ 30 fps = 10 s loop).
    :param fps: 30 or 60 per the HLD media spec.
    :param orbit: Total turntable rotation in degrees (360 loops).
    :param shadow: Paint a soft contact shadow under the volume (the HLD
        guidelines call floor shadows crucial for the depth effect).
    :param zoom: Extra camera zoom applied after safe-area framing.  1.0 =
        no extra zoom; values > 1 enlarge the subject (useful for portrait
        subjects in the 16:9 frame).
    :return: Path of the MP4 written.
    """
    from quiltwright.hld import add_floor_shadow, render_hld_video, style_plotter_for_hld

    _require_viz("render_hld_single")

    p = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    _compose_single_scene(
        p,
        grid,
        pf,
        scalar,
        show_points,
        show_volume,
        vol_opacity,
        vol_threshold,
        pca_info,
        sliceable=False,
        scalar_bar=False,
    )
    if shadow:
        add_floor_shadow(p, grid.bounds)
    style_plotter_for_hld(p, zoom=zoom)

    saved = render_hld_video(p, out_path, n_frames=n_frames, fps=fps, orbit_degrees=orbit)
    p.close()
    print(
        f"  Saved HLD video {saved}  ({n_frames} frames, {n_frames / fps:.1f}s loop)\n"
        "  Next: open it in HLD Author to encode for the display's USB player."
    )
    return saved


# ---------------------------------------------------------------------------
# CT / MRI biomedical demo
# ---------------------------------------------------------------------------

#: Per-dataset presets: isovalues, colors, and per-layer opacities tuned to
#: each modality's intensity distribution.  Colors deliberately avoid pure
#: white so they remain visible on an HLD (white = transparent on-device).
CT_PRESETS: dict[str, dict] = {
    "brain": {
        "loader": "download_brain",
        "scalar": "image_data",
        "description": "181×217×181 T1 MRI brain (1 mm isotropic)",
        "isovalues": [40, 100, 180],
        "colors": ["#e8c4a0", "#e07030", "#c02020"],
        "opacities": [0.25, 0.60, 0.90],
        "smooth_iters": 30,
    },
    "full_head": {
        "loader": "download_full_head",
        "scalar": "MetaImage",
        "description": "256×256×94 CT head (12-bit Hounsfield-ish)",
        "isovalues": [400, 1500],
        "colors": ["#f4c8a0", "#f0e8c0"],
        "opacities": [0.20, 0.85],
        "smooth_iters": 20,
    },
    "head_2": {
        "loader": "download_head_2",
        "scalar": "Scalars_",
        "description": "256×256×94 CT head alt (12-bit Hounsfield-ish)",
        "isovalues": [400, 1500],
        "colors": ["#f4c8a0", "#f0e8c0"],
        "opacities": [0.20, 0.85],
        "smooth_iters": 20,
    },
    "whole_body_ct_male": {
        "loader": "download_whole_body_ct_male",
        "scalar": "NIFTI",
        "description": "160×160×273 whole-body CT male (Hounsfield units)",
        "isovalues": [-100, 300, 700],
        "colors": ["#f0c080", "#d08040", "#f0e8c0"],
        "opacities": [0.15, 0.45, 0.85],
        "smooth_iters": 20,
        "multiblock_key": "ct",
    },
    "whole_body_ct_female": {
        "loader": "download_whole_body_ct_female",
        "scalar": "NIFTI",
        "description": "160×160×271 whole-body CT female (Hounsfield units)",
        "isovalues": [-100, 300, 700],
        "colors": ["#f0c080", "#d08040", "#f0e8c0"],
        "opacities": [0.15, 0.45, 0.85],
        "smooth_iters": 20,
        "multiblock_key": "ct",
    },
}


def render_ct_hld(
    ct_dataset: str = "brain",
    out_path: Path | str = "ct_demo",
    n_frames: int = 300,
    fps: int = 30,
    orbit: float = 360.0,
    shadow: bool = True,
    smooth_iters: int | None = None,
    isovalues: list[float] | None = None,
    zoom: float = 1.8,
) -> Path:
    """Render a PyVista biomedical CT/MRI dataset as a Hololuminescent Display video.

    Loads one of PyVista's built-in medical volumes, extracts layered
    isosurfaces (skin / tissue / bone), and renders a slow turntable orbit
    to the official HLD master spec (3840×2160 HEVC bt709).  White = the
    HLD background is invisible on-device, so all colors avoid pure white.

    Requires ``pyvista`` + ``pillow`` + ffmpeg (``poetry install --with viz``).

    :param ct_dataset: Dataset key from :data:`CT_PRESETS`
        (``"brain"``, ``"full_head"``, ``"head_2"``,
        ``"whole_body_ct_male"``, ``"whole_body_ct_female"``).
    :param out_path: Output stem; ``_hld.mp4`` is appended.
    :param n_frames: Frame count (default 300 @ 30 fps = 10 s loop).
    :param fps: 30 or 60 per the HLD media spec.
    :param orbit: Total turntable rotation in degrees (360 loops seamlessly).
    :param shadow: Paint a soft contact shadow under the volume.
    :param smooth_iters: Surface-smoothing iterations (``None`` uses preset).
    :param isovalues: Override isovalue thresholds (``None`` uses preset).
    :param zoom: Camera zoom applied after safe-area framing (default 1.8).
        Portrait subjects (brains, bodies) appear small in the 16:9 landscape
        frame after ``reset_camera()``; this zooms them up to fill the alcove.
        Increase for taller subjects (e.g. 2.2 for whole-body CT).
    :return: Path of the MP4 written.
    """
    from quiltwright.hld import add_floor_shadow, render_hld_video, style_plotter_for_hld

    _require_viz("render_ct_hld")

    data = _load_ct_volume(ct_dataset, isovalues)
    preset = CT_PRESETS[ct_dataset]
    isos = isovalues if isovalues is not None else preset["isovalues"]
    n_smooth = smooth_iters if smooth_iters is not None else preset["smooth_iters"]

    p = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    _compose_ct_scene(p, data, preset, isos, n_smooth)

    if shadow:
        add_floor_shadow(p, data.bounds)
    style_plotter_for_hld(p, zoom=zoom)

    saved = render_hld_video(p, out_path, n_frames=n_frames, fps=fps, orbit_degrees=orbit)
    p.close()
    print(
        f"  Saved HLD video {saved}  ({n_frames} frames, {n_frames / fps:.1f}s loop)\n"
        "  Next: open in HLD Author to encode for the display's USB player."
    )
    return saved


def render_ct_quilt(
    ct_dataset: str = "brain",
    out_path: Path | str = "ct_demo",
    device: str = "portrait",
    view_cone: float | None = None,
    smooth_iters: int | None = None,
    isovalues: list[float] | None = None,
    cast: bool = False,
    quilt_grid: tuple[int, int] | None = None,
    video: bool = False,
    n_frames: int = 180,
    fps: int = 24,
    orbit: float = 360.0,
    zoom: float = 1.6,
) -> Path:
    """Render a PyVista biomedical CT/MRI dataset as a Looking Glass quilt (PNG or MP4).

    Composes the same isosurface layers as :func:`render_ct_hld` (skin /
    tissue / bone), then sweeps the camera across the device's view cone
    with off-axis projections and tiles the views into a quilt.  Unlike HLD
    output, no white background or safe-area framing is needed — the quilt's
    view-cone sweep and off-axis cameras handle framing on their own.

    Requires ``pyvista`` + ``pillow`` (``poetry install --with viz``);
    video additionally needs ffmpeg (or ``pip install imageio-ffmpeg``).

    :param ct_dataset: Dataset key from :data:`CT_PRESETS`.
    :param out_path: Output stem; the quilt suffix + extension are appended.
    :param device: Key into :data:`quiltwright.lfd.QUILT_PRESETS`
        (e.g. ``"portrait"``, ``"go"``, ``"16-landscape"``).
    :param view_cone: Override the preset's view cone in degrees.
    :param smooth_iters: Surface-smoothing iterations (``None`` uses preset).
    :param isovalues: Override isovalue thresholds (``None`` uses preset).
    :param cast: If ``True``, also send the quilt to a connected Looking
        Glass via the local Looking Glass Bridge service.
    :param quilt_grid: Optional ``(columns, rows)`` override of the preset's
        view grid — more views = smoother look-around, lower per-view
        resolution.
    :param video: Render a turntable quilt video instead of a still.
    :param n_frames: Video frame count (with *fps* sets loop duration).
    :param fps: Video frame rate.
    :param orbit: Total camera orbit in degrees over the clip (360 loops).
    :param zoom: Camera dolly factor; see :func:`render_quilt`.
    :return: Path of the quilt PNG/MP4 written.
    """
    _require_viz("render_ct_quilt")

    data = _load_ct_volume(ct_dataset, isovalues)
    preset = CT_PRESETS[ct_dataset]
    isos = isovalues if isovalues is not None else preset["isovalues"]
    n_smooth = smooth_iters if smooth_iters is not None else preset["smooth_iters"]

    spec = QUILT_PRESETS[device]
    if quilt_grid is not None:
        spec = spec.with_grid(*quilt_grid)

    p = pv.Plotter(off_screen=True)
    _compose_ct_scene(p, data, preset, isos, n_smooth)

    if video:
        saved = render_quilt_video(
            p,
            spec,
            out_path,
            n_frames=n_frames,
            fps=fps,
            orbit_degrees=orbit,
            view_cone=view_cone,
            zoom=zoom,
        )
        p.close()
        print(
            f"  Saved quilt video {saved}  "
            f"({n_frames} frames x {spec.n_views} views, {n_frames / fps:.1f}s loop)"
        )
    else:
        quilt = render_quilt(p, spec, view_cone=view_cone, zoom=zoom)
        p.close()
        saved = save_quilt(quilt, out_path, spec)
        print(f"  Saved quilt {saved}  ({spec.n_views} views, {spec.columns}x{spec.rows})")

    if cast:
        cast_quilt(saved, spec)
        print("  Cast to Looking Glass via Bridge")
    return saved


def render_ct_still(
    ct_dataset: str = "brain",
    out_path: Path | str = "ct_demo",
    shadow: bool = True,
    smooth_iters: int | None = None,
    isovalues: list[float] | None = None,
    zoom: float = 1.8,
) -> Path:
    """Render a single HLD-resolution PNG of a CT/MRI dataset.

    Same scene as :func:`render_ct_hld` (white background, safe-area framing,
    layered isosurfaces, optional contact shadow) but exports one
    ``*_hld.png`` at 3840×2160 instead of a video.  No ffmpeg required.
    Useful for previews and signage systems that accept still images.

    :param ct_dataset: Dataset key from :data:`CT_PRESETS`.
    :param out_path: Output stem; ``_hld.png`` is appended.
    :param shadow: Paint a soft contact shadow under the volume.
    :param smooth_iters: Surface-smoothing iterations (``None`` uses preset).
    :param isovalues: Override isovalue thresholds (``None`` uses preset).
    :param zoom: Camera zoom after safe-area framing (default 1.8).
    :return: Path of the PNG written.
    """
    from quiltwright.hld import add_floor_shadow, render_hld_still, style_plotter_for_hld

    _require_viz("render_ct_still")

    data = _load_ct_volume(ct_dataset, isovalues)
    preset = CT_PRESETS[ct_dataset]
    isos = isovalues if isovalues is not None else preset["isovalues"]
    n_smooth = smooth_iters if smooth_iters is not None else preset["smooth_iters"]

    p = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    _compose_ct_scene(p, data, preset, isos, n_smooth)

    if shadow:
        add_floor_shadow(p, data.bounds)
    style_plotter_for_hld(p, zoom=zoom)

    saved = render_hld_still(p, out_path)
    p.close()
    print(f"  Saved HLD still {saved}  (3840×2160, white background)")
    return saved


def render_ct_viewer(
    ct_dataset: str = "brain",
    off_screen: bool = False,
    out_path: Path | str | None = None,
    shadow: bool = False,
    smooth_iters: int | None = None,
    isovalues: list[float] | None = None,
) -> None:
    """Interactive PyVista viewer for a biomedical CT/MRI isosurface scene.

    Opens an interactive window (rotate / zoom / pan) or, with
    *off_screen* / *out_path*, saves a PNG screenshot.  Uses the same
    isosurface layers as :func:`render_ct_hld` but on a dark background
    so the translucent layers read clearly on screen.

    Requires ``pyvista`` (``poetry install --with viz``).

    :param ct_dataset: Dataset key from :data:`CT_PRESETS`.
    :param off_screen: Render headless (for PNG export).
    :param out_path: PNG path; implies *off_screen*.
    :param shadow: Add a contact shadow disc under the volume.
    :param smooth_iters: Surface-smoothing iterations (``None`` = preset).
    :param isovalues: Override isovalue thresholds (``None`` = preset).
    """
    from quiltwright.hld import add_floor_shadow

    _require_viz("render_ct_viewer")

    if out_path is not None:
        off_screen = True

    data = _load_ct_volume(ct_dataset, isovalues)
    preset = CT_PRESETS[ct_dataset]
    isos = isovalues if isovalues is not None else preset["isovalues"]
    n_smooth = smooth_iters if smooth_iters is not None else preset["smooth_iters"]

    p = pv.Plotter(off_screen=off_screen, title=f"CT/MRI — {ct_dataset}")
    p.set_background("black")
    _compose_ct_scene(p, data, preset, isos, n_smooth)

    if shadow:
        add_floor_shadow(p, data.bounds, dark_bg=True)

    if not off_screen:
        _add_nav_help(p)
    p.add_title(f"{ct_dataset}  —  {preset['description']}", font_size=10, color="white")
    p.add_axes()

    if off_screen and out_path:
        p.show(auto_close=False)
        p.screenshot(str(out_path))
        p.close()
        print(f"  Saved {out_path}")
    else:
        p.show()


def _load_ct_volume(ct_dataset: str, isovalues: list[float] | None = None):
    """Load and return the raw ``pv.ImageData`` for *ct_dataset*."""
    import pyvista.examples as pv_examples

    if ct_dataset not in CT_PRESETS:
        raise ValueError(f"Unknown ct_dataset '{ct_dataset}'.  Choose from: {list(CT_PRESETS)}")

    preset = CT_PRESETS[ct_dataset]
    print(f"  Loading {ct_dataset}: {preset['description']} ...", flush=True)

    loader = getattr(pv_examples, preset["loader"])
    data = loader()
    if "multiblock_key" in preset:
        data = data[preset["multiblock_key"]]
    data.set_active_scalars(preset["scalar"])
    return data


def _compose_ct_scene(p, data, preset: dict, isos: list[float], n_smooth: int) -> None:
    """Add CT isosurface layers to an existing plotter."""
    colors = preset["colors"]
    opacities = preset["opacities"]

    for iso, color, opacity in zip(isos, colors, opacities):
        surf = data.contour([iso])
        if surf.n_points == 0:
            print(f"    isovalue {iso}: no surface — skipping")
            continue
        if n_smooth > 0:
            surf = surf.smooth(n_iter=n_smooth, relaxation_factor=0.1)
        p.add_mesh(surf, color=color, opacity=opacity, smooth_shading=True, show_scalar_bar=False)
        print(f"    isovalue {iso}: {surf.n_points:,} pts  color={color}  opacity={opacity}")


# ---------------------------------------------------------------------------
# TVB brain demo
# ---------------------------------------------------------------------------

#: Per-dataset scene recipes for The Virtual Brain data.  Three scene kinds:
#:
#: ``surface``
#:     One triangulated surface, optionally coloured by a parcellation.
#: ``connectome``
#:     Region centres as degree-scaled spheres plus the strongest tracts as
#:     weight-coloured tubes, inside a translucent cortex for context.
#: ``layers``
#:     Nested head shells (cortex inside skull inside skin), each with its
#:     own colour and opacity.
#:
#: As with :data:`CT_PRESETS`, colours avoid pure white so the scene stays
#: visible on an HLD, where white renders as transparent.
TVB_PRESETS: dict[str, dict] = {
    "cortex": {
        "kind": "surface",
        "description": "16k-vertex cortical surface, coloured by 76-region parcellation",
        "surface": "cortex_16384",
        "region_mapping": "regionMapping_16k_76",
        "cmap": "turbo",
        "opacity": 1.0,
        "smooth_iters": 30,
    },
    "cortex_80k": {
        "kind": "surface",
        "description": "82k-vertex cortical surface, coloured by 80-region parcellation",
        "surface": "cortex_80k",
        "region_mapping": "regionMapping_80k_80",
        "cmap": "turbo",
        "opacity": 1.0,
        "smooth_iters": 20,
    },
    "cortex_hires": {
        "kind": "surface",
        "description": "283k-vertex two-hemisphere cortex (decimated for view sweeps)",
        "surface": "cortex_2x120k",
        "region_mapping": None,
        "color": "#d8b0a0",
        "cmap": None,
        "opacity": 1.0,
        "smooth_iters": 10,
        "decimate": 0.5,
    },
    "connectome": {
        "kind": "connectome",
        "description": "76-region structural connectome inside a translucent cortex",
        "connectivity": "connectivity_76",
        "surface": "cortex_16384",
        "percentile": 90.0,
        "cmap": "autumn",
        "node_color": "#ffe8a0",
        "shell_color": "#4477aa",
        "shell_opacity": 0.06,
        "tube_radius": 1.1,
        "node_radius": 4.5,
        "smooth_iters": 30,
    },
    "connectome_998": {
        "kind": "connectome",
        "description": "998-region high-resolution connectome (top 1% of tracts)",
        "connectivity": "connectivity_998",
        "surface": "cortex_16384",
        "percentile": 99.0,
        "cmap": "autumn",
        "node_color": "#ffe8a0",
        "shell_color": "#4477aa",
        "shell_opacity": 0.05,
        "tube_radius": 0.5,
        "node_radius": 2.2,
        "smooth_iters": 30,
    },
    "head_layers": {
        "kind": "layers",
        "description": "Nested head shells — cortex, inner/outer skull, scalp",
        "layers": [
            {"surface": "cortex_16384", "color": "#e07030", "opacity": 1.00},
            {"surface": "inner_skull_4096", "color": "#c8a878", "opacity": 0.22},
            {"surface": "outer_skull_4096", "color": "#e8dcc0", "opacity": 0.16},
            {"surface": "outer_skin_4096", "color": "#f0c8a8", "opacity": 0.12},
        ],
        "smooth_iters": 20,
    },
    "macaque": {
        "kind": "surface",
        "description": "147k-vertex macaque cortex, coloured by 84-region parcellation",
        "surface": "macaque_147k",
        "region_mapping": "regionMapping_147k_84",
        "cmap": "turbo",
        "opacity": 1.0,
        "smooth_iters": 20,
        "decimate": 0.4,
    },
    "macaque_connectome": {
        "kind": "connectome",
        "description": "84-region macaque connectome inside a translucent macaque cortex",
        "connectivity": "macaque_84",
        "surface": "macaque_147k",
        "percentile": 92.0,
        "cmap": "autumn",
        "node_color": "#ffe8a0",
        "shell_color": "#4477aa",
        "shell_opacity": 0.07,
        "tube_radius": 0.6,
        "node_radius": 3.0,
        "smooth_iters": 15,
        "decimate": 0.6,
    },
}


def _compose_tvb_scene(
    p,
    preset: dict,
    *,
    percentile: float | None = None,
    decimate: float | None = None,
    smooth_iters: int | None = None,
    quiet: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Add a TVB scene to an existing plotter and return its bounds.

    :param p: Target ``pv.Plotter``.
    :param preset: Entry from :data:`TVB_PRESETS`.
    :param percentile: Connectome edge threshold override.
    :param decimate: Triangle-decimation fraction override.
    :param smooth_iters: Smoothing-iteration override.
    :param quiet: Suppress download progress output.
    :return: Scene bounds, for shadow placement.
    """
    from waverider.tvb_data import connectome_polydata, surface_polydata

    kind = preset["kind"]
    n_smooth = smooth_iters if smooth_iters is not None else preset.get("smooth_iters", 0)
    n_decimate = decimate if decimate is not None else preset.get("decimate", 0.0)

    if kind == "surface":
        mesh = surface_polydata(
            preset["surface"],
            region_mapping=preset.get("region_mapping"),
            smooth_iters=n_smooth,
            decimate=n_decimate,
            quiet=quiet,
        )
        if preset.get("region_mapping"):
            p.add_mesh(
                mesh,
                scalars="region",
                cmap=preset.get("cmap", "turbo"),
                opacity=preset.get("opacity", 1.0),
                smooth_shading=True,
                show_scalar_bar=False,
            )
        else:
            p.add_mesh(
                mesh,
                color=preset.get("color", "#d8b0a0"),
                opacity=preset.get("opacity", 1.0),
                smooth_shading=True,
                show_scalar_bar=False,
            )
        print(f"    surface {preset['surface']}: {mesh.n_points:,} pts, {mesh.n_cells:,} tris")
        return mesh.bounds

    if kind == "connectome":
        pct = percentile if percentile is not None else preset.get("percentile", 90.0)
        shell = surface_polydata(
            preset["surface"], smooth_iters=n_smooth, decimate=n_decimate, quiet=quiet
        )
        p.add_mesh(
            shell,
            color=preset.get("shell_color", "#4477aa"),
            opacity=preset.get("shell_opacity", 0.06),
            smooth_shading=True,
            show_scalar_bar=False,
        )
        nodes, edges = connectome_polydata(
            preset["connectivity"],
            percentile=pct,
            tube_radius=preset.get("tube_radius", 1.1),
            node_radius=preset.get("node_radius", 4.5),
            quiet=quiet,
        )
        p.add_mesh(
            edges, scalars="weight", cmap=preset.get("cmap", "autumn"), show_scalar_bar=False
        )
        p.add_mesh(nodes, color=preset.get("node_color", "#ffe8a0"), show_scalar_bar=False)
        print(
            f"    shell {preset['surface']}: {shell.n_points:,} pts\n"
            f"    connectome {preset['connectivity']}: {edges.n_cells:,} tube cells "
            f"above weight percentile {pct:g}"
        )
        return shell.bounds

    if kind == "layers":
        bounds = None
        for layer in preset["layers"]:
            mesh = surface_polydata(layer["surface"], smooth_iters=n_smooth, quiet=quiet)
            p.add_mesh(
                mesh,
                color=layer["color"],
                opacity=layer["opacity"],
                smooth_shading=True,
                show_scalar_bar=False,
            )
            print(
                f"    layer {layer['surface']}: {mesh.n_points:,} pts  "
                f"color={layer['color']}  opacity={layer['opacity']}"
            )
            # The outermost shell (last, largest) defines the scene extent.
            bounds = mesh.bounds
        return bounds

    raise ValueError(f"Unknown TVB scene kind '{kind}'.")


def _tvb_preset(tvb_dataset: str) -> dict:
    """Look up a TVB preset, raising a helpful error for unknown keys."""
    if tvb_dataset not in TVB_PRESETS:
        raise ValueError(
            f"Unknown tvb_dataset '{tvb_dataset}'.  Choose from: {sorted(TVB_PRESETS)}"
        )
    return TVB_PRESETS[tvb_dataset]


def render_tvb_hld(
    tvb_dataset: str = "cortex",
    out_path: Path | str = "tvb_demo",
    n_frames: int = 300,
    fps: int = 30,
    orbit: float = 360.0,
    shadow: bool = True,
    percentile: float | None = None,
    decimate: float | None = None,
    smooth_iters: int | None = None,
    zoom: float = 1.8,
) -> Path:
    """Render a TVB brain dataset as a Hololuminescent Display video.

    Downloads ``tvb-data`` on first use (see :mod:`waverider.tvb_data`),
    composes the preset's scene, and renders a slow turntable orbit to the
    official HLD master spec (3840×2160 HEVC bt709).

    Requires ``pyvista`` + ``pillow`` + ffmpeg (``poetry install --with viz``).

    :param tvb_dataset: Key from :data:`TVB_PRESETS`.
    :param out_path: Output stem; ``_hld.mp4`` is appended.
    :param n_frames: Frame count (default 300 @ 30 fps = 10 s loop).
    :param fps: 30 or 60 per the HLD media spec.
    :param orbit: Total turntable rotation in degrees (360 loops seamlessly).
    :param shadow: Paint a soft contact shadow under the brain.
    :param percentile: Connectome edge threshold override.
    :param decimate: Triangle-decimation fraction override.
    :param smooth_iters: Smoothing-iteration override.
    :param zoom: Camera zoom applied after safe-area framing.
    :return: Path of the MP4 written.
    """
    from quiltwright.hld import add_floor_shadow, render_hld_video, style_plotter_for_hld

    _require_viz("render_tvb_hld")
    preset = _tvb_preset(tvb_dataset)

    p = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    bounds = _compose_tvb_scene(
        p, preset, percentile=percentile, decimate=decimate, smooth_iters=smooth_iters
    )

    if shadow:
        add_floor_shadow(p, bounds)
    style_plotter_for_hld(p, zoom=zoom)

    saved = render_hld_video(p, out_path, n_frames=n_frames, fps=fps, orbit_degrees=orbit)
    p.close()
    print(
        f"  Saved HLD video {saved}  ({n_frames} frames, {n_frames / fps:.1f}s loop)\n"
        "  Next: open in HLD Author to encode for the display's USB player."
    )
    return saved


def render_tvb_quilt(
    tvb_dataset: str = "cortex",
    out_path: Path | str = "tvb_demo",
    device: str = "portrait",
    view_cone: float | None = None,
    percentile: float | None = None,
    decimate: float | None = None,
    smooth_iters: int | None = None,
    cast: bool = False,
    quilt_grid: tuple[int, int] | None = None,
    video: bool = False,
    n_frames: int = 180,
    fps: int = 24,
    orbit: float = 360.0,
    zoom: float = 1.6,
) -> Path:
    """Render a TVB brain dataset as a Looking Glass quilt (PNG or MP4).

    Composes the same scene as :func:`render_tvb_hld`, then sweeps the camera
    across the device's view cone with off-axis projections and tiles the
    views into a quilt.

    Requires ``pyvista`` + ``pillow`` (``poetry install --with viz``);
    video additionally needs ffmpeg.

    :param tvb_dataset: Key from :data:`TVB_PRESETS`.
    :param out_path: Output stem; the quilt suffix + extension are appended.
    :param device: Key into :data:`quiltwright.lfd.QUILT_PRESETS`.
    :param view_cone: Override the preset's view cone in degrees.
    :param percentile: Connectome edge threshold override.
    :param decimate: Triangle-decimation fraction override.  Raise this for
        the dense surfaces — every extra triangle is paid for once per view.
    :param smooth_iters: Smoothing-iteration override.
    :param cast: Also send the quilt to a connected Looking Glass via Bridge.
    :param quilt_grid: Optional ``(columns, rows)`` override of the preset grid.
    :param video: Render a turntable quilt video instead of a still.
    :param n_frames: Video frame count.
    :param fps: Video frame rate.
    :param orbit: Total camera orbit in degrees over the clip.
    :param zoom: Camera dolly factor.
    :return: Path of the quilt PNG/MP4 written.
    """
    _require_viz("render_tvb_quilt")
    preset = _tvb_preset(tvb_dataset)

    spec = QUILT_PRESETS[device]
    if quilt_grid is not None:
        spec = spec.with_grid(*quilt_grid)

    p = pv.Plotter(off_screen=True)
    _compose_tvb_scene(
        p, preset, percentile=percentile, decimate=decimate, smooth_iters=smooth_iters
    )

    if video:
        saved = render_quilt_video(
            p,
            spec,
            out_path,
            n_frames=n_frames,
            fps=fps,
            orbit_degrees=orbit,
            view_cone=view_cone,
            zoom=zoom,
        )
        p.close()
        print(
            f"  Saved quilt video {saved}  "
            f"({n_frames} frames x {spec.n_views} views, {n_frames / fps:.1f}s loop)"
        )
    else:
        quilt = render_quilt(p, spec, view_cone=view_cone, zoom=zoom)
        p.close()
        saved = save_quilt(quilt, out_path, spec)
        print(f"  Saved quilt {saved}  ({spec.n_views} views, {spec.columns}x{spec.rows})")

    if cast:
        cast_quilt(saved, spec)
        print("  Cast to Looking Glass via Bridge")
    return saved


def render_tvb_still(
    tvb_dataset: str = "cortex",
    out_path: Path | str = "tvb_demo",
    shadow: bool = True,
    percentile: float | None = None,
    decimate: float | None = None,
    smooth_iters: int | None = None,
    zoom: float = 1.8,
) -> Path:
    """Render a single HLD-resolution PNG of a TVB brain scene.

    Same scene as :func:`render_tvb_hld` (white background, safe-area
    framing, optional contact shadow) but exports one ``*_hld.png`` at
    3840×2160 instead of a video.  No ffmpeg required.

    :param tvb_dataset: Key from :data:`TVB_PRESETS`.
    :param out_path: Output stem; ``_hld.png`` is appended.
    :param shadow: Paint a soft contact shadow under the brain.
    :param percentile: Connectome edge threshold override.
    :param decimate: Triangle-decimation fraction override.
    :param smooth_iters: Smoothing-iteration override.
    :param zoom: Camera zoom after safe-area framing.
    :return: Path of the PNG written.
    """
    from quiltwright.hld import add_floor_shadow, render_hld_still, style_plotter_for_hld

    _require_viz("render_tvb_still")
    preset = _tvb_preset(tvb_dataset)

    p = pv.Plotter(off_screen=True, theme=pv.themes.DocumentTheme())
    bounds = _compose_tvb_scene(
        p, preset, percentile=percentile, decimate=decimate, smooth_iters=smooth_iters
    )

    if shadow:
        add_floor_shadow(p, bounds)
    style_plotter_for_hld(p, zoom=zoom)

    saved = render_hld_still(p, out_path)
    p.close()
    print(f"  Saved HLD still {saved}  (3840×2160, white background)")
    return saved


def render_tvb_viewer(
    tvb_dataset: str = "cortex",
    off_screen: bool = False,
    out_path: Path | str | None = None,
    shadow: bool = False,
    percentile: float | None = None,
    decimate: float | None = None,
    smooth_iters: int | None = None,
) -> None:
    """Interactive PyVista viewer for a TVB brain scene.

    Opens an interactive window (rotate / zoom / pan) or, with *off_screen*
    / *out_path*, saves a PNG screenshot.  Uses a dark background so
    translucent shells and connectome tubes read clearly on screen.

    Requires ``pyvista`` (``poetry install --with viz``).

    :param tvb_dataset: Key from :data:`TVB_PRESETS`.
    :param off_screen: Render headless (for PNG export).
    :param out_path: PNG path; implies *off_screen*.
    :param shadow: Add a contact shadow disc under the brain.
    :param percentile: Connectome edge threshold override.
    :param decimate: Triangle-decimation fraction override.
    :param smooth_iters: Smoothing-iteration override.
    """
    from quiltwright.hld import add_floor_shadow

    _require_viz("render_tvb_viewer")
    preset = _tvb_preset(tvb_dataset)

    if out_path is not None:
        off_screen = True

    p = pv.Plotter(off_screen=off_screen, title=f"TVB — {tvb_dataset}")
    p.set_background("black")
    bounds = _compose_tvb_scene(
        p, preset, percentile=percentile, decimate=decimate, smooth_iters=smooth_iters
    )

    if shadow:
        add_floor_shadow(p, bounds, dark_bg=True)

    if not off_screen:
        _add_nav_help(p)
    p.add_title(f"{tvb_dataset}  —  {preset['description']}", font_size=10, color="white")
    p.add_axes()

    if off_screen and out_path:
        p.show(auto_close=False)
        p.screenshot(str(out_path))
        p.close()
        print(f"  Saved {out_path}")
    else:
        p.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the voxel visualizer."""
    p = argparse.ArgumentParser(
        description="Manifold voxel visualizer — project, rasterize, slice."
    )

    # Dataset
    p.add_argument(
        "--dataset",
        choices=[
            "helix",
            "swiss_roll",
            "torus",
            "iris",
            "wine",
            "breast_cancer",
            "digits",
            "mnist",
            "cifar10",
            "cifar100",
            "load",
        ],
        default="helix",
        help="Dataset to visualize.",
    )
    p.add_argument("--X-file", default="X.npy", help="Path to X.npy (--dataset load).")
    p.add_argument(
        "--y-file", default=None, help="Path to y.npy (--dataset load); omit for all-zeros."
    )
    p.add_argument(
        "--n-points",
        type=int,
        default=800,
        help="Max points to use (stratified subsample for large datasets).",
    )
    p.add_argument("--seed", type=int, default=42)

    # ManifoldModel
    p.add_argument("--k-graph", type=int, default=10)
    p.add_argument("--k-pca", type=int, default=20)
    p.add_argument("--k-vote", type=int, default=7)
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument(
        "--pre-pca",
        type=int,
        default=0,
        help="Pre-reduce to this many dims via PCA before ManifoldModel "
        "(0 = disabled; recommended 40–50 for MNIST/CIFAR).",
    )
    p.add_argument(
        "--pca-components",
        type=str,
        default="1,2,3",
        help="Which 3 principal components to visualize, as comma-separated "
        "1-based indices (default: '1,2,3').  Use e.g. '4,5,6' to explore "
        "deeper manifold subspaces.",
    )

    # Voxelization
    p.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="Voxel grid resolution per axis (N³ total cells).",
    )
    p.add_argument("--padding", type=float, default=0.05, help="Fractional bounding-box padding.")

    # Rendering
    p.add_argument(
        "--scalar",
        choices=["density", "curvature", "height", "intrinsic_dim", "class_vote"],
        default="density",
        help="Scalar field to display in single-panel mode.",
    )
    p.add_argument(
        "--multi-scalar", action="store_true", help="Show all four fields in a 2×2 panel layout."
    )
    p.add_argument(
        "--no-points", action="store_true", help="Suppress scatter overlay of raw training points."
    )
    p.add_argument(
        "--volume", action="store_true", help="Render the full voxel cloud behind the slice planes."
    )
    p.add_argument(
        "--vol-opacity",
        type=float,
        default=0.12,
        help="Opacity of the voxel cloud (0–1, default 0.12).",
    )
    p.add_argument(
        "--vol-threshold",
        type=float,
        default=0.04,
        help="Density threshold as fraction of max (default 0.04).",
    )
    p.add_argument(
        "--off-screen",
        action="store_true",
        help="Render headless and write a PNG instead of opening a window.",
    )
    p.add_argument("--out", default=None, help="Output PNG path (implies --off-screen).")

    # Looking Glass holographic output
    p.add_argument(
        "--quilt",
        choices=sorted(QUILT_PRESETS),
        default=None,
        help="Render a Looking Glass quilt for this device instead of a flat "
        "PNG (implies --off-screen). Output name comes from --out (suffix "
        "and extension are replaced by the quilt convention).",
    )
    p.add_argument(
        "--view-cone",
        type=float,
        default=None,
        help="Override the quilt view cone in degrees (default: the device "
        "preset's cone — 35 for most, 50 for the Gen3 16\" landscape).",
    )
    p.add_argument(
        "--quilt-grid",
        type=str,
        default=None,
        metavar="CxR",
        help="Override the quilt view grid, e.g. '11x6' for 66 views. More "
        "views = smoother look-around but lower per-view resolution "
        "(quilt pixel size is fixed per device).",
    )
    p.add_argument(
        "--quilt-video",
        action="store_true",
        help="Render a looping turntable quilt video (MP4) instead of a "
        "still quilt. Requires ffmpeg (or pip install imageio-ffmpeg).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Video frame count (default: 180 for --quilt-video, 300 for --hld).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Video frame rate (default: 24 for --quilt-video; 30 for --hld, "
        "which only allows 30 or 60).",
    )
    p.add_argument(
        "--orbit",
        type=float,
        default=360.0,
        help="Total turntable rotation in degrees over the clip (default 360).",
    )
    p.add_argument(
        "--quilt-zoom",
        type=float,
        default=1.6,
        help="Camera dolly factor for quilt output (default: 1.6). >1 makes "
        "the volume fill more of each view, which increases both effective "
        "resolution and perceived depth. Use 1.0 for PyVista's own framing.",
    )
    p.add_argument(
        "--quilt-scalar-bar",
        action="store_true",
        help="Keep the colour scale bar in quilt output. Off by default: a "
        "2-D overlay sits on the focal plane and fights the depth cue.",
    )
    p.add_argument(
        "--cast",
        action="store_true",
        help="After rendering the quilt, display it on the connected Looking "
        "Glass via the local Looking Glass Bridge service.",
    )

    # Hololuminescent Display (HLD) output — the 16"/27"/86" HLD line plays
    # plain 2-D video (white = invisible), not quilts.
    p.add_argument(
        "--hld",
        action="store_true",
        help="Render a turntable video for a Looking Glass Hololuminescent "
        "Display (white background, safe-area framing, 3840x2160 HEVC "
        "master). Finish it in the HLD Author app.",
    )
    p.add_argument(
        "--no-shadow",
        action="store_true",
        help="HLD mode: skip the contact shadow under the volume.",
    )

    # CT / MRI biomedical demo — bypasses ManifoldModel entirely
    p.add_argument(
        "--ct-demo",
        action="store_true",
        help="Render a real PyVista biomedical CT/MRI dataset directly to HLD "
        "or a Looking Glass quilt (skips ManifoldModel; combine with "
        "--ct-dataset, --hld or --quilt, --frames, --fps, --orbit, "
        "--no-shadow, --out).",
    )
    p.add_argument(
        "--ct-dataset",
        choices=sorted(CT_PRESETS),
        default="brain",
        help="Which PyVista medical dataset to render (default: brain).",
    )
    p.add_argument(
        "--ct-isovalues",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help="Comma-separated isovalue thresholds for CT demo (overrides preset). "
        "E.g. '400,1500' for a two-layer head CT.",
    )
    # TVB brain demo — real brain geometry, also bypasses ManifoldModel
    p.add_argument(
        "--tvb-demo",
        action="store_true",
        help="Render a real brain dataset from The Virtual Brain directly to "
        "HLD or a Looking Glass quilt (skips ManifoldModel; combine with "
        "--tvb-dataset, --hld or --quilt, --frames, --fps, --orbit, "
        "--no-shadow, --out).  Downloads ~337 MB of tvb-data on first use.",
    )
    p.add_argument(
        "--tvb-dataset",
        choices=sorted(TVB_PRESETS),
        default="cortex",
        help="Which TVB brain scene to render (default: cortex).",
    )
    p.add_argument(
        "--tvb-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Connectome scenes: keep only tracts at or above this percentile "
        "of non-zero connection weights (preset default is 90–99 depending "
        "on node count).  Lower values draw more edges.",
    )
    p.add_argument(
        "--tvb-decimate",
        type=float,
        default=None,
        metavar="F",
        help="Fraction of triangles to remove from TVB surfaces, 0.0–1.0.  "
        "Worth raising for --quilt, where every triangle is rendered once "
        "per view (e.g. 0.7 for cortex_hires on a 48-view device).",
    )
    p.add_argument(
        "--tvb-smooth",
        type=int,
        default=None,
        metavar="N",
        help="Laplacian smoothing iterations for TVB surfaces (overrides preset).",
    )
    p.add_argument(
        "--tvb-clear-cache",
        action="store_true",
        help="Delete the cached tvb-data archive and exit.",
    )

    p.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="Camera zoom applied after safe-area framing for HLD output.  "
        "Default: 1.8 for --ct-demo / --tvb-demo (portrait subjects in a 16:9 "
        "frame), 1.0 for manifold mode.  Values > 1 enlarge the subject.",
    )
    p.add_argument(
        "--still",
        action="store_true",
        help="Export a single HLD-resolution PNG (3840×2160, white background) "
        "instead of a video.  No ffmpeg required.  Only applies with --hld.",
    )

    return p.parse_args()


def main() -> None:
    """Entry point for the ``waverider-voxel-viz`` command."""
    args = parse_args()

    if args.tvb_clear_cache:
        from waverider.tvb_data import clear_cache

        clear_cache()
        return

    if args.out:
        args.off_screen = True

    out_path = Path(args.out) if args.out else None

    # ---- TVB brain demo (short-circuit — no ManifoldModel) ---------------
    if args.tvb_demo:
        from waverider.tvb_data import TVB_DATA_DOI

        preset = TVB_PRESETS[args.tvb_dataset]
        print("=" * 60)
        print("TVB BRAIN DEMO")
        print(f"  dataset  : {args.tvb_dataset}")
        print(f"  desc     : {preset['description']}")
        print(f"  kind     : {preset['kind']}")
        print(f"  source   : tvb-data, Zenodo doi:{TVB_DATA_DOI} (GPL-3.0)")
        print("=" * 60)

        if args.hld and args.quilt:
            print("ERROR: --hld and --quilt are mutually exclusive (different device families).")
            sys.exit(1)

        stem = out_path if out_path else Path(f"tvb_{args.tvb_dataset}")

        if args.hld and args.still:
            print(f"  mode     : HLD still → {stem}_hld.png")
            render_tvb_still(
                tvb_dataset=args.tvb_dataset,
                out_path=stem,
                shadow=not args.no_shadow,
                percentile=args.tvb_percentile,
                decimate=args.tvb_decimate,
                smooth_iters=args.tvb_smooth,
                zoom=args.zoom if args.zoom is not None else 1.8,
            )
        elif args.hld:
            print(f"  mode     : HLD video → {stem}_hld.mp4")
            print(f"  frames   : {args.frames or 300}  fps={args.fps or 30}  orbit={args.orbit}°")
            render_tvb_hld(
                tvb_dataset=args.tvb_dataset,
                out_path=stem,
                n_frames=args.frames if args.frames is not None else 300,
                fps=args.fps if args.fps is not None else 30,
                orbit=args.orbit,
                shadow=not args.no_shadow,
                percentile=args.tvb_percentile,
                decimate=args.tvb_decimate,
                smooth_iters=args.tvb_smooth,
                zoom=args.zoom if args.zoom is not None else 1.8,
            )
        elif args.quilt:
            quilt_grid = None
            if args.quilt_grid:
                try:
                    cols, rows = (int(v) for v in args.quilt_grid.lower().split("x"))
                    quilt_grid = (cols, rows)
                except ValueError:
                    print(f"ERROR: --quilt-grid must look like '11x6', got '{args.quilt_grid}'")
                    sys.exit(1)
            kind = "video" if args.quilt_video else "still"
            print(f"  mode     : LFD quilt {kind} ({args.quilt}) → {stem}_qs...")
            render_tvb_quilt(
                tvb_dataset=args.tvb_dataset,
                out_path=stem,
                device=args.quilt,
                view_cone=args.view_cone,
                percentile=args.tvb_percentile,
                decimate=args.tvb_decimate,
                smooth_iters=args.tvb_smooth,
                cast=args.cast,
                quilt_grid=quilt_grid,
                video=args.quilt_video,
                n_frames=args.frames if args.frames is not None else 180,
                fps=args.fps if args.fps is not None else 24,
                orbit=args.orbit,
                zoom=args.quilt_zoom,
            )
        else:
            print(f"  mode     : {'headless PNG' if args.off_screen else 'interactive viewer'}")
            # No contact shadow here: it is an HLD device affordance that
            # reads correctly against that mode's white background, but on
            # the viewer's black background it renders as a bright halo.
            render_tvb_viewer(
                tvb_dataset=args.tvb_dataset,
                off_screen=args.off_screen,
                out_path=out_path,
                shadow=False,
                percentile=args.tvb_percentile,
                decimate=args.tvb_decimate,
                smooth_iters=args.tvb_smooth,
            )
        print("\nDone.")
        return

    # ---- CT / MRI biomedical demo (short-circuit — no ManifoldModel) ------
    if args.ct_demo:
        preset = CT_PRESETS[args.ct_dataset]
        isovalues = None
        if args.ct_isovalues:
            try:
                isovalues = [float(v) for v in args.ct_isovalues.split(",")]
            except ValueError:
                print(
                    f"ERROR: --ct-isovalues must be comma-separated numbers, got '{args.ct_isovalues}'"
                )
                sys.exit(1)
        isos_display = isovalues if isovalues is not None else preset["isovalues"]
        print("=" * 60)
        print("CT / MRI BIOMEDICAL DEMO")
        print(f"  dataset  : {args.ct_dataset}")
        print(f"  desc     : {preset['description']}")
        print(f"  isovalues: {isos_display}")
        print("=" * 60)

        if args.hld and args.quilt:
            print("ERROR: --hld and --quilt are mutually exclusive (different device families).")
            sys.exit(1)

        if args.hld and args.still:
            stem = out_path if out_path else Path(f"ct_{args.ct_dataset}")
            print(f"  mode     : HLD still → {stem}_hld.png")
            render_ct_still(
                ct_dataset=args.ct_dataset,
                out_path=stem,
                shadow=not args.no_shadow,
                isovalues=isovalues,
                zoom=args.zoom if args.zoom is not None else 1.8,
            )
        elif args.hld:
            stem = out_path if out_path else Path(f"ct_{args.ct_dataset}")
            print(f"  mode     : HLD video → {stem}_hld.mp4")
            print(f"  frames   : {args.frames or 300}  fps={args.fps or 30}  orbit={args.orbit}°")
            render_ct_hld(
                ct_dataset=args.ct_dataset,
                out_path=stem,
                n_frames=args.frames if args.frames is not None else 300,
                fps=args.fps if args.fps is not None else 30,
                orbit=args.orbit,
                shadow=not args.no_shadow,
                isovalues=isovalues,
                zoom=args.zoom if args.zoom is not None else 1.8,
            )
        elif args.quilt:
            stem = out_path if out_path else Path(f"ct_{args.ct_dataset}")
            quilt_grid = None
            if args.quilt_grid:
                try:
                    cols, rows = (int(v) for v in args.quilt_grid.lower().split("x"))
                    quilt_grid = (cols, rows)
                except ValueError:
                    print(f"ERROR: --quilt-grid must look like '11x6', got '{args.quilt_grid}'")
                    sys.exit(1)
            kind = "video" if args.quilt_video else "still"
            print(f"  mode     : LFD quilt {kind} ({args.quilt}) → {stem}_qs...")
            render_ct_quilt(
                ct_dataset=args.ct_dataset,
                out_path=stem,
                device=args.quilt,
                view_cone=args.view_cone,
                isovalues=isovalues,
                cast=args.cast,
                quilt_grid=quilt_grid,
                video=args.quilt_video,
                n_frames=args.frames if args.frames is not None else 180,
                fps=args.fps if args.fps is not None else 24,
                orbit=args.orbit,
                zoom=args.quilt_zoom,
            )
        else:
            print(f"  mode     : {'headless PNG' if args.off_screen else 'interactive viewer'}")
            render_ct_viewer(
                ct_dataset=args.ct_dataset,
                off_screen=args.off_screen,
                out_path=out_path,
                shadow=not args.no_shadow,
                isovalues=isovalues,
            )
        print("\nDone.")
        return

    # -----------------------------------------------------------------------

    try:
        pca_components = tuple(int(x) for x in args.pca_components.split(","))
        assert len(pca_components) == 3 and all(c >= 1 for c in pca_components)
    except (ValueError, AssertionError):
        print(
            f"ERROR: --pca-components must be 3 comma-separated positive integers, "
            f"got '{args.pca_components}'"
        )
        sys.exit(1)

    # Auto-suggest pre-pca for high-dimensional real datasets
    if args.pre_pca == 0 and args.dataset in {"mnist", "cifar10", "cifar100"}:
        args.pre_pca = 50
        print(
            f"[auto] --pre-pca set to {args.pre_pca} for {args.dataset} (override with --pre-pca N)"
        )

    pc_label = ",".join(str(c) for c in pca_components)
    print("=" * 60)
    print("MANIFOLD VOXEL VISUALIZER")
    pre = f"  pre-pca={args.pre_pca}D  " if args.pre_pca > 0 else "  "
    print(
        f"  dataset={args.dataset}  n={args.n_points}{pre}"
        f"res={args.resolution}\u00b3  PCs=[{pc_label}]"
    )
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Loading dataset ...")
    X, y = load_dataset(args)
    n_classes = len(np.unique(y))
    print(f"      X shape: {X.shape}   classes: {n_classes}")

    # 2. Pre-PCA + dimensionality discovery
    if args.pre_pca > 0 and X.shape[1] > args.pre_pca:
        print(f"\n[2/5] Pre-PCA: {X.shape[1]}D \u2192 {args.pre_pca}D ...")
        pre_reducer = PCA(n_components=args.pre_pca, random_state=42)
        X = pre_reducer.fit_transform(X).astype("d")
        ev_retained = pre_reducer.explained_variance_ratio_.sum()
        print(f"      Variance retained: {ev_retained:.1%}")
    else:
        print(f"\n[2/5] Pre-PCA: skipped (ambient dim = {X.shape[1]})")

    ambient = X.shape[1]
    n_disc = min(200, len(X))
    k_disc = min(args.k_pca, len(X) - 1)
    taus = (0.95, 0.90, 0.85, 0.80)
    print(f"\n      Discovering intrinsic dimensionality  (n_samples={n_disc}, k={k_disc}) ...")
    dim_report = discover_dimensionality(
        X,
        n_samples=n_disc,
        k=k_disc,
        variance_thresholds=taus,
    )

    print(f"\n      Ambient dim: {ambient}")
    for tau in taus:
        r = dim_report[tau]
        marker = " <--" if tau == args.tau else ""
        print(
            f"      \u03c4={tau:.2f}:  d* = {r['mean']:.1f} \u00b1 {r['std']:.1f}  "
            f"(median {r['median']:.0f}, range [{r['min']}, {r['max']}]){marker}"
        )

    ref = dim_report.get(args.tau, dim_report[0.90])
    noise_pct = 100.0 * (1.0 - ref["mean"] / ambient)
    print(
        f"      Noise reduction (\u03c4={args.tau}): {noise_pct:.1f}%  "
        f"({ambient}D \u2192 {ref['mean']:.1f}D)"
    )

    min_class_size = min(np.bincount(y)) if n_classes <= len(y) else 0
    if min_class_size >= 8:
        n_per = max(5, min(50 if n_classes <= 20 else 15, min_class_size))
        class_dims = discover_per_class_dimensionality(
            X,
            y,
            k=k_disc,
            tau=args.tau,
            n_samples_per_class=n_per,
        )
        class_means = [v["mean"] for v in class_dims.values()]
        if n_classes <= 20:
            print(f"      Per-class d* (\u03c4={args.tau}):")
            for c in sorted(class_dims):
                v = class_dims[c]
                print(f"        Class {c:>3d}:  d* = {v['mean']:.1f} \u00b1 {v['std']:.1f}")
        else:
            print(f"      Per-class d* ({n_classes} classes, \u03c4={args.tau}):")
            print(
                f"        mean = {np.mean(class_means):.1f}  "
                f"std = {np.std(class_means):.1f}  "
                f"range [{min(class_means):.1f}, {max(class_means):.1f}]"
            )
    else:
        print(f"      Per-class analysis: skipped (smallest class has {min_class_size} samples)")

    # 3. Fit + observe
    print("\n[3/5] Fitting ManifoldModel + ManifoldObserver ...")
    subject, observer, pf, pca_info = fit_and_observe(
        X,
        y,
        k_graph=args.k_graph,
        k_pca=args.k_pca,
        k_vote=args.k_vote,
        tau=args.tau,
        pre_pca=0,
        pca_components=pca_components,
    )
    print(f"      Field entries: {len(pf.curvature)}")
    print(f"      curvature  mean={pf.curvature.mean():.4f}  max={pf.curvature.max():.4f}")
    print(f"      height     mean={pf.height.mean():.4f}  max={pf.height.max():.4f}")
    print(f"      d*         mean={pf.intrinsic_dim.mean():.2f}")
    if pca_info is not None:
        evr = pca_info.explained_variance_ratio
        c = pca_info.components
        print(
            f"      3D PCA:  PC{c[0]}={evr[0]:.1%}  PC{c[1]}={evr[1]:.1%}  "
            f"PC{c[2]}={evr[2]:.1%}  "
            f"(total={pca_info.total_explained:.1%} of {pca_info.ambient_dim}D)"
        )

    # 4. Voxelize
    print(f"\n[4/5] Voxelizing to {args.resolution}\u00b3 grid ...")
    vox = voxelize(pf, resolution=args.resolution, padding=args.padding)
    total = args.resolution**3
    occupied = int((vox["density"] > 0).sum())
    print(f"      Occupied voxels: {occupied}/{total} ({100 * occupied / total:.1f}%)")

    # 5. Render
    print("\n[5/5] Rendering ...")
    grid = build_grid(vox)

    if args.hld and args.quilt:
        print("ERROR: --hld and --quilt are mutually exclusive (different device families).")
        sys.exit(1)

    if args.hld:
        stem = out_path if out_path else Path(f"manifold_{args.dataset}_{args.scalar}")
        render_hld_single(
            grid,
            pf,
            scalar=args.scalar,
            out_path=stem,
            show_points=not args.no_points,
            show_volume=args.volume,
            vol_opacity=args.vol_opacity,
            vol_threshold=args.vol_threshold,
            pca_info=pca_info,
            n_frames=args.frames if args.frames is not None else 300,
            fps=args.fps if args.fps is not None else 30,
            orbit=args.orbit,
            shadow=not args.no_shadow,
            zoom=args.zoom if args.zoom is not None else 1.0,
        )
    elif args.quilt:
        stem = out_path if out_path else Path(f"manifold_{args.dataset}_{args.scalar}")
        quilt_grid = None
        if args.quilt_grid:
            try:
                cols, rows = (int(v) for v in args.quilt_grid.lower().split("x"))
                quilt_grid = (cols, rows)
            except ValueError:
                print(f"ERROR: --quilt-grid must look like '11x6', got '{args.quilt_grid}'")
                sys.exit(1)
        render_quilt_single(
            grid,
            pf,
            scalar=args.scalar,
            out_path=stem,
            device=args.quilt,
            view_cone=args.view_cone,
            show_points=not args.no_points,
            show_volume=args.volume,
            vol_opacity=args.vol_opacity,
            vol_threshold=args.vol_threshold,
            pca_info=pca_info,
            cast=args.cast,
            quilt_grid=quilt_grid,
            video=args.quilt_video,
            n_frames=args.frames if args.frames is not None else 180,
            fps=args.fps if args.fps is not None else 24,
            orbit=args.orbit,
            zoom=args.quilt_zoom,
            scalar_bar=args.quilt_scalar_bar,
        )
    elif args.multi_scalar:
        render_multi(
            grid,
            pf,
            off_screen=args.off_screen,
            out_path=out_path,
            show_volume=args.volume,
            vol_opacity=args.vol_opacity,
            vol_threshold=args.vol_threshold,
            pca_info=pca_info,
        )
    else:
        render_single(
            grid,
            pf,
            scalar=args.scalar,
            off_screen=args.off_screen,
            out_path=out_path,
            show_points=not args.no_points,
            show_volume=args.volume,
            vol_opacity=args.vol_opacity,
            vol_threshold=args.vol_threshold,
            pca_info=pca_info,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
