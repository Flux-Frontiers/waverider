#!/usr/bin/env python3
"""
Manifold Voxel Visualizer
=========================

Projects a fitted ManifoldModel + ManifoldObserver into a 3-D PCA subspace,
rasterizes the observer's geometric fields (density, curvature, height,
intrinsic dimensionality, class vote) into a uniform voxel grid, and opens
an interactive PyVista viewer where you can slice, clip, and volume-render
the manifold from any angle.

Pipeline
--------
1. Load or generate embeddings + integer labels.
2. Optional: pre-reduce with PCA to ``--pre-pca`` dims before ManifoldModel
   (recommended for high-dimensional datasets like MNIST/CIFAR).
3. Discover intrinsic dimensionality via local PCA (thin SVD) at multiple
   variance thresholds, with per-class analysis.
4. Fit a :class:`ManifoldModel`, then wrap it in a :class:`ManifoldObserver`
   and call ``observe()`` to populate the geometric field.
5. Project training points to 3-D via PCA for visualization, annotating
   axes with explained-variance ratios and rendering direction arrows.
6. Rasterize each scalar field (density / curvature / height / d* / class)
   onto a uniform ``pv.ImageData`` voxel grid using ``numpy.histogramdd``
   accumulation.
7. Launch a PyVista plotter with an orthogonal-slice widget and optional
   scatter overlay of the raw training points.

Supported datasets
------------------
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
    load          --X-file X.npy  [--y-file y.npy]

Usage
-----
    # Synthetic helix (default)
    python manifold_voxel_viz.py

    # Real: Iris dataset
    python manifold_voxel_viz.py --dataset iris

    # Real: sklearn Digits, curvature field, 2×2 panel
    python manifold_voxel_viz.py --dataset digits --multi-scalar

    # Real: MNIST — subsample 1 500 pts, pre-reduce to 50 D
    python manifold_voxel_viz.py --dataset mnist --n-points 1500 --pre-pca 50

    # Real: CIFAR-10 — subsample 1 000 pts, pre-reduce to 40 D
    python manifold_voxel_viz.py --dataset cifar10 --n-points 1000 --pre-pca 40

    # Real: CIFAR-100 — subsample 1 000 pts, pre-reduce to 50 D
    python manifold_voxel_viz.py --dataset cifar100 --n-points 1000 --pre-pca 50

    # Custom .npy files
    python manifold_voxel_viz.py --dataset load --X-file X.npy --y-file y.npy

    # Headless PNG export
    python manifold_voxel_viz.py --dataset iris --off-screen --out iris_voxels.png

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyvista as pv
import scipy.ndimage
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
    """Stratified subsample to at most n points."""
    if len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    # Stratified: keep class proportions
    classes, counts = np.unique(y, return_counts=True)
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
    """Return (X, y) based on CLI args."""
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
    """Per-point geometric scalars extracted from a ManifoldObserver field."""

    X3: np.ndarray  # (n, 3) PCA projection
    density_w: np.ndarray  # (n,)  weight = 1 (for histogramdd)
    curvature: np.ndarray  # (n,)  scalar curvature
    height: np.ndarray  # (n,)  height above tangent plane
    intrinsic_dim: np.ndarray  # (n,)  local d*
    labels: np.ndarray  # (n,)  integer class labels


class PCAInfo(NamedTuple):
    """Metadata from the 3-D PCA projection for axis annotation."""

    explained_variance_ratio: np.ndarray  # (3,) per-component ratio
    total_explained: float  # sum of the 3 ratios
    ambient_dim: int  # dimensionality before projection
    components: tuple[int, int, int]  # 1-based PC indices shown (e.g. (1,2,3))


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
    """Fit ManifoldModel, run ManifoldObserver, project to 3-D PCA.

    :param X: Training embeddings, shape (n, d).
    :param y: Integer labels, shape (n,).
    :param k_graph: KNN graph degree.
    :param k_pca: Neighbours used for local PCA.
    :param k_vote: Neighbours used for classification vote.
    :param tau: Variance threshold for intrinsic dim selection.
    :param pre_pca: If > 0, reduce X to this many dims via PCA before fitting
        ManifoldModel.  Recommended for high-dimensional data (MNIST, CIFAR).
    :param pca_components: Which 3 principal components to project onto,
        as 1-based indices.  Default ``(1, 2, 3)`` selects the three highest-
        variance axes.  Use e.g. ``(4, 5, 6)`` to explore deeper subspaces.
    :return: Tuple of (subject, observer, PointField, PCAInfo | None).
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

    # Extract per-node arrays in graph order
    curvatures = np.array([o.curvature for o in field], dtype="d")
    heights = np.array([o.height for o in field], dtype="d")
    idims = np.array([o.intrinsic_dim for o in field], dtype="d")

    # Node labels from subject geometry store (same order as field)
    node_labels = np.array([(subject._geometries[o.node_id].label or 0) for o in field], dtype=int)

    # Project to 3-D via PCA (selectable components)
    pca_info = None
    if X.shape[1] > 3:
        # Compute enough components to cover the requested indices
        n_comp = min(max(pca_components), X.shape[1], len(X))
        pc_label = ",".join(str(c) for c in pca_components)
        print(f"  PCA projection to 3-D  (PC{pc_label}) ...", flush=True)
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X).astype("d")
        # Select the 3 requested components (convert 1-based to 0-based)
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

    :param pf: :class:`PointField` from :func:`fit_and_observe`.
    :param resolution: Number of voxels along each axis.
    :param padding: Fractional padding beyond bounding box (e.g., 0.05 = 5%).
    :return: Dict with keys ``origin``, ``spacing``, ``dims``, and one
        key per scalar field, each a (resolution**3,) float32 array.
    """
    res = resolution
    X3 = pf.X3

    lo = X3.min(axis=0)
    hi = X3.max(axis=0)
    span = hi - lo
    lo -= span * padding
    hi += span * padding
    spacing = (hi - lo) / res

    # Bin indices per point: shape (n, 3), values in [0, res-1]
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
        out = np.where(count > 0, s / np.maximum(count, 1), 0.0).astype("f4")
        return out

    def _majority_vote(labels: np.ndarray) -> np.ndarray:
        n_classes = int(labels.max()) + 1
        votes = np.zeros((total_cells, n_classes), dtype="f4")
        for c in range(n_classes):
            mask = labels == c
            if mask.any():
                votes[:, c] = np.bincount(flat_idx[mask], minlength=total_cells)
        return votes.argmax(axis=1).astype("f4")

    # Smooth density with a simple 3-D Gaussian blur
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
# PyVista rendering
# ---------------------------------------------------------------------------


def build_grid(vox: dict):
    """Construct a ``pv.ImageData`` from a voxelization dict."""
    res = vox["dims"][0]
    grid = pv.ImageData()
    grid.dimensions = (res, res, res)  # point data → res^3 points
    grid.origin = vox["origin"].tolist()
    grid.spacing = vox["spacing"].tolist()

    for key in ("density", "curvature", "height", "intrinsic_dim", "class_vote"):
        arr = vox[key]
        # VTK expects Fortran (column-major) order for ImageData
        grid.point_data[key] = arr.reshape(res, res, res).ravel(order="F")

    return grid


CMAP_MAP = {
    "density": "plasma",
    "curvature": "coolwarm",
    "height": "viridis",
    "intrinsic_dim": "tab10",
    "class_vote": "Set1",
}


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
    colors = ["#e74c3c", "#2ecc71", "#3498db"]  # red, green, blue

    for i in range(3):
        direction = np.zeros(3, dtype="f4")
        direction[i] = 1.0
        # Scale arrow length by sqrt(variance ratio) — compresses dynamic range
        # while preserving order; floor at 0.30 so no axis gets lost.
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
        widget in the lower-left corner.  Clicking a face jumps to that
        standard view (top / front / right / isometric).
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

    Thresholds the density field at ``threshold_frac`` × max-density, then
    renders the surviving voxel cells coloured by ``scalar`` at low opacity.
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

    :param show_volume: If ``True``, render the full voxel cloud behind the slices.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as a fraction of max (filters empty voxels).
    :param pca_info: If provided, label axes with PCA variance and add direction arrows.
    """
    p = pv.Plotter(off_screen=off_screen, title=f"Manifold Voxels — {scalar}")

    if show_volume:
        _add_voxel_cloud(p, grid, scalar, opacity=vol_opacity, threshold_frac=vol_threshold)

    p.add_mesh_slice_orthogonal(
        grid,
        scalars=scalar,
        cmap=CMAP_MAP.get(scalar, "plasma"),
        show_scalar_bar=True,
    )

    if show_points:
        cloud = pv.PolyData(pf.X3.astype("f4"))
        cloud.point_data["label"] = pf.labels.astype("f4")
        p.add_points(
            cloud, scalars="label", cmap="Set1", point_size=8, opacity=0.7, show_scalar_bar=False
        )

    _add_pca_axes(p, pca_info)
    _add_pca_arrows(p, pf, pca_info)
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
) -> None:
    """2×2 panel: density / curvature / height / class_vote.

    :param show_volume: If ``True``, render the full voxel cloud in each panel.
    :param vol_opacity: Opacity of the voxel cloud (0–1).
    :param vol_threshold: Density threshold as fraction of max.
    :param pca_info: If provided, label axes with PCA variance and add direction arrows.
    """
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manifold voxel visualizer — project, rasterize, slice."
    )

    # Dataset
    p.add_argument(
        "--dataset",
        choices=[
            "helix",
            "swiss_roll",
            "torus",  # synthetic
            "iris",
            "wine",
            "breast_cancer",
            "digits",  # real (sklearn)
            "mnist",
            "cifar10",
            "cifar100",  # real (keras)
            "load",  # custom .npy
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
        "(0 = disabled). Recommended: 40-50 for MNIST/CIFAR).",
    )
    p.add_argument(
        "--pca-components",
        type=str,
        default="1,2,3",
        help="Which 3 principal components to visualize, as comma-separated "
        "1-based indices (default: '1,2,3' = three highest-variance axes). "
        "Use e.g. '4,5,6' to explore deeper manifold subspaces.",
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
    # Voxel cloud
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
        help="Density threshold as fraction of max to include in cloud "
        "(default 0.04 — filters near-empty voxels).",
    )
    p.add_argument(
        "--off-screen",
        action="store_true",
        help="Render headless and write a PNG instead of opening a window.",
    )
    p.add_argument("--out", default=None, help="Output PNG path (implies --off-screen).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.out:
        args.off_screen = True

    out_path = Path(args.out) if args.out else None

    # Parse PCA component selection
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
    #    Apply pre-PCA first so discovery runs in the fitting space.
    if args.pre_pca > 0 and X.shape[1] > args.pre_pca:
        print(f"\n[2/5] Pre-PCA: {X.shape[1]}D \u2192 {args.pre_pca}D ...")
        pre_reducer = PCA(n_components=args.pre_pca, random_state=42)
        X = pre_reducer.fit_transform(X).astype("d")
        ev_retained = pre_reducer.explained_variance_ratio_.sum()
        print(f"      Variance retained: {ev_retained:.1%}")
    else:
        print(f"\n[2/5] Pre-PCA: skipped (ambient dim = {X.shape[1]})")

    # Intrinsic dimensionality discovery via local PCA (thin SVD)
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

    # Per-class dimensionality discovery
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

    # 3. Fit + observe  (pre_pca=0 because we already reduced above)
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
            f"      3D PCA:  PC{c[0]}={evr[0]:.1%}  PC{c[1]}={evr[1]:.1%}  PC{c[2]}={evr[2]:.1%}  "
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

    if args.multi_scalar:
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
