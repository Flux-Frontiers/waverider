#!/usr/bin/env python3
"""
Manifold-Informed Architecture Benchmark: MNIST and sklearn Digits
===================================================================

Tests the hypothesis that MLP architectures whose bottleneck width
is set to the data's intrinsic dimensionality (d) outperform standard
fixed-width architectures — achieving comparable or higher accuracy
with fewer parameters.

Supports two datasets via ``--dataset {mnist,digits}``:
  - MNIST:  70 000 28×28 grayscale digit images (784-dimensional input)
  - Digits: sklearn's 8×8 toy digit dataset (64-dimensional input)

Three phases
------------
Phase 1 — Manifold Discovery
    Estimates intrinsic dimensionality via local PCA: for each of
    ``--discovery-samples`` randomly drawn points, the k-nearest
    neighbors (``--k-pca``) are gathered and a local covariance matrix
    is built. The number of eigenvalues needed to explain fraction τ
    (``--tau``) of local variance gives the local intrinsic dimension.
    Both global statistics and per-class statistics are reported.
    The bottleneck d is set to the maximum per-class intrinsic dim,
    ensuring the hardest digit manifold is fully accommodated.

Phase 2 — Architecture Comparison
    Five architectures are built and summarised:

    - Standard (128→64):            input → 128 → 64 → n_classes
    - Manifold (2d→d):              input → 2d  → d  → n_classes
    - Wide Manifold (4d→2d→d):      input → 4d  → 2d → d → n_classes
    - PCA→dD + MLP (2d→d):          PCA-projected input → 2d → d → n_classes
    - Intrinsic Dim (PCA→dD→output): PCA-projected input → d → n_classes

    The PCA models receive input already compressed to d dimensions by
    global sklearn PCA; the remaining models operate in raw pixel space.

Phase 3 — Training and Evaluation
    All architectures are trained with Adam for ``--epochs`` epochs
    across ``--trials`` independent random seeds.  Per-trial metrics
    include test accuracy, test loss, wall time, and the epoch at which
    95 % training accuracy was first reached.  Aggregate results are
    printed in a comparative table and saved to JSON alongside a
    four-panel matplotlib figure (validation accuracy curves, training
    loss curves, final test accuracy bar chart, parameter count chart).

Results are written next to the script as
``{mnist,digits}_architecture_results.{json,png}``.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
Last Revision: 2026-03-28

Usage
-----
    python benchmarks/canonical_tests/mnist_manifold_architecture.py
    python benchmarks/canonical_tests/mnist_manifold_architecture.py --dataset digits
    python benchmarks/canonical_tests/mnist_manifold_architecture.py --epochs 30 --trials 5 --tau 0.90
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Force CPU — Metal GPU per-op sync overhead dominates for small MLPs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

DEVICE_INFO = {
    "tensorflow_version": tf.__version__,
    "device_used": "CPU (forced)",
}
print(f"TensorFlow {tf.__version__} | Device: {DEVICE_INFO['device_used']}")

STRATEGY = tf.distribute.OneDeviceStrategy("/CPU:0")

import keras  # noqa: E402

# ---------------------------------------------------------------------------
# waverider import
# ---------------------------------------------------------------------------

from waverider.manifold_model import ManifoldModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)


# ---------------------------------------------------------------------------
# Phase 2: Model Builders
# ---------------------------------------------------------------------------


def _compile(model, lr):
    """Compile with logit-based cross-entropy and gradient clipping.

    Using from_logits=True fuses log-softmax into the loss — numerically
    stable on Metal GPU where the separate softmax→log path overflows float32.
    clipnorm=1.0 prevents gradient explosion from any remaining overflow.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_standard_model(input_dim, n_classes, lr=0.001):
    """Standard architecture: 784 → 128 → 64 → 10. Common default."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_classes),  # logits — from_logits=True in loss
        ]
    )
    return _compile(model, lr)


def build_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold-informed architecture: input → 2d → d → output.

    The bottleneck width matches the discovered intrinsic dimensionality.
    The layer before it is 2d to give the network room to learn the
    projection down to the manifold.
    """
    d = max(intrinsic_dim, n_classes)  # at least as wide as output
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ]
    )
    return _compile(model, lr)


def build_manifold_observer_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold-informed architecture: input → d + 1 → d + 1 → output.

    The bottleneck width matches the discovered intrinsic dimensionality.
    The layer before it is 2d to give the network room to learn the
    projection down to the manifold.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(intrinsic_dim + 1, activation="relu"),
            keras.layers.Dense(intrinsic_dim + 1, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ]
    )
    return _compile(model, lr)


def build_wide_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Wider manifold-informed: input → 4d → 2d → d → output.

    Three hidden layers with progressive compression toward the
    manifold bottleneck. More capacity for learning the projection.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(4 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ]
    )
    return _compile(model, lr)


def build_pca_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → 2d → d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    The network only needs to learn the nonlinear classification
    in the manifold subspace, not the projection itself.
    """
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ]
    )
    return _compile(model, lr)


def build_pca_intrinsic_dim_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected model: d → output.

    Input is already PCA-projected to intrinsic_dim dimensions.
    The network only needs to learn the nonlinear classification
    in the manifold subspace, not the projection itself.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(intrinsic_dim, activation="relu"),
            keras.layers.Dense(n_classes),  # logits
        ]
    )
    return _compile(model, lr)


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    """Train a model and return metrics."""
    with STRATEGY.scope():
        model = build_fn()
    n_params = count_params(model)

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    wall_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Convergence epoch (first epoch hitting 95% train accuracy)
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.95:
            conv_epoch = i
            break

    return {
        "trial": trial,
        "n_params": n_params,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "wall_time": wall_time,
        "convergence_epoch": conv_epoch,
        "train_acc": [float(a) for a in history.history["accuracy"]],
        "val_acc": [float(a) for a in history.history["val_accuracy"]],
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }


def run_trial_sklearn(build_fn, X_train, y_train, X_test, y_test, trial):
    """Train a sklearn/ManifoldModel classifier and return metrics."""
    clf = build_fn()
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    acc = float(clf.score(X_test, y_test))
    pred_time = time.perf_counter() - t1

    geometry = None
    if hasattr(clf, "geometry_summary"):
        geometry = clf.geometry_summary()

    return {
        "trial": trial,
        "n_params": 0,
        "test_loss": None,
        "test_acc": acc,
        "wall_time": fit_time + pred_time,
        "fit_time": fit_time,
        "pred_time": pred_time,
        "convergence_epoch": None,
        "geometry": geometry,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_arch_schematics(ax, arch_layers, colors):
    """Draw schematic network diagrams as a key panel.

    Each architecture is rendered as a column of rounded rectangles (layers)
    connected by arrows.  Box width is proportional to log(layer_size) so
    that the large-input bar does not dwarf the narrow bottleneck layers.

    :param ax: Matplotlib axes to draw into (axis is turned off).
    :param arch_layers: Dict mapping architecture name → list of layer sizes
        (input first, output last).
    :param colors: Dict mapping architecture name → colour string.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Network Architecture Schematics", fontsize=11, fontweight="bold")

    names = list(arch_layers.keys())
    n = len(names)
    col_w = 1.0 / n

    max_layers = max(len(v) for v in arch_layers.values())
    max_size = max(s for layers in arch_layers.values() for s in layers)

    box_h = 0.09
    y_top = 0.82
    y_bot = 0.08
    y_step = (y_top - y_bot) / max(max_layers - 1, 1)

    for i, name in enumerate(names):
        layers = arch_layers[name]
        color = colors.get(name, "gray")
        x_ctr = (i + 0.5) * col_w
        max_box_w = col_w * 0.82

        short = name.split("(")[0].strip()
        ax.text(
            x_ctr, 0.95, short, ha="center", va="top", fontsize=8, fontweight="bold", color=color
        )

        prev_y = None
        for j, size in enumerate(layers):
            yc = y_top - j * y_step
            w = max_box_w * math.log(size + 1) / math.log(max_size + 1)

            rect = FancyBboxPatch(
                (x_ctr - w / 2, yc - box_h / 2),
                w,
                box_h,
                boxstyle="round,pad=0.008",
                facecolor=color,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(rect)

            label = f"{size:,}" if size < 10000 else f"{size // 1000}K"
            ax.text(
                x_ctr,
                yc,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
                fontweight="bold",
            )

            if prev_y is not None:
                ax.annotate(
                    "",
                    xy=(x_ctr, yc + box_h / 2 + 0.005),
                    xytext=(x_ctr, prev_y - box_h / 2 - 0.005),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )
            prev_y = yc


def plot_results(all_results, intrinsic_dim, save_path, elapsed=None, input_dim=784, n_classes=10):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    d = intrinsic_dim
    colors = {
        "Standard (128→64)": "steelblue",
        f"Wide Manifold (4d→2d→d, d={d})": "forestgreen",
        f"Manifold (2d→d, d={d})": "firebrick",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
    }

    arch_layers = {
        "Standard (128→64)": [input_dim, 128, 64, n_classes],
        f"Wide Manifold (4d→2d→d, d={d})": [input_dim, 4 * d, 2 * d, d, n_classes],
        f"Manifold (2d→d, d={d})": [input_dim, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
    }

    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.38, wspace=0.3)
    ax_val = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_par = fig.add_subplot(gs[1, 1])
    ax_arch = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"MNIST: Manifold-Informed Architecture (d={d}) vs Standard{elapsed_str}",
        fontsize=14,
        fontweight="bold",
    )

    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short_names = [n.split("(")[0].strip() for n in names]

    # Validation accuracy curves
    for name, results in all_results.items():
        if not results or "val_acc" not in results[0]:
            continue
        accs = np.array([r["val_acc"] for r in results])
        ep = np.arange(1, accs.shape[1] + 1)
        color = colors.get(name, "gray")
        ax_val.plot(ep, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax_val.fill_between(
            ep, accs.mean(0) - accs.std(0), accs.mean(0) + accs.std(0), alpha=0.15, color=color
        )
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Accuracy")
    ax_val.set_title("Validation Accuracy")
    ax_val.legend().set_visible(False)
    ax_val.grid(True, alpha=0.3)

    # Training loss curves
    for name, results in all_results.items():
        if not results or "train_loss" not in results[0]:
            continue
        losses = np.array([r["train_loss"] for r in results])
        ep = np.arange(1, losses.shape[1] + 1)
        color = colors.get(name, "gray")
        ax_loss.plot(ep, losses.mean(0), "-", label=name, linewidth=2, color=color)
        ax_loss.fill_between(
            ep,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=color,
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)

    # Final test accuracy comparison
    bars = ax_acc.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Final Test Accuracy")
    ax_acc.grid(True, alpha=0.3, axis="y")
    ax_acc.set_ylim(0.85, 1.0)

    # Parameter efficiency
    param_counts = [all_results[n][0]["n_params"] for n in names]
    efficiencies = [m / p * 1000 if p > 0 else 0 for m, p in zip(means, param_counts)]
    bars = ax_par.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m, eff in zip(bars, param_counts, means, efficiencies):
        label = f"geometric\nacc={m:.4f}" if p == 0 else f"{p:,}\n{eff:.2f}acc/Kp"
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() * 1.3, 2),
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_par.set_ylabel("Parameters")
    ax_par.set_title("Parameter Count (lower is better at same accuracy)")
    ax_par.set_yscale("log")
    ax_par.grid(True, alpha=0.3, axis="y")

    # Architecture schematics panel
    _draw_arch_schematics(ax_arch, arch_layers, colors)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MNIST: Manifold-Informed Architecture vs Standard"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--tau", type=float, default=0.90, help="Variance threshold for intrinsic dim"
    )
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=500,
        help="Points to sample for dimensionality discovery",
    )
    parser.add_argument("--k-pca", type=int, default=50, help="Neighborhood size for local PCA")
    parser.add_argument(
        "--k-graph",
        type=int,
        default=15,
        help="Neighborhood size for ManifoldModel graph construction",
    )
    parser.add_argument("--k-vote", type=int, default=7, help="Voting neighbors for ManifoldModel")
    parser.add_argument(
        "--manifold-samples",
        type=int,
        default=5000,
        help="Max training samples for sklearn methods (ManifoldModel/KNN). "
        "ManifoldModel is O(n²) — keep this well below full MNIST.",
    )
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    if args.dataset == "mnist":
        print("\nLoading MNIST...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype("float32")
        X_test = X_test.reshape(-1, 784).astype("float32")
        # Normalize
        scaler = StandardScaler()
        # nan_to_num: constant pixels (std=0) produce NaN after scaling → zero them
        X_train = np.nan_to_num(scaler.fit_transform(X_train)).astype("float32")
        X_test = np.nan_to_num(scaler.transform(X_test)).astype("float32")
    else:
        print("\nLoading sklearn digits...")
        from sklearn.datasets import load_digits

        data = load_digits()
        X, y = data.data.astype("float32"), data.target
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = np.nan_to_num(scaler.fit_transform(X_train)).astype("float32")
        X_test = np.nan_to_num(scaler.transform(X_test)).astype("float32")

    input_dim = X_train.shape[1]
    n_classes = len(set(y_train))
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes}, Input dim: {input_dim}")

    # -----------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    print(f"\nSampling {args.discovery_samples} points, k={args.k_pca} neighbors...")
    t0 = time.perf_counter()
    dim_report = discover_dimensionality(
        X_train,
        n_samples=args.discovery_samples,
        k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    discovery_time = time.perf_counter() - t0
    print(f"Discovery time: {discovery_time:.1f}s\n")

    print(f"{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f} {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    # Per-class dimensionality
    print(f"\nPer-class intrinsic dimensionality (τ={args.tau}):")
    class_dims = discover_per_class_dimensionality(
        X_train, y_train, k=args.k_pca, tau=args.tau, n_samples_per_class=50
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        print(f"  Digit {c}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    # Use max of per-class maxima as the bottleneck — accommodates the hardest sample
    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    print(
        f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}"
        f"  →  using d = {intrinsic_dim} (τ={args.tau})"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    d = intrinsic_dim

    # PCA projection (needed for intrinsic-dim and PCA models)
    from sklearn.decomposition import PCA as skPCA

    # svd_solver='full' uses LAPACK GESDD and avoids the covariance_eigh path
    # that triggers spurious Apple BLAS float warnings on macOS.
    pca = skPCA(n_components=d, svd_solver="full")
    # cast to float64 for PCA matmul to avoid Apple BLAS float32 overflow
    X_train_pca = pca.fit_transform(X_train.astype(np.float64)).astype("float32")
    X_test_pca = pca.transform(X_test.astype(np.float64)).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    # Stratified subsample for O(n²) sklearn methods (ManifoldModel, KNN).
    n_sk = min(args.manifold_samples, len(X_train))
    if n_sk < len(X_train):
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_sk, random_state=42)
        sk_idx, _ = next(sss.split(X_train, y_train))
        X_sk, y_sk = X_train[sk_idx], y_train[sk_idx]
        print(f"  sklearn methods: stratified subsample {n_sk:,} / {len(X_train):,} train samples")
    else:
        X_sk, y_sk = X_train, y_train

    # Each entry: (build_fn, X_tr, y_tr, X_te, is_sklearn)
    architectures = {
        f"Euclidean KNN (k={args.k_vote})": (
            lambda: KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean"),
            X_sk,
            y_sk,
            X_test,
            True,
        ),
        f"ManifoldModel (τ={args.tau})": (
            lambda: ManifoldModel(
                k_graph=args.k_graph,
                k_pca=args.k_pca,
                k_vote=args.k_vote,
                variance_threshold=args.tau,
            ),
            X_sk,
            y_sk,
            X_test,
            True,
        ),
        "Standard (128→64)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            y_train,
            X_test,
            False,
        ),
        f"Wide Manifold (4d→2d→d, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            y_train,
            X_test,
            False,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            y_train,
            X_test,
            False,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            X_train_pca,
            y_train,
            X_test_pca,
            False,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca,
            y_train,
            X_test_pca,
            False,
        ),
    }

    # Show architecture details
    for name, entry in architectures.items():
        build_fn, is_sklearn = entry[0], entry[4]
        if is_sklearn:
            print(f"\n{name}:")
            print("  Parameters: 0 (non-parametric)")
            continue
        model = build_fn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")
        for layer in model.layers:
            if hasattr(layer, "units"):
                print(f"  {layer.name}: → {layer.units}")

    # -----------------------------------------------------------------------
    # Phase 3: Train and compare
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING")
    print("=" * 70)

    all_results = {}

    for name, entry in architectures.items():
        build_fn, X_tr, y_tr, X_te, is_sklearn = entry
        print(f"\n{name}")
        trial_results = []

        if is_sklearn:
            result = run_trial_sklearn(build_fn, X_tr, y_tr, X_te, y_test, trial=0)
            extra = ""
            if result.get("geometry"):
                extra = f"  id={result['geometry']['mean_intrinsic_dim']:.1f}"
            print(f"  acc={result['test_acc']:.4f}  time={result['wall_time']:.1f}s{extra}")
            trial_results.append(result)
        else:
            for trial in range(args.trials):
                np.random.seed(trial * 42)
                tf.random.set_seed(trial * 42)

                result = run_trial(
                    build_fn,
                    X_tr,
                    y_tr,
                    X_te,
                    y_test,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    trial=trial,
                )

                conv_str = (
                    f"conv@{result['convergence_epoch']}"
                    if result["convergence_epoch"] is not None
                    else "no conv"
                )
                print(
                    f"  Trial {trial + 1}/{args.trials}: "
                    f"acc={result['test_acc']:.4f}  "
                    f"loss={result['test_loss']:.4f}  "
                    f"{conv_str}  "
                    f"time={result['wall_time']:.1f}s"
                )
                trial_results.append(result)

        all_results[name] = trial_results

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"Dataset: {'MNIST' if args.dataset == 'mnist' else 'Digits'} "
        f"({input_dim}D, {n_classes} classes)"
    )
    print(f"Intrinsic dimensionality: d = {intrinsic_dim} (τ={args.tau})")
    print(f"Noise dimensions: {100 * (1 - intrinsic_dim / input_dim):.1f}%")
    print(f"Epochs: {args.epochs}, Trials: {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}")
    print("-" * 70)

    col_w = 22
    header = f"{'Metric':<25}"
    for name in all_results:
        short = name.split("(")[0].strip()
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * 70)

    for label, key, fmt in [
        ("Test Accuracy", "test_acc", ".4f"),
        ("Test Loss", "test_loss", ".4f"),
        ("Parameters", "n_params", ",d"),
        ("Wall Time (s)", "wall_time", ".1f"),
    ]:
        row = f"{label:<25}"
        for name, results in all_results.items():
            vals = [r[key] for r in results]
            if fmt == ",d":
                row += f"{vals[0]:>{col_w},}"
            else:
                valid = [v for v in vals if v is not None]
                if valid:
                    m, s = np.mean(valid), np.std(valid)
                    row += f"  {m:{fmt}} ± {s:{fmt}}  "
                else:
                    row += f"{'N/A':>{col_w}}"
        print(row)

    # Convergence
    row = f"{'Epochs to 95%':<25}"
    for name, results in all_results.items():
        convs = [r["convergence_epoch"] for r in results if r["convergence_epoch"] is not None]
        if convs:
            row += f"  {np.mean(convs):.1f} ± {np.std(convs):.1f} ({len(convs)}/{len(results)})  "
        else:
            row += f"{'N/A':>{col_w}}"
    print(row)

    # ManifoldModel geometry footnote
    print("-" * 70)
    print("MANIFOLD GEOMETRY (ManifoldModel):")
    for name, results in all_results.items():
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in results if r.get("geometry")]
            if geoms:
                mean_ids = [g["mean_intrinsic_dim"] for g in geoms if g]
                ambient = geoms[0].get("ambient_dim", input_dim)
                noise_pct = 100 * (1 - np.mean(mean_ids) / ambient)
                print(
                    f"  {name}: mean intrinsic dim = {np.mean(mean_ids):.1f}"
                    f" / {ambient}  ({noise_pct:.0f}% noise suppressed)"
                )

    # Parameter efficiency
    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        n_params = results[0]["n_params"]
        if n_params == 0:
            continue
        mean_acc = np.mean([r["test_acc"] for r in results])
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner
    print("-" * 70)
    best_name = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_acc = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_name = "Standard (128→64)"
    std_acc = np.mean([r["test_acc"] for r in all_results[std_name]])

    if best_name != std_name:
        delta = best_acc - std_acc
        print(f">> WINNER: {best_name}")
        print(f"   {best_acc:.4f} vs {std_acc:.4f} (standard)")
        print(f"   Delta: +{delta:.4f} ({delta * 100:.2f} pp)")

        best_params = all_results[best_name][0]["n_params"]
        std_params = all_results[std_name][0]["n_params"]
        if best_params == 0:
            print("   Uses ZERO learned parameters — pure manifold geometry")
        elif best_params < std_params:
            reduction = 100 * (1 - best_params / std_params)
            print(f"   With {reduction:.0f}% FEWER parameters ({best_params:,} vs {std_params:,})")
        elif best_params > std_params:
            increase = 100 * (best_params / std_params - 1)
            print(f"   With {increase:.0f}% more parameters ({best_params:,} vs {std_params:,})")
    else:
        print(f">> Standard architecture wins: {std_acc:.4f}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "intrinsic_dim": intrinsic_dim,
        "global_intrinsic_dim_mean": global_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "k_pca": args.k_pca,
        "k_graph": args.k_graph,
        "k_vote": args.k_vote,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = (
        Path(__file__).resolve().parent
        / f"{'mnist' if args.dataset == 'mnist' else 'digits'}_architecture_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(
            Path(__file__).resolve().parent
            / f"{'mnist' if args.dataset == 'mnist' else 'digits'}_architecture_results.png"
        )
        plot_results(
            all_results,
            intrinsic_dim,
            plot_path,
            elapsed=time.perf_counter() - t_start,
            input_dim=input_dim,
            n_classes=n_classes,
        )


if __name__ == "__main__":
    main()
