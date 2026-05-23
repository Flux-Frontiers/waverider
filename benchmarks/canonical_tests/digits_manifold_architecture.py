#!/usr/bin/env python3
"""
Digits Benchmark: Manifold-Informed Architecture — Comprehensive Comparison
============================================================================

Benchmarks ALL approaches on sklearn digits — the canonical small dataset
where manifold geometry is completely transparent:

  sklearn digits: 1,797 samples, 64-dimensional (8×8 pixels), 10 classes.

Because the dataset is small enough for complete evaluation, all methods
are assessed with 5-fold stratified cross-validation, giving statistically
robust results without subsampling.

Seven approaches compared
--------------------------
  1. Standard MLP (1024→512)          — fixed-width baseline neural network
  2. Manifold MLP (2d→d)              — bottleneck at intrinsic dimensionality d
  3. Manifold + ManifoldAdam (2d→d)   — manifold bottleneck + projected gradient
  4. Wide Manifold MLP (4d→2d→d)      — progressive compression to manifold dim
  5. PCA→dD + MLP (2d→d)              — global PCA projection + nonlinear head
  6. Intrinsic Dim (PCA→dD→output)    — PCA projection + minimal head
  7. ManifoldModel (τ=0.90)           — zero learned parameters, pure geometry
  8. Euclidean KNN (k=7)              — classic Euclidean baseline

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples points (k=--k-pca neighbors each).
    Global and per-class intrinsic dimensionality reported.
    Bottleneck d = max per-class max intrinsic dim at τ=--tau.

Phase 2 — Architecture Summary
    All architectures are described with parameter counts.

Phase 3 — 5-Fold CV
    All methods run on the same folds.  Aggregate mean ± std accuracy,
    timing, and geometry stats are printed and saved to
    ``digits_manifold_architecture_results.json``.  A five-panel
    matplotlib figure is saved alongside (``digits_manifold_architecture_results.png``).

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
Last Revision: 2026-03-30

Usage
-----
    python benchmarks/canonical_tests/digits_manifold_architecture.py
    python benchmarks/canonical_tests/digits_manifold_architecture.py --epochs 50 --trials 3 --tau 0.85
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow()
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.decomposition import PCA as skPCA  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ---------------------------------------------------------------------------
# waverider imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model_builder import (  # noqa: E402
    build_manifold_model,
    build_pca_intrinsic_dim_model,
    build_pca_model,
    build_standard_model,
    build_wide_manifold_model,
)
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)
from waverider.manifold_model import ManifoldModel  # noqa: E402
from waverider.manifold_optimizer import ManifoldAdam, make_basis  # noqa: E402


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Phase 3: Fold Runner (handles both Keras and sklearn)
# ---------------------------------------------------------------------------


def run_fold(
    build_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    batch_size,
    fold,
    is_sklearn=False,
):
    """Run one CV fold — works for both Keras models and sklearn classifiers.

    :param build_fn: Zero-arg callable that returns a fresh model or classifier.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param epochs: Training epochs (Keras only).
    :param batch_size: Batch size (Keras only).
    :param fold: Fold index (0-based), recorded in result.
    :param is_sklearn: If True, treat as a sklearn estimator.
    :returns: Dict of per-fold metrics.
    """
    if is_sklearn:
        clf = build_fn()
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        acc = float(clf.score(X_test, y_test))
        pred_time = time.perf_counter() - t1
        # geometry stats for ManifoldModel
        geometry = None
        if hasattr(clf, "geometry_summary"):
            geometry = clf.geometry_summary()
        return {
            "fold": fold,
            "n_params": 0,
            "test_loss": None,
            "test_acc": acc,
            "wall_time": fit_time + pred_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "convergence_epoch": None,
            "geometry": geometry,
        }
    else:
        # Keras model
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
        conv_epoch = None
        for i, a in enumerate(history.history["accuracy"]):
            if a >= 0.95:
                conv_epoch = i
                break
        return {
            "fold": fold,
            "n_params": n_params,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "wall_time": wall_time,
            "fit_time": wall_time,
            "pred_time": None,
            "convergence_epoch": conv_epoch,
            "geometry": None,
            "train_acc": [float(a) for a in history.history["accuracy"]],
            "val_acc": [float(a) for a in history.history["val_accuracy"]],
            "train_loss": [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
        }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_arch_schematics(ax, arch_layers, colors):
    """Draw schematic network diagrams as a key panel.

    Each architecture is rendered as a column of rounded rectangles (layers)
    connected by arrows.  Box width is proportional to log(layer_size) so
    that the input bar does not dwarf the narrow bottleneck layers.

    :param ax: Matplotlib axes to draw into (axis is turned off).
    :param arch_layers: Dict mapping architecture name → list of layer sizes.
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
            x_ctr,
            0.95,
            short,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color=color,
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
                fontsize=6,
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


def plot_results(all_results, intrinsic_dim, save_path, elapsed=None, input_dim=64, n_classes=10):
    """Save a five-panel comparison figure with architecture schematics key.

    :param all_results: Dict mapping architecture name → list of fold result dicts.
    :param intrinsic_dim: Discovered bottleneck dimension d.
    :param save_path: Filesystem path for the PNG output.
    :param elapsed: Optional total wall time in seconds (for figure title).
    :param input_dim: Raw input dimensionality (default 64 for digits).
    :param n_classes: Number of output classes (default 10 for digits).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    d = intrinsic_dim
    colors = {
        "Standard (1024→512)": "steelblue",
        f"Wide Manifold (4d→2d→d, d={d})": "forestgreen",
        f"Manifold (2d→d, d={d})": "firebrick",
        f"Manifold + ManifoldAdam (d={d})": "darkred",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
        "ManifoldModel (τ=0.9)": "teal",
        "ManifoldModel": "teal",
        "Euclidean KNN (k=7)": "saddlebrown",
        "Euclidean KNN": "saddlebrown",
    }
    # Also map by exact key from all_results
    for name in all_results:
        if name not in colors:
            if "ManifoldModel" in name:
                colors[name] = "teal"
            elif "KNN" in name:
                colors[name] = "saddlebrown"

    arch_layers = {
        "Standard (1024→512)": [input_dim, 1024, 512, n_classes],
        f"Wide Manifold (4d→2d→d, d={d})": [input_dim, 4 * d, 2 * d, d, n_classes],
        f"Manifold (2d→d, d={d})": [input_dim, 2 * d, d, n_classes],
        f"Manifold + ManifoldAdam (d={d})": [input_dim, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
        # ManifoldModel and KNN are non-parametric — excluded from schematics
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.38, wspace=0.3)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_par = fig.add_subplot(gs[1, 1])
    ax_arch = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"Digits: Manifold Architecture Comparison (d={d})\n"
        f"64D digits → manifold discovery → architecture"
        f"  |  10 classes{elapsed_str}  |  {timestamp}",
        fontsize=14,
        fontweight="bold",
    )

    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short_names = [n.split("(")[0].strip() for n in names]

    # --- Top-left: Per-fold accuracy boxplot (digits uses 5-fold CV) ---
    keras_names = [n for n in names if any(r.get("train_acc") for r in all_results[n])]
    per_fold_accs = []
    box_labels = []
    box_colors_list = []
    for name in names:
        accs = [r["test_acc"] for r in all_results[name]]
        per_fold_accs.append(accs)
        short = name.split("(")[0].strip()
        box_labels.append(short)
        box_colors_list.append(colors.get(name, "gray"))

    bp = ax_box.boxplot(per_fold_accs, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], box_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_box.set_xticks(range(1, len(names) + 1))
    ax_box.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=8)
    ax_box.set_ylabel("CV Fold Accuracy")
    ax_box.set_title("Per-Fold Accuracy Distribution (5-Fold CV)")
    ax_box.grid(True, alpha=0.3, axis="y")

    # --- Top-right: Training loss curves (Keras methods only) ---
    for name in keras_names:
        results = all_results[name]
        loss_curves = [r["train_loss"] for r in results if r.get("train_loss")]
        if not loss_curves:
            continue
        max_len = max(len(c) for c in loss_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in loss_curves]
        losses = np.array(padded)
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
    ax_loss.set_title("Training Loss (Keras models, mean ± std across folds×trials)")
    ax_loss.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)

    # --- Bottom-left: Final test accuracy bars ---
    bars = ax_acc.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    ax_acc.set_ylabel("Mean CV Test Accuracy")
    ax_acc.set_title("Final Test Accuracy (mean ± std)")
    ax_acc.set_ylim(0, float(max(means)) * 1.25)
    ax_acc.tick_params(axis="x", labelsize=7, rotation=30)
    ax_acc.grid(True, alpha=0.3, axis="y")

    # --- Bottom-right: Parameter count bars ---
    param_counts = []
    for name in names:
        n_params = all_results[name][0]["n_params"]
        param_counts.append(max(n_params, 1))  # avoid log(0)
    bars_p = ax_par.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, name, p in zip(bars_p, names, param_counts):
        actual = all_results[name][0]["n_params"]
        label = "geometric" if actual == 0 else f"{actual:,}"
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_par.set_ylabel("Parameters")
    ax_par.set_title("Parameter Count (log scale)\nsklearn methods: non-parametric")
    ax_par.set_yscale("log")
    ax_par.tick_params(axis="x", labelsize=7, rotation=30)
    ax_par.grid(True, alpha=0.3, axis="y")

    # --- Architecture schematics (Keras models only) ---
    _draw_arch_schematics(ax_arch, arch_layers, colors)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Digits: Manifold-Informed Architecture — Comprehensive Comparison"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for Keras models")
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Random-seed trials per fold for Keras models",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (small — dataset is tiny)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.90,
        help="Variance threshold τ for intrinsic dimensionality",
    )
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=200,
        help="Points to sample for dimensionality discovery (out of 1797)",
    )
    parser.add_argument("--k-pca", type=int, default=30, help="Neighborhood size for local PCA")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help="Samples per class for per-class dimensionality",
    )
    parser.add_argument(
        "--k-graph",
        type=int,
        default=15,
        help="Neighborhood size for ManifoldModel graph construction",
    )
    parser.add_argument("--k-vote", type=int, default=7, help="Voting neighbors for ManifoldModel")
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load and scale data
    # -----------------------------------------------------------------------

    print("\nLoading sklearn digits...")
    data = load_digits()
    X, y = data.data.astype("float32"), data.target

    scaler = StandardScaler()
    # nan_to_num: constant pixels (std=0) produce NaN after scaling → zero them
    X = np.nan_to_num(scaler.fit_transform(X)).astype("float32")

    input_dim = X.shape[1]
    n_classes = len(set(y))

    print(f"  Dataset: {X.shape[0]} samples, {input_dim} dims, {n_classes} classes")
    print("  Evaluation: 5-fold stratified cross-validation")

    # -----------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    print(f"\nSampling {args.discovery_samples} points, k={args.k_pca} neighbors...")
    print("(Using SVD-based local PCA — efficient for high-dimensional data)")
    t0 = time.perf_counter()
    dim_report = discover_dimensionality(
        X,
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
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    # Per-class dimensionality
    print(
        f"\nPer-class intrinsic dimensionality (τ={args.tau}, {args.samples_per_class} samples/class):"
    )
    class_dims = discover_per_class_dimensionality(
        X, y, k=args.k_pca, tau=args.tau, n_samples_per_class=args.samples_per_class
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        print(f"  Digit {c}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    # Bottleneck = max of per-class maxima — accommodates the hardest manifold
    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = intrinsic_dim
    print(
        f"\n>> Global intrinsic dim (mean): {global_dim}"
        f"  |  Max per-class max: {intrinsic_dim}"
        f"  →  using d = {d} (τ={args.tau})"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Architecture summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    # PCA projection (needed for make_basis and var_explained reporting;
    # fold-level PCA is re-fit per fold to avoid data leakage)
    pca = skPCA(n_components=d)
    pca.fit(X)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    V_d = make_basis(pca)  # (input_dim, d) — top-d principal axes

    # Build one instance of each Keras architecture to print param counts
    _sample_builds = {
        "Standard (1024→512)": lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
        f"Wide Manifold (4d→2d→d, d={d})": lambda: build_wide_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"Manifold (2d→d, d={d})": lambda: build_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"Manifold + ManifoldAdam (d={d})": lambda: build_manifold_model(
            input_dim,
            n_classes,
            d,
            lr=args.lr,
            optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
        ),
        f"PCA→{d}D + MLP (2d→d)": lambda: build_pca_model(n_classes, d, lr=args.lr),
        f"Intrinsic Dim (PCA→{d}D→output)": lambda: build_pca_intrinsic_dim_model(
            n_classes, d, lr=args.lr
        ),
    }

    for name, bfn in _sample_builds.items():
        model = bfn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")
        for layer in model.layers:
            if hasattr(layer, "units"):
                print(f"  {layer.name}: → {layer.units}")

    print(f"\nManifoldModel (τ={args.tau}): 0 learned parameters")
    print("  The manifold IS the model. Pure geometry.")
    print(f"\nEuclidean KNN (k={args.k_vote}): 0 learned parameters")
    print("  Classic non-parametric baseline.")

    # -----------------------------------------------------------------------
    # Phase 3: 5-Fold CV
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Folds: 5  |  Keras trials per fold: {args.trials}  |  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}  |  τ: {args.tau}")

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Build the architectures dict now that we know d.
    # Each entry: (build_fn, is_sklearn, needs_pca)
    # needs_pca=True means the fold's training data must be PCA-projected.
    architectures = {
        f"Euclidean KNN (k={args.k_vote})": (
            lambda: KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean"),
            True,
            False,
        ),
        f"ManifoldModel (τ={args.tau})": (
            lambda: ManifoldModel(
                k_graph=args.k_graph,
                k_pca=args.k_pca,
                k_vote=args.k_vote,
                variance_threshold=args.tau,
            ),
            True,
            False,
        ),
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            False,
            False,
        ),
        f"Wide Manifold (4d→2d→d, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"Manifold + ManifoldAdam (d={d})": (
            lambda: build_manifold_model(
                input_dim,
                n_classes,
                d,
                lr=args.lr,
                optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
            ),
            False,
            False,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
    }

    all_results = {}

    fold_splits = list(skf.split(X, y))

    for name, (build_fn, is_sklearn, needs_pca) in architectures.items():
        print(f"\n{name}...")
        fold_results = []

        for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # PCA is fit on the training fold only (no data leakage)
            if needs_pca:
                pca_fold = skPCA(n_components=d)
                # cast to float64 for PCA matmul to avoid Apple BLAS float32 overflow
                X_tr_use = pca_fold.fit_transform(X_tr.astype(np.float64)).astype("float32")
                X_te_use = pca_fold.transform(X_te.astype(np.float64)).astype("float32")
            else:
                X_tr_use, X_te_use = X_tr, X_te

            if is_sklearn:
                result = run_fold(
                    build_fn,
                    X_tr_use,
                    y_tr,
                    X_te_use,
                    y_te,
                    args.epochs,
                    args.batch_size,
                    fold_i,
                    is_sklearn=True,
                )
                fold_results.append(result)
            else:
                for trial in range(args.trials):
                    np.random.seed(trial * 100 + fold_i)
                    tf.random.set_seed(trial * 100 + fold_i)
                    result = run_fold(
                        build_fn,
                        X_tr_use,
                        y_tr,
                        X_te_use,
                        y_te,
                        args.epochs,
                        args.batch_size,
                        fold_i,
                        is_sklearn=False,
                    )
                    fold_results.append(result)

        all_results[name] = fold_results

        # Single summary line per method
        mean_acc = np.mean([r["test_acc"] for r in fold_results])
        std_acc = np.std([r["test_acc"] for r in fold_results])
        total_time = sum(r["wall_time"] for r in fold_results)
        extra = ""
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in fold_results if r.get("geometry")]
            if geoms:
                mean_id = np.mean([g["mean_intrinsic_dim"] for g in geoms if g])
                extra = f"  id={mean_id:.1f}"
        print(f"  {mean_acc:.4f} ± {std_acc:.4f}  ({total_time:.1f}s total{extra})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — 5-FOLD STRATIFIED CV")
    print("=" * 70)
    print(f"Dataset: sklearn digits ({input_dim}D, {n_classes} classes, 1797 samples)")
    print(f"Intrinsic dimensionality: d = {intrinsic_dim} (τ={args.tau})")
    print(
        f"Noise dimensions: {100 * (1 - intrinsic_dim / input_dim):.1f}%"
        f"  ({input_dim - intrinsic_dim} of {input_dim} dims suppressed)"
    )
    print(f"Epochs: {args.epochs}  |  Trials/fold (Keras): {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}  |  Total time: {elapsed:.1f}s")
    print("-" * 70)

    # Determine best mean accuracy for marker
    best_acc = max(np.mean([r["test_acc"] for r in all_results[n]]) for n in all_results)

    print(f"\n{'Architecture':<40} {'Mean Acc':>10} {'Std':>8} {'Params':>10} {'Time (s)':>10}")
    print("-" * 80)

    for name in all_results:
        results = all_results[name]
        accs = [r["test_acc"] for r in results]
        times = [r["wall_time"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_time = np.mean(times)
        n_params = results[0]["n_params"]
        params_str = "non-param" if n_params == 0 else f"{n_params:,}"
        marker = "  << BEST" if abs(mean_acc - best_acc) < 1e-9 else ""
        print(
            f"{name:<40} {mean_acc:>10.4f} {std_acc:>8.4f}"
            f" {params_str:>10} {mean_time:>9.2f}s{marker}"
        )

    # ManifoldModel geometry footnote
    print("\n" + "-" * 70)
    print("MANIFOLD GEOMETRY (ManifoldModel folds)")
    print("-" * 70)
    for name in all_results:
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in all_results[name] if r.get("geometry")]
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
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters, Keras models only):")
    for name in all_results:
        results = all_results[name]
        n_params = results[0]["n_params"]
        if n_params > 0:
            mean_acc = np.mean([r["test_acc"] for r in results])
            eff = mean_acc / n_params * 1000
            print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner callout
    print("-" * 70)
    std_name = "Standard (1024→512)"
    best_name = max(
        all_results,
        key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]),
    )
    best_mean = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_mean = np.mean([r["test_acc"] for r in all_results[std_name]])

    if best_name != std_name:
        delta = best_mean - std_mean
        print(f">> WINNER: {best_name}")
        print(f"   {best_mean:.4f} vs {std_mean:.4f} (standard MLP)")
        print(f"   Delta: +{delta:.4f} ({delta * 100:.2f} pp)")
        n_params_best = all_results[best_name][0]["n_params"]
        n_params_std = all_results[std_name][0]["n_params"]
        if n_params_best == 0:
            print("   Uses ZERO learned parameters — pure manifold geometry")
        elif n_params_best < n_params_std:
            reduction = 100 * (1 - n_params_best / n_params_std)
            print(
                f"   With {reduction:.0f}% FEWER parameters ({n_params_best:,} vs {n_params_std:,})"
            )
        elif n_params_best > n_params_std:
            increase = 100 * (n_params_best / n_params_std - 1)
            print(
                f"   With {increase:.0f}% more parameters ({n_params_best:,} vs {n_params_std:,})"
            )
    else:
        print(f">> Standard architecture wins: {std_mean:.4f}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": "digits",
        "n_samples": int(X.shape[0]),
        "input_dim": input_dim,
        "n_classes": n_classes,
        "intrinsic_dim": intrinsic_dim,
        "global_intrinsic_dim_mean": global_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "n_folds": n_folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "k_pca": args.k_pca,
        "k_graph": args.k_graph,
        "k_vote": args.k_vote,
        "elapsed_s": elapsed,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = Path(__file__).resolve().parent / "digits_manifold_architecture_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    if args.plot:
        plot_path = str(
            Path(__file__).resolve().parent / "digits_manifold_architecture_results.png"
        )
        plot_results(
            all_results,
            intrinsic_dim,
            plot_path,
            elapsed=elapsed,
            input_dim=input_dim,
            n_classes=n_classes,
        )


if __name__ == "__main__":
    main()
