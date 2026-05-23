#!/usr/bin/env python3
"""
CIFAR Architecture Sweep: MLP Manifold-Informed Architectures
=============================================================

Unified benchmark that runs the full MLP architecture suite on either
CIFAR-10 or CIFAR-100, selected via ``--dataset``.  This script consolidates
``cifar10_manifold_architecture.py`` and ``cifar100_manifold_architecture.py``
into a single entry point and adds the **Class-Augmented PCA** architecture.

Motivation for unification
---------------------------
Both CIFAR-10 and CIFAR-100 share 3,072-dimensional pixel space and ~99.1%
noise (d*≈28 for both datasets).  Comparing their sweep results in a common
format reveals how bottleneck geometry scales with class count.

Class-Augmented PCA architecture
---------------------------------
CIFAR probes show d*≈28 for both datasets in 3,072-dim pixel space.  The key
insight: the manifold embedding space is geometrically insufficient to separate
all C classes no matter what.  The fix: project into d*+C dimensions via PCA —
enough room for manifold geometry (d* dims) AND one coordinate per class (C
dims) — then classify directly with a single linear layer.  No hidden
bottleneck.

    PCA(d*+C) → Dense(C, softmax)

This removes the Shannon bottleneck entirely: classes are not squeezed; they
are explicitly allocated a dedicated coordinate.

Architecture suite (MLP-only, no ResNet)
-----------------------------------------
  - Standard (1024→512)         raw input, over-parameterized baseline
  - Manifold (2d→d)             raw input, manifold-width bottleneck
  - PCA→dD + MLP (2d→d)        PCA-d* input, the star architecture
  - PCA→dD + MLP-wide (4d→2d)  PCA-d* input, wider first layer
  - PCA→dD + MLP-deep (2d→2d→d) PCA-d* input, deeper variant
  - Intrinsic Dim (PCA→dD→out) PCA-d* input, minimal single-layer
  - UB-PCA (PCA→d*→w*→C)       PCA-d* input, UB theorem single-layer
  - UB-PCA-deep (w*→w*→C)      PCA-d* input, UB theorem two-layer
  - LDA+PCA Aug (d*+C-1→C)     linear ceiling marker

Output files (compatible with existing per-dataset scripts)
------------------------------------------------------------
  cifar10_architecture_results.json  / cifar10_architecture_results.png
  cifar100_architecture_results.json / cifar100_architecture_results.png

Part of WaveRider, https://github.com/flux-frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/cifar_architecture_sweep.py --dataset cifar10
    python benchmarks/canonical_tests/cifar_architecture_sweep.py --dataset cifar100 \\
        --epochs 60 --trials 3 --plot
    python benchmarks/canonical_tests/cifar_architecture_sweep.py --dataset cifar10 \\
        --only "Class-Aug"
    python benchmarks/canonical_tests/cifar_architecture_sweep.py --dataset cifar100 \\
        --metal --epochs 80 --trials 3

Author: Eric G. Suchanek, PhD
Last Revision: 2026-04-12 16:46:44
License: Elastic 2.0
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorFlow / Metal setup — must happen before importing tf
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow(gpu_flag="--metal")
import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model_builder import (  # noqa: E402
    build_manifold_model,
    build_pca_intrinsic_dim_model,
    build_pca_mlp_wide,
    build_pca_model,
    build_standard_model,
    build_universal_bottleneck_pca,
)
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)

# ---------------------------------------------------------------------------
# Class name lists
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial, conv_thresh):
    """Train a model for one trial and return metrics.

    :param build_fn: Zero-arg callable returning a compiled Keras model.
    :param X_train: Training features array.
    :param y_train: Training labels array.
    :param X_test: Test features array.
    :param y_test: Test labels array.
    :param epochs: Number of training epochs.
    :param batch_size: Mini-batch size.
    :param trial: Trial index (0-based), recorded in result dict.
    :param conv_thresh: Accuracy threshold used to determine convergence epoch.
    :returns: Dict of per-trial metrics including training curves.
    """
    model = build_fn()
    n_params = count_params(model)

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    wall_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= conv_thresh:
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
        "train_loss": [float(v) for v in history.history["loss"]],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_arch_schematics(ax, arch_layers, colors):
    """Draw schematic network diagrams as a key panel.

    Each architecture is rendered as a column of rounded rectangles (layers)
    connected by arrows.  Box width is proportional to log(layer_size) so
    that the 3072-input bar does not dwarf the narrow bottleneck layers.

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


def plot_results(
    all_results,
    intrinsic_dim,
    save_path,
    dataset_label,
    elapsed=None,
    input_dim=3072,
    n_classes=10,
    d_star=None,
):
    """Save a five-panel comparison figure with architecture schematics key.

    :param all_results: Dict mapping architecture name → list of trial result dicts.
    :param intrinsic_dim: Discovered bottleneck dimension d (after max(d, n_classes)).
    :param save_path: Filesystem path for the PNG output.
    :param dataset_label: Human-readable dataset name for figure title.
    :param elapsed: Optional total wall time in seconds.
    :param input_dim: Raw input dimensionality (for schematic input-layer size).
    :param n_classes: Number of output classes (for schematic output-layer size).
    :param d_star: Unclamped global intrinsic dim (global mean from local PCA). Used
        for the LDA+PCA schematic.  Defaults to intrinsic_dim when not provided.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    d = intrinsic_dim
    w_star = d + n_classes - 1

    colors = {
        "Standard (1024→512)": "steelblue",
        f"Manifold (2d→d, d={d})": "firebrick",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"PCA→{d}D + MLP-wide (4d→2d)": "mediumvioletred",
        f"PCA→{d}D + MLP-deep (2d→2d→d)": "indigo",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
        "UB-PCA (PCA→d*→w*→C)": "slategray",
        "UB-PCA-deep (w*→w*→C)": "dimgray",
        "LDA+PCA Aug (d*+C-1→C)": "darkcyan",
    }

    _d_star = d_star if d_star is not None else d
    _proj_lda_pca = _d_star + n_classes - 1

    arch_layers = {
        "Standard (1024→512)": [input_dim, 1024, 512, n_classes],
        f"Manifold (2d→d, d={d})": [input_dim, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP-wide (4d→2d)": [d, 4 * d, 2 * d, n_classes],
        f"PCA→{d}D + MLP-deep (2d→2d→d)": [d, 2 * d, 2 * d, d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
        "UB-PCA (PCA→d*→w*→C)": [d, w_star, n_classes],
        "UB-PCA-deep (w*→w*→C)": [d, w_star, w_star, n_classes],
        "LDA+PCA Aug (d*+C-1→C)": [_proj_lda_pca, n_classes],
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.38, wspace=0.3)
    ax_val = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_par = fig.add_subplot(gs[1, 1])
    ax_arch = fig.add_subplot(gs[2, :])

    fig.suptitle(
        f"{dataset_label}: MLP Architecture Sweep (d*={d})\n"
        f"3,072D color images → manifold discovery → architecture comparison"
        f"{elapsed_str}  |  {timestamp}",
        fontsize=14,
        fontweight="bold",
    )

    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short_names = [n.split("(")[0].strip() for n in names]

    # --- Training accuracy curves ---
    for name, results in all_results.items():
        accs = np.array([r["train_acc"] for r in results])
        ep = np.arange(1, accs.shape[1] + 1)
        color = colors.get(name, "gray")
        ax_val.plot(ep, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax_val.fill_between(
            ep,
            accs.mean(0) - accs.std(0),
            accs.mean(0) + accs.std(0),
            alpha=0.15,
            color=color,
        )
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Training Accuracy")
    ax_val.set_title("Training Accuracy")
    ax_val.legend().set_visible(False)
    ax_val.grid(True, alpha=0.3)

    # --- Training loss curves ---
    for name, results in all_results.items():
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
    ax_loss.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)

    # --- Final test accuracy bars ---
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
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Final Test Accuracy")
    ax_acc.set_ylim(0, float(max(means)) * 1.25)
    ax_acc.tick_params(axis="x", labelsize=7, rotation=30)
    ax_acc.grid(True, alpha=0.3, axis="y")

    # --- Parameter count bars ---
    param_counts = [all_results[n][0]["n_params"] for n in names]
    bars = ax_par.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m in zip(bars, param_counts, means):
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{p:,}\nacc={m:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_par.set_ylabel("Parameters")
    ax_par.set_title("Parameter Count (lower is better at same accuracy)")
    ax_par.set_yscale("log")
    ax_par.tick_params(axis="x", labelsize=7, rotation=30)
    ax_par.grid(True, alpha=0.3, axis="y")

    # --- Architecture schematics key ---
    # Only include architectures that were actually run
    run_arch_layers = {k: v for k, v in arch_layers.items() if k in all_results}
    run_colors = {k: v for k, v in colors.items() if k in all_results}
    _draw_arch_schematics(ax_arch, run_arch_layers, run_colors)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR MLP Architecture Sweep: unified CIFAR-10 / CIFAR-100 benchmark"
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        required=True,
        help="Dataset to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of independent trials (default: 4 for cifar10, 3 for cifar100)",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--tau", type=float, default=0.90, help="Variance threshold for intrinsic dim"
    )
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=2000,
        help="Points to sample for dimensionality discovery",
    )
    parser.add_argument("--k-pca", type=int, default=50, help="Neighborhood size for local PCA")
    parser.add_argument(
        "--metal",
        action="store_true",
        default=False,
        help="Use Metal GPU (M-series Mac); default is CPU-forced",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="Load existing JSON results and regenerate plot without training",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the architecture whose key contains this substring (case-insensitive)",
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    # Default trial counts per dataset
    if args.trials is None:
        args.trials = 4 if args.dataset == "cifar10" else 3

    # -----------------------------------------------------------------------
    # Dataset configuration
    # -----------------------------------------------------------------------

    if args.dataset == "cifar10":
        dataset_label = "CIFAR-10"
        class_names = CIFAR10_CLASSES
        conv_thresh = 0.40
        json_stem = "cifar10_architecture_results"
        png_stem = "cifar10_architecture_results"
    else:
        dataset_label = "CIFAR-100"
        class_names = CIFAR100_CLASSES
        conv_thresh = 0.15
        json_stem = "cifar100_architecture_results"
        png_stem = "cifar100_architecture_results"

    # -----------------------------------------------------------------------
    # --plot-only: reload saved JSON and regenerate plot, then exit
    # -----------------------------------------------------------------------

    if args.plot_only:
        results_path = Path(__file__).resolve().parent / f"{json_stem}.json"
        if not results_path.exists():
            print(f"ERROR: No results file found at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        all_results = saved["results"]
        d = saved.get("intrinsic_dim", saved.get("d_star", 30))
        input_dim = saved.get("input_dim", 3072)
        n_classes_saved = saved.get("n_classes", 10)
        plot_path = str(Path(__file__).resolve().parent / f"{png_stem}.png")
        plot_results(
            all_results,
            d,
            plot_path,
            dataset_label=dataset_label,
            input_dim=input_dim,
            n_classes=n_classes_saved,
            d_star=saved.get("global_dim"),
        )
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print(f"\nLoading {dataset_label}...")
    if args.dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

    # Flatten: (N, 32, 32, 3) → (N, 3072)  — keep float64 through sklearn
    X_train = X_train.reshape(-1, 3072).astype("float64")
    X_test = X_test.reshape(-1, 3072).astype("float64")
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]
    n_classes = len(set(y_train))
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes}  |  Input dim: {input_dim} (32×32×3 color images)")

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
        X_train,
        n_samples=args.discovery_samples,
        k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    discovery_time = time.perf_counter() - t0
    print(f"\nDiscovery time: {discovery_time:.1f}s\n")

    print(f"{'tau':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    # Per-class dimensionality
    samples_per_class = max(5, args.discovery_samples // n_classes)
    print(
        f"\nPer-class intrinsic dimensionality (tau={args.tau}, {samples_per_class} samples/class):"
    )
    class_dims = discover_per_class_dimensionality(
        X_train,
        y_train,
        k=args.k_pca,
        tau=args.tau,
        n_samples_per_class=samples_per_class,
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        label = class_names[c] if c < len(class_names) else str(c)
        print(
            f"  {label:>20}: d = {cd['mean']:.1f} +/- {cd['std']:.1f}  [{cd['min']}, {cd['max']}]"
        )

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    # Clamp d to n_classes — need at least that many dims to separate all classes
    d = max(intrinsic_dim, n_classes)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={n_classes})")
    print(f"   d = {d / input_dim * 100:.1f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures & compute PCA projections
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    from sklearn.decomposition import PCA as skPCA

    def _fpe_ctx():
        """Fresh errstate each call — Apple Accelerate BLAS fires spurious FPE
        warnings (divide-by-zero, overflow, invalid) on large float64 matmuls."""
        return np.errstate(divide="ignore", over="ignore", invalid="ignore")

    # PCA to d* dims — for UB-PCA and Intrinsic Dim architectures
    # Use full SVD — randomized solver's power iterations overflow on 3072-dim data
    pca_d = skPCA(n_components=d, svd_solver="full")
    with _fpe_ctx():
        X_train_pca_d = np.nan_to_num(
            pca_d.fit_transform(X_train), nan=0.0, posinf=0.0, neginf=0.0
        ).astype("float32")
        X_test_pca_d = np.nan_to_num(
            pca_d.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0
        ).astype("float32")
    var_d = pca_d.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_d * 100:.1f}% of global variance")

    # Cast all arrays to float32 for Keras (sklearn work is done)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # Architecture registry: name → (build_fn, X_tr, X_te)
    all_architectures = {
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            X_train_pca_d,
            X_test_pca_d,
        ),
        f"PCA→{d}D + MLP-wide (4d→2d)": (
            lambda: build_pca_mlp_wide(n_classes, d, lr=args.lr),
            X_train_pca_d,
            X_test_pca_d,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca_d,
            X_test_pca_d,
        ),
        "UB-PCA (PCA→d*→w*→C)": (
            lambda: build_universal_bottleneck_pca(n_classes, d, lr=args.lr),
            X_train_pca_d,
            X_test_pca_d,
        ),
    }

    # Apply --only filter (substring match, case-insensitive)
    if args.only is not None:
        needle = args.only.lower()
        architectures = {k: v for k, v in all_architectures.items() if needle in k.lower()}
        if not architectures:
            available = "\n  ".join(all_architectures.keys())
            print(
                f"ERROR: --only '{args.only}' matched no architecture.  Available:\n  {available}"
            )
            sys.exit(1)
        print(
            f"\n--only filter: running {len(architectures)} architecture(s): {list(architectures.keys())}"
        )
    else:
        architectures = all_architectures

    # Show architecture details
    for name, (build_fn, _, _) in architectures.items():
        model = build_fn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")
        for layer in model.layers:
            if hasattr(layer, "units"):
                print(f"  {layer.name}: -> {layer.units}")

    # -----------------------------------------------------------------------
    # Phase 3: Train and compare
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING")
    print("=" * 70)

    all_results = {}

    for name, (build_fn, X_tr, X_te) in architectures.items():
        print(f"\n{name}")
        trial_results = []

        for trial in range(args.trials):
            np.random.seed(trial * 42)
            tf.random.set_seed(trial * 42)

            result = run_trial(
                build_fn,
                X_tr,
                y_train,
                X_te,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                trial=trial,
                conv_thresh=conv_thresh,
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

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {dataset_label} ({input_dim}D, {n_classes} classes)")
    print(
        f"Intrinsic dimensionality: d = {d}"
        f" (local-PCA max: {intrinsic_dim}, global mean: {global_dim}, tau={args.tau})"
    )
    print(f"Noise dimensions: {100 * (1 - d / input_dim):.1f}%")
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
                m, s = np.mean(vals), np.std(vals)
                row += f"  {m:{fmt}} +/- {s:{fmt}}  "
        print(row)

    # Convergence row
    conv_label = f"Epochs to {conv_thresh * 100:.0f}%"
    row = f"{conv_label:<25}"
    for name, results in all_results.items():
        convs = [r["convergence_epoch"] for r in results if r["convergence_epoch"] is not None]
        if convs:
            row += f"  {np.mean(convs):.1f} +/- {np.std(convs):.1f} ({len(convs)}/{len(results)})  "
        else:
            row += f"{'N/A':>{col_w}}"
    print(row)

    # Parameter efficiency
    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner
    print("-" * 70)
    best_name = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_acc = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_name = "Standard (1024→512)"
    if std_name in all_results:
        std_acc = np.mean([r["test_acc"] for r in all_results[std_name]])
        if best_name != std_name:
            delta = best_acc - std_acc
            print(f">> MANIFOLD-INFORMED WINS: {best_name}")
            print(f"   {best_acc:.4f} vs {std_acc:.4f} (standard)")
            print(f"   Delta: +{delta:.4f} ({delta * 100:.2f}%)")
            best_params = all_results[best_name][0]["n_params"]
            std_params = all_results[std_name][0]["n_params"]
            if best_params < std_params:
                reduction = 100 * (1 - best_params / std_params)
                print(
                    f"   With {reduction:.0f}% FEWER parameters ({best_params:,} vs {std_params:,})"
                )
            elif best_params > std_params:
                increase = 100 * (best_params / std_params - 1)
                print(
                    f"   With {increase:.0f}% more parameters ({best_params:,} vs {std_params:,})"
                )
        else:
            print(f">> Standard architecture wins: {std_acc:.4f}")
    else:
        print(f">> Best architecture: {best_name}  ({best_acc:.4f})")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "d": d,
        "elapsed_s": elapsed,
        "class_names": class_names,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {
            str(k): {
                **v,
                "class_name": (class_names[k] if k < len(class_names) else str(k)),
            }
            for k, v in class_dims.items()
        },
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = Path(__file__).resolve().parent / f"{json_stem}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(Path(__file__).resolve().parent / f"{png_stem}.png")
        plot_results(
            all_results,
            d,
            plot_path,
            dataset_label=dataset_label,
            elapsed=elapsed,
            input_dim=input_dim,
            n_classes=n_classes,
            d_star=global_dim,
        )


if __name__ == "__main__":
    main()
