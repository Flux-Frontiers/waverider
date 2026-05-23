#!/usr/bin/env python3
"""
CIFAR-100 Benchmark: Manifold-Informed Architecture vs Standard Architecture
=============================================================================

The hardest step in the manifold series: 3,072-dimensional color images
(32×32×3), 50K training / 10K test samples, 100 fine-grained classes.

Same ambient space as CIFAR-10 but 10× more classes crammed onto the same
manifold.  The hypothesis is under maximum stress: does a manifold-informed
bottleneck still hold when 100 classes must be separated in ~35D?

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples random training points (k=--k-pca
    neighbors each).  Intrinsic dimensionality d is set to the maximum
    per-class intrinsic dim at τ=--tau (default 0.90), accommodating the
    hardest class.

Phase 2 — Architecture Comparison
    Seven architectures are built and summarised:

    - Standard (1024→512):              input(3072) → 1024 → 512 → 100
    - Wide Manifold (d+1):              input(3072) → d+1  → 100
    - Manifold (d):                     input(3072) → d    → 100
    - Manifold + ManifoldAdam (d):      input(3072) → d    → 100  [ManifoldAdam optimizer]
    - ManifoldAdam (1024→512):          input(3072) → 1024 → 512 → 100  [ManifoldAdam optimizer]
    - PCA→dD + MLP (2d→d):             PCA(d) → 2d → d → 100
    - Intrinsic Dim (PCA→dD→d→output): PCA(d) → d  → 100

    The PCA models receive input already compressed to d dimensions by
    global sklearn PCA; the remaining models operate in raw pixel space.

Phase 3 — Training and Evaluation
    All architectures are trained with Adam for --epochs epochs across
    --trials independent random seeds.  Aggregate results (accuracy, loss,
    parameter counts, convergence epochs, parameter efficiency) are printed
    and saved to ``cifar100_architecture_results.json`` alongside a four-panel
    matplotlib figure (``cifar100_architecture_results.png``).

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/cifar100_manifold_architecture.py [--epochs 50] [--trials 3]
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
import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

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
from waverider.manifold_optimizer import ManifoldAdam, make_basis  # noqa: E402

# ---------------------------------------------------------------------------
# CIFAR-100 class names (fine labels)
# ---------------------------------------------------------------------------

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


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    """Train a model and return metrics."""
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

    # Convergence epoch (first epoch hitting 15% train accuracy —
    # CIFAR-100 MLPs top out around 25-35%)
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.15:
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_arch_schematics(ax, arch_layers, colors):
    """Draw schematic network diagrams as a key panel.

    Each architecture is rendered as a column of rounded rectangles (layers)
    connected by arrows.  Box width is proportional to log(layer_size) so
    that the 3072-input bar does not dwarf the narrow bottleneck layers.

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


def plot_results(
    all_results, intrinsic_dim, save_path, elapsed=None, input_dim=3072, n_classes=100
):
    """Save a five-panel comparison figure with architecture schematics key.

    :param all_results: Dict mapping architecture name → list of trial result dicts.
    :param intrinsic_dim: Bottleneck dimension d (already max(local_pca, n_classes)).
    :param save_path: Filesystem path for the PNG output.
    :param elapsed: Optional total wall time in seconds.
    :param input_dim: Raw input dimensionality.
    :param n_classes: Number of output classes.
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
        f"Wide Manifold (d+1, d={d})": "forestgreen",
        f"Manifold (d={d})": "firebrick",
        f"Manifold + ManifoldAdam (d={d})": "darkred",
        f"ManifoldAdam (1024→512, proj→{d}D)": "mediumpurple",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
    }

    arch_layers = {
        "Standard (1024→512)": [input_dim, 1024, 512, n_classes],
        f"Wide Manifold (d+1, d={d})": [input_dim, d + 1, n_classes],
        f"Manifold (d={d})": [input_dim, d, n_classes],
        f"Manifold + ManifoldAdam (d={d})": [input_dim, d, n_classes],
        f"ManifoldAdam (1024→512, proj→{d}D)": [input_dim, 1024, 512, n_classes],
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
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
        f"CIFAR-100: Manifold-Informed Architecture (d={d}) vs Standard\n"
        f"3,072D color images → manifold discovery → architecture"
        f"  |  100 fine-grained classes{elapsed_str}  |  {timestamp}",
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

    # --- Architecture schematics ---
    _draw_arch_schematics(ax_arch, arch_layers, colors)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-100: Manifold-Informed Architecture vs Standard"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--tau", type=float, default=0.90, help="Variance threshold for intrinsic dim"
    )
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=500,
        help="Points to sample for dimensionality discovery",
    )
    parser.add_argument("--k-pca", type=int, default=25, help="Neighborhood size for local PCA")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Samples per class for per-class dimensionality (default 10 for speed)",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print("\nLoading CIFAR-100...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")
    # Flatten: (N, 32, 32, 3) → (N, 3072)
    X_train = X_train.reshape(-1, 3072).astype("float32")
    X_test = X_test.reshape(-1, 3072).astype("float32")
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

    print(f"{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f} {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    # Per-class dimensionality
    print(
        f"\nPer-class intrinsic dimensionality (τ={args.tau}, {args.samples_per_class} samples/class):"
    )
    class_dims = discover_per_class_dimensionality(
        X_train,
        y_train,
        k=args.k_pca,
        tau=args.tau,
        n_samples_per_class=args.samples_per_class,
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        label = CIFAR100_CLASSES[c] if c < len(CIFAR100_CLASSES) else str(c)
        print(f"  {label:>20}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    # Use max of per-class maxima as the bottleneck — accommodates the hardest sample
    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    # We know there are 100 classes — need at least n_classes dims to separate them.
    # Clamp d to n_classes so PCA projection and network widths are consistent.
    d = max(intrinsic_dim, n_classes)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={n_classes})")
    print(f"   d = {d / input_dim * 100:.1f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    # PCA projection (needed for intrinsic-dim and PCA models)
    from sklearn.decomposition import PCA as skPCA

    pca = skPCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_test_pca = pca.transform(X_test).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    V_d = make_basis(pca)  # (n_dims, d) — top-d principal axes

    # Each entry: (build_fn, X_tr, X_te) — raw or PCA-projected as appropriate
    architectures = {
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Wide Manifold (d+1, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Manifold (d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Manifold + ManifoldAdam (d={d})": (
            lambda: build_manifold_model(
                input_dim,
                n_classes,
                d,
                lr=args.lr,
                optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
            ),
            X_train,
            X_test,
        ),
        f"ManifoldAdam (1024→512, proj→{d}D)": (
            lambda: build_standard_model(
                input_dim,
                n_classes,
                lr=args.lr,
                optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
            ),
            X_train,
            X_test,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
    }

    # Show architecture details
    for name, (build_fn, _, _) in architectures.items():
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
    print(f"Dataset: CIFAR-100 ({input_dim}D, {n_classes} fine-grained classes)")
    print(
        f"Intrinsic dimensionality: d = {d} (local-PCA max: {intrinsic_dim}, global mean: {global_dim}, τ={args.tau})"
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
                row += f"  {m:{fmt}} ± {s:{fmt}}  "
        print(row)

    # Convergence
    row = f"{'Epochs to 15%':<25}"
    for name, results in all_results.items():
        convs = [r["convergence_epoch"] for r in results if r["convergence_epoch"] is not None]
        if convs:
            row += f"  {np.mean(convs):.1f} ± {np.std(convs):.1f} ({len(convs)}/{len(results)})  "
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

    elapsed = time.perf_counter() - t_start
    save_data = {
        "device": DEVICE_INFO,
        "dataset": "cifar100",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "d": d,
        "elapsed_s": elapsed,
        "class_names": CIFAR100_CLASSES,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
        "k_pca": args.k_pca,
        "discovery_samples": args.discovery_samples,
        "samples_per_class": args.samples_per_class,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "trials": args.trials,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {
            str(k): {
                **v,
                "class_name": (CIFAR100_CLASSES[k] if k < len(CIFAR100_CLASSES) else str(k)),
            }
            for k, v in class_dims.items()
        },
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = Path(__file__).resolve().parent / "cifar100_architecture_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(Path(__file__).resolve().parent / "cifar100_architecture_results.png")
        plot_results(
            all_results,
            d,
            plot_path,
            elapsed=elapsed,
            input_dim=input_dim,
            n_classes=n_classes,
        )


if __name__ == "__main__":
    main()
