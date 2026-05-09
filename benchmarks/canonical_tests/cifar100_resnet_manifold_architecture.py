#!/usr/bin/env python3
"""
CIFAR-100 ResNet Benchmark: Universal Bottleneck Theorem Validation
====================================================================

Extends the CIFAR-10 ResNet benchmark to CIFAR-100: same 3,072-dimensional
color images (32×32×3), 50K training / 10K test samples, 100 fine-grained
classes.

The central question: does the Universal Bottleneck Theorem (w* = d* + C − 1)
hold in the high-C regime?  For CIFAR-100, C=100 >> d*≈19, so
w* = d* + 99 ≈ 118.  The theorem predicts ManifoldResNet-UB outperforms both
ManifoldResNet-d (19 filters) and ManifoldResNet-2d (38 filters).

Twelve architectures are compared:

    - Standard (1024→512):              flat MLP baseline
    - Manifold (2d→d):                  manifold bottleneck MLP
    - ResNet (Adam):                    flat → reshape → 3 ResBlocks(32) → GAP → 100
    - ManifoldResNet-d:                 ResNet with d* filters per block
    - ManifoldResNet-2d:                ResNet with 2d* filters per block
    - ManifoldResNet-UB:                ResNet with w*=d*+C-1 filters (UB theorem)
    - PCA(100)→C:                       PCA-100 → hidden(100) → output
    - PCA(100) linear→C:               PCA-100 → output (linear baseline)
    - PCA(100) Whitney(2d)→C:          PCA-100 → hidden(2d*) → output
    - Intrinsic Dim (PCA→d*D→output):  PCA-d* → d* → output
    - UB-PCA (PCA→d*D→w*→C):          PCA-d* → w* → output
    - UB-raw (→w*→C):                  raw input → w* → output

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/cifar100_resnet_manifold_architecture.py \\
        [--epochs 50] [--trials 3] [--metal]
    python benchmarks/canonical_tests/cifar100_resnet_manifold_architecture.py \\
        --only ManifoldResNet-UB
    python benchmarks/canonical_tests/cifar100_resnet_manifold_architecture.py \\
        --plot-only
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------
_USE_METAL = "--metal" in sys.argv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if not _USE_METAL:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

_device_label = f"Metal GPU ({gpus[0].name})" if (_USE_METAL and gpus) else "CPU (forced)"
DEVICE_INFO = {
    "tensorflow_version": tf.__version__,
    "device_used": _device_label,
}
print(f"TensorFlow {tf.__version__} | Device: {DEVICE_INFO['device_used']}")

import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model_builder import (  # noqa: E402
    build_manifold_model,
    build_manifold_resnet,
    build_manifold_resnet_2d,
    build_pca_intrinsic_dim_model,
    build_pca_linear_model,
    build_pca_nc_model,
    build_standard_model,
    build_ub_pca_mlp,
    build_universal_bottleneck_mlp,
    build_universal_bottleneck_pca,
    build_universal_bottleneck_raw,
)
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)

# ---------------------------------------------------------------------------
# CIFAR-100 class names (fine labels, 100 classes)
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
# ResNet builder (32 filters — standard baseline)
# ---------------------------------------------------------------------------


def _residual_block(x, filters):
    """One residual block: Conv→BN→ReLU→Conv→BN→add skip→ReLU.

    :param x: Input tensor.
    :param filters: Number of Conv2D filters.
    :returns: Output tensor after residual connection.
    """
    skip = x
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    if skip.shape[-1] != filters:
        skip = keras.layers.Conv2D(filters, (1, 1), padding="same")(skip)
    x = keras.layers.Add()([x, skip])
    x = keras.layers.ReLU()(x)
    return x


def build_resnet(input_dim, n_classes, lr=0.001, optimizer=None):
    """Small ResNet: 3 residual blocks → global average pool → output.

    :param input_dim: Flat input dimensionality.
    :param n_classes: Number of output classes.
    :param lr: Learning rate (used when ``optimizer`` is None).
    :param optimizer: Optional pre-configured optimizer.
    :returns: Compiled Keras model.
    """
    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Reshape((32, 32, 3))(inp)

    x = _residual_block(x, 32)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, 32)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = _residual_block(x, 32)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizer if optimizer is not None else keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Phase 3: Benchmark
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    """Train a model for one trial and return metrics.

    :param build_fn: Zero-arg callable returning a compiled Keras model.
    :param X_train: Training features array.
    :param y_train: Training labels array.
    :param X_test: Test features array.
    :param y_test: Test labels array.
    :param epochs: Number of training epochs.
    :param batch_size: Mini-batch size.
    :param trial: Trial index (0-based), recorded in result.
    :returns: Dict of per-trial metrics including train curves.
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

    # Convergence epoch: first epoch hitting 20% train accuracy (100-class task)
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.20:
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
    connected by arrows.  Box width is proportional to log(layer_size).

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
            x_ctr, 0.95, short, ha="center", va="top", fontsize=7, fontweight="bold", color=color
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
    """Save a six-panel comparison figure with architecture schematics key.

    :param all_results: Dict mapping architecture name → list of trial result dicts.
    :param intrinsic_dim: Bottleneck dimension d_arch.
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
    w_ub = d + n_classes - 1  # Universal Bottleneck width

    _known_colors = {
        "Standard (1024→512)": "steelblue",
        f"Manifold (2d→d, d={d})": "firebrick",
        "ResNet (Adam)": "mediumseagreen",
        f"ManifoldResNet-d (d={d})": "darkorchid",
        f"ManifoldResNet-2d (2d={2 * d})": "indigo",
        f"ManifoldResNet-UB (w*={w_ub})": "crimson",
        f"PCA(100)→{n_classes}": "goldenrod",
        f"PCA(100) linear→{n_classes}": "darkgoldenrod",
        f"PCA(100) Whitney(2d={2 * d})→{n_classes}": "saddlebrown",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
        f"UB-PCA (PCA→{d}D→{w_ub}→{n_classes})": "deepskyblue",
        f"UB-raw (→{w_ub}→{n_classes})": "tomato",
        f"UB-MLP (→{w_ub}→{w_ub}→{n_classes})": "darkcyan",
        f"UB-PCA-MLP (→{w_ub + 1}→PCA→{w_ub + 1}→{n_classes})": "mediumslateblue",
    }
    _palette = ["slategray", "olive", "teal", "hotpink", "peru", "cadetblue"]
    _unknown = [n for n in all_results if n not in _known_colors]
    colors = dict(_known_colors)
    for i, name in enumerate(_unknown):
        colors[name] = _palette[i % len(_palette)]

    arch_layers = {
        "Standard (1024→512)": [input_dim, 1024, 512, n_classes],
        f"Manifold (2d→d, d={d})": [input_dim, 2 * d, d, n_classes],
        "ResNet (Adam)": [input_dim, 32, 32, 32, n_classes],
        f"ManifoldResNet-d (d={d})": [input_dim, d, d, d, n_classes],
        f"ManifoldResNet-2d (2d={2 * d})": [input_dim, 2 * d, 2 * d, 2 * d, n_classes],
        f"ManifoldResNet-UB (w*={w_ub})": [input_dim, w_ub, w_ub, w_ub, n_classes],
        f"PCA(100)→{n_classes}": [100, 100, n_classes],
        f"PCA(100) linear→{n_classes}": [100, n_classes],
        f"PCA(100) Whitney(2d={2 * d})→{n_classes}": [100, 2 * d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
        f"UB-PCA (PCA→{d}D→{w_ub}→{n_classes})": [d, w_ub, n_classes],
        f"UB-raw (→{w_ub}→{n_classes})": [input_dim, w_ub, n_classes],
        f"UB-MLP (→{w_ub}→{w_ub}→{n_classes})": [input_dim, w_ub, w_ub, n_classes],
        f"UB-PCA-MLP (→{w_ub + 1}→PCA→{w_ub + 1}→{n_classes})": [
            input_dim,
            w_ub + 1,
            w_ub + 1,
            n_classes,
        ],
    }
    arch_layers = {k: v for k, v in arch_layers.items() if k in all_results}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = f"  |  total run time: {elapsed:.0f}s" if elapsed is not None else ""

    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.6, 0.85], hspace=0.42, wspace=0.3)
    ax_val = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_par = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[2, :])
    ax_arch = fig.add_subplot(gs[3, :])

    fig.suptitle(
        f"ManifoldResNet Architecture Study: CIFAR-100  (d*={d}, w*=d*+C-1={w_ub})\n"
        f"input={input_dim}D  |  {n_classes} classes{elapsed_str}\n"
        f"Generated: {timestamp}",
        fontsize=13,
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
            ep, accs.mean(0) - accs.std(0), accs.mean(0) + accs.std(0), alpha=0.15, color=color
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

    # --- Mean wall time per architecture ---
    wall_times = [np.mean([r["wall_time"] for r in all_results[n]]) for n in names]
    wall_stds = [np.std([r["wall_time"] for r in all_results[n]]) for n in names]
    bars = ax_time.bar(
        short_names, wall_times, yerr=wall_stds, color=bar_colors, alpha=0.8, capsize=4
    )
    for bar, t in zip(bars, wall_times):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax_time.set_ylabel("Wall Time per Trial (s)")
    ax_time.set_title(f"Mean Training Time per Trial  |  {timestamp}")
    ax_time.tick_params(axis="x", labelsize=7, rotation=30)
    ax_time.grid(True, alpha=0.3, axis="y")

    # --- Architecture schematics ---
    _draw_arch_schematics(ax_arch, arch_layers, colors)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ResNet vs ManifoldResNet on CIFAR-100 (Universal Bottleneck Theorem)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--metal",
        action="store_true",
        default=False,
        help="Allow TF-Metal GPU (default: CPU-forced). "
        "Recommended for Conv2D-heavy architectures on Apple Silicon.",
    )
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
        default=5,
        help="Samples per class for per-class dimensionality (default 5 for 100 classes)",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ARCH",
        help="Run only these architectures (by name fragment), merge into existing results",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figure from existing results JSON without running any training",
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print("\nLoading CIFAR-100...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")
    X_train = X_train.reshape(-1, 3072).astype("float32")
    X_test = X_test.reshape(-1, 3072).astype("float32")
    y_train = y_train.ravel()
    y_test = y_test.ravel()

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
        print(f"  {label:>16}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    # For CIFAR-100, n_classes=100 >> d*≈19.  Use manifold dim for architecture
    # sizing; n_classes governs the output layer only.
    d_arch = intrinsic_dim
    w_ub = d_arch + n_classes - 1  # Universal Bottleneck: d* + C − 1
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   d_arch = {d_arch}  (manifold dimension — drives filter/bottleneck count)")
    print(f"   w*     = {w_ub}  (Universal Bottleneck: d* + C − 1 = {d_arch} + {n_classes} − 1)")
    print(f"   d_arch = {d_arch / input_dim * 100:.1f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Results path — used by --plot-only and --only
    # -----------------------------------------------------------------------

    results_path = (
        Path(__file__).resolve().parent / "cifar100_resnet_manifold_architecture_results.json"
    )
    plot_path = str(results_path.with_suffix(".png"))

    # --plot-only: load existing JSON and regenerate figure without training
    if args.plot_only:
        if not results_path.exists():
            print(f"ERROR: no existing results at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        d_saved = saved.get("d_arch", saved.get("d", d_arch))
        print(f"\nRegenerating figure from {results_path} (d_arch={d_saved})")
        plot_results(
            saved["results"],
            d_saved,
            plot_path,
            elapsed=saved.get("elapsed_s", 0),
            input_dim=saved.get("input_dim", input_dim),
            n_classes=saved.get("n_classes", n_classes),
        )
        print(f"Figure saved to {plot_path}")
        return

    # Load existing results when --only is given (incremental mode)
    existing_results: dict = {}
    if args.only and results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
        existing_results = saved.get("results", {})
        print(f"\nIncremental mode: loaded {len(existing_results)} existing architectures.")
        print(f"  Running only: {args.only}")

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    pca = skPCA(n_components=d_arch)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_test_pca = pca.transform(X_test).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d_arch}D captures {var_explained * 100:.1f}% of global variance")

    pca100 = skPCA(n_components=100)
    X_train_pca100 = pca100.fit_transform(X_train).astype("float32")
    X_test_pca100 = pca100.transform(X_test).astype("float32")
    var_explained100 = pca100.explained_variance_ratio_.sum()
    print(f"  PCA to 100D captures {var_explained100 * 100:.1f}% of global variance")

    architectures = {
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Manifold (2d→d, d={d_arch})": (
            lambda: build_manifold_model(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
        "ResNet (Adam)": (
            lambda: build_resnet(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-d (d={d_arch})": (
            lambda: build_manifold_resnet(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-2d (2d={2 * d_arch})": (
            lambda: build_manifold_resnet_2d(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-UB (w*={w_ub})": (
            lambda: build_manifold_resnet(input_dim, n_classes, w_ub, lr=args.lr),
            X_train,
            X_test,
        ),
        f"PCA(100)→{n_classes}": (
            lambda: build_pca_intrinsic_dim_model(n_classes, 100, lr=args.lr),
            X_train_pca100,
            X_test_pca100,
        ),
        f"PCA(100) linear→{n_classes}": (
            lambda: build_pca_linear_model(n_classes, 100, lr=args.lr),
            X_train_pca100,
            X_test_pca100,
        ),
        f"PCA(100) Whitney(2d={2 * d_arch})→{n_classes}": (
            lambda: build_pca_nc_model(n_classes, 100, hidden_width=2 * d_arch, lr=args.lr),
            X_train_pca100,
            X_test_pca100,
        ),
        f"Intrinsic Dim (PCA→{d_arch}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
        f"UB-PCA (PCA→{d_arch}D→{w_ub}→{n_classes})": (
            lambda: build_universal_bottleneck_pca(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
        f"UB-raw (→{w_ub}→{n_classes})": (
            lambda: build_universal_bottleneck_raw(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
        f"UB-MLP (→{w_ub}→{w_ub}→{n_classes})": (
            lambda: build_universal_bottleneck_mlp(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
        f"UB-PCA-MLP (→{w_ub + 1}→PCA→{w_ub + 1}→{n_classes})": (
            lambda: build_ub_pca_mlp(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_test,
        ),
    }

    for name, (build_fn, _, _) in architectures.items():
        model = build_fn()
        n_params = count_params(model)
        print(f"  {name}: {n_params:,} params")

    # -----------------------------------------------------------------------
    # Phase 3: Train and compare
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING")
    print("=" * 70)

    all_results = dict(existing_results)

    for name, (build_fn, X_tr, X_te) in architectures.items():
        # --only filter: skip architectures not matching any fragment
        if args.only and not any(frag.lower() in name.lower() for frag in args.only):
            continue

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
    print(f"Dataset: CIFAR-100 ({input_dim}D, {n_classes} classes)")
    print(
        f"Intrinsic dimensionality: d_arch = {d_arch} "
        f"(local-PCA max: {intrinsic_dim}, global mean: {global_dim}, τ={args.tau})"
    )
    print(f"Universal Bottleneck: w* = d* + C − 1 = {d_arch} + {n_classes} − 1 = {w_ub}")
    print(f"Noise dimensions: {100 * (1 - d_arch / input_dim):.1f}%")
    print(f"Epochs: {args.epochs}, Trials: {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}")
    print("-" * 70)

    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        std_acc = np.std([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        mean_time = np.mean([r["wall_time"] for r in results])
        print(
            f"  {name:<50}  acc={mean_acc:.4f}±{std_acc:.4f}  "
            f"params={n_params:>8,}  time={mean_time:.1f}s"
        )

    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    print("-" * 70)
    # UB theorem comparison
    resnet_name = "ResNet (Adam)"
    ub_name = f"ManifoldResNet-UB (w*={w_ub})"
    d_name = f"ManifoldResNet-d (d={d_arch})"
    if resnet_name in all_results and ub_name in all_results:
        resnet_acc = np.mean([r["test_acc"] for r in all_results[resnet_name]])
        ub_acc = np.mean([r["test_acc"] for r in all_results[ub_name]])
        resnet_params = all_results[resnet_name][0]["n_params"]
        ub_params = all_results[ub_name][0]["n_params"]
        print(
            f">> ManifoldResNet-UB vs ResNet:  {ub_acc:.4f} vs {resnet_acc:.4f}"
            f" ({(ub_acc - resnet_acc) * 100:+.2f} pp)"
            f"  |  {resnet_params:,} → {ub_params:,} params"
        )
    if d_name in all_results and ub_name in all_results:
        d_acc = np.mean([r["test_acc"] for r in all_results[d_name]])
        ub_acc = np.mean([r["test_acc"] for r in all_results[ub_name]])
        d_params = all_results[d_name][0]["n_params"]
        ub_params = all_results[ub_name][0]["n_params"]
        print(
            f">> ManifoldResNet-UB vs d-only:  {ub_acc:.4f} vs {d_acc:.4f}"
            f" ({(ub_acc - d_acc) * 100:+.2f} pp)"
            f"  |  {d_params:,} → {ub_params:,} params"
        )

    best_name = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_acc = np.mean([r["test_acc"] for r in all_results[best_name]])
    print(f">> BEST: {best_name}: {best_acc:.4f}")
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
        "d_arch": d_arch,
        "w_ub": w_ub,
        "elapsed_s": elapsed,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
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
        "notes": (
            f"Universal Bottleneck Theorem: w* = d* + C − 1 = {d_arch} + {n_classes} − 1 = {w_ub}. "
            "ManifoldResNet-UB uses w* as filter count at every residual block. "
            "For CIFAR-100, C=100 >> d*≈19, so w* is driven by the information floor, not Whitney. "
            "ManifoldResNet-d uses d_arch=intrinsic_dim; ManifoldResNet-2d uses 2*d_arch. "
            "Conv2D kernels are (3,3,C,C) so ManifoldAdam projection does not apply."
        ),
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_results(
            all_results,
            d_arch,
            plot_path,
            elapsed=elapsed,
            input_dim=input_dim,
            n_classes=n_classes,
        )


if __name__ == "__main__":
    main()
