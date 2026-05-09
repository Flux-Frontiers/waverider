#!/usr/bin/env python3
"""
ResNet Benchmark: ManifoldAdam with Residual Architectures
===========================================================

Tests whether gradient projection onto the manifold basis improves a
small ResNet-style architecture vs standard ResNet on CIFAR-10.

The key scientific question: does ManifoldAdam help ResNet training, where
skip connections already provide their own form of information preservation?
For a ResNet with a flat-input → reshape → Conv2D pipeline, ManifoldAdam
can only project weights whose first dimension matches the ambient data
dimensionality (3072). Conv2D kernels are shaped (3,3,3,32) — they do NOT
match — so ManifoldAdam is a no-op for those weights.  The Dense output
layer also does not match.  This is intentional and scientifically
interesting: by running ResNet + ManifoldAdam we confirm whether projecting
*input-space* gradients (if any dense layer sees the full 3072-D input)
adds any benefit vs standard Adam.  The answer is likely "no" since the
first operation is a reshape → Conv, but the experiment measures it directly.

For contrast, the flat MLP ``Intrinsic Dim (PCA→dD→output)`` benchmark is
included — it benefits from ManifoldAdam because its first dense layer has
shape (3072, d) and the projection applies.

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples random training points (k=--k-pca
    neighbors each).  Intrinsic dimensionality d is set to the maximum
    per-class intrinsic dim at τ=--tau (default 0.90).

Phase 2 — Architecture Comparison
    Six architectures are built and summarised:

    - Standard (1024→512):            flat MLP baseline
    - Manifold (2d→d):                manifold bottleneck MLP
    - Manifold + ManifoldAdam (d):    manifold bottleneck + projected Adam
    - ResNet (Adam):                  flat 3072 → reshape(32,32,3) → 3 ResBlocks → GAP → 10
    - ResNet + ManifoldAdam:          same, with manifold-projected gradient optimizer
    - Intrinsic Dim (PCA→dD→output):  PCA-projected input → d → 10

Phase 3 — Training and Evaluation
    All architectures are trained for --epochs epochs across --trials
    independent random seeds.  Results are saved to
    ``resnet_manifold_architecture_results.json`` and a five-panel figure
    ``resnet_manifold_architecture_results.png``.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/resnet_manifold_architecture.py [--epochs 50] [--trials 3]
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
# Check for --metal before importing TensorFlow (env vars must be set first).
_USE_METAL = "--metal" in sys.argv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if not _USE_METAL:
    # Force CPU — Metal GPU per-op sync overhead dominates for small dense nets.
    # Pass --metal to allow TF-Metal for Conv2D-heavy architectures.
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
    build_universal_bottleneck_pca,
    build_universal_bottleneck_raw,
)
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)

# ---------------------------------------------------------------------------
# CIFAR-10 class names
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


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# ResNet builder
# ---------------------------------------------------------------------------


def _residual_block(x, filters):
    """One residual block: Conv→BN→ReLU→Conv→BN→add skip→ReLU.

    If the skip connection has a different number of channels, a 1×1 Conv
    is inserted to match shapes before the addition.

    :param x: Input tensor.
    :param filters: Number of Conv2D filters.
    :returns: Output tensor after residual connection.
    """
    skip = x
    # Main path
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    # Match skip channels if needed
    if skip.shape[-1] != filters:
        skip = keras.layers.Conv2D(filters, (1, 1), padding="same")(skip)
    x = keras.layers.Add()([x, skip])
    x = keras.layers.ReLU()(x)
    return x


def build_resnet(input_dim, n_classes, lr=0.001, optimizer=None):
    """Small ResNet: 3 residual blocks → global average pool → output.

    Input is a flat vector of length ``input_dim`` (e.g., 3072 for CIFAR-10).
    The first layer reshapes it to (32, 32, 3) so the convolutional blocks
    can operate on the spatial structure.

    Architecture::

        Input(3072) → Reshape(32,32,3)
        → ResBlock(32) → MaxPool(2×2)
        → ResBlock(32) → MaxPool(2×2)
        → ResBlock(32) → MaxPool(2×2)
        → GlobalAveragePool
        → Dense(n_classes, softmax)

    Note on ManifoldAdam compatibility: Conv2D kernels have shape
    (kH, kW, C_in, C_out) — none of these dimensions equal 3072, so
    ManifoldAdam's gradient projection is a no-op for all convolutional
    weights.  The Dense output layer shape (32, n_classes) also does not
    match.  ManifoldAdam therefore reduces to standard Adam for this
    architecture.  This is the intended experimental baseline: does
    manifold-projecting gradients of any matching weight help, and what
    is the overhead if no weights match?

    :param input_dim: Flat input dimensionality (must equal H×W×C).
    :param n_classes: Number of output classes.
    :param lr: Learning rate (used when ``optimizer`` is None).
    :param optimizer: Optional pre-configured optimizer.  If None,
        ``keras.optimizers.Adam(lr)`` is used.
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

    # Convergence epoch: first epoch hitting 40% train accuracy
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.40:
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


def plot_results(all_results, intrinsic_dim, save_path, elapsed=None, input_dim=3072, n_classes=10):
    """Save a six-panel comparison figure with architecture schematics key.

    :param all_results: Dict mapping architecture name → list of trial result dicts.
    :param intrinsic_dim: Bottleneck dimension d.
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

    # Known colors — unknown architectures fall back to a rotating palette
    _known_colors = {
        "Standard (1024→512)": "steelblue",
        f"Manifold (2d→d, d={d})": "firebrick",
        "ResNet (Adam)": "mediumseagreen",
        f"ManifoldResNet-d (d={d})": "darkorchid",
        f"ManifoldResNet-2d (2d={2 * d})": "indigo",
        f"ManifoldResNet-UB (w*={w_ub})": "crimson",
        f"ManifoldResNet-UB+Drop (w*={w_ub})": "darkred",
        f"PCA(100)→{n_classes}": "goldenrod",
        f"PCA(100) linear→{n_classes}": "darkgoldenrod",
        f"PCA(100) Whitney(2d={2 * d})→{n_classes}": "saddlebrown",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
        f"UB-PCA (PCA→{d}D→{w_ub}→{n_classes})": "deepskyblue",
        f"UB-raw (→{w_ub}→{n_classes})": "tomato",
    }
    _palette = ["slategray", "olive", "teal", "hotpink", "peru", "cadetblue"]
    _unknown = [n for n in all_results if n not in _known_colors]
    colors = dict(_known_colors)
    for i, name in enumerate(_unknown):
        colors[name] = _palette[i % len(_palette)]

    # Known arch layer shapes — unknown archs are omitted from schematics.
    # For ResNet variants the values are total activation counts at each spatial
    # stage, not filter counts:
    #   Reshape(32,32,3) → ResBlock(F)+MaxPool → 16×16×F activations
    #                    → ResBlock(F)+MaxPool →  8×8×F activations
    #                    → ResBlock(F)+MaxPool →  4×4×F activations
    #                    → GlobalAveragePool   →      F activations
    # Showing filter counts alone (e.g. [32,32,32]) would misrepresent the
    # per-stage capacity and look identical to a dense MLP diagram.
    arch_layers = {
        # MLPs: each entry is a dense-layer width
        "Standard (1024→512)": [input_dim, 1024, 512, n_classes],
        # build_manifold_model: single hidden layer at d (not 2d→d)
        f"Manifold (2d→d, d={d})": [input_dim, d, n_classes],
        # ResNets: activation volume at each MaxPool output stage, then GAP→F
        "ResNet (Adam)": [input_dim, 16 * 16 * 32, 8 * 8 * 32, 4 * 4 * 32, 32, n_classes],
        f"ManifoldResNet-d (d={d})": [input_dim, 16 * 16 * d, 8 * 8 * d, 4 * 4 * d, d, n_classes],
        f"ManifoldResNet-2d (2d={2 * d})": [
            input_dim,
            16 * 16 * 2 * d,
            8 * 8 * 2 * d,
            4 * 4 * 2 * d,
            2 * d,
            n_classes,
        ],
        f"ManifoldResNet-UB (w*={w_ub})": [
            input_dim,
            16 * 16 * w_ub,
            8 * 8 * w_ub,
            4 * 4 * w_ub,
            w_ub,
            n_classes,
        ],
        f"ManifoldResNet-UB+Drop (w*={w_ub})": [
            input_dim,
            16 * 16 * w_ub,
            8 * 8 * w_ub,
            4 * 4 * w_ub,
            w_ub,
            n_classes,
        ],
        # PCA-seeded MLPs: each entry is a dense-layer width
        f"PCA(100)→{n_classes}": [100, 100, n_classes],
        f"PCA(100) linear→{n_classes}": [100, n_classes],
        f"PCA(100) Whitney(2d={2 * d})→{n_classes}": [100, 2 * d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
        f"UB-PCA (PCA→{d}D→{w_ub}→{n_classes})": [d, w_ub, n_classes],
        f"UB-raw (→{w_ub}→{n_classes})": [input_dim, w_ub, n_classes],
    }
    # Only schematise architectures that were actually run
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
        f"ManifoldResNet Architecture Study: CIFAR-10  (d*={d}, w*=d*+C-1={w_ub})\n"
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
    parser = argparse.ArgumentParser(description="ResNet vs ManifoldAdam on CIFAR-10")
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
        default=10,
        help="Samples per class for per-class dimensionality",
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

    print("\nLoading CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
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
        label = CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c)
        print(f"  {label:>12}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = max(intrinsic_dim, n_classes)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={n_classes})")
    print(f"   d = {d / input_dim * 100:.1f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Results path — used by --plot-only and --only
    # -----------------------------------------------------------------------

    results_path = Path(__file__).resolve().parent / "resnet_manifold_architecture_results.json"
    plot_path = str(results_path.with_suffix(".png"))

    # --plot-only: load existing JSON and regenerate figure without training
    if args.plot_only:
        if not results_path.exists():
            print(f"ERROR: no existing results at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        d_saved = saved.get("d", d)
        print(f"\nRegenerating figure from {results_path} (d={d_saved})")
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

    # PCA projection (for the flat MLP baseline)
    pca = skPCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_test_pca = pca.transform(X_test).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

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
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        "ResNet (Adam)": (
            lambda: build_resnet(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-d (d={d})": (
            lambda: build_manifold_resnet(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-2d (2d={2 * d})": (
            lambda: build_manifold_resnet_2d(input_dim, n_classes, d, lr=args.lr),
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
        f"PCA(100) Whitney(2d={2 * d})→{n_classes}": (
            lambda: build_pca_nc_model(n_classes, 100, hidden_width=2 * d, lr=args.lr),
            X_train_pca100,
            X_test_pca100,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
        f"UB-PCA (PCA→{d}D→{d + n_classes - 1}→{n_classes})": (
            lambda: build_universal_bottleneck_pca(n_classes, d, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
        f"UB-raw (→{d + n_classes - 1}→{n_classes})": (
            lambda: build_universal_bottleneck_raw(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-UB (w*={d + n_classes - 1})": (
            lambda: build_manifold_resnet(input_dim, n_classes, d + n_classes - 1, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-UB+Drop (w*={d + n_classes - 1})": (
            lambda: build_manifold_resnet(
                input_dim, n_classes, d + n_classes - 1, lr=args.lr, dropout=0.3
            ),
            X_train,
            X_test,
        ),
    }

    # Filter by --only (incremental mode): keep only requested architectures
    if args.only:
        architectures = {
            name: val
            for name, val in architectures.items()
            if any(fragment.lower() in name.lower() for fragment in args.only)
        }
        if not architectures:
            print(f"ERROR: no architectures matched {args.only}")
            print(f"  Available: {list({**dict.fromkeys(existing_results)})}")
            sys.exit(1)
        print(f"  Matched architectures: {list(architectures)}")

    # Show architecture details — build one of each to print param counts
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

    # Merge with existing results (incremental mode)
    if existing_results:
        merged = dict(existing_results)
        merged.update(all_results)
        all_results = merged
        print(
            f"\nMerged: {len(existing_results)} existing + {len(architectures)} new = {len(all_results)} total"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: CIFAR-10 ({input_dim}D, {n_classes} classes)")
    print(
        f"Intrinsic dimensionality: d = {d} (local-PCA max: {intrinsic_dim}, global mean: {global_dim}, τ={args.tau})"
    )
    print(f"Noise dimensions: {100 * (1 - d / input_dim):.1f}%")
    print(f"Epochs: {args.epochs}, Trials: {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}")
    print("-" * 70)

    col_w = 22
    header = f"{'Metric':<30}"
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
        row = f"{label:<30}"
        for name, results in all_results.items():
            vals = [r[key] for r in results]
            if fmt == ",d":
                row += f"{vals[0]:>{col_w},}"
            else:
                m, s = np.mean(vals), np.std(vals)
                row += f"  {m:{fmt}} ± {s:{fmt}}  "
        print(row)

    # Convergence
    row = f"{'Epochs to 40%':<30}"
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

    # Key comparisons
    print("-" * 70)

    def _acc(name):
        """Return mean test accuracy for a named architecture, or None if not present."""
        if name not in all_results:
            return None
        return np.mean([r["test_acc"] for r in all_results[name]])

    def _params(name):
        if name not in all_results:
            return None
        return all_results[name][0]["n_params"]

    def _pp(name_a, name_b):
        a, b = _acc(name_a), _acc(name_b)
        if a is None or b is None:
            return None
        return (a - b) * 100

    std_name = "Standard (1024→512)"
    resnet_name = "ResNet (Adam)"
    # Tolerate cached results with a different d value (incremental --only runs)
    mresnet_name = next(
        (k for k in all_results if k.startswith("ManifoldResNet-d (")), f"ManifoldResNet-d (d={d})"
    )
    pca100_name = f"PCA(100)→{n_classes}"
    pca100_lin_name = f"PCA(100) linear→{n_classes}"
    ub_name = next((k for k in all_results if k.startswith("ManifoldResNet-UB (")), None)
    ub_drop_name = next((k for k in all_results if k.startswith("ManifoldResNet-UB+Drop")), None)

    resnet_acc = _acc(resnet_name)
    std_acc = _acc(std_name)
    mresnet_acc = _acc(mresnet_name)
    resnet_params = _params(resnet_name)
    mresnet_params = _params(mresnet_name)

    if resnet_acc is not None and std_acc is not None:
        print(
            f">> ResNet vs Standard MLP:        {resnet_acc:.4f} vs {std_acc:.4f}"
            f" ({(resnet_acc - std_acc) * 100:+.2f} pp)"
        )
    if mresnet_acc is not None and resnet_acc is not None:
        print(
            f">> ManifoldResNet-d vs ResNet:    {mresnet_acc:.4f} vs {resnet_acc:.4f}"
            f" ({(mresnet_acc - resnet_acc) * 100:+.2f} pp)"
            f"  |  {resnet_params:,} → {mresnet_params:,} params"
            f" ({resnet_params / max(mresnet_params, 1):.1f}× reduction)"
        )
    if mresnet_acc is not None and std_acc is not None:
        print(
            f">> ManifoldResNet-d vs Standard:  {mresnet_acc:.4f} vs {std_acc:.4f}"
            f" ({(mresnet_acc - std_acc) * 100:+.2f} pp)"
        )
    if ub_name and ub_drop_name:
        ub_acc = _acc(ub_name)
        ub_drop_acc = _acc(ub_drop_name)
        ub_params = _params(ub_name)
        ub_drop_params = _params(ub_drop_name)
        print(
            f">> ManifoldResNet-UB+Drop vs UB:  {ub_drop_acc:.4f} vs {ub_acc:.4f}"
            f" ({(ub_drop_acc - ub_acc) * 100:+.2f} pp)"
            f"  |  {ub_params:,} → {ub_drop_params:,} params"
        )
        if resnet_acc is not None:
            print(
                f">> ManifoldResNet-UB+Drop vs ResNet: {ub_drop_acc:.4f} vs {resnet_acc:.4f}"
                f" ({(ub_drop_acc - resnet_acc) * 100:+.2f} pp)"
                f"  |  {resnet_params:,} → {ub_drop_params:,} params"
            )
    for name in (pca100_name, pca100_lin_name):
        acc = _acc(name)
        if acc is not None and resnet_acc is not None:
            print(
                f">> {name} vs ResNet:  {acc:.4f} vs {resnet_acc:.4f}"
                f" ({(acc - resnet_acc) * 100:+.2f} pp)"
                f"  |  {_params(name):,} params"
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
        "dataset": "cifar10",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "d": d,
        "elapsed_s": elapsed,
        "class_names": CIFAR10_CLASSES,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {
            str(k): {
                **v,
                "class_name": (CIFAR10_CLASSES[k] if k < len(CIFAR10_CLASSES) else str(k)),
            }
            for k, v in class_dims.items()
        },
        "results": {name: results for name, results in all_results.items()},
        "notes": (
            "ManifoldResNet-d uses the manifold intrinsic dimension d* as the Conv2D filter "
            "count at every residual block, replacing the arbitrary 32-filter baseline. "
            "After GlobalAveragePooling the network produces a d*-dimensional feature vector "
            "aligned with the data manifold before the final softmax classifier. "
            "PCA(100) architectures use 100 principal components as input; "
            "PCA(100)→10 adds a hidden layer of 100 units, PCA(100) linear→10 is a direct "
            "linear classifier on the 100-component PCA projection."
        ),
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_results(
            all_results, d, plot_path, elapsed=elapsed, input_dim=input_dim, n_classes=n_classes
        )


if __name__ == "__main__":
    main()
