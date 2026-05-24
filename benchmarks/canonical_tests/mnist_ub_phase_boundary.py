#!/usr/bin/env python3
"""
MNIST Universal Bottleneck Phase Boundary Test
===============================================

Tests the Universal Bottleneck Theorem phase boundary on MNIST and
Fashion-MNIST.  Both datasets have C=10 classes and expected d* ≈ 10–22,
placing them in the **Whitney-dominated regime** (C ≤ d*).

Hypothesis: ManifoldResNet-UB+Drop(w*, dropout=0.3) should win cleanly here,
confirming the regime analysis from CIFAR-10.

Four architectures are compared (focused phase boundary test):

    - ResNet (Adam):                  baseline, 32 filters
    - ManifoldResNet-d (d={d}):       d* filters
    - ManifoldResNet-UB (w*={w*}):    w* = d* + C - 1 filters, no dropout
    - ManifoldResNet-UB+Drop (w*={w*}): w* filters, dropout=0.3

UB Theorem regime:
    Whitney-dominated (C ≤ d*) → UB+Drop predicted winner

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    # Both datasets (default)
    python benchmarks/canonical_tests/mnist_ub_phase_boundary.py --epochs 30 --trials 3 --metal

    # MNIST only
    python benchmarks/canonical_tests/mnist_ub_phase_boundary.py --dataset mnist --epochs 30 --trials 3 --metal

    # Fashion-MNIST only
    python benchmarks/canonical_tests/mnist_ub_phase_boundary.py --dataset fashion_mnist --epochs 30 --trials 3 --metal

    # Plot only (regenerate figure from saved JSON)
    python benchmarks/canonical_tests/mnist_ub_phase_boundary.py --dataset mnist --plot-only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow(gpu_flag="--metal")
import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model_builder import build_manifold_resnet  # noqa: E402
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM = 784  # 28 × 28 × 1
SPATIAL_SHAPE = (28, 28, 1)
N_CLASSES = 10
CONVERGENCE_THRESHOLD = 0.70  # MNIST converges faster than CIFAR-10

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_params(model):
    """Return total trainable parameter count for a Keras model.

    :param model: Compiled Keras model.
    :returns: Integer parameter count.
    """
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


class _ThrottledProgbar(keras.callbacks.Callback):
    """Keras-style progress bar that redraws every 5% of steps per epoch."""

    def on_epoch_begin(self, epoch, logs=None):
        self._progbar = None
        self._stride = None
        self._steps = None

    def on_train_batch_end(self, batch, logs=None):
        if self._progbar is None:
            self._steps = self.params.get("steps") or 1
            self._stride = max(1, self._steps // 20)
            self._progbar = keras.utils.Progbar(self._steps, unit_name="step")
        seen = batch + 1
        if seen % self._stride == 0 or seen == self._steps:
            self._progbar.update(seen, list((logs or {}).items()))

    def on_epoch_end(self, epoch, logs=None):
        if self._progbar is not None and self._steps is not None:
            self._progbar.update(self._steps, list((logs or {}).items()), finalize=True)


def _build_resnet_baseline(input_dim, n_classes, lr=0.001):
    """Conventional small ResNet with 32 filters (Adam optimizer).

    Uses build_manifold_resnet with intrinsic_dim=32 so that the ResNet
    architecture family is fully consistent; 32 is the conventional filter
    count for a small ResNet rather than a manifold-derived value.

    :param input_dim: Flat input dimensionality (784 for MNIST).
    :param n_classes: Number of output classes (10).
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    return build_manifold_resnet(
        input_dim,
        n_classes,
        intrinsic_dim=32,
        lr=lr,
        spatial_shape=SPATIAL_SHAPE,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Training
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
        callbacks=[_ThrottledProgbar()],
    )
    wall_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Convergence epoch: first epoch hitting CONVERGENCE_THRESHOLD train accuracy
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= CONVERGENCE_THRESHOLD:
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


def plot_results(all_results, intrinsic_dim, save_path, dataset_label, elapsed=None, n_classes=10):
    """Save a four-panel comparison figure.

    Panels:
        1. Training accuracy curves (mean ± std band)
        2. Final test accuracy bars with values labeled
        3. Parameter count bars (log scale)
        4. Wall time per trial

    :param all_results: Dict mapping architecture name → list of trial result dicts.
    :param intrinsic_dim: Bottleneck dimension d*.
    :param save_path: Filesystem path for the PNG output.
    :param dataset_label: Human-readable dataset name for the figure title.
    :param elapsed: Optional total wall time in seconds.
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
    w_ub = d + n_classes - 1

    colors = {
        "ResNet (Adam)": "mediumseagreen",
        f"ManifoldResNet-d (d={d})": "darkorchid",
        f"ManifoldResNet-UB (w*={w_ub})": "crimson",
        f"ManifoldResNet-UB+Drop (w*={w_ub})": "darkred",
    }
    # Fallback for any unexpected architecture names
    _palette = ["slategray", "olive", "teal", "hotpink"]
    for i, name in enumerate(n for n in all_results if n not in colors):
        colors[name] = _palette[i % len(_palette)]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = f"  |  total run time: {elapsed:.0f}s" if elapsed is not None else ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.42, wspace=0.32)
    ax_train, ax_acc, ax_par, ax_time = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    fig.suptitle(
        f"{dataset_label}  (d*={d}, w*=d*+C-1={w_ub}, C={n_classes})\n"
        f"input={INPUT_DIM}D  |  {n_classes} classes{elapsed_str}\n"
        f"Generated: {timestamp}",
        fontsize=13,
        fontweight="bold",
    )

    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short_names = [n.split("(")[0].strip() for n in names]

    # --- Panel 1: Training accuracy curves ---
    for name, results in all_results.items():
        accs = np.array([r["train_acc"] for r in results])
        ep = np.arange(1, accs.shape[1] + 1)
        color = colors.get(name, "gray")
        ax_train.plot(ep, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax_train.fill_between(
            ep,
            accs.mean(0) - accs.std(0),
            accs.mean(0) + accs.std(0),
            alpha=0.15,
            color=color,
        )
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training Accuracy")
    ax_train.set_title("Training Accuracy (mean ± std)")
    ax_train.legend(fontsize=7, loc="lower right")
    ax_train.grid(True, alpha=0.3)

    # --- Panel 2: Final test accuracy bars ---
    bars = ax_acc.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_title("Final Test Accuracy")
    y_min = max(0.0, float(min(means)) - 0.02)
    y_max = min(1.0, float(max(means)) + 0.02)
    ax_acc.set_ylim(y_min, y_max)
    ax_acc.tick_params(axis="x", labelsize=7, rotation=20)
    ax_acc.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: Parameter count bars (log scale) ---
    param_counts = [all_results[n][0]["n_params"] for n in names]
    bars = ax_par.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m in zip(bars, param_counts, means):
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{p:,}\nacc={m:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_par.set_ylabel("Parameters")
    ax_par.set_title("Parameter Count (log scale)")
    ax_par.set_yscale("log")
    ax_par.tick_params(axis="x", labelsize=7, rotation=20)
    ax_par.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Wall time per trial ---
    wall_times = [np.mean([r["wall_time"] for r in all_results[n]]) for n in names]
    wall_stds = [np.std([r["wall_time"] for r in all_results[n]]) for n in names]
    bars = ax_time.bar(
        short_names, wall_times, yerr=wall_stds, color=bar_colors, alpha=0.8, capsize=4
    )
    for bar, t in zip(bars, wall_times):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax_time.set_ylabel("Wall Time per Trial (s)")
    ax_time.set_title(f"Mean Training Time per Trial  |  {timestamp}")
    ax_time.tick_params(axis="x", labelsize=7, rotation=20)
    ax_time.grid(True, alpha=0.3, axis="y")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------


def load_dataset(dataset_name):
    """Load and preprocess MNIST or Fashion-MNIST.

    Flattens (N, 28, 28) → (N, 784) and applies StandardScaler normalization.

    :param dataset_name: One of ``"mnist"`` or ``"fashion_mnist"``.
    :returns: Tuple (X_train, y_train, X_test, y_test) as float32 arrays.
    """
    print(f"\nLoading {DATASET_LABELS[dataset_name]}...")
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Flatten: (N, 28, 28) → (N, 784)
    X_train = X_train.reshape(-1, INPUT_DIM).astype("float32")
    X_test = X_test.reshape(-1, INPUT_DIM).astype("float32")
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {N_CLASSES}  |  Input dim: {INPUT_DIM} (28×28×1 greyscale images)")
    return X_train, y_train, X_test, y_test


def run_dataset(dataset_name, args, results_dir):
    """Execute the full phase boundary experiment for one dataset.

    Runs all four architectures for ``args.trials`` trials, saves JSON and
    PNG results.

    :param dataset_name: ``"mnist"`` or ``"fashion_mnist"``.
    :param args: Parsed argparse namespace.
    :param results_dir: Directory in which to write JSON/PNG outputs.
    """
    dataset_label = DATASET_LABELS[dataset_name]
    results_path = results_dir / f"mnist_ub_phase_boundary_{dataset_name}_results.json"
    plot_path = str(results_path.with_suffix(".png"))

    # ------------------------------------------------------------------
    # Load data (needed even for --plot-only to satisfy path construction)
    # ------------------------------------------------------------------

    if args.plot_only:
        if not results_path.exists():
            print(f"ERROR: no existing results at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        d_saved = saved.get("d", 16)
        print(f"\nRegenerating figure from {results_path} (d={d_saved})")
        plot_results(
            saved["results"],
            d_saved,
            plot_path,
            dataset_label=dataset_label,
            elapsed=saved.get("elapsed_s", 0),
            n_classes=saved.get("n_classes", N_CLASSES),
        )
        return

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"PHASE 1: MANIFOLD DISCOVERY — {dataset_label}")
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
        noise_pct = 100 * (1 - r["mean"] / INPUT_DIM)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f} "
            f"{r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    print(
        f"\nPer-class intrinsic dimensionality "
        f"(τ={args.tau}, {args.samples_per_class} samples/class):"
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
        print(f"  class {c:>2}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = max(intrinsic_dim, N_CLASSES)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={N_CLASSES})")
    print(f"   d = {d / INPUT_DIM * 100:.2f}% of ambient dimensions")

    w_ub = d + N_CLASSES - 1

    # ------------------------------------------------------------------
    # UB Theorem summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"UB THEOREM: w* = d* + C - 1 = {d} + {N_CLASSES} - 1 = {w_ub}")
    print(f"  Whitney bound:     d* = {d}  (manifold embedding)")
    print(f"  Information floor: C-1 = {N_CLASSES - 1}  (class simplex)")
    print(f"  Regime: Whitney-dominated (C={N_CLASSES} ≤ d*={d}) → UB+Drop predicted winner")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 2: Build architectures
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"PHASE 2: ARCHITECTURE COMPARISON — {dataset_label}")
    print("=" * 70)

    # Each architecture maps name → (build_fn, X_train_arr, X_test_arr)
    # build_manifold_resnet takes flat input and applies Reshape internally.
    architectures = {
        "ResNet (Adam)": (
            lambda: _build_resnet_baseline(INPUT_DIM, N_CLASSES, lr=args.lr),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-d (d={d})": (
            lambda d_=d: build_manifold_resnet(
                INPUT_DIM,
                N_CLASSES,
                d_,
                lr=args.lr,
                spatial_shape=SPATIAL_SHAPE,
                dropout=0.0,
            ),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-UB (w*={w_ub})": (
            lambda w=w_ub: build_manifold_resnet(
                INPUT_DIM,
                N_CLASSES,
                w,
                lr=args.lr,
                spatial_shape=SPATIAL_SHAPE,
                dropout=0.0,
            ),
            X_train,
            X_test,
        ),
        f"ManifoldResNet-UB+Drop (w*={w_ub})": (
            lambda w=w_ub: build_manifold_resnet(
                INPUT_DIM,
                N_CLASSES,
                w,
                lr=args.lr,
                spatial_shape=SPATIAL_SHAPE,
                dropout=0.3,
            ),
            X_train,
            X_test,
        ),
    }

    # Filter by --only (incremental / partial run mode)
    existing_results: dict = {}
    if args.only:
        if results_path.exists():
            with open(results_path) as f:
                saved = json.load(f)
            existing_results = saved.get("results", {})
            print(f"\nIncremental mode: loaded {len(existing_results)} existing architectures.")
        architectures = {
            name: val
            for name, val in architectures.items()
            if any(fragment.lower() in name.lower() for fragment in args.only)
        }
        if not architectures:
            print(f"ERROR: no architectures matched {args.only}")
            sys.exit(1)
        print(f"  Running only: {list(architectures)}")

    # Print parameter counts
    for name, (build_fn, _, _) in architectures.items():
        model = build_fn()
        n_params = count_params(model)
        print(f"\n{name}:")
        print(f"  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Phase 3: Train
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"PHASE 3: TRAINING — {dataset_label}")
    print("=" * 70)
    print(f"Epochs: {args.epochs}  |  Trials: {args.trials}  |  Batch: {args.batch_size}")

    all_results: dict = {}

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
            f"\nMerged: {len(existing_results)} existing + {len(architectures)} new "
            f"= {len(all_results)} total"
        )

    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------

    output = {
        "dataset": dataset_name,
        "d": d,
        "n_classes": N_CLASSES,
        "input_dim": INPUT_DIM,
        "w_star": w_ub,
        "elapsed_s": elapsed,
        "results": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    if args.plot:
        plot_results(
            all_results,
            d,
            plot_path,
            dataset_label=dataset_label,
            elapsed=elapsed,
            n_classes=N_CLASSES,
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"RESULTS SUMMARY — {dataset_label.upper()}")
    print("=" * 70)
    print(f"  w* = d* + C - 1 = {d} + {N_CLASSES} - 1 = {w_ub}")
    print(f"  {'Architecture':<36} {'Acc (mean ± std)':>20}  {'Params':>10}")
    print("  " + "-" * 68)

    best_name = None
    best_acc = -1.0
    for name, results in all_results.items():
        accs = [r["test_acc"] for r in results]
        m = float(np.mean(accs))
        s = float(np.std(accs))
        p = results[0]["n_params"]
        print(f"  {name:<36} {m:.4f} ± {s:.4f}  {p:>10,}")
        if m > best_acc:
            best_acc = m
            best_name = name

    print("  " + "-" * 68)
    ub_drop_key = next((k for k in all_results if k.startswith("ManifoldResNet-UB+Drop")), None)
    expected = ub_drop_key if ub_drop_key else "ManifoldResNet-UB+Drop (predicted)"
    print(f"  Winner: {best_name}")
    if best_name == ub_drop_key:
        print(
            f"  CONFIRMED: UB+Drop wins as predicted in Whitney-dominated regime (C={N_CLASSES} ≤ d*={d})"
        )
    else:
        print(
            f"  NOTE: Expected winner was {expected} "
            f"(Whitney-dominated regime, C={N_CLASSES} ≤ d*={d})"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MNIST / Fashion-MNIST Universal Bottleneck phase boundary test"
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist", "both"],
        default="both",
        help="Dataset to run (default: both, runs sequentially)",
    )
    parser.add_argument("--epochs", type=int, default=30)
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

    results_dir = Path(__file__).resolve().parent

    if args.dataset == "both":
        datasets = ["mnist", "fashion_mnist"]
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        run_dataset(dataset_name, args, results_dir)

    print("Done.")


if __name__ == "__main__":
    main()
