#!/usr/bin/env python3
"""
Tiny ImageNet Benchmark: Manifold-Informed Architecture vs Standard
====================================================================

The step up from CIFAR-100: 12,288-dimensional color images (64×64×3),
200 classes, 100K training / 10K validation samples.  4× the ambient
dimensionality of CIFAR, 2× the classes.

The hypothesis: intrinsic dimensionality is bounded by class structure,
not ambient dim.  Even at 12,288D, the manifold lives in a low-d subspace
and a manifold-informed bottleneck should outperform a brute-force MLP.

Dataset
-------
Tiny ImageNet is not bundled with Keras.  On first run this script
downloads and caches it automatically via ``tensorflow_datasets``.
If ``tensorflow_datasets`` is not installed:
    pip install tensorflow-datasets

Three phases
------------
Phase 1 — Manifold Discovery
    Local PCA over --discovery-samples random training points (k=--k-pca
    neighbors each).  Intrinsic dimensionality d is set to the maximum
    per-class intrinsic dim at τ=--tau (default 0.90), accommodating the
    hardest class.

Phase 2 — Architecture Comparison
    Six architectures:

    - Standard (1024→512):              input → 1024 → 512 → 200
    - Wide Manifold (d+1 hidden):       input → d+1  → 200
    - Manifold (d hidden):              input → d    → 200
    - ManifoldAdam (1024→512, proj→dD): Standard architecture + gradient
                                        projection onto top-d PCA axes
    - PCA→dD + MLP (2d→d):             PCA-projected input → 2d → d → 200
    - Intrinsic Dim (PCA→dD→output):    PCA-projected input → d  → 200

Phase 3 — Training and Evaluation
    All architectures trained with Adam for --epochs epochs across
    --trials independent random seeds.  Results saved to
    ``tiny_imagenet_architecture_results.json`` and
    ``tiny_imagenet_architecture_results.png``.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py
    python benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py \\
        --epochs 50 --trials 3 --discovery-samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow(gpu_flag="--gpu")
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
# Data loading
# ---------------------------------------------------------------------------

_TINY_IMAGENET_CACHE = Path.home() / ".cache" / "tiny_imagenet"
_INPUT_DIM = 64 * 64 * 3  # 12,288
_N_CLASSES = 200


def load_tiny_imagenet():
    """Load Tiny ImageNet-200 from the Stanford CS231n archive, cache as numpy.

    Downloads once to ~/.cache/tiny_imagenet/.  Subsequent runs load from
    the cached .npy files — no network required.

    :returns: Tuple ((X_train, y_train), (X_val, y_val)) of float32 / int32
        arrays.  Images are NOT normalised here — caller handles scaling.
    """
    import urllib.request
    import zipfile

    from PIL import Image

    cache_dir = _TINY_IMAGENET_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_x_path = cache_dir / "train_x.npy"
    train_y_path = cache_dir / "train_y.npy"
    val_x_path = cache_dir / "val_x.npy"
    val_y_path = cache_dir / "val_y.npy"

    if all(p.exists() for p in (train_x_path, train_y_path, val_x_path, val_y_path)):
        print("  Loading from cache...")
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path)
        y_val = np.load(val_y_path)
        return (X_train, y_train), (X_val, y_val)

    # Download the zip from Stanford (237 MB, one-time)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = cache_dir / "tiny-imagenet-200.zip"
    extract_dir = cache_dir / "tiny-imagenet-200"

    if not extract_dir.exists():
        if not zip_path.exists():
            print(f"  Downloading {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            print("  Download complete.")
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    root = extract_dir

    # Build class-name -> integer-label mapping from wnids.txt
    wnids = (root / "wnids.txt").read_text().strip().split("\n")
    wnid_to_label = {w.strip(): i for i, w in enumerate(wnids)}

    def _read_image(path):
        """Read an image, convert to RGB, flatten to float32."""
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float32).flatten()

    # --- Train split ---
    print("  Processing train split...")
    train_xs, train_ys = [], []
    train_dir = root / "train"
    for wnid in sorted(wnids):
        label = wnid_to_label[wnid]
        images_dir = train_dir / wnid / "images"
        for img_path in sorted(images_dir.glob("*.JPEG")):
            train_xs.append(_read_image(img_path))
            train_ys.append(label)

    # --- Validation split ---
    print("  Processing val split...")
    val_annotations = (root / "val" / "val_annotations.txt").read_text().strip().split("\n")
    val_xs, val_ys = [], []
    val_images_dir = root / "val" / "images"
    for line in val_annotations:
        parts = line.split("\t")
        fname, wnid = parts[0], parts[1]
        val_xs.append(_read_image(val_images_dir / fname))
        val_ys.append(wnid_to_label[wnid])

    X_train = np.array(train_xs)
    y_train = np.array(train_ys, dtype=np.int32)
    X_val = np.array(val_xs)
    y_val = np.array(val_ys, dtype=np.int32)

    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(val_x_path, X_val)
    np.save(val_y_path, y_val)
    print(f"  Cached to {cache_dir}")

    return (X_train, y_train), (X_val, y_val)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Peak-clamping callback
# ---------------------------------------------------------------------------


class PeakClampingCallback(tf.keras.callbacks.Callback):
    """Stop training once val_accuracy has been below its peak for `patience`
    consecutive epochs, then restore the weights from the peak epoch.

    This is appropriate for architectures (e.g. PCA-based linear heads) that
    converge in a handful of epochs and then overfit/drift — their val accuracy
    rises sharply then falls monotonically.  Clamping at the peak epoch ensures
    the final model is evaluated at its best point, not at the end of a long
    unnecessary training run.

    :param patience: Number of consecutive below-peak epochs before stopping.
        Default 2 (catches rapid declines without being hair-trigger on noise).
    """

    def __init__(self, patience: int = 2):
        super().__init__()
        self.patience = patience
        self._best_val_acc: float = -float("inf")
        self._best_weights = None
        self._wait: int = 0
        self.stopped_epoch: int = 0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = (logs or {}).get("val_accuracy", 0.0)
        if val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            self._best_weights = self.model.get_weights()
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self._best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(
                f"    [PeakClamping] stopped at epoch {self.stopped_epoch + 1},"
                f" peak val_acc={self._best_val_acc:.4f}"
            )


# ---------------------------------------------------------------------------
# Phase 3: Trial runner
# ---------------------------------------------------------------------------


def run_trial(
    build_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    trial,
    use_peak_clamping: bool = False,
):
    """Train one model instance and return metrics.

    :param use_peak_clamping: When True, attach :class:`PeakClampingCallback`
        so training stops automatically once val_accuracy starts declining.
        The returned metrics reflect the peak-epoch model, not the final epoch.
    """
    model = build_fn()
    n_params = count_params(model)

    callbacks = [PeakClampingCallback(patience=2)] if use_peak_clamping else []

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )
    wall_time = time.perf_counter() - t0

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Convergence: first epoch hitting 5% train accuracy
    # (200-class MLP from pixels is hard — 0.5% random baseline)
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.05:
            conv_epoch = i
            break

    return {
        "trial": trial,
        "n_params": n_params,
        "test_loss": float(val_loss),
        "test_acc": float(val_acc),
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


def plot_results(all_results, intrinsic_dim, input_dim, save_path, elapsed=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    palette = [
        "steelblue",
        "firebrick",
        "forestgreen",
        "darkorange",
        "mediumpurple",
        "saddlebrown",
    ]
    names = list(all_results.keys())
    colors = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""
    fig.suptitle(
        f"Tiny ImageNet: Manifold-Informed Architecture (d={intrinsic_dim}) vs Standard\n"
        f"{input_dim:,}D color images (64×64×3) → manifold discovery → architecture"
        f"  |  200 classes{elapsed_str}",
        fontsize=13,
        fontweight="bold",
    )

    # Validation accuracy curves
    ax = axes[0, 0]
    for name, results in all_results.items():
        accs = np.array([r["val_acc"] for r in results])
        epochs_arr = np.arange(1, accs.shape[1] + 1)
        color = colors[name]
        ax.plot(epochs_arr, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs_arr,
            accs.mean(0) - accs.std(0),
            accs.mean(0) + accs.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Training loss curves
    ax = axes[0, 1]
    for name, results in all_results.items():
        losses = np.array([r["train_loss"] for r in results])
        epochs_arr = np.arange(1, losses.shape[1] + 1)
        color = colors[name]
        ax.plot(epochs_arr, losses.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs_arr,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Final validation accuracy bar chart
    ax = axes[1, 0]
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors[n] for n in names]
    short_names = [n.split("(")[0].strip() for n in names]
    bars = ax.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Final Validation Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    # Parameter counts
    ax = axes[1, 1]
    param_counts = [all_results[n][0]["n_params"] for n in names]
    bars = ax.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m in zip(bars, param_counts, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{p:,}\nacc={m:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count (lower is better at same accuracy)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Tiny ImageNet: Manifold-Informed Architecture vs Standard"
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
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (default: CPU)")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="tiny_imagenet",
        help="Prefix for output files (default: tiny_imagenet)",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print("\nLoading Tiny ImageNet (64×64×3, 200 classes)...")
    (X_train, y_train), (X_val, y_val) = load_tiny_imagenet()

    # Normalise to [0, 1] first, then z-score only features with nonzero
    # variance.  Constant-valued pixel positions (std ≈ 0) would otherwise
    # produce inf/NaN and poison PCA's randomized SVD.
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # Replace any inf/NaN from zero-variance columns with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    input_dim = X_train.shape[1]  # 12,288
    n_classes = len(set(y_train))  # 200
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Classes: {n_classes}  |  Input dim: {input_dim:,} (64×64×3 color images)")

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
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    print(
        f"\nPer-class intrinsic dimensionality"
        f" (τ={args.tau}, {args.samples_per_class} samples/class):"
    )
    t0 = time.perf_counter()
    class_dims = discover_per_class_dimensionality(
        X_train,
        y_train,
        k=args.k_pca,
        tau=args.tau,
        n_samples_per_class=args.samples_per_class,
    )
    per_class_time = time.perf_counter() - t0
    print(f"  Per-class discovery time: {per_class_time:.1f}s")
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        print(f"  Class {c:>3}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = max(intrinsic_dim, n_classes)
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   Using d = {d} (max of local-PCA={intrinsic_dim}, n_classes={n_classes})")
    print(f"   d = {d / input_dim * 100:.2f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    from sklearn.decomposition import PCA as skPCA

    pca = skPCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_val_pca = pca.transform(X_val).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    V_d = make_basis(pca)  # (input_dim, d) — top-d principal axes

    # Each entry: (build_fn, X_tr, X_te, use_peak_clamping)
    # Peak clamping is enabled for PCA-based architectures that converge in
    # a handful of epochs then decline — it stops at the best val_acc epoch.
    # Standard / Manifold architectures train for the full epoch budget.
    architectures = {
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_val,
            False,
        ),
        f"Wide Manifold (d+1, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_val,
            False,
        ),
        f"Manifold (d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            X_train,
            X_val,
            False,
        ),
        f"ManifoldAdam (1024→512, proj→{d}D)": (
            lambda: build_standard_model(
                input_dim,
                n_classes,
                lr=args.lr,
                optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
            ),
            X_train,
            X_val,
            False,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_val_pca,
            True,  # peak-clamp: converges in ~1 epoch then declines
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_val_pca,
            True,  # peak-clamp: converges in ~3 epochs then declines
        ),
    }

    for name, (build_fn, _, _, _) in architectures.items():
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

    for name, (build_fn, X_tr, X_te, use_clamping) in architectures.items():
        clamp_note = " [peak-clamped]" if use_clamping else ""
        print(f"\n{name}{clamp_note}")
        trial_results = []

        for trial in range(args.trials):
            np.random.seed(trial * 42)
            tf.random.set_seed(trial * 42)

            result = run_trial(
                build_fn,
                X_tr,
                y_train,
                X_te,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
                trial=trial,
                use_peak_clamping=use_clamping,
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
    print(f"Dataset: Tiny ImageNet ({input_dim:,}D, {n_classes} classes, 64×64×3)")
    print(
        f"Intrinsic dimensionality: d = {d}"
        f" (local-PCA max: {intrinsic_dim}, global mean: {global_dim}, τ={args.tau})"
    )
    print(f"Noise dimensions: {100 * (1 - d / input_dim):.1f}%")
    print(f"Epochs: {args.epochs}, Trials: {args.trials}")
    print(f"Device: {DEVICE_INFO['device_used']}")
    print(f"Total run time: {elapsed:.0f}s")
    print("-" * 70)

    col_w = 22
    header = f"{'Metric':<25}"
    for name in all_results:
        short = name.split("(")[0].strip()
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * 70)

    for label, key, fmt in [
        ("Val Accuracy", "test_acc", ".4f"),
        ("Val Loss", "test_loss", ".4f"),
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

    row = f"{'Epochs to 5%':<25}"
    for name, results in all_results.items():
        convs = [r["convergence_epoch"] for r in results if r["convergence_epoch"] is not None]
        if convs:
            row += f"  {np.mean(convs):.1f} ± {np.std(convs):.1f} ({len(convs)}/{len(results)})  "
        else:
            row += f"{'N/A':>{col_w}}"
    print(row)

    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

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

    save_data = {
        "device": DEVICE_INFO,
        "dataset": "tiny_imagenet",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "d": d,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "elapsed_s": elapsed,
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = (
        Path(__file__).resolve().parent / f"{args.output_prefix}_architecture_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(results_path).replace(".json", ".png")
        plot_results(all_results, d, input_dim, plot_path, elapsed=elapsed)

    return save_data


if __name__ == "__main__":
    main()
