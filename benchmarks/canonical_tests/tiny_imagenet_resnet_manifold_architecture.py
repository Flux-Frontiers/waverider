#!/usr/bin/env python3
"""
Tiny ImageNet ResNet Benchmark: Universal Bottleneck Theorem Validation
=======================================================================

Extends the CIFAR-100 ResNet benchmark to Tiny ImageNet: 12,288-dimensional
color images (64×64×3), 100K training / 10K validation samples, 200 classes.
A second floor-dominated data point (C ≫ d*) for the UB theorem paper.

The central question: does the Universal Bottleneck Theorem (w* = d* + C − 1)
continue to hold in the extreme floor-dominated regime?  For Tiny ImageNet,
C=200 ≫ d*, so w* is driven almost entirely by the information floor.  The
theorem predicts ManifoldResNet-d is the parameter-efficiency champion while
ManifoldResNet-UB uses its extra capacity sub-optimally (as on CIFAR-100).

Architectures (MLP suite matches cifar_architecture_sweep.py for cross-dataset comparison):

    MLP architectures (shared with CIFAR sweep):
    - Standard (1024→512):              flat MLP baseline
    - Manifold (2d→d):                  manifold bottleneck MLP
    - PCA→d*D + MLP (2d→d):            PCA-d* input, manifold-width MLP
    - PCA→d*D + MLP-wide (4d→2d):      PCA-d* input, wider first layer
    - Intrinsic Dim (PCA→d*D→output):  PCA-d* → d* → output
    - UB-PCA (PCA→d*→w*→C):           PCA-d* → w* → output (UB theorem)

    ResNet architectures (Tiny ImageNet specific):
    - ResNet (Adam):                    reshape → 3 ResBlocks(32) → GAP → 200
    - ManifoldResNet-d:                 ResNet with d* filters per block
    - ManifoldResNet-2d:                ResNet with 2d* filters per block
    - ManifoldResNet-UB:                ResNet with w*=d*+C-1 filters (UB theorem)
    - PCA(200)→C:                       PCA-200 → hidden(200) → output
    - PCA(200) linear→C:               PCA-200 → output (linear baseline)
    - PCA(200) Whitney(2d)→C:          PCA-200 → hidden(2d*) → output
    - UB-raw (→w*→C):                  raw input → w* → output
    - UB-MLP (→w*→w*→C):              raw input → w* → w* → output
    - UB-PCA-MLP:                      raw → w*+1 → PCA → w*+1 → output

Dataset
-------
Tiny ImageNet is not bundled with Keras.  On first run this script downloads
the canonical ``tiny-imagenet-200.zip`` from the CS231n mirror (~237 MB),
extracts it, parses the standard ``wnids.txt`` + ``val_annotations.txt``
layout, and caches the result as ``.npy`` files in
``~/.cache/tiny_imagenet/``.  Subsequent runs load from the cache — no
network required.  Only stdlib + Pillow (already a Keras dependency) is
needed for loading; no ``tensorflow-datasets`` dependency.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/tiny_imagenet_resnet_manifold_architecture.py \\
        [--epochs 30] [--trials 3] [--metal]
    python benchmarks/canonical_tests/tiny_imagenet_resnet_manifold_architecture.py \\
        --only ManifoldResNet-UB
    python benchmarks/canonical_tests/tiny_imagenet_resnet_manifold_architecture.py \\
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
    build_pca_mlp_wide,
    build_pca_model,
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
# Constants / dataset caching
# ---------------------------------------------------------------------------

_TINY_IMAGENET_CACHE = Path.home() / ".cache" / "tiny_imagenet"
_SPATIAL_SHAPE = (64, 64, 3)
_INPUT_DIM = 64 * 64 * 3  # 12,288
_N_CLASSES = 200


_TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_with_progress(url, dest):
    """Stream-download ``url`` to ``dest`` with a simple progress indicator."""
    import urllib.request

    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Downloading: {mb:6.1f} / {total_mb:6.1f} MB ({pct:5.1f}%)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()  # newline after progress


def _build_arrays_from_extracted(root: Path):
    """Walk an extracted ``tiny-imagenet-200/`` tree, return numpy arrays.

    :param root: Path to the extracted ``tiny-imagenet-200`` directory.
    :returns: Tuple ``((X_train, y_train), (X_val, y_val))`` of flat float32
        feature arrays and int32 label arrays.
    """
    from PIL import Image

    # Class → integer index mapping from wnids.txt (200 lines, one wnid each).
    wnids_path = root / "wnids.txt"
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    print(f"  Found {len(wnids)} classes in wnids.txt")

    def _load_image(path):
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1)

    # Train: train/<wnid>/images/*.JPEG  (500 per class, 100K total)
    print("  Loading training images (100K)...")
    train_dir = root / "train"
    xs, ys = [], []
    for i, wnid in enumerate(wnids):
        img_dir = train_dir / wnid / "images"
        for img_path in sorted(img_dir.glob("*.JPEG")):
            xs.append(_load_image(img_path))
            ys.append(wnid_to_idx[wnid])
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(wnids)} classes loaded")
    X_train = np.stack(xs).astype(np.float32)
    y_train = np.array(ys, dtype=np.int32)

    # Val: val/images/*.JPEG with val_annotations.txt mapping filename → wnid.
    print("  Loading validation images (10K)...")
    val_dir = root / "val"
    val_ann = val_dir / "val_annotations.txt"
    val_labels = {}
    with open(val_ann) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                val_labels[parts[0]] = parts[1]

    xs, ys = [], []
    for fname in sorted(val_labels.keys()):
        img_path = val_dir / "images" / fname
        xs.append(_load_image(img_path))
        ys.append(wnid_to_idx[val_labels[fname]])
    X_val = np.stack(xs).astype(np.float32)
    y_val = np.array(ys, dtype=np.int32)

    return (X_train, y_train), (X_val, y_val)


def load_tiny_imagenet():
    """Load Tiny ImageNet from the CS231n mirror, cached as numpy arrays.

    On first call, downloads ``tiny-imagenet-200.zip`` (~237 MB) from the
    CS231n Stanford mirror, extracts it, walks the standard train / val
    directory layout, and caches flat float32 / int32 arrays under
    ``~/.cache/tiny_imagenet/``.  Subsequent calls load straight from the
    cache (no network, no image decoding).

    :returns: Tuple ``((X_train, y_train), (X_val, y_val))`` of float32 /
        int32 arrays.  Images are flattened (``64*64*3 = 12288``) but NOT
        normalised — caller handles scaling.
    """
    cache_dir = _TINY_IMAGENET_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_x_path = cache_dir / "train_x.npy"
    train_y_path = cache_dir / "train_y.npy"
    val_x_path = cache_dir / "val_x.npy"
    val_y_path = cache_dir / "val_y.npy"

    if all(p.exists() for p in (train_x_path, train_y_path, val_x_path, val_y_path)):
        print("  Loading from cache (mmap_mode='r')...")
        # Memory-map the big feature arrays so the 4.7 GB train_x.npy does not
        # need to be resident. Labels are small and can load normally.
        X_train = np.load(train_x_path, mmap_mode="r")
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path, mmap_mode="r")
        y_val = np.load(val_y_path)
        return (X_train, y_train), (X_val, y_val)

    # Need to download + extract.
    import zipfile

    zip_path = cache_dir / "tiny-imagenet-200.zip"
    extracted_root = cache_dir / "tiny-imagenet-200"

    if not extracted_root.exists():
        if not zip_path.exists():
            print(f"  Downloading {_TINY_IMAGENET_URL} (~237 MB)...")
            _download_with_progress(_TINY_IMAGENET_URL, zip_path)
        print(f"  Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)

    (X_train, y_train), (X_val, y_val) = _build_arrays_from_extracted(extracted_root)

    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(val_x_path, X_val)
    np.save(val_y_path, y_val)
    print(f"  Cached numpy arrays to {cache_dir}")

    return (X_train, y_train), (X_val, y_val)


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Memory-safe feature standardisation
# ---------------------------------------------------------------------------
#
# StandardScaler.fit_transform on a (100K, 12288) float32 array peaks at
# ~14 GB because it holds the input, the mean-centered copy, and the scaled
# output simultaneously.  These helpers stream the computation in chunks so
# peak resident memory for scaling is one ~4.7 GB output buffer plus a small
# chunk temporary — the input is memory-mapped and paged on demand.


def _streaming_mean_std(arr_mmap, chunk=4096):
    """Compute per-feature mean and std from an mmap'd array in chunks.

    :param arr_mmap: ``np.memmap`` or ndarray of shape (N, D).
    :param chunk: Number of rows to process per chunk.
    :returns: Tuple ``(mean, std)`` of float32 ndarrays, shape (D,).
    """
    n, d = arr_mmap.shape
    sum_ = np.zeros(d, dtype=np.float64)
    sumsq = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        b = np.asarray(arr_mmap[i : i + chunk], dtype=np.float64)
        sum_ += b.sum(0)
        sumsq += (b * b).sum(0)
    mean = sum_ / n
    var = sumsq / n - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def _scale_into_new(arr_mmap, mean, std, chunk=4096):
    """Return a newly-allocated float32 (x − mean)/std array, filled in chunks.

    :param arr_mmap: ``np.memmap`` or ndarray input of shape (N, D).
    :param mean: Per-feature mean, shape (D,).
    :param std: Per-feature std, shape (D,).
    :param chunk: Number of rows to process per chunk.
    :returns: Float32 ndarray of shape (N, D) with standardised features.
    """
    n = arr_mmap.shape[0]
    out = np.empty(arr_mmap.shape, dtype=np.float32)
    for i in range(0, n, chunk):
        out[i : i + chunk] = (arr_mmap[i : i + chunk].astype(np.float32) - mean) / std
    return out


# ---------------------------------------------------------------------------
# ResNet builder (32 filters baseline — mirrors CIFAR-100 recipe)
# ---------------------------------------------------------------------------


def _residual_block(x, filters):
    """One residual block: Conv→BN→ReLU→Conv→BN→add skip→ReLU."""
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


def build_resnet(input_dim, n_classes, lr=0.001, optimizer=None, spatial_shape=_SPATIAL_SHAPE):
    """Small ResNet: 3 residual blocks → GAP → output. 32 filters baseline."""
    inp = keras.layers.Input(shape=(input_dim,))
    x = keras.layers.Reshape(spatial_shape)(inp)

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
# Phase 3: Benchmark trial runner
# ---------------------------------------------------------------------------


def run_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    """Train a model for one trial and return metrics."""
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

    # Convergence: first epoch hitting 10% train accuracy.
    # Random baseline is 0.5% for 200 classes, so 10% is a meaningful threshold.
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.10:
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
    """Draw schematic network diagrams as a key panel."""
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
    all_results, intrinsic_dim, save_path, elapsed=None, input_dim=_INPUT_DIM, n_classes=_N_CLASSES
):
    """Save a six-panel comparison figure with architecture schematics key."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    d = intrinsic_dim
    w_ub = d + n_classes - 1

    _known_colors = {
        "Standard (1024→512)": "steelblue",
        f"Manifold (2d→d, d={d})": "firebrick",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"PCA→{d}D + MLP-wide (4d→2d)": "mediumvioletred",
        "ResNet (Adam)": "mediumseagreen",
        f"ManifoldResNet-d (d={d})": "darkgreen",
        f"ManifoldResNet-2d (2d={2 * d})": "indigo",
        f"ManifoldResNet-UB (w*={w_ub})": "crimson",
        f"PCA({n_classes})→{n_classes}": "goldenrod",
        f"PCA({n_classes}) linear→{n_classes}": "darkgoldenrod",
        f"PCA({n_classes}) Whitney(2d={2 * d})→{n_classes}": "saddlebrown",
        f"Intrinsic Dim (PCA→{d}D→output)": "darkorange",
        "UB-PCA (PCA→d*→w*→C)": "deepskyblue",
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
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP-wide (4d→2d)": [d, 4 * d, 2 * d, n_classes],
        "ResNet (Adam)": [input_dim, 32, 32, 32, n_classes],
        f"ManifoldResNet-d (d={d})": [input_dim, d, d, d, n_classes],
        f"ManifoldResNet-2d (2d={2 * d})": [input_dim, 2 * d, 2 * d, 2 * d, n_classes],
        f"ManifoldResNet-UB (w*={w_ub})": [input_dim, w_ub, w_ub, w_ub, n_classes],
        f"PCA({n_classes})→{n_classes}": [n_classes, n_classes, n_classes],
        f"PCA({n_classes}) linear→{n_classes}": [n_classes, n_classes],
        f"PCA({n_classes}) Whitney(2d={2 * d})→{n_classes}": [n_classes, 2 * d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→output)": [d, d, n_classes],
        "UB-PCA (PCA→d*→w*→C)": [d, w_ub, n_classes],
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

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.6, 0.85], hspace=0.42, wspace=0.3)
    ax_val = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_par = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[2, :])
    ax_arch = fig.add_subplot(gs[3, :])

    fig.suptitle(
        f"ManifoldResNet Architecture Study: Tiny ImageNet  (d*={d}, w*=d*+C-1={w_ub})\n"
        f"input={input_dim:,}D (64×64×3)  |  {n_classes} classes{elapsed_str}\n"
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
    ax_acc.set_title("Final Validation Accuracy")
    ax_acc.set_ylim(0, float(max(means)) * 1.25 if max(means) > 0 else 1)
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

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ResNet vs ManifoldResNet on Tiny ImageNet (Universal Bottleneck Theorem)"
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
        default=3,
        help="Samples per class for per-class dimensionality (default 3 for 200 classes)",
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
    parser.add_argument(
        "--results-suffix",
        type=str,
        default="",
        help="Optional suffix for the results JSON/PNG filenames (e.g., '_smoke', '_metal')",
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print("\nLoading Tiny ImageNet (64×64×3, 200 classes)...")
    (X_train_mm, y_train), (X_val_mm, y_val) = load_tiny_imagenet()

    input_dim = X_train_mm.shape[1]
    n_classes = int(y_train.max()) + 1
    print(f"  Train: {X_train_mm.shape}, Val: {X_val_mm.shape}  (memory-mapped)")
    print(f"  Classes: {n_classes}  |  Input dim: {input_dim:,} (64×64×3 color images)")

    # Chunked per-feature standardisation — peak resident memory ≈ 4.7 GB
    # (the output) + small chunk temporaries, vs ~14 GB for StandardScaler.
    print("  Computing mean/std (streaming, chunk=4096)...")
    t_scale = time.perf_counter()
    mean, std = _streaming_mean_std(X_train_mm, chunk=4096)
    print(f"  Scaling train into new float32 array ({X_train_mm.nbytes / 1e9:.1f} GB)...")
    X_train = _scale_into_new(X_train_mm, mean, std, chunk=4096)
    print(f"  Scaling val into new float32 array ({X_val_mm.nbytes / 1e9:.2f} GB)...")
    X_val = _scale_into_new(X_val_mm, mean, std, chunk=4096)
    # Release the memmap handles so the kernel can reclaim their page cache.
    del X_train_mm, X_val_mm
    print(f"  Standardisation done in {time.perf_counter() - t_scale:.1f}s")

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
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    print(
        f"\nPer-class intrinsic dimensionality "
        f"(τ={args.tau}, {args.samples_per_class} samples/class):"
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
    # Skip per-class printout for 200 classes — too much spam. Summarise.
    all_maxes = [cd["max"] for cd in class_dims.values()]
    all_means = [cd["mean"] for cd in class_dims.values()]
    print(
        f"  Per-class summary: mean d = {np.mean(all_means):.1f}"
        f"  |  min max = {min(all_maxes)}  |  max max = {max(all_maxes)}"
    )

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(all_maxes)
    d_arch = intrinsic_dim
    w_ub = d_arch + n_classes - 1  # Universal Bottleneck: d* + C − 1
    print(f"\n>> Global intrinsic dim (mean): {global_dim}  |  Max per-class max: {intrinsic_dim}")
    print(f"   d_arch = {d_arch}  (manifold dimension — drives filter/bottleneck count)")
    print(f"   w*     = {w_ub}  (Universal Bottleneck: d* + C − 1 = {d_arch} + {n_classes} − 1)")
    print(f"   d_arch = {d_arch / input_dim * 100:.2f}% of ambient dimensions")

    # -----------------------------------------------------------------------
    # Results path — used by --plot-only and --only
    # -----------------------------------------------------------------------

    base = f"tiny_imagenet_resnet_manifold_architecture_results{args.results_suffix}"
    results_path = Path(__file__).resolve().parent / f"{base}.json"
    plot_path = str(results_path.with_suffix(".png"))

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
    X_val_pca = pca.transform(X_val).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d_arch}D captures {var_explained * 100:.1f}% of global variance")

    pca_nc = skPCA(n_components=n_classes)
    X_train_pca_nc = pca_nc.fit_transform(X_train).astype("float32")
    X_val_pca_nc = pca_nc.transform(X_val).astype("float32")
    var_explained_nc = pca_nc.explained_variance_ratio_.sum()
    print(f"  PCA to {n_classes}D captures {var_explained_nc * 100:.1f}% of global variance")

    architectures = {
        "Standard (1024→512)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_val,
        ),
        f"Manifold (2d→d, d={d_arch})": (
            lambda: build_manifold_model(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_val,
        ),
        f"PCA→{d_arch}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_val_pca,
        ),
        f"PCA→{d_arch}D + MLP-wide (4d→2d)": (
            lambda: build_pca_mlp_wide(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_val_pca,
        ),
        "ResNet (Adam)": (
            lambda: build_resnet(input_dim, n_classes, lr=args.lr, spatial_shape=_SPATIAL_SHAPE),
            X_train,
            X_val,
        ),
        f"ManifoldResNet-d (d={d_arch})": (
            lambda: build_manifold_resnet(
                input_dim, n_classes, d_arch, lr=args.lr, spatial_shape=_SPATIAL_SHAPE
            ),
            X_train,
            X_val,
        ),
        f"ManifoldResNet-2d (2d={2 * d_arch})": (
            lambda: build_manifold_resnet_2d(
                input_dim, n_classes, d_arch, lr=args.lr, spatial_shape=_SPATIAL_SHAPE
            ),
            X_train,
            X_val,
        ),
        f"ManifoldResNet-UB (w*={w_ub})": (
            lambda: build_manifold_resnet(
                input_dim, n_classes, w_ub, lr=args.lr, spatial_shape=_SPATIAL_SHAPE
            ),
            X_train,
            X_val,
        ),
        f"PCA({n_classes})→{n_classes}": (
            lambda: build_pca_intrinsic_dim_model(n_classes, n_classes, lr=args.lr),
            X_train_pca_nc,
            X_val_pca_nc,
        ),
        f"PCA({n_classes}) linear→{n_classes}": (
            lambda: build_pca_linear_model(n_classes, n_classes, lr=args.lr),
            X_train_pca_nc,
            X_val_pca_nc,
        ),
        f"PCA({n_classes}) Whitney(2d={2 * d_arch})→{n_classes}": (
            lambda: build_pca_nc_model(n_classes, n_classes, hidden_width=2 * d_arch, lr=args.lr),
            X_train_pca_nc,
            X_val_pca_nc,
        ),
        f"Intrinsic Dim (PCA→{d_arch}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_val_pca,
        ),
        "UB-PCA (PCA→d*→w*→C)": (
            lambda: build_universal_bottleneck_pca(n_classes, d_arch, lr=args.lr),
            X_train_pca,
            X_val_pca,
        ),
        f"UB-raw (→{w_ub}→{n_classes})": (
            lambda: build_universal_bottleneck_raw(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_val,
        ),
        f"UB-MLP (→{w_ub}→{w_ub}→{n_classes})": (
            lambda: build_universal_bottleneck_mlp(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_val,
        ),
        f"UB-PCA-MLP (→{w_ub + 1}→PCA→{w_ub + 1}→{n_classes})": (
            lambda: build_ub_pca_mlp(input_dim, n_classes, d_arch, lr=args.lr),
            X_train,
            X_val,
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
                y_val,
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
    print(f"Dataset: Tiny ImageNet ({input_dim:,}D, {n_classes} classes, 64×64×3)")
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
            f"  {name:<55}  acc={mean_acc:.4f}±{std_acc:.4f}  "
            f"params={n_params:>9,}  time={mean_time:.1f}s"
        )

    print("-" * 70)
    print("PARAMETER EFFICIENCY (accuracy per 1K parameters):")
    for name, results in all_results.items():
        mean_acc = np.mean([r["test_acc"] for r in results])
        n_params = results[0]["n_params"]
        eff = mean_acc / n_params * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    print("-" * 70)
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
        "dataset": "tiny_imagenet",
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
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
        "notes": (
            f"Universal Bottleneck Theorem: w* = d* + C − 1 = "
            f"{d_arch} + {n_classes} − 1 = {w_ub}. "
            "ManifoldResNet-UB uses w* as filter count at every residual block. "
            "For Tiny ImageNet, C=200 ≫ d*, so w* is driven almost entirely by "
            "the information floor (second floor-dominated data point after CIFAR-100). "
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
