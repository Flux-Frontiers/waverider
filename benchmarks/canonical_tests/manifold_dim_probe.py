#!/usr/bin/env python3
"""
Manifold Dimension Probe
========================

Train ManifoldResNet-d (our best model, d*=19 on CIFAR-10) with NO suppression
of any learned feature dimensions.  After training, forward-pass the entire
dataset through the GlobalAveragePooling layer (the d*-dimensional learned
feature vector) and analyze what the network encoded in EVERY dimension —
especially the "extra" ones that manifold suppression would normally zero out.

Scientific question
-------------------
The ManifoldWalker spec (Algorithm 1, Phase 5) suppresses off-manifold
gradient components during descent.  In ManifoldResNet-d, the filter count d*
implicitly limits representation to the intrinsic subspace — but the network
is free to use all d* dimensions however it likes.

Local PCA of raw CIFAR-10 at τ=0.90 needs k≈16 dims to capture 90% of
variance.  The network has 19 filters.  What does it put in dims 16–18?

Hypotheses
----------
H1 (noise):         Extra dims are near-zero — the network self-suppresses.
H2 (uncertainty):   Extra-dim magnitude correlates with misclassification.
H3 (inter-class):   Extra dims cluster visually similar class pairs
                    (vehicles together, animals together).
H4 (curvature):     Extra dims are high for samples near class boundaries
                    (high-entropy softmax predictions).

Outputs
-------
    manifold_dim_probe_results.json   — per-sample activations + labels + metrics
    manifold_dim_probe.png            — 6-panel analysis figure

Part of WaveRider.
Author: Eric G. Suchanek, PhD
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# TensorFlow / Metal setup  (must happen before import)
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
print(f"TensorFlow {tf.__version__} | Device: {_device_label}")

MACHINE_INFO = "M5 Max, MacBook Pro, 64GB RAM, 2TB SSD"

DEVICE_INFO = {
    "tensorflow_version": tf.__version__,
    "device_used": _device_label,
    "machine": MACHINE_INFO,
    "invocation": " ".join(sys.argv),
}

import keras  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from model_builder import build_manifold_resnet  # noqa: E402
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
)

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
    # aquatic mammals
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    # fish
    "aquarium fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    # flowers
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip",
    # food containers
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate",
    # fruit & vegetables
    "apple",
    "mushroom",
    "orange",
    "pear",
    "sweet pepper",
    # household electrical
    "clock",
    "keyboard",
    "lamp",
    "telephone",
    "television",
    # household furniture
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    # insects
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    # large carnivores
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    # large outdoor things
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    # large outdoor scenes
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
    # large omnivores/herbivores
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    # medium mammals
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    # non-insect invertebrates
    "crab",
    "lobster",
    "snail",
    "spider",
    "worm",
    # people
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    # reptiles
    "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    # small mammals
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    # trees
    "maple tree",
    "oak tree",
    "palm tree",
    "pine tree",
    "willow tree",
    # vehicles 1
    "bicycle",
    "bus",
    "motorcycle",
    "pickup truck",
    "train",
    # vehicles 2
    "lawn-mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
]

# ---------------------------------------------------------------------------
# Probe model: tap the GlobalAveragePooling2D layer
# ---------------------------------------------------------------------------


def build_probe_model(trained_model):
    """Return a model with the same input but GAP-layer output.

    The GlobalAveragePooling2D layer is the last pooling layer before the
    Dense classifier — it produces the d*-dimensional learned feature vector.

    :param trained_model: A fully trained ManifoldResNet-d Keras model.
    :returns: Keras Model whose output is the GAP activation (d*, float32).
    """
    gap_layer = None
    for layer in trained_model.layers:
        if isinstance(layer, keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
    if gap_layer is None:
        raise ValueError("No GlobalAveragePooling2D layer found in model.")
    return keras.Model(inputs=trained_model.input, outputs=gap_layer.output)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def activation_pca(acts):
    """PCA of the d*-dimensional learned representations.

    :param acts: Array (N, d*) of GAP activations.
    :returns: Tuple (pca, acts_pca, explained_var_ratio).
    """
    pca = skPCA(n_components=acts.shape[1])
    acts_pca = pca.fit_transform(acts)
    return pca, acts_pca, pca.explained_variance_ratio_


def find_k_tau(explained_var_ratio, tau):
    """Minimum number of PCs capturing fraction tau of activation variance.

    :param explained_var_ratio: Array of explained variance ratios (sorted descending).
    :param tau: Variance threshold in (0, 1].
    :returns: Integer k.
    """
    cumvar = np.cumsum(explained_var_ratio)
    hits = np.where(cumvar >= tau)[0]
    return int(hits[0] + 1) if len(hits) else len(explained_var_ratio)


def per_class_mean_activations(acts, labels, n_classes):
    """Mean activation per dimension per class.

    :param acts: Array (N, d*).
    :param labels: Integer class labels (N,).
    :param n_classes: Number of classes.
    :returns: Array (n_classes, d*).
    """
    return np.array(
        [
            acts[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(acts.shape[1])
            for c in range(n_classes)
        ]
    )


def extra_dim_magnitude(acts, k_on):
    """L2 norm of the off-manifold (extra) dimensions per sample.

    :param acts: Array (N, d*) — activations in PCA space (already transformed).
    :param k_on: Number of on-manifold dimensions (first k_on PCs).
    :returns: Array (N,) of extra-dim L2 magnitudes.
    """
    if k_on >= acts.shape[1]:
        return np.zeros(acts.shape[0])
    return np.linalg.norm(acts[:, k_on:], axis=1)


def softmax_entropy(probs):
    """Shannon entropy of softmax probability vectors (in nats).

    :param probs: Array (N, C) of class probabilities.
    :returns: Array (N,) of entropies.
    """
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_probe_results(
    acts_pca,
    evr,
    labels,
    per_class_acts,
    extra_mag,
    is_correct,
    entropy,
    k_90,
    k_95,
    d_star,
    save_path,
    dataset="CIFAR-10",
    class_names=None,
    elapsed=None,
):
    """Six-panel figure summarising the dimension probe.

    Panel 1 — Eigen spectrum of learned representations (log scale).
    Panel 2 — Per-class mean activation heatmap (n_classes × d*).
    Panel 3 — Per-class extra-dim magnitude (box plots).
    Panel 4 — Extra-dim magnitude: correct vs incorrect predictions.
    Panel 5 — t-SNE of full d*-dim activations, coloured by class.
    Panel 6 — t-SNE of extra dims only (indices k_90 … d*-1).

    :param acts_pca: (N, d*) activations in PCA space.
    :param evr: Explained variance ratio array (d*,).
    :param labels: Integer class labels (N,).
    :param per_class_acts: (n_classes, d*) mean activations in original act space.
    :param extra_mag: (N,) L2 magnitude of extra dims.
    :param is_correct: (N,) bool — True if prediction matches label.
    :param entropy: (N,) softmax entropy per sample.
    :param k_90: Number of dims capturing 90% activation variance.
    :param k_95: Number of dims capturing 95% activation variance.
    :param d_star: Intrinsic dimensionality (total filter/feature count).
    :param save_path: Output PNG path.
    :param dataset: Dataset name string for titles.
    :param class_names: List of class name strings; defaults to CIFAR-10 names.
    :param elapsed: Total run time in seconds for the suptitle.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    n_classes = per_class_acts.shape[0]
    if class_names is None:
        class_names = CIFAR10_CLASSES[:n_classes]
    # For large class counts (e.g. CIFAR-100) use a wider colormap
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    colors = cmap(np.linspace(0, 1, n_classes))

    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = f"  |  run time: {elapsed:.0f}s" if elapsed is not None else ""
    fig.suptitle(
        f"Manifold Dimension Probe — ManifoldResNet-d* on {dataset}\n"
        f"d*={d_star} learned features | k_90={k_90} on-manifold dims | "
        f"{d_star - k_90} extra dims{elapsed_str}\n"
        f"Generated: {timestamp}  |  {MACHINE_INFO}\n"
        f"Invocation: {DEVICE_INFO['invocation']}",
        fontsize=12,
        fontweight="bold",
    )

    # --- Panel 1: Eigen spectrum ---
    dims = np.arange(1, d_star + 1)
    cumvar = np.cumsum(evr) * 100
    bar_colors = ["darkorchid" if i < k_90 else "tomato" for i in range(d_star)]
    ax1.bar(dims, evr * 100, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax1_r = ax1.twinx()
    ax1_r.plot(dims, cumvar, "k--", linewidth=1.5, label="Cumulative %")
    ax1_r.axhline(90, color="steelblue", linestyle=":", linewidth=1, label="90%")
    ax1_r.axhline(95, color="firebrick", linestyle=":", linewidth=1, label="95%")
    ax1_r.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax1_r.legend(fontsize=8, loc="center right")
    ax1.axvline(k_90 + 0.5, color="steelblue", linestyle="--", linewidth=1.5, label=f"k_90={k_90}")
    ax1.axvline(k_95 + 0.5, color="firebrick", linestyle="--", linewidth=1.5, label=f"k_95={k_95}")
    ax1.set_xlabel("PC index (learned feature)")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(
        "Eigen Spectrum of Learned d*-Dim Representations\n(purple=on-manifold, red=extra)"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: Per-class mean activation heatmap ---
    im = ax2.imshow(
        per_class_acts,
        aspect="auto",
        cmap="RdBu_r",
        norm=Normalize(vmin=per_class_acts.min(), vmax=per_class_acts.max()),
    )
    ax2.set_yticks(range(n_classes))
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel("Learned feature dimension")
    ax2.set_title("Per-Class Mean Activation Heatmap\n(rows=classes, cols=feature dims)")
    ax2.axvline(k_90 - 0.5, color="steelblue", linestyle="--", linewidth=1.5, label=f"k_90={k_90}")
    ax2.legend(fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Mean activation")
    for col in range(d_star):
        ax2.axvline(col - 0.5, color="gray", linewidth=0.2, alpha=0.4)

    # --- Panel 3: Extra-dim magnitude per class (box plots) ---
    extra_by_class = [extra_mag[labels == c] for c in range(n_classes)]
    bp = ax3.boxplot(extra_by_class, patch_artist=True, notch=False, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax3.set_xticks(range(1, n_classes + 1))
    ax3.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel(f"L2 magnitude (dims {k_90}–{d_star - 1})")
    ax3.set_title(f"Extra-Dim Magnitude per Class\n(dims {k_90}…{d_star - 1}, beyond 90% variance)")
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Extra-dim magnitude vs correctness + entropy ---
    correct_mag = extra_mag[is_correct]
    wrong_mag = extra_mag[~is_correct]
    ax4.hist(
        correct_mag,
        bins=40,
        alpha=0.6,
        color="mediumseagreen",
        label=f"Correct (n={is_correct.sum():,})",
        density=True,
    )
    ax4.hist(
        wrong_mag,
        bins=40,
        alpha=0.6,
        color="tomato",
        label=f"Wrong (n={(~is_correct).sum():,})",
        density=True,
    )
    ax4.set_xlabel(f"Extra-dim L2 magnitude (dims {k_90}–{d_star - 1})")
    ax4.set_ylabel("Density")
    ax4.set_title(
        "H2: Extra-Dim Magnitude vs Classification Outcome\n"
        "(right-shifted wrong → magnitude encodes uncertainty)"
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Correlation annotation
    corr = np.corrcoef(extra_mag.astype(float), (~is_correct).astype(float))[0, 1]
    ax4.text(
        0.97,
        0.95,
        f"ρ = {corr:.3f}",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # --- Panel 5: t-SNE of full d*-dim activations ---
    try:
        from sklearn.manifold import TSNE

        # Subsample for speed: 5K points
        rng = np.random.default_rng(42)
        idx = rng.choice(len(acts_pca), min(5000, len(acts_pca)), replace=False)
        tsne_in = acts_pca[idx]
        tsne_labels = labels[idx]
        tsne_acts = TSNE(
            n_components=2, perplexity=40, random_state=42, max_iter=1000
        ).fit_transform(tsne_in)
        for c in range(n_classes):
            mask = tsne_labels == c
            ax5.scatter(
                tsne_acts[mask, 0],
                tsne_acts[mask, 1],
                s=4,
                alpha=0.5,
                color=colors[c],
                label=class_names[c],
            )
        ax5.set_title(f"t-SNE: Full d*={d_star} Learned Features\n(5K samples, coloured by class)")
        ax5.legend(fontsize=7, markerscale=3, ncol=2, loc="upper right")
        ax5.axis("off")

        # --- Panel 6: t-SNE of extra dims only ---
        n_extra = d_star - k_90
        if n_extra >= 2:
            extra_slice = acts_pca[idx, k_90:]
            tsne_extra = TSNE(
                n_components=2, perplexity=40, random_state=42, max_iter=1000
            ).fit_transform(extra_slice)
            for c in range(n_classes):
                mask = tsne_labels == c
                ax6.scatter(
                    tsne_extra[mask, 0],
                    tsne_extra[mask, 1],
                    s=4,
                    alpha=0.5,
                    color=colors[c],
                    label=class_names[c],
                )
            ax6.set_title(
                f"t-SNE: EXTRA dims only (dims {k_90}–{d_star - 1})\n"
                f"H3: Do similar classes co-cluster here?"
            )
            ax6.legend(fontsize=7, markerscale=3, ncol=2, loc="upper right")
            ax6.axis("off")
        else:
            ax6.text(
                0.5,
                0.5,
                f"Only {n_extra} extra dim(s)\nt-SNE not applicable",
                ha="center",
                va="center",
                transform=ax6.transAxes,
                fontsize=12,
            )
            ax6.axis("off")
    except ImportError:
        ax5.text(
            0.5,
            0.5,
            "scikit-learn TSNE not available",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )
        ax6.text(
            0.5,
            0.5,
            "scikit-learn TSNE not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Manifold Dimension Probe — ManifoldResNet-d*")
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10",
        help="Dataset to probe (default: cifar10)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--tau", type=float, default=0.90, help="Variance threshold for intrinsic dim discovery"
    )
    parser.add_argument("--discovery-samples", type=int, default=500)
    parser.add_argument("--k-pca", type=int, default=25)
    parser.add_argument(
        "--metal",
        action="store_true",
        default=False,
        help="Allow TF-Metal GPU (recommended on Apple Silicon)",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        default=False,
        help="Re-render figure from saved JSON + npz without re-running",
    )
    args = parser.parse_args()
    t_start = time.perf_counter()

    out_dir = Path(__file__).resolve().parent
    _slug = args.dataset  # e.g. "cifar10" or "cifar100"
    json_path = out_dir / f"manifold_dim_probe_{_slug}_results.json"
    npz_path = out_dir / f"manifold_dim_probe_{_slug}_arrays.npz"
    png_path = out_dir / f"manifold_dim_probe_{_slug}.png"

    # -----------------------------------------------------------------------
    # Replot-only path
    # -----------------------------------------------------------------------
    if args.replot:
        if not json_path.exists() or not npz_path.exists():
            print(f"ERROR: need both {json_path} and {npz_path} to replot.")
            return
        with open(json_path) as f:
            results = json.load(f)
        arrays = np.load(npz_path)
        print("Generating figure from saved results...")
        plot_probe_results(
            acts_pca=arrays["acts_pca"],
            evr=np.array(list(results["eigenspectrum_summary"].values())),
            labels=arrays["labels"],
            per_class_acts=np.array(results["per_class_mean_activations"]),
            extra_mag=arrays["extra_mag"],
            is_correct=arrays["is_correct"],
            entropy=arrays["entropy"],
            k_90=results["activation_pca"]["k_90"],
            k_95=results["activation_pca"]["k_95"],
            d_star=results["d_star"],
            save_path=str(png_path),
        )
        return

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    if args.dataset == "cifar100":
        print("\nLoading CIFAR-100...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
        class_names = CIFAR100_CLASSES
        dataset_label = "CIFAR-100"
        n_classes = 100
    else:
        print("\nLoading CIFAR-10...")
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        class_names = CIFAR10_CLASSES
        dataset_label = "CIFAR-10"
        n_classes = 10

    X_train = X_train.reshape(-1, 3072).astype("float32")
    X_test = X_test.reshape(-1, 3072).astype("float32")
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    input_dim = 3072
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  Classes: {n_classes}")

    # Concatenated set for the forward pass
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    split = len(X_train)

    # -----------------------------------------------------------------------
    # Phase 1: Manifold discovery
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)
    dim_report = discover_dimensionality(
        X_train,
        n_samples=args.discovery_samples,
        k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    d_star = int(round(dim_report[args.tau]["mean"]))
    print(f"\n  Intrinsic dimensionality d* = {d_star}  (τ={args.tau})")
    print(f"  Raw pixel N={input_dim}, compression N/d* = {input_dim / d_star:.0f}×")

    # -----------------------------------------------------------------------
    # Phase 2: Train ManifoldResNet-d*  (single run — this is a probe)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PHASE 2: TRAINING ManifoldResNet-d* (d*={d_star}, {args.epochs} epochs)")
    print("=" * 70)

    model = build_manifold_resnet(input_dim, n_classes, d_star, lr=args.lr)
    model.summary()
    n_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"\n  Parameters: {n_params:,}")

    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    train_time = time.perf_counter() - t0
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test accuracy: {test_acc:.4f}  ({train_time:.1f}s)")

    # -----------------------------------------------------------------------
    # Phase 3: Extract d*-dimensional GAP activations for ALL data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: PROBING — extracting GAP activations (no suppression)")
    print("=" * 70)

    probe = build_probe_model(model)
    print(f"  Probe output shape: {probe.output_shape}")

    print(f"  Forward pass: {len(X_all):,} samples...")
    acts = probe.predict(X_all, batch_size=512, verbose=1)  # (N, d*)
    print(f"  Activations shape: {acts.shape}")

    # Softmax predictions (for correctness + entropy)
    probs = model.predict(X_all, batch_size=512, verbose=0)  # (N, n_classes)
    preds = probs.argmax(axis=1)
    is_correct = preds == y_all
    entropy = softmax_entropy(probs)

    train_acc_probe = is_correct[:split].mean()
    test_acc_probe = is_correct[split:].mean()
    print(f"  Probe-confirmed accuracy — train: {train_acc_probe:.4f}  test: {test_acc_probe:.4f}")

    # -----------------------------------------------------------------------
    # Phase 4: PCA of learned representations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: PCA OF LEARNED d*-DIM REPRESENTATIONS")
    print("=" * 70)

    pca, acts_pca, evr = activation_pca(acts)
    k_90 = find_k_tau(evr, 0.90)
    k_95 = find_k_tau(evr, 0.95)
    n_extra = d_star - k_90

    print("\n  Eigen spectrum (% variance per PC):")
    for i, v in enumerate(evr):
        marker = " ← k_90" if i == k_90 - 1 else (" ← k_95" if i == k_95 - 1 else "")
        tag = "[on-manifold]" if i < k_90 else "[EXTRA]"
        print(f"    PC{i:02d}: {v * 100:6.2f}%  {tag}{marker}")
    print(f"\n  k_90 = {k_90}  (dims 0–{k_90 - 1} capture 90% of activation variance)")
    print(f"  k_95 = {k_95}  (dims 0–{k_95 - 1} capture 95%)")
    print(f"  Extra dims: {n_extra}  (indices {k_90}–{d_star - 1})")

    # -----------------------------------------------------------------------
    # Phase 5: Hypothesis tests
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: HYPOTHESIS TESTS")
    print("=" * 70)

    extra_mag = extra_dim_magnitude(acts_pca, k_90)

    # H1: Are extra dims near-zero?
    extra_mean_mag = extra_mag.mean()
    on_mean_mag = np.linalg.norm(acts_pca[:, :k_90], axis=1).mean()
    h1_ratio = extra_mean_mag / (on_mean_mag + 1e-9)
    print(f"\nH1 (noise): extra-dim mean magnitude = {extra_mean_mag:.4f}")
    print(f"            on-manifold mean magnitude = {on_mean_mag:.4f}")
    print(
        f"            ratio (extra/on) = {h1_ratio:.4f}  ({'near-zero ✓' if h1_ratio < 0.1 else 'non-trivial ✗'})"
    )

    # H2: Extra-dim magnitude vs misclassification
    correct_extra = extra_mag[is_correct].mean()
    wrong_extra = extra_mag[~is_correct].mean()
    h2_corr = np.corrcoef(extra_mag, (~is_correct).astype(float))[0, 1]
    print(f"\nH2 (uncertainty): mean extra-mag (correct) = {correct_extra:.4f}")
    print(f"                  mean extra-mag (wrong)   = {wrong_extra:.4f}")
    print(
        f"                  correlation with error   = {h2_corr:.4f}",
        "  ✓ encodes uncertainty" if h2_corr > 0.05 else "  ✗ no uncertainty signal",
    )

    # H3: Per-class extra-dim magnitude
    per_class_extra = [extra_mag[y_all == c].mean() for c in range(n_classes)]
    print("\nH3 (inter-class): extra-dim magnitude per class:")
    for c, (name, mag) in enumerate(zip(class_names, per_class_extra)):
        bar = "#" * int(mag * 20 / (max(per_class_extra) + 1e-9))
        print(f"    {name:12s}: {mag:.4f}  {bar}")

    # H4: Correlation with softmax entropy
    h4_corr = np.corrcoef(extra_mag, entropy)[0, 1]
    print(
        f"\nH4 (curvature/entropy): ρ(extra-mag, softmax entropy) = {h4_corr:.4f}",
        "  ✓ boundary signal" if h4_corr > 0.1 else "  ✗ no boundary signal",
    )

    # Per-class mean activations (original act space, not PCA)
    per_class_acts = per_class_mean_activations(acts, y_all, n_classes)

    # -----------------------------------------------------------------------
    # Phase 6: Save results
    # -----------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start

    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": float(elapsed),
        "device": DEVICE_INFO,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "d_star": d_star,
        "tau_discovery": args.tau,
        "epochs": args.epochs,
        "n_params": n_params,
        "test_accuracy": float(test_acc),
        "train_time_s": float(train_time),
        "n_total_samples": int(len(X_all)),
        "activation_pca": {
            "k_90": int(k_90),
            "k_95": int(k_95),
            "n_extra": int(n_extra),
            "explained_variance_ratio": [float(v) for v in evr],
            "cumulative_variance": [float(v) for v in np.cumsum(evr)],
        },
        "hypotheses": {
            "H1_noise": {
                "extra_mean_magnitude": float(extra_mean_mag),
                "on_mean_magnitude": float(on_mean_mag),
                "ratio": float(h1_ratio),
                "verdict": "near-zero (self-suppressed)"
                if h1_ratio < 0.1
                else "non-trivial signal",
            },
            "H2_uncertainty": {
                "mean_extra_mag_correct": float(correct_extra),
                "mean_extra_mag_wrong": float(wrong_extra),
                "correlation_with_error": float(h2_corr),
                "verdict": "encodes uncertainty" if h2_corr > 0.05 else "no uncertainty signal",
            },
            "H3_interclass": {
                "per_class_extra_magnitude": {
                    name: float(mag) for name, mag in zip(class_names, per_class_extra)
                },
            },
            "H4_entropy": {
                "correlation_extra_mag_entropy": float(h4_corr),
                "verdict": "boundary signal" if h4_corr > 0.1 else "no boundary signal",
            },
        },
        "per_class_mean_activations": per_class_acts.tolist(),
        "eigenspectrum_summary": {f"PC{i:02d}": float(v) for i, v in enumerate(evr)},
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    np.savez_compressed(
        npz_path,
        acts_pca=acts_pca,
        labels=y_all,
        extra_mag=extra_mag,
        is_correct=is_correct,
        entropy=entropy,
    )
    print(f"Arrays saved to {npz_path}")

    # -----------------------------------------------------------------------
    # Phase 7: Plot
    # -----------------------------------------------------------------------
    print("\nGenerating figure...")
    plot_probe_results(
        acts_pca=acts_pca,
        evr=evr,
        labels=y_all,
        per_class_acts=per_class_acts,
        extra_mag=extra_mag,
        is_correct=is_correct,
        entropy=entropy,
        k_90=k_90,
        k_95=k_95,
        d_star=d_star,
        save_path=str(png_path),
        dataset=dataset_label,
        class_names=class_names,
        elapsed=elapsed,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)
    print(f"  Run date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total run time:       {elapsed:.1f}s")
    print(f"  Device:               {DEVICE_INFO['device_used']}")
    print(f"  Machine:              {MACHINE_INFO}")
    print(f"  Invocation:           {DEVICE_INFO['invocation']}")
    print(f"  d* = {d_star}  |  k_90 = {k_90}  |  extra = {n_extra} dims")
    print(f"  Test accuracy:        {test_acc:.4f}")
    print(f"  Activation PCA k_90:  {k_90} dims capture 90% of learned representation variance")
    print(f"  Extra-dim ratio:      {h1_ratio:.4f}  (0 = pure noise, 1 = equal signal)")
    print(f"  ρ(extra, error):      {h2_corr:.4f}  (>0 = extra dims encode uncertainty)")
    print(f"  ρ(extra, entropy):    {h4_corr:.4f}  (>0 = extra dims track boundary ambiguity)")
    print("=" * 70)


if __name__ == "__main__":
    main()
