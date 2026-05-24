#!/usr/bin/env python3
"""
Iris Benchmark: Manifold-Informed Architecture vs Standard Architecture
=======================================================================

The entry point to the manifold series: 4-dimensional flower measurements,
150 samples, 3 classes.  The iris dataset famously lives on a ~2-dimensional
manifold inside 4D space — the compression story is 4→2 (50% noise reduction).

Small scale makes this fast and interpretable.  The hypothesis: a bottleneck
matching the intrinsic dimension should match or exceed a standard architecture
with fewer parameters.

Phase 1: Discover intrinsic dimensionality via local PCA
Phase 2: Build architectures:
  - Standard:        4 → 16 → 8 → 3
  - Wide Manifold:   4 → 4d → 2d → d → 3   (bottleneck = max per-class dim)
  - Manifold:        4 → 2d → d → 3
  - PCA→dD + MLP:    d → 2d → d → 3         (PCA-projected input)
  - Intrinsic Dim:   PCA→dD → d → 3          (raw manifold subspace)
Phase 3: Train all with Adam, compare accuracy & parameter efficiency

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD

Usage
-----
    python benchmarks/canonical_tests/iris_manifold_architecture.py [--epochs 300] [--trials 10]
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

tf, DEVICE_INFO = setup_tensorflow()
import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


# ---------------------------------------------------------------------------
# Phase 2: Model Builders
# ---------------------------------------------------------------------------


def build_standard_model(input_dim, n_classes, lr=0.001):
    """Standard MLP for Iris: 4 → 16 → 8 → 3."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Manifold-informed: input → 2d → d → output."""
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_wide_manifold_model(input_dim, n_classes, intrinsic_dim, lr=0.001):
    """Wider manifold-informed: input → 4d → 2d → d → output."""
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(4 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected: d → 2d → d → output."""
    d = max(intrinsic_dim, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pca_intrinsic_dim_model(n_classes, intrinsic_dim, lr=0.001):
    """PCA pre-projected: d → d → output."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(intrinsic_dim,)),
            keras.layers.Dense(intrinsic_dim, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


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

    # Convergence: first epoch hitting 90% train accuracy
    conv_epoch = None
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.90:
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


def plot_results(all_results, intrinsic_dim, save_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    palette = [
        "steelblue",
        "forestgreen",
        "firebrick",
        "darkorange",
        "mediumpurple",
    ]
    names = list(all_results.keys())
    colors = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Iris: Manifold-Informed Architecture (d={intrinsic_dim}) vs Standard\n"
        f"4D flower measurements → manifold discovery → architecture  |  3 classes",
        fontsize=13,
        fontweight="bold",
    )

    # Validation accuracy curves
    ax = axes[0, 0]
    for name, results in all_results.items():
        accs = np.array([r["val_acc"] for r in results])
        epochs = np.arange(1, accs.shape[1] + 1)
        color = colors[name]
        ax.plot(epochs, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs,
            accs.mean(0) - accs.std(0),
            accs.mean(0) + accs.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Training loss curves
    ax = axes[0, 1]
    for name, results in all_results.items():
        losses = np.array([r["train_loss"] for r in results])
        epochs = np.arange(1, losses.shape[1] + 1)
        color = colors[name]
        ax.plot(epochs, losses.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Final test accuracy
    ax = axes[1, 0]
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors[n] for n in names]
    short_names = [n.split("(")[0].strip() for n in names]
    bars = ax.bar(short_names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.7, 1.05)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    # Parameter count (log scale)
    ax = axes[1, 1]
    param_counts = [all_results[n][0]["n_params"] for n in names]
    bars = ax.bar(short_names, param_counts, color=bar_colors, alpha=0.8)
    for bar, p, m in zip(bars, param_counts, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{p:,}\nacc={m:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Parameters")
    ax.set_title("Parameter Count (lower is better at same accuracy)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Iris: Manifold-Informed Architecture vs Standard")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--k-pca", type=int, default=10)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figure from existing results JSON without running any training",
    )
    args = parser.parse_args()

    results_path = Path(__file__).resolve().parent / "iris_architecture_results.json"
    plot_path = str(results_path.with_suffix(".png"))

    if args.plot_only:
        if not results_path.exists():
            print(f"No results file found: {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            saved = json.load(f)
        plot_results(saved["results"], saved["intrinsic_dim"], plot_path)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    print("\nLoading Iris...")
    iris = load_iris()
    X_all = iris.data.astype("float32")
    y_all = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, random_state=42, stratify=y_all
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    input_dim = X_train.shape[1]
    n_classes = len(set(y_train))
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {n_classes} ({', '.join(IRIS_CLASSES)})")
    print(f"  Input dim: {input_dim} (sepal/petal length+width)")

    # -----------------------------------------------------------------------
    # Phase 1: Discover intrinsic dimensionality
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    print(f"\nSampling all {len(X_train)} points, k={args.k_pca} neighbors...")
    dim_report = discover_dimensionality(
        X_train,
        n_samples=len(X_train),
        k=args.k_pca,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )

    print(f"\n{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
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
        X_train,
        y_train,
        k=args.k_pca,
        tau=args.tau,
        n_samples_per_class=10,
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        label = IRIS_CLASSES[c] if c < len(IRIS_CLASSES) else str(c)
        print(f"  {label:>12}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = global_dim
    print(
        f"\n>> Global intrinsic dim (mean): {global_dim}  |  Using d = {intrinsic_dim} (rounded mean)"
    )
    print(
        f"   Using d = {intrinsic_dim} (τ={args.tau})  =  {intrinsic_dim / input_dim * 100:.0f}% of ambient dimensions"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Build architectures
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    d = intrinsic_dim

    from sklearn.decomposition import PCA as skPCA

    pca = skPCA(n_components=d)
    X_train_pca = pca.fit_transform(X_train).astype("float32")
    X_test_pca = pca.transform(X_test).astype("float32")
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    architectures = {
        "Standard (16→8)": (
            lambda: build_standard_model(input_dim, n_classes, lr=args.lr),
            X_train,
            X_test,
        ),
        f"Wide Manifold (4d→2d→d, d={d})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
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
            X_train_pca,
            X_test_pca,
        ),
        f"Intrinsic Dim (PCA→{d}D→output)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            X_train_pca,
            X_test_pca,
        ),
    }

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
                f"time={result['wall_time']:.2f}s"
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
        f"Dataset: Iris ({input_dim}D, {n_classes} classes, {len(X_train)} train / {len(X_test)} test)"
    )
    print(
        f"Intrinsic dimensionality: d = {intrinsic_dim} (global mean: {global_dim}, τ={args.tau})"
    )
    print(f"Noise dimensions: {100 * (1 - intrinsic_dim / input_dim):.0f}%")
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
        ("Wall Time (s)", "wall_time", ".2f"),
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
    row = f"{'Epochs to 90%':<25}"
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
        eff = mean_acc / max(n_params, 1) * 1000
        print(f"  {name}: {eff:.4f} acc/Kparam  ({mean_acc:.4f} / {n_params:,})")

    # Winner
    print("-" * 70)
    best_name = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_acc = np.mean([r["test_acc"] for r in all_results[best_name]])
    std_name = "Standard (16→8)"
    std_acc = np.mean([r["test_acc"] for r in all_results[std_name]])

    if best_name != std_name:
        delta = best_acc - std_acc
        print(f">> MANIFOLD-INFORMED WINS: {best_name}")
        print(f"   {best_acc:.4f} vs {std_acc:.4f} (standard)  Δ={delta:+.4f}")
        best_params = all_results[best_name][0]["n_params"]
        std_params = all_results[std_name][0]["n_params"]
        if best_params < std_params:
            reduction = 100 * (1 - best_params / std_params)
            print(f"   With {reduction:.0f}% FEWER parameters ({best_params:,} vs {std_params:,})")
    else:
        print(f">> Standard architecture wins: {std_acc:.4f}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------

    save_data = {
        "device": DEVICE_INFO,
        "dataset": "iris",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "class_names": IRIS_CLASSES,
        "global_dim": global_dim,
        "intrinsic_dim": intrinsic_dim,
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {
            str(k): {
                **v,
                "class_name": IRIS_CLASSES[k] if k < len(IRIS_CLASSES) else str(k),
            }
            for k, v in class_dims.items()
        },
        "results": {name: results for name, results in all_results.items()},
    }

    results_path = Path(__file__).resolve().parent / "iris_architecture_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.plot:
        plot_path = str(Path(__file__).resolve().parent / "iris_architecture_results.png")
        plot_results(all_results, intrinsic_dim, plot_path)


if __name__ == "__main__":
    main()
