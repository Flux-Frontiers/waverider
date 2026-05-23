#!/usr/bin/env python3
"""
MNIST Benchmark: ManifoldModel — The Ultimate Test on Real Data
===============================================================

Tests ManifoldModel — zero learned parameters, pure geometry — against
Euclidean KNN on the MNIST handwritten-digit dataset.

MNIST provides the ideal proving ground:
  - 784 ambient dimensions (28×28 pixels)
  - Real manifold structure (digits are smooth curves embedded in pixel space)
  - 60,000 training / 10,000 test images
  - 10 digit classes with varying geometric complexity

Because ManifoldModel's O(n²·d) graph construction is prohibitive at full
scale, training data is stratified-subsampled (--n-train, default 5,000).
Test evaluation uses a separate stratified subsample (--n-test, default 2,000).

Three comparisons (apples-to-apples on the same subsamples)
------------------------------------------------------------
  ManifoldModel (zero params):
    fit()     — discovers local geometry via local PCA (--k-pca=50),
                builds a k-NN graph (--k-graph=15) with manifold-weighted
                edges (--manifold-weight=0.8)
    predict() — classifies via graph-walk + manifold-projected voting
                (--k-vote=7), at variance threshold --tau (default 0.90)
  Euclidean KNN (subsampled):   sklearn KNN on the same n_train samples
  Euclidean KNN (full train):   sklearn KNN on all 60K training images
                                (upper-bound reference)

Results (including per-class accuracy and geometry statistics) are saved to
``benchmarks/mnist_manifold_model_results.json``.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/mnist_manifold_model.py [--n-train 5000] [--tau 0.90]
"""

import argparse
import json
import time

from benchmarks.tf_setup import setup_tensorflow

# ---------------------------------------------------------------------------
# TensorFlow (dataset loading only)
# ---------------------------------------------------------------------------


tf, _DEVICE_INFO = setup_tensorflow(tf_log_level="3")
import numpy as np  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from waverider.manifold_model import ManifoldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Bootstrap ManifoldModel imports
# ---------------------------------------------------------------------------


DIGIT_NAMES = [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# Stratified subsample
# ---------------------------------------------------------------------------


def stratified_subsample(X, y, n_samples, seed=42):
    """Subsample maintaining class balance."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per_class = n_samples // len(classes)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MNIST: ManifoldModel — The Ultimate Test on Real Data"
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5000,
        help="Training subsample size (default 5000)",
    )
    parser.add_argument(
        "--n-test", type=int, default=2000, help="Test subsample size (default 2000)"
    )
    parser.add_argument("--k-graph", type=int, default=15, help="Graph neighbors per node")
    parser.add_argument("--k-pca", type=int, default=50, help="PCA neighborhood size")
    parser.add_argument("--k-vote", type=int, default=7, help="Voting neighbors for prediction")
    parser.add_argument(
        "--manifold-weight",
        type=float,
        default=0.8,
        help="Manifold vs Euclidean distance blend",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MNIST: ManifoldModel — THE ULTIMATE TEST ON REAL DATA")
    print("Zero learned parameters. The manifold IS the model.")
    print("=" * 70)

    # Load MNIST
    tf.get_logger().setLevel("ERROR")
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

    # Flatten: (N, 28, 28) → (N, 784)
    X_train_full = X_train_full.reshape(-1, 784).astype("float64")
    X_test_full = X_test_full.reshape(-1, 784).astype("float64")

    # Normalize
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test_full = scaler.transform(X_test_full)

    # Subsample
    X_train, y_train = stratified_subsample(X_train_full, y_train_full, args.n_train, seed=42)
    X_test, y_test = stratified_subsample(X_test_full, y_test_full, args.n_test, seed=99)

    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print("\nDataset: MNIST (real handwritten digits)")
    print(f"  Full: {X_train_full.shape[0]} train, {X_test_full.shape[0]} test")
    print(f"  Subsample: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"  Dimensions: {input_dim} (28×28 pixels), Classes: {n_classes}")

    # -----------------------------------------------------------------------
    # Experiment: ManifoldModel at multiple tau values vs KNN
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT: ManifoldModel vs Euclidean KNN on REAL MNIST")
    print("=" * 70)

    tau_values = [0.95, 0.90, 0.85, 0.80]
    results = {}

    # Baseline: Euclidean KNN on subsample
    print(f"\nEuclidean KNN (k={args.k_vote}) on subsample ({args.n_train})...")
    t0 = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)
    knn_time = time.perf_counter() - t0
    results["Euclidean KNN (subsample)"] = {
        "accuracy": knn_acc,
        "time": knn_time,
        "params": 0,
    }
    print(f"  Accuracy: {knn_acc:.4f}  Time: {knn_time:.1f}s")

    # Baseline: Euclidean KNN on full data
    print(f"\nEuclidean KNN (k={args.k_vote}) on FULL training data (60K)...")
    t0 = time.perf_counter()
    knn_full = KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean")
    knn_full.fit(X_train_full, y_train_full)
    knn_full_acc = knn_full.score(X_test, y_test)
    knn_full_time = time.perf_counter() - t0
    results["Euclidean KNN (full 60K)"] = {
        "accuracy": knn_full_acc,
        "time": knn_full_time,
        "params": 0,
    }
    print(f"  Accuracy: {knn_full_acc:.4f}  Time: {knn_full_time:.1f}s")

    # ManifoldModel at each tau
    for tau in tau_values:
        name = f"ManifoldModel (tau={tau})"
        print(f"\n{name}...")
        print(
            f"  Building: k_graph={args.k_graph}, k_pca={args.k_pca}, "
            f"k_vote={args.k_vote}, manifold_weight={args.manifold_weight}"
        )

        model = ManifoldModel(
            k_graph=args.k_graph,
            k_pca=args.k_pca,
            k_vote=args.k_vote,
            variance_threshold=tau,
            manifold_weight=args.manifold_weight,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        print(f"  Fit time: {fit_time:.1f}s")

        # Geometry report
        summary = model.geometry_summary()
        print(f"  Nodes: {summary['n_nodes']}, Edges: {summary['n_edges']}")
        print(
            f"  Intrinsic dim: {summary['mean_intrinsic_dim']:.1f} ± "
            f"{summary['std_intrinsic_dim']:.1f} "
            f"[{summary['min_intrinsic_dim']}, {summary['max_intrinsic_dim']}]"
        )
        noise_pct = 100 * (1 - summary["mean_intrinsic_dim"] / input_dim)
        print(f"  Noise dimensions: {noise_pct:.1f}%")

        # Predict
        print(f"  Predicting {len(X_test)} test samples...")
        t0 = time.perf_counter()
        preds = model.predict(X_test)
        pred_time = time.perf_counter() - t0
        acc = float(np.mean(preds == y_test))
        total_time = fit_time + pred_time
        print(f"  Predict time: {pred_time:.1f}s")
        print(f"  Accuracy: {acc:.4f}  Total time: {total_time:.1f}s")

        results[name] = {
            "accuracy": acc,
            "time": total_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "params": 0,
            "geometry": summary,
        }

        # Per-class accuracy
        print("  Per-class accuracy:")
        for c in range(n_classes):
            mask = y_test == c
            if mask.sum() > 0:
                c_acc = float(np.mean(preds[mask] == y_test[mask]))
                print(f"    digit {c}: {c_acc:.4f} ({mask.sum()} samples)")

    # -----------------------------------------------------------------------
    # Fly demo
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FLY MODE DEMO: Navigating the MNIST manifold")
    print("=" * 70)

    # Use the best-performing model's tau (refit on subsample)
    best_tau = max(
        (t for t in tau_values),
        key=lambda t: results[f"ManifoldModel (tau={t})"]["accuracy"],
    )
    print(f"\nUsing best tau={best_tau}")

    fly_model = ManifoldModel(
        k_graph=args.k_graph,
        k_pca=args.k_pca,
        k_vote=args.k_vote,
        variance_threshold=best_tau,
        manifold_weight=args.manifold_weight,
    )
    fly_model.fit(X_train, y_train)

    # Fly from digit 0 to digit 9
    start_class, end_class = 0, 9
    start_idx = np.where(y_train == start_class)[0][0]
    end_idx = np.where(y_train == end_class)[0][0]

    fly_model.fly_to(f"n{start_idx}")
    geom = fly_model.get_geometry(f"n{start_idx}")
    print(f"\nStarting at n{start_idx} (digit {start_class})")
    print(f"  Local intrinsic dim: {geom.intrinsic_dim}")

    print(f"\nFlying toward digit {end_class} (n{end_idx})...")
    path = fly_model.fly_toward(X_train[end_idx], max_steps=25)
    for step, node_id in enumerate(path):
        idx = int(node_id[1:])
        node_geom = fly_model.get_geometry(node_id)
        dist = np.linalg.norm(X_train[idx] - X_train[end_idx])
        digit = y_train[idx]
        print(
            f"  Step {step + 1:>2}: {node_id} (digit {digit}, "
            f"intrinsic_dim={node_geom.intrinsic_dim:>2}, "
            f"dist={dist:.2f})"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: MNIST (real handwritten digits, {input_dim}D, {n_classes} classes)")
    print(f"Training subsample: {args.n_train}, Test: {args.n_test}")
    print(f"{'Method':<35} {'Accuracy':>10} {'Params':>10} {'Time':>10}")
    print("-" * 70)

    best_acc = max(v["accuracy"] for v in results.values())
    for name, r in results.items():
        marker = " << BEST" if r["accuracy"] == best_acc else ""
        print(f"{name:<35} {r['accuracy']:>10.4f} {r['params']:>10} {r['time']:>9.1f}s{marker}")

    # ManifoldModel vs KNN delta
    best_mm_name = max(
        (n for n in results if n.startswith("ManifoldModel")),
        key=lambda n: results[n]["accuracy"],
    )
    mm_acc = results[best_mm_name]["accuracy"]
    knn_sub_acc = results["Euclidean KNN (subsample)"]["accuracy"]
    delta = mm_acc - knn_sub_acc

    print("-" * 70)
    if delta > 0:
        print(
            f">> ManifoldModel BEATS Euclidean KNN (same data): "
            f"{mm_acc:.4f} vs {knn_sub_acc:.4f} (+{delta:.4f})"
        )
        print(f"   Winner: {best_mm_name}")
    elif delta == 0:
        print(f">> ManifoldModel TIES Euclidean KNN: {mm_acc:.4f}")
    else:
        print(f">> Euclidean KNN leads by {-delta:.4f}: {knn_sub_acc:.4f} vs {mm_acc:.4f}")

    # Geometry summary
    print(f"\nMANIFOLD GEOMETRY (real MNIST, {input_dim}D):")
    for name, r in results.items():
        if "geometry" in r:
            g = r["geometry"]
            noise = 100 * (1 - g["mean_intrinsic_dim"] / input_dim)
            print(f"  {name}: d = {g['mean_intrinsic_dim']:.1f}/{input_dim} ({noise:.1f}% noise)")

    print("=" * 70)

    # Save results
    save_data = {
        "dataset": "mnist",
        "input_dim": input_dim,
        "n_classes": n_classes,
        "n_train_subsample": args.n_train,
        "n_test_subsample": args.n_test,
        "results": {},
    }
    for name, r in results.items():
        save_r = {k: v for k, v in r.items() if k != "geometry"}
        if "geometry" in r:
            save_r["geometry"] = r["geometry"]
        save_data["results"][name] = save_r

    results_path = "benchmarks/mnist_manifold_model_results.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
