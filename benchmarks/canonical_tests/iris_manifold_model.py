#!/usr/bin/env python3
"""
Iris Benchmark: ManifoldModel vs Standard KNN
==============================================

Tests ManifoldModel — where the manifold IS the classifier (zero learned
parameters) — against standard Euclidean KNN on the sklearn Iris dataset.

Dataset: sklearn Iris — 150 samples of 4-dimensional flower measurements,
3 classes (setosa, versicolor, virginica).

Benchmark (5-fold stratified cross-validation, seed=42)
-------------------------------------------------------
  Euclidean KNN (k=5):     sklearn KNeighborsClassifier, Euclidean distance
  ManifoldModel (τ sweep): fit via local PCA (k_pca=20), k-NN graph with
                           k_graph=10, voting with k_vote=5

Per-fold mean intrinsic dimensionality is summarised at the end.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD

Usage
-----
    python benchmarks/canonical_tests/iris_manifold_model.py
"""

import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from waverider.manifold_model import ManifoldModel


def main() -> None:
    print("=" * 70)
    print("IRIS BENCHMARK: ManifoldModel vs Standard KNN")
    print("The manifold IS the model. No weights. Just geometry.")
    print("=" * 70)

    data = load_iris()
    X, y = data.data.astype("float64"), data.target
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} dims, {len(set(y))} classes")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    methods = {
        "Euclidean KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "ManifoldModel (tau=0.95)": lambda: ManifoldModel(
            k_graph=10, k_pca=20, k_vote=5, variance_threshold=0.95
        ),
        "ManifoldModel (tau=0.90)": lambda: ManifoldModel(
            k_graph=10, k_pca=20, k_vote=5, variance_threshold=0.90
        ),
        "ManifoldModel (tau=0.85)": lambda: ManifoldModel(
            k_graph=10, k_pca=20, k_vote=5, variance_threshold=0.85
        ),
        "ManifoldModel (tau=0.80)": lambda: ManifoldModel(
            k_graph=10, k_pca=20, k_vote=5, variance_threshold=0.80
        ),
    }

    results: dict[str, tuple[float, float, float]] = {}
    geom_reports: dict[str, float] = {}

    for name, make_clf in methods.items():
        fold_accs = []
        fold_times = []
        all_dims = []

        print(f"\n{name}...")
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf = make_clf()
            t0 = time.perf_counter()
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_te, y_te)
            elapsed = time.perf_counter() - t0

            fold_accs.append(acc)
            fold_times.append(elapsed)

            if hasattr(clf, "intrinsic_dim") and clf.intrinsic_dim is not None:
                all_dims.append(clf.intrinsic_dim)

            print(f"  Fold {fold_i + 1}: {acc:.4f} ({elapsed:.2f}s)")

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_time = float(np.mean(fold_times))
        results[name] = (mean_acc, std_acc, mean_time)

        if all_dims:
            geom_reports[name] = float(np.mean(all_dims))

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'Accuracy':>18} {'Time':>10}")
    print("-" * 70)

    best_acc = max(v[0] for v in results.values())
    for name, (mean_acc, std_acc, mean_time) in results.items():
        marker = " << BEST" if mean_acc == best_acc else ""
        print(f"{name:<35} {mean_acc:.4f} +/- {std_acc:.4f} {mean_time:>8.2f}s{marker}")

    if geom_reports:
        print("\n" + "-" * 70)
        print("MANIFOLD GEOMETRY")
        print("-" * 70)
        for name, mean_d in geom_reports.items():
            noise_pct = 100 * (1 - mean_d / X.shape[1])
            print(f"  {name}: intrinsic dim = {mean_d:.2f}/{X.shape[1]} ({noise_pct:.0f}% noise)")

    print("=" * 70)


if __name__ == "__main__":
    main()
