#!/usr/bin/env python3
"""
Digits Benchmark: Manifold-Aware KNN vs Standard KNN
=====================================================

No neural network. No training. No Adam. Just geometry.

Question: does understanding the manifold structure of raw pixel space
give you better classification than treating the space as isotropic?

Dataset: sklearn digits — 1797 samples of 8×8 pixel images (64-dimensional
input), 10 classes (digits 0-9).  Evaluation uses 5-fold stratified
cross-validation (k_vote=7, k_pca=50).

Seven methods are compared
--------------------------
  Euclidean KNN:                  standard sklearn KNN, Euclidean distance
  Cosine KNN:                     standard sklearn KNN, cosine distance
  ManifoldKNN (τ=0.95/0.90/0.85): at each query point, find k_pca nearest
                                   neighbors, compute local PCA to discover
                                   the tangent space (d dims), project
                                   neighbors onto it, and vote in the
                                   projected subspace.  Off-manifold
                                   dimensions (noise) are suppressed.
  EigenWeightedManifoldKNN (τ=0.95/0.90): like ManifoldKNN but weights
                                   projected distances by 1/sqrt(λ) —
                                   a Mahalanobis-like metric on the local
                                   tangent space that emphasises fine-
                                   structure directions over coarse ones.

Both ManifoldKNN and EigenWeightedManifoldKNN are implemented here without
depending on waverider's ManifoldModel (lightweight, no graph construction).

Results are printed to stdout; no JSON or plot output is produced.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/digits_manifold_knn.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Import ManifoldWalker directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Manifold-Aware KNN Classifier
# ---------------------------------------------------------------------------


class ManifoldKNN:
    """K-nearest-neighbor classifier using manifold-projected distances.

    At each query point:
    1. Find k_pca nearest neighbors in the training set
    2. Compute local PCA to discover the tangent space (d dimensions)
    3. Project the query and its k_vote nearest neighbors onto this tangent space
    4. Measure distances in the projected d-dim space (not the ambient 64-dim space)
    5. Vote by majority among k_vote nearest in projected space

    The key insight: in the full 64-dim space, many dimensions are noise
    (background pixels, irrelevant variation). The manifold projection
    strips these away, measuring distance only along dimensions where
    the data actually varies — the strokes, curves, and edges that
    distinguish digits.
    """

    def __init__(
        self,
        k_vote=5,
        k_pca=50,
        variance_threshold=0.95,
    ):
        self.k_vote = k_vote
        self.k_pca = k_pca
        self.variance_threshold = variance_threshold
        self.intrinsic_dims = []

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype="d")
        self.y_train = np.asarray(y)
        return self

    def _local_pca(self, point, neighbors):
        """PCA on a neighborhood, return projection matrix and intrinsic dim."""
        centered = (neighbors - neighbors.mean(axis=0)).astype(np.float64)
        cov = np.einsum("ij,ik->jk", centered, centered) / (len(neighbors) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Intrinsic dimensionality
        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            d = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        else:
            d = len(eigenvalues)

        d = max(1, min(d, len(eigenvalues)))
        return eigenvectors[:, :d], d, eigenvalues

    def predict(self, X):
        X = np.asarray(X, dtype="d")
        predictions = np.zeros(len(X), dtype=self.y_train.dtype)

        for i, query in enumerate(X):
            # Find k_pca nearest neighbors for PCA
            dists = np.linalg.norm(self.X_train - query, axis=1)
            pca_idx = np.argpartition(dists, self.k_pca)[: self.k_pca]
            pca_neighbors = self.X_train[pca_idx]

            # Local PCA — discover tangent space
            V_d, d, eigenvalues = self._local_pca(query, pca_neighbors)
            self.intrinsic_dims.append(d)

            # Project query and ALL training points onto tangent space
            # (more efficient: just project the k_pca neighborhood)
            query_proj = V_d.T @ (query - pca_neighbors.mean(axis=0))
            neighbor_proj = (pca_neighbors - pca_neighbors.mean(axis=0)) @ V_d

            # Distances in projected space
            proj_dists = np.linalg.norm(neighbor_proj - query_proj, axis=1)

            # k_vote nearest in projected space
            vote_idx = np.argpartition(proj_dists, min(self.k_vote, len(proj_dists) - 1))[
                : self.k_vote
            ]
            vote_labels = self.y_train[pca_idx[vote_idx]]

            # Majority vote
            counts = np.bincount(vote_labels.astype(int), minlength=10)
            predictions[i] = np.argmax(counts)

        return predictions

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# ---------------------------------------------------------------------------
# Eigenvalue-Weighted Manifold KNN
# ---------------------------------------------------------------------------


class EigenWeightedManifoldKNN(ManifoldKNN):
    """Manifold KNN with eigenvalue-weighted distances.

    Instead of treating all on-manifold directions equally, weight
    distances by inverse eigenvalue — take large steps in low-variance
    directions (fine structure) and small steps in high-variance
    directions (coarse structure). This is a Mahalanobis-like distance
    on the local tangent space.
    """

    def predict(self, X):
        X = np.asarray(X, dtype="d")
        predictions = np.zeros(len(X), dtype=self.y_train.dtype)

        for i, query in enumerate(X):
            dists = np.linalg.norm(self.X_train - query, axis=1)
            pca_idx = np.argpartition(dists, self.k_pca)[: self.k_pca]
            pca_neighbors = self.X_train[pca_idx]

            V_d, d, eigenvalues = self._local_pca(query, pca_neighbors)
            self.intrinsic_dims.append(d)

            centroid = pca_neighbors.mean(axis=0)
            query_proj = V_d.T @ (query - centroid)
            neighbor_proj = (pca_neighbors - centroid) @ V_d

            # Eigenvalue weighting: scale each dimension by 1/sqrt(λ)
            # This gives Mahalanobis-like distance on the tangent space
            weights = np.ones(d)
            for j in range(d):
                if eigenvalues[j] > 0:
                    weights[j] = 1.0 / np.sqrt(eigenvalues[j])

            query_weighted = query_proj * weights
            neighbor_weighted = neighbor_proj * weights

            proj_dists = np.linalg.norm(neighbor_weighted - query_weighted, axis=1)

            vote_idx = np.argpartition(proj_dists, min(self.k_vote, len(proj_dists) - 1))[
                : self.k_vote
            ]
            vote_labels = self.y_train[pca_idx[vote_idx]]

            counts = np.bincount(vote_labels.astype(int), minlength=10)
            predictions[i] = np.argmax(counts)

        return predictions


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("DIGITS BENCHMARK: Manifold-Aware KNN vs Standard KNN")
    print("No neural network. No training. Just geometry.")
    print("=" * 70)

    # Load data
    data = load_digits()
    X, y = data.data.astype("float64"), data.target
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} dimensions, {len(set(y))} classes")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 5-fold cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    k_vote = 7
    k_pca = 50

    methods = {
        "Euclidean KNN": lambda: KNeighborsClassifier(n_neighbors=k_vote, metric="euclidean"),
        "Cosine KNN": lambda: KNeighborsClassifier(n_neighbors=k_vote, metric="cosine"),
        "Manifold KNN (tau=0.95)": lambda: ManifoldKNN(
            k_vote=k_vote, k_pca=k_pca, variance_threshold=0.95
        ),
        "Manifold KNN (tau=0.90)": lambda: ManifoldKNN(
            k_vote=k_vote, k_pca=k_pca, variance_threshold=0.90
        ),
        "Manifold KNN (tau=0.85)": lambda: ManifoldKNN(
            k_vote=k_vote, k_pca=k_pca, variance_threshold=0.85
        ),
        "EigenWeighted Manifold (tau=0.95)": lambda: EigenWeightedManifoldKNN(
            k_vote=k_vote, k_pca=k_pca, variance_threshold=0.95
        ),
        "EigenWeighted Manifold (tau=0.90)": lambda: EigenWeightedManifoldKNN(
            k_vote=k_vote, k_pca=k_pca, variance_threshold=0.90
        ),
    }

    results = {}
    dims_report = {}

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

            if hasattr(clf, "intrinsic_dims") and clf.intrinsic_dims:
                all_dims.extend(clf.intrinsic_dims)

            print(f"  Fold {fold_i + 1}: {acc:.4f} ({elapsed:.2f}s)")

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_time = np.mean(fold_times)
        results[name] = (mean_acc, std_acc, mean_time)

        if all_dims:
            dims_report[name] = (np.mean(all_dims), np.std(all_dims), min(all_dims), max(all_dims))

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<38} {'Accuracy':>12} {'Time':>10}")
    print("-" * 70)

    best_acc = max(v[0] for v in results.values())

    for name, (mean_acc, std_acc, mean_time) in results.items():
        marker = " << BEST" if mean_acc == best_acc else ""
        print(f"{name:<38} {mean_acc:.4f} +/- {std_acc:.4f} {mean_time:>8.2f}s{marker}")

    # Intrinsic dimensionality report
    if dims_report:
        print("\n" + "-" * 70)
        print("INTRINSIC DIMENSIONALITY OF DIGIT MANIFOLD")
        print("-" * 70)
        for name, (mean_d, std_d, min_d, max_d) in dims_report.items():
            noise_pct = 100 * (1 - mean_d / X.shape[1])
            print(f"  {name}:")
            print(f"    Mean: {mean_d:.1f} / {X.shape[1]} dims")
            print(f"    Range: [{min_d}, {max_d}]")
            print(f"    Noise dimensions: {noise_pct:.1f}%")

    print("\n" + "=" * 70)

    # Determine winner
    euclidean_acc = results["Euclidean KNN"][0]
    manifold_accs = {k: v[0] for k, v in results.items() if "Manifold" in k}
    best_manifold_name = max(manifold_accs, key=manifold_accs.get)
    best_manifold_acc = manifold_accs[best_manifold_name]

    if best_manifold_acc > euclidean_acc:
        delta = best_manifold_acc - euclidean_acc
        print(f"MANIFOLD WINS: {best_manifold_name}")
        print(f"  {best_manifold_acc:.4f} vs {euclidean_acc:.4f} (euclidean)")
        print(f"  Improvement: +{delta:.4f} ({delta * 100:.2f}%)")
    else:
        print(f"Euclidean KNN wins: {euclidean_acc:.4f} vs {best_manifold_acc:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
