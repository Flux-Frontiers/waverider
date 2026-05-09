"""
Manifold dimensionality discovery utilities.

Shared by all canonical benchmark scripts.  Provides local-PCA-based
intrinsic dimensionality estimation, both globally and per-class.
"""

import numpy as np
from scipy.linalg import svd


def _local_eigenvalues(neighbors: np.ndarray) -> np.ndarray:
    """Return eigenvalues of the local covariance via thin SVD.

    Avoids forming the (n_dims × n_dims) covariance matrix.  The centered
    neighbor matrix has shape (k, n_dims); its thin SVD gives singular values
    s where eigenvalues of cov = s² / (k - 1).  Cost is O(k² × n_dims)
    rather than O(n_dims³).

    :param neighbors: Array of shape (k, n_dims), float64.
    :returns: Eigenvalues in descending order, shape (k,).
    """
    centered = (neighbors - neighbors.mean(axis=0)).astype(np.float64)
    _, s, _ = svd(centered, full_matrices=False, check_finite=False)
    return (s**2) / max(len(neighbors) - 1, 1)


def discover_dimensionality(X, n_samples=500, k=50, variance_thresholds=(0.95, 0.90, 0.85)):
    """Discover intrinsic dimensionality of the data manifold via local PCA.

    Samples n_samples random points, computes local PCA at each using thin
    SVD on the (k × n_dims) neighbor matrix, and returns statistics on
    intrinsic dimensionality at each variance threshold.

    :param X: Data matrix of shape (n_points, n_dims).
    :param n_samples: Number of random points to sample.
    :param k: Neighborhood size for local PCA.
    :param variance_thresholds: Iterable of τ values to report.
    :returns: Dict mapping each τ to a statistics dict with keys
        mean, std, median, min, max.
    """
    n_points, _ = X.shape
    sample_idx = np.random.choice(n_points, size=min(n_samples, n_points), replace=False)
    k_use = min(k, n_points - 1)

    results = {tau: [] for tau in variance_thresholds}
    n_sample = len(sample_idx)

    for i, idx in enumerate(sample_idx):
        if (i + 1) % 10 == 0 or (i + 1) == n_sample:
            end = "\n" if (i + 1) == n_sample else "\r"
            print(f"  Local PCA: {i + 1}/{n_sample}", end=end, flush=True)

        point = X[idx]
        dists = np.linalg.norm(X - point, axis=1)
        knn_idx = np.argpartition(dists, k_use)[:k_use]

        eigenvalues = _local_eigenvalues(X[knn_idx])
        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            for tau in variance_thresholds:
                d = int(np.searchsorted(cumulative, tau) + 1)
                results[tau].append(d)

    report = {}
    for tau in variance_thresholds:
        dims = results[tau]
        if not dims:
            report[tau] = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0, "max": 0}
        else:
            report[tau] = {
                "mean": float(np.mean(dims)),
                "std": float(np.std(dims)),
                "median": float(np.median(dims)),
                "min": int(np.min(dims)),
                "max": int(np.max(dims)),
            }
    return report


def discover_per_class_dimensionality(X, y, k=50, tau=0.90, n_samples_per_class=50):
    """Discover intrinsic dimensionality per class via local PCA.

    For each class, samples n_samples_per_class points and estimates the
    local intrinsic dimensionality using thin SVD on the (k × n_dims)
    neighbor matrix.

    :param X: Data matrix of shape (n_points, n_dims).
    :param y: Class labels of shape (n_points,).
    :param k: Neighborhood size for local PCA.
    :param tau: Variance threshold.
    :param n_samples_per_class: Number of random points to sample per class.
    :returns: Dict mapping class label to a statistics dict with keys
        mean, std, min, max.
    """
    classes = sorted(set(y))
    class_dims = {}

    for c in classes:
        X_c = X[y == c]
        n_sample = min(n_samples_per_class, len(X_c))
        sample_idx = np.random.choice(len(X_c), size=n_sample, replace=False)
        k_use = min(k, len(X_c) - 1)

        dims = []
        for idx in sample_idx:
            point = X_c[idx]
            dists = np.linalg.norm(X_c - point, axis=1)
            knn_idx = np.argpartition(dists, k_use)[:k_use]

            eigenvalues = _local_eigenvalues(X_c[knn_idx])
            total = eigenvalues.sum()
            if total > 0:
                cumulative = np.cumsum(eigenvalues) / total
                d = int(np.searchsorted(cumulative, tau) + 1)
                dims.append(d)

        class_dims[c] = {
            "mean": float(np.mean(dims)),
            "std": float(np.std(dims)),
            "min": int(np.min(dims)),
            "max": int(np.max(dims)),
        }

    return class_dims
