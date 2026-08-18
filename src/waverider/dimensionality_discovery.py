"""
Manifold dimensionality discovery utilities.

Shared by all canonical benchmark scripts.  Provides local-PCA-based
intrinsic dimensionality estimation, both globally and per-class.
"""

import numpy as np
from scipy.linalg import svd

#: Canonical neighborhood size for local PCA across the benchmark scripts.
#:
#: ``k`` is not a free knob: it sets the *physical scale* the estimator probes,
#: and local PCA recovers a tangent space only while that scale stays small
#: relative to the manifold's reach.  Probe wider and the local covariance mixes
#: tangent directions with curvature.  On identical synthetic data with three
#: regions of true local dimension 8/16/24, moving ``k`` from 10 to 100 moved the
#: reported per-class maximum from 8 to 25 -- a 3.1x swing with the data held
#: fixed.  That is the same mechanism that produced CIFAR-10 = 34 at ``k=50``
#: (``cifar_architecture_sweep.py``) and 19 at ``k=25``
#: (``resnet_manifold_architecture.py``), which for years read as a preprocessing
#: difference and is not one.
#:
#: 25 is chosen over 50 because Levina & Bickel (2004) and Pope et al. (ICLR
#: 2021, Table 5) both find that estimates rise monotonically with ``k`` and that
#: large ``k`` over-estimates, and because it is the value behind the published
#: ``d* = 19 -> w* = 28`` CIFAR-10 result.  Scripts on small-``n`` or low-ambient
#: data (iris, the clinical set) deviate deliberately and say so inline.
#:
#: This is a *convention*, not a measurement.  Quote ``k`` with every ``d*``.
DEFAULT_K_PCA = 25


def _bootstrap_mean_ci(dims, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of a per-point dimension sample.

    The point estimate of ``d*`` is not a constant: at the shipped defaults,
    fifteen repeat runs moved the mean statistic over 11.9-14.6 (sd 0.77), which
    is enough to move ``w* = d* + C - 1`` by one or two.  Reporting an interval
    makes that visible in the artifact instead of leaving it to be rediscovered.

    Deliberately *not* offered for the per-class maximum.  The maximum is an
    order statistic over the probe budget -- it drifts upward as the budget
    grows -- and the bootstrap of a sample maximum is inconsistent, so a CI on it
    would be a number that looks like a guarantee and is not one.

    :param dims: Per-probe-point dimension estimates.
    :param n_boot: Bootstrap resamples.
    :param alpha: Two-sided level; 0.05 gives a 95% interval.
    :param seed: Seed for the resampling; fixed so the interval is reproducible
        given the same per-point sample.
    :returns: ``(low, high)``, or ``(nan, nan)`` for an empty sample.
    """
    a = np.asarray(dims, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def estimator_params(
    k,
    tau=None,
    variance_thresholds=None,
    n_samples=None,
    n_samples_per_class=None,
    seed=None,
    aggregation=None,
    preprocessing=None,
    n_points=None,
    n_dims=None,
):
    """Build the provenance block that must travel with every ``d*``.

    ``resnet_manifold_architecture_results.json`` records ``tau`` but not ``k``,
    so ``d* = 19 -> w* = 28`` -- the headline of the bottleneck-width paper --
    is not reproducible from its own artifact.  Every benchmark should write this
    block beside its results so that the number can be recomputed without
    reading the script that produced it.

    Keys whose value is unknown are kept and set to ``None`` rather than dropped:
    an explicit ``"seed": null`` records that a run was unseeded, which is the
    fact a reader needs.

    :param k: Neighborhood size actually used.
    :param tau: The cumulative-variance threshold ``d*`` was read at.
    :param variance_thresholds: All thresholds reported, if several.
    :param n_samples: Global probe budget.
    :param n_samples_per_class: Per-class probe budget.
    :param seed: Seed passed to the estimator, or ``None`` if unseeded.
    :param aggregation: Which statistic became ``d*`` -- ``"per_class_max"``,
        ``"global_mean"``, ``"global_median"``.  The two conventions differ by 3
        on CIFAR-10 (19 vs 16), so naming it is not optional.
    :param preprocessing: e.g. ``"StandardScaler"``.
    :param n_points: Rows of the matrix the estimate was taken on.
    :param n_dims: Ambient dimension.
    :returns: JSON-serialisable dict.
    """
    return {
        "estimator": "local_pca_cumulative_variance",
        "k": None if k is None else int(k),
        "tau": None if tau is None else float(tau),
        "variance_thresholds": (
            None if variance_thresholds is None else [float(t) for t in variance_thresholds]
        ),
        "n_samples": None if n_samples is None else int(n_samples),
        "n_samples_per_class": (None if n_samples_per_class is None else int(n_samples_per_class)),
        "seed": None if seed is None else int(seed),
        "aggregation": aggregation,
        "preprocessing": preprocessing,
        "n_points": None if n_points is None else int(n_points),
        "n_dims": None if n_dims is None else int(n_dims),
    }


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


def discover_dimensionality(
    X, n_samples=500, k=50, variance_thresholds=(0.95, 0.90, 0.85), seed=None
):
    """Discover intrinsic dimensionality of the data manifold via local PCA.

    Samples n_samples random points, computes local PCA at each using thin
    SVD on the (k × n_dims) neighbor matrix, and returns statistics on
    intrinsic dimensionality at each variance threshold.

    :param X: Data matrix of shape (n_points, n_dims).
    :param n_samples: Number of random points to sample.
    :param k: Neighborhood size for local PCA.
    :param variance_thresholds: Iterable of τ values to report.
    :param seed: Seed for probe-point selection.  ``None`` draws from the
        global NumPy RNG, preserving the historical behaviour; pass an int to
        make the measurement reproducible.  The estimate moves by roughly a
        dimension between draws at default settings, so an unseeded call is
        not repeatable and should not be quoted as a measured constant.
    :returns: Dict mapping each τ to a statistics dict with keys mean, std,
        median, min, max, n_probe, and a percentile-bootstrap 95% interval on
        the mean (mean_ci95_low / mean_ci95_high).  No interval is given for
        ``max``: it is an order statistic over the probe budget and the
        bootstrap of a sample maximum is inconsistent.  Keys are the thresholds
        themselves, so the mapping stays sortable -- call
        :func:`estimator_params` for the provenance block rather than adding
        string keys here.
    """
    n_points, _ = X.shape
    size = min(n_samples, n_points)
    if seed is None:
        sample_idx = np.random.choice(n_points, size=size, replace=False)
    else:
        sample_idx = np.random.default_rng(seed).choice(n_points, size=size, replace=False)
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
            report[tau] = {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0,
                "n_probe": 0,
                "mean_ci95_low": float("nan"),
                "mean_ci95_high": float("nan"),
            }
        else:
            lo, hi = _bootstrap_mean_ci(dims)
            report[tau] = {
                "mean": float(np.mean(dims)),
                "std": float(np.std(dims)),
                "median": float(np.median(dims)),
                "min": int(np.min(dims)),
                "max": int(np.max(dims)),
                "n_probe": len(dims),
                "mean_ci95_low": lo,
                "mean_ci95_high": hi,
            }
    return report


def discover_per_class_dimensionality(X, y, k=50, tau=0.90, n_samples_per_class=50, seed=None):
    """Discover intrinsic dimensionality per class via local PCA.

    For each class, samples n_samples_per_class points and estimates the
    local intrinsic dimensionality using thin SVD on the (k × n_dims)
    neighbor matrix.

    :param X: Data matrix of shape (n_points, n_dims).
    :param y: Class labels of shape (n_points,).
    :param k: Neighborhood size for local PCA.
    :param tau: Variance threshold.
    :param n_samples_per_class: Number of random points to sample per class.
    :param seed: Seed for probe-point selection.  ``None`` draws from the global
        NumPy RNG, preserving the historical behaviour.  Note that the per-class
        *maximum* is an order statistic over ``n_samples_per_class`` draws, so it
        drifts upward as that budget grows and is not comparable across runs
        with different budgets.
    :returns: Dict mapping class label to a statistics dict with keys mean,
        std, min, max, n_probe, mean_ci95_low, mean_ci95_high.
    """
    classes = sorted(set(y))
    class_dims = {}
    rng = None if seed is None else np.random.default_rng(seed)

    for c in classes:
        X_c = X[y == c]
        n_sample = min(n_samples_per_class, len(X_c))
        chooser = np.random if rng is None else rng
        sample_idx = chooser.choice(len(X_c), size=n_sample, replace=False)
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

        lo, hi = _bootstrap_mean_ci(dims)
        class_dims[c] = {
            "mean": float(np.mean(dims)),
            "std": float(np.std(dims)),
            "min": int(np.min(dims)),
            "max": int(np.max(dims)),
            "n_probe": len(dims),
            "mean_ci95_low": lo,
            "mean_ci95_high": hi,
        }

    return class_dims
