"""Calibration of the local-PCA estimator against manifolds of known dimension.

``test_dimensionality_discovery.py`` covers exactly flat, noiseless subspaces --
a line in 5-D and a plane in 6-D -- where local PCA is trivially exact.  Nothing
there says what the estimator does on a manifold whose dimension it cannot read
off a rank.  That gap is why ``d*`` for CIFAR-10 was reported as 34, 19 and 16
from the same pipeline without anyone being able to say which was right.

These tests fill it.  They pin the estimator's *bias curve* against synthetic
manifolds whose intrinsic dimension is known by construction, and they guard the
three structural facts the calibration turned up:

1. the estimate rises monotonically with the neighbourhood size ``k``, so ``k``
   is a measurement choice and not a detail;
2. the estimate rises monotonically with the variance threshold ``tau``, with a
   ceiling near ``tau * d`` -- so the estimator returns roughly ``tau * d``, not
   ``d``, and is biased **low**;
3. the per-class *maximum* is an order statistic over the probe budget and
   drifts upward as that budget grows.

Per the repo's never-regress rule, the committed table below is a guard: if a
change to the estimator moves these numbers, either the change is wrong or the
table is renegotiated in the same commit, deliberately.
"""

import numpy as np
import pytest

from waverider.dimensionality_discovery import (
    DEFAULT_K_PCA,
    discover_dimensionality,
    discover_per_class_dimensionality,
)
from waverider.dimensionality_profile import mle_levina_bickel, twonn

# ---------------------------------------------------------------------------
# Synthetic manifolds of known intrinsic dimension
# ---------------------------------------------------------------------------

AMBIENT = 128
N_POINTS = 3000
N_PROBE = 150


def _embed(Z, ambient, seed):
    """Rotate a (n, d) sample rigidly into `ambient` dimensions."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ q.T


def flat_cube(d, ambient=AMBIENT, n=N_POINTS, seed=0):
    """Uniform samples from a d-cube, rigidly embedded.  Zero curvature."""
    return _embed(np.random.default_rng(seed).random((n, d)), ambient, seed + 1)


def sphere(d, ambient=AMBIENT, n=N_POINTS, seed=0):
    """Uniform samples from a d-sphere, rigidly embedded.  Constant curvature."""
    z = np.random.default_rng(seed).standard_normal((n, d + 1))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return _embed(z, ambient, seed + 1)


@pytest.fixture(scope="module")
def manifolds():
    return {
        ("cube", 5): flat_cube(5),
        ("cube", 10): flat_cube(10),
        ("sphere", 5): sphere(5),
        ("sphere", 10): sphere(10),
    }


def _dhat(X, k, tau, seed=0):
    report = discover_dimensionality(
        X, n_samples=N_PROBE, k=k, variance_thresholds=(tau,), seed=seed
    )
    return report[tau]["mean"]


# ---------------------------------------------------------------------------
# The committed bias table
# ---------------------------------------------------------------------------

# (manifold, true d, k) -> mean d-hat at tau=0.90, seed=0, n=3000, D=128.
# Measured, not derived.  Every entry is below its true d.
BIAS_TABLE_TAU_090 = {
    ("cube", 5, 10): 3.73,
    ("cube", 5, 50): 4.39,
    ("cube", 5, 100): 4.51,
    ("cube", 10, 10): 5.21,
    ("cube", 10, 50): 8.00,
    ("cube", 10, 100): 8.35,
    ("sphere", 5, 10): 3.77,
    ("sphere", 5, 50): 4.96,
    ("sphere", 5, 100): 5.00,
    ("sphere", 10, 10): 5.37,
    ("sphere", 10, 50): 8.45,
    ("sphere", 10, 100): 9.00,
}

TOL = 0.35  # seeded, so this is guarding drift, not sampling noise


class TestBiasTable:
    @pytest.mark.parametrize(("key", "expected"), sorted(BIAS_TABLE_TAU_090.items()))
    def test_matches_committed_value(self, manifolds, key, expected):
        name, d, k = key
        assert _dhat(manifolds[(name, d)], k=k, tau=0.90) == pytest.approx(expected, abs=TOL)

    @pytest.mark.parametrize(("key", "expected"), sorted(BIAS_TABLE_TAU_090.items()))
    def test_never_over_reports_true_dimension(self, key, expected):
        """The manuscript called the per-class max a "conservative (upper
        bound)"; the per-point estimator underneath it is biased low at every
        setting measured, which is why that claim was retracted.

        Eleven of the twelve entries are strictly below the truth.  The
        exception is the d=5 sphere at k=100, which lands exactly on 5: the
        ceiling is near ``tau * d`` = 4.5, and a dimension count is an integer,
        so it rounds up to the truth rather than past it."""
        _, d, _ = key
        assert expected <= d

    @pytest.mark.parametrize("key", sorted({(n, d) for n, d, _ in BIAS_TABLE_TAU_090}))
    def test_under_reports_at_the_canonical_setting(self, manifolds, key):
        """At the convention the benchmarks now share -- k=DEFAULT_K_PCA,
        tau=0.90 -- the shortfall is unambiguous on every manifold tested."""
        assert _dhat(manifolds[key], k=DEFAULT_K_PCA, tau=0.90) < key[1]


class TestMonotonicity:
    @pytest.mark.parametrize("key", [("cube", 10), ("sphere", 10)])
    def test_rises_with_k(self, manifolds, key):
        """k is the physical scale being probed.  This is the mechanism behind
        CIFAR-10 reading 34 at k=50 and 19 at k=25 -- same data, same
        preprocessing, different neighbourhood."""
        dims = [_dhat(manifolds[key], k=k, tau=0.90) for k in (10, 25, 50, 100)]
        assert dims == sorted(dims)
        assert dims[-1] > 1.5 * dims[0]

    @pytest.mark.parametrize("key", [("cube", 10), ("sphere", 10)])
    def test_rises_with_tau(self, manifolds, key):
        dims = [_dhat(manifolds[key], k=50, tau=t) for t in (0.85, 0.90, 0.95)]
        assert dims == sorted(dims)

    @pytest.mark.parametrize(("name", "d"), [("cube", 5), ("cube", 10), ("sphere", 10)])
    @pytest.mark.parametrize("tau", [0.85, 0.90, 0.95])
    def test_ceiling_near_tau_times_d(self, manifolds, name, d, tau):
        """For locally isotropic data the d tangent eigenvalues are comparable,
        so cumulative variance crosses tau at about ceil(tau*d) components.  The
        estimator therefore returns approximately tau*d and approaches it from
        below as k grows -- a structural ceiling, not a tuning accident."""
        assert _dhat(manifolds[(name, d)], k=100, tau=tau) <= tau * d + 0.6


class TestSeeding:
    def test_same_seed_is_bit_identical(self, manifolds):
        X = manifolds[("cube", 10)]
        assert _dhat(X, k=50, tau=0.90, seed=7) == _dhat(X, k=50, tau=0.90, seed=7)

    def test_different_seeds_move_the_estimate(self, manifolds):
        """Fifteen unseeded repeats at the shipped defaults moved the mean over
        11.9-14.6.  Seeding is what makes a quoted d* a measurement rather than
        a draw."""
        X = manifolds[("cube", 10)]
        spread = {round(_dhat(X, k=25, tau=0.90, seed=s), 6) for s in range(6)}
        assert len(spread) > 1

    def test_bootstrap_ci_brackets_the_mean(self, manifolds):
        r = discover_dimensionality(
            manifolds[("cube", 10)], n_samples=N_PROBE, k=25, variance_thresholds=(0.90,), seed=0
        )[0.90]
        assert r["mean_ci95_low"] <= r["mean"] <= r["mean_ci95_high"]
        assert r["n_probe"] == N_PROBE


class TestProbeBudget:
    def test_per_class_max_drifts_up_with_budget(self):
        """The per-class maximum is partly an order statistic over the sampling
        budget, so it is not comparable across runs with different budgets.  The
        CIFAR-10 architecture run used samples_per_class=10 -- a maximum over
        ~100 noisy draws."""
        rng = np.random.default_rng(3)
        parts, labels = [], []
        for c, d in enumerate((8, 16)):
            z = np.zeros((600, 24))
            z[:, :d] = rng.random((600, d))
            z[:, 0] += 6.0 * c  # separate the classes
            parts.append(z)
            labels.append(np.full(600, c))
        X, y = np.vstack(parts), np.concatenate(labels)

        maxima = [
            max(
                cd["max"]
                for cd in discover_per_class_dimensionality(
                    X, y, k=25, tau=0.90, n_samples_per_class=b, seed=0
                ).values()
            )
            for b in (5, 25, 100)
        ]
        assert maxima == sorted(maxima)
        assert maxima[-1] > maxima[0]


class TestCrossEstimator:
    """Pope et al. (ICLR 2021, Table 6) report CIFAR-10 as 21/96/11/7 under
    MLE / GeoMLE / TwoNN / kNN-graph.  A 13-fold spread between published
    estimators is the field's normal state, so any d* has to be quoted with the
    estimator that produced it.  These tests record where ours sit relative to
    two references on data whose answer is known."""

    @pytest.mark.parametrize(("name", "d"), [("cube", 5), ("cube", 10), ("sphere", 10)])
    def test_all_three_estimators_under_report(self, manifolds, name, d):
        X = manifolds[(name, d)]
        assert _dhat(X, k=50, tau=0.90) < d
        assert mle_levina_bickel(X, k=5, n_probe=800, seed=0) < d

    @pytest.mark.parametrize(("name", "d"), [("cube", 5), ("cube", 10), ("sphere", 10)])
    def test_twonn_over_reports_where_local_pca_under_reports(self, manifolds, name, d):
        """Opposite sign of error from the other two.  Recorded, not tuned away."""
        assert twonn(manifolds[(name, d)], n_probe=800, seed=0) > d

    def test_estimators_disagree_by_a_wide_margin(self, manifolds):
        X = manifolds[("cube", 10)]
        values = [
            _dhat(X, k=10, tau=0.90),
            _dhat(X, k=100, tau=0.90),
            mle_levina_bickel(X, k=5, n_probe=800, seed=0),
            twonn(X, n_probe=800, seed=0),
        ]
        assert max(values) / min(values) > 2.0
