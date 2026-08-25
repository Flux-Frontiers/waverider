"""Tests for dimensionality_profile: scale-resolved intrinsic dimension.

These are calibration tests, not smoke tests.  The estimators are exercised
against manifolds whose intrinsic dimension is known by construction, including
curved ones, because the failure mode being guarded against is a plausible
number that is simply wrong.  Tolerances are deliberately loose: the point is to
catch a regression that changes the estimator's behaviour, not to pin down a
value the estimator is not entitled to.
"""

import numpy as np
import pytest

from waverider.dimensionality_profile import (
    dimension_profile,
    find_plateau,
    knn_radii,
    local_pca_dimension,
    mle_levina_bickel,
    reach_proxy,
    tau_corrected,
    twonn,
)

# ---------------------------------------------------------------------------
# Manifolds of known intrinsic dimension
# ---------------------------------------------------------------------------


def _embed(Z, ambient, seed):
    """Rotate a low-dimensional parameterisation into an ambient space."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ Q.T


def cube(d, ambient=64, n=2000, seed=0):
    """Uniform d-cube: flat, so local PCA has no curvature to contend with."""
    return _embed(np.random.default_rng(seed).uniform(0, 1, (n, d)), ambient, seed + 1)


def sphere(d, ambient=64, n=2000, seed=0):
    """Uniform d-sphere: curved, so wide neighbourhoods inflate the estimate."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d + 1))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return _embed(Z, ambient, seed + 1)


@pytest.fixture(scope="module")
def sphere8():
    return sphere(8, ambient=64, n=3000, seed=1)


@pytest.fixture(scope="module")
def cube5():
    return cube(5, ambient=32, n=2000, seed=2)


# ---------------------------------------------------------------------------
# knn_radii — the scale must be reported, and must grow with k
# ---------------------------------------------------------------------------


class TestKnnRadii:
    def test_reports_scale_and_spread(self, sphere8):
        r = knn_radii(sphere8, k=20, n_probe=80, seed=0)
        assert r["p10"] <= r["median"] <= r["p90"]
        assert r["min"] <= r["median"] <= r["max"]
        assert r["k"] == 20

    def test_radius_increases_with_k(self, sphere8):
        small = knn_radii(sphere8, k=10, n_probe=80, seed=0)["median"]
        large = knn_radii(sphere8, k=100, n_probe=80, seed=0)["median"]
        assert large > small, "a larger neighbourhood must probe a wider scale"


# ---------------------------------------------------------------------------
# local_pca_dimension — the k-dependence this module exists to expose
# ---------------------------------------------------------------------------


class TestLocalPcaDimension:
    def test_requires_exactly_one_scale_parameter(self, cube5):
        with pytest.raises(ValueError):
            local_pca_dimension(cube5, tau=0.90, n_probe=10)
        with pytest.raises(ValueError):
            local_pca_dimension(cube5, k=10, radius=1.0, tau=0.90, n_probe=10)

    def test_estimate_grows_with_neighbourhood(self, sphere8):
        """The documented failure mode: k alone moves the answer."""
        narrow = local_pca_dimension(sphere8, k=5, tau=0.90, n_probe=60, seed=0)["median"]
        wide = local_pca_dimension(sphere8, k=80, tau=0.90, n_probe=60, seed=0)["median"]
        assert wide > narrow

    def test_estimate_grows_with_tau(self, cube5):
        low = local_pca_dimension(cube5, k=40, tau=0.80, n_probe=60, seed=0)["mean"]
        high = local_pca_dimension(cube5, k=40, tau=0.95, n_probe=60, seed=0)["mean"]
        assert high >= low

    def test_seed_makes_the_measurement_reproducible(self, cube5):
        a = local_pca_dimension(cube5, k=30, tau=0.90, n_probe=50, seed=7)
        b = local_pca_dimension(cube5, k=30, tau=0.90, n_probe=50, seed=7)
        assert np.array_equal(a["dims"], b["dims"])

    def test_different_seeds_sample_different_points(self, cube5):
        a = local_pca_dimension(cube5, k=30, tau=0.90, n_probe=50, seed=1)
        b = local_pca_dimension(cube5, k=30, tau=0.90, n_probe=50, seed=2)
        assert not np.array_equal(a["dims"], b["dims"])

    def test_radius_mode_records_the_scale(self, sphere8):
        r = knn_radii(sphere8, k=40, n_probe=60, seed=0)["median"]
        out = local_pca_dimension(sphere8, radius=r, tau=0.90, n_probe=60, seed=0)
        assert out["radius"] == pytest.approx(r)
        assert out["k"] is None
        assert out["n_used"] + out["n_skipped"] == out["n_probe"]

    def test_tiny_radius_skips_sparse_probe_points(self, sphere8):
        out = local_pca_dimension(sphere8, radius=1e-9, tau=0.90, n_probe=30, seed=0)
        assert out["n_skipped"] == 30, "no point has neighbours at this scale"


# ---------------------------------------------------------------------------
# The profile and its plateau — the recommended measurement
# ---------------------------------------------------------------------------


class TestProfileAndPlateau:
    def test_profile_is_ordered_and_scale_annotated(self, sphere8):
        prof = dimension_profile(sphere8, k_values=(10, 40, 80), tau=0.90, n_probe=60, seed=0)
        assert [p["k"] for p in prof] == [10, 40, 80]
        radii = [p["radius_median"] for p in prof]
        assert radii == sorted(radii), "radius must increase with k"

    def test_plateau_recovers_dimension_after_tau_correction(self, sphere8):
        """The end-to-end claim: profile + plateau + tau-correction ~ true d."""
        prof = dimension_profile(
            sphere8, k_values=(5, 10, 20, 40, 80, 160), tau=0.90, n_probe=100, seed=0
        )
        plateau = find_plateau(prof, tol=1.0)
        assert plateau is not None, "a clean manifold should show a plateau"
        corrected = tau_corrected(plateau["dimension"], 0.90)
        assert 6.0 <= corrected <= 10.0, f"expected ~8, got {corrected}"

    def test_no_plateau_on_full_rank_noise(self):
        """Isotropic noise has no tangent structure; report that, don't invent one."""
        X = np.random.default_rng(0).standard_normal((1500, 40))
        prof = dimension_profile(X, k_values=(5, 10, 20, 40, 80), tau=0.90, n_probe=60, seed=0)
        plateau = find_plateau(prof, tol=1.0, min_points=3)
        assert plateau is None or plateau["dimension"] > 10

    def test_find_plateau_needs_enough_points(self):
        assert find_plateau([{"k": 5, "median": 3.0, "radius_median": 1.0}], min_points=3) is None

    def test_reach_proxy_is_none_without_a_plateau(self, sphere8):
        prof = dimension_profile(sphere8, k_values=(10, 40), tau=0.90, n_probe=40, seed=0)
        assert reach_proxy(prof, None) is None


# ---------------------------------------------------------------------------
# Reference estimators
# ---------------------------------------------------------------------------


class TestReferenceEstimators:
    def test_mle_recovers_known_dimension(self, sphere8):
        est = mle_levina_bickel(sphere8, k=5, n_probe=400, seed=0)
        assert 5.0 <= est <= 11.0, f"expected ~8, got {est}"

    def test_mle_is_k_stable_where_local_pca_is_not(self, sphere8):
        """The reason MLE is worth carrying as a cross-check.

        On a clean manifold the MLE barely moves with the neighbourhood size,
        while local PCA at a fixed threshold roughly doubles over the same
        range.  Measured here: MLE 7.7 -> 7.2 across k = 3..80, against local
        PCA 3 -> 7.  (Pope et al. report MLE *rising* with k on real image
        data; that is a property of heterogeneous real data, not of a clean
        synthetic manifold, so it is deliberately not asserted here.)
        """
        mle_small = mle_levina_bickel(sphere8, k=3, n_probe=300, seed=0)
        mle_large = mle_levina_bickel(sphere8, k=40, n_probe=300, seed=0)
        mle_drift = abs(mle_large - mle_small) / mle_small

        pca_small = local_pca_dimension(sphere8, k=5, tau=0.90, n_probe=60, seed=0)["median"]
        pca_large = local_pca_dimension(sphere8, k=80, tau=0.90, n_probe=60, seed=0)["median"]
        pca_drift = abs(pca_large - pca_small) / pca_small

        assert mle_drift < 0.20, f"MLE drifted {mle_drift:.0%} across k"
        assert pca_drift > mle_drift, "local PCA should be the more scale-sensitive of the two"

    def test_probe_point_is_not_its_own_nearest_neighbour(self, cube5):
        """Regression guard for the expanded-form distance computation.

        Distances are computed as |p|^2 - 2p.x + |x|^2, which leaves a probe
        point's distance to itself at a small positive residual rather than
        exactly zero.  MLE and TwoNN exclude the probe point by dropping zero
        distances, so an unpinned residual becomes the nearest neighbour and
        the estimate collapses toward zero.
        """
        est = mle_levina_bickel(cube5, k=5, n_probe=200, seed=0)
        assert est > 1.0, f"estimate collapsed to {est}; self-distance not excluded"

    def test_twonn_returns_a_finite_estimate(self, sphere8):
        est = twonn(sphere8, n_probe=400, seed=0)
        assert np.isfinite(est) and est > 0

    def test_estimators_agree_on_a_flat_manifold(self, cube5):
        mle = mle_levina_bickel(cube5, k=5, n_probe=400, seed=0)
        assert 3.0 <= mle <= 8.0, f"expected ~5, got {mle}"


class TestTauCorrection:
    def test_removes_the_threshold_factor(self):
        assert tau_corrected(9.0, 0.90) == pytest.approx(10.0)

    def test_rejects_invalid_tau(self):
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                tau_corrected(10.0, bad)
