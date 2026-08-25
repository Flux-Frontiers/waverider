"""Tests for manifold_reach: the rank-k local-vs-global reconstruction gap.

These check the instrument's mechanics -- held-out evaluation, per-fit
centering, and the null-subtraction arithmetic -- independent of whether any
real manifold shows curvature. Whether it *recovers a known reach* is a
calibration question, covered separately in
``test_manifold_reach_calibration.py``.
"""

import numpy as np
import pytest

from waverider.manifold_reach import (
    corrected_gap,
    gaussian_null,
    held_out_split,
    local_rank_k_reconstruction_mse,
    rank_k_reconstruction_mse,
    reconstruction_gap,
)


def _embed(Z, ambient, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ q.T


def linear_subspace(d, ambient=32, n=800, noise=0.0, seed=0):
    """Flat d-subspace, rigidly embedded, with optional isotropic noise."""
    rng = np.random.default_rng(seed)
    X = _embed(rng.standard_normal((n, d)), ambient, seed + 1)
    if noise > 0:
        X = X + rng.normal(0, noise, X.shape)
    return X


def sphere(d, ambient=32, n=800, radius=1.0, seed=0):
    """Uniform d-sphere of the given radius, rigidly embedded."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d + 1))
    Z = radius * Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return _embed(Z, ambient, seed + 1)


# ---------------------------------------------------------------------------
# held_out_split
# ---------------------------------------------------------------------------


class TestHeldOutSplit:
    def test_disjoint_and_covers_all_points(self):
        fit_idx, holdout_idx = held_out_split(100, 20, seed=0)
        assert len(fit_idx) == 80
        assert len(holdout_idx) == 20
        assert set(fit_idx).isdisjoint(holdout_idx)
        assert set(fit_idx) | set(holdout_idx) == set(range(100))

    def test_clamps_holdout_below_n_points(self):
        fit_idx, holdout_idx = held_out_split(10, 50, seed=0)
        assert len(fit_idx) >= 1
        assert len(holdout_idx) == 9

    def test_reproducible_with_seed(self):
        a = held_out_split(100, 20, seed=3)
        b = held_out_split(100, 20, seed=3)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
# rank_k_reconstruction_mse -- global fit
# ---------------------------------------------------------------------------


class TestRankKReconstructionMse:
    def test_exact_rank_recovers_zero_error(self):
        """A noiseless d-subspace fit at k=d reconstructs held-out points exactly."""
        X = linear_subspace(d=5, ambient=32, n=500, noise=0.0, seed=1)
        fit_idx, holdout_idx = held_out_split(len(X), 100, seed=0)
        mse = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=5)
        assert mse == pytest.approx(0.0, abs=1e-8)

    def test_underranked_fit_has_positive_error(self):
        X = linear_subspace(d=5, ambient=32, n=500, noise=0.0, seed=1)
        fit_idx, holdout_idx = held_out_split(len(X), 100, seed=0)
        mse = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=2)
        assert mse > 1e-6

    def test_error_shrinks_as_k_grows(self):
        X = linear_subspace(d=8, ambient=32, n=500, noise=0.05, seed=2)
        fit_idx, holdout_idx = held_out_split(len(X), 100, seed=0)
        small_k = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=2)
        large_k = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=8)
        assert large_k < small_k


# ---------------------------------------------------------------------------
# local_rank_k_reconstruction_mse
# ---------------------------------------------------------------------------


class TestLocalRankKReconstructionMse:
    def test_skips_points_below_min_neighbors(self):
        X = sphere(d=5, ambient=32, n=300, seed=0)
        fit_idx, holdout_idx = held_out_split(len(X), 40, seed=0)
        out = local_rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=5, radius=1e-9)
        assert out["n_skipped"] == 40
        assert out["n_used"] == 0
        assert np.isnan(out["mse"])

    def test_used_plus_skipped_equals_holdout(self):
        X = sphere(d=5, ambient=32, n=300, seed=0)
        fit_idx, holdout_idx = held_out_split(len(X), 40, seed=0)
        out = local_rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=5, radius=0.5)
        assert out["n_used"] + out["n_skipped"] == 40

    def test_small_neighborhood_on_sphere_beats_global(self):
        """At small radius the tangent plane is nearly exact; the global fit,
        forced to average over the whole sphere, is not."""
        X = sphere(d=5, ambient=32, n=1500, radius=1.0, seed=3)
        fit_idx, holdout_idx = held_out_split(len(X), 100, seed=0)
        global_mse = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=5)
        local = local_rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k=5, radius=0.5)
        assert local["n_used"] > 0
        assert local["mse"] < global_mse


# ---------------------------------------------------------------------------
# reconstruction_gap
# ---------------------------------------------------------------------------


class TestReconstructionGap:
    def test_output_ordered_by_radius(self):
        X = sphere(d=4, ambient=24, n=600, seed=0)
        out = reconstruction_gap(X, k=4, radii=[0.8, 0.2, 0.5], n_holdout=50, seed=0)
        radii = [e["radius"] for e in out["by_radius"]]
        assert radii == sorted(radii)

    def test_gap_is_global_minus_local(self):
        X = sphere(d=4, ambient=24, n=600, seed=0)
        out = reconstruction_gap(X, k=4, radii=[0.75], n_holdout=50, seed=0)
        entry = out["by_radius"][0]
        assert entry["gap"] == pytest.approx(out["mse_global"] - entry["mse_local"])

    def test_zero_gap_on_a_flat_subspace(self):
        """No curvature to exploit: local and global fits should agree closely."""
        X = linear_subspace(d=4, ambient=24, n=600, noise=0.02, seed=5)
        out = reconstruction_gap(X, k=4, radii=[0.5, 1.0, 2.0], n_holdout=80, seed=0)
        for entry in out["by_radius"]:
            if entry["n_used"] > 5:
                assert abs(entry["gap"]) < 0.05


# ---------------------------------------------------------------------------
# gaussian_null / corrected_gap
# ---------------------------------------------------------------------------


class TestGaussianNullAndCorrectedGap:
    def test_null_matches_moments(self):
        X = sphere(d=5, ambient=16, n=2000, seed=0)
        null = gaussian_null(X, seed=1)
        assert null.shape == X.shape
        assert np.allclose(null.mean(axis=0), X.mean(axis=0), atol=0.2)

    def test_corrected_gap_arithmetic(self):
        real = {"k": 3, "by_radius": [{"radius": 1.0, "gap": 5.0}, {"radius": 2.0, "gap": 3.0}]}
        null = {"k": 3, "by_radius": [{"radius": 1.0, "gap": 1.0}, {"radius": 2.0, "gap": 4.0}]}
        out = corrected_gap(real, null)
        assert out[0] == {"radius": 1.0, "gap_real": 5.0, "gap_null": 1.0, "gap_corrected": 4.0}
        assert out[1] == {"radius": 2.0, "gap_real": 3.0, "gap_null": 4.0, "gap_corrected": -1.0}

    def test_mismatched_k_rejected(self):
        real = {"k": 3, "by_radius": [{"radius": 1.0, "gap": 5.0}]}
        null = {"k": 4, "by_radius": [{"radius": 1.0, "gap": 1.0}]}
        with pytest.raises(ValueError):
            corrected_gap(real, null)

    def test_mismatched_radii_rejected(self):
        real = {"k": 3, "by_radius": [{"radius": 1.0, "gap": 5.0}]}
        null = {"k": 3, "by_radius": [{"radius": 2.0, "gap": 1.0}]}
        with pytest.raises(ValueError):
            corrected_gap(real, null)
