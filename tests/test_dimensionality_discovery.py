"""Tests for dimensionality_discovery: local-PCA intrinsic dimension estimators."""

import numpy as np
import pytest

from waverider.dimensionality_discovery import (
    discover_dimensionality,
    discover_per_class_dimensionality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def line_in_5d():
    """Points along a 1-D line embedded in 5-D — intrinsic dim should be 1."""
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 10, size=200)
    X = np.zeros((200, 5))
    X[:, 0] = t
    return X


@pytest.fixture
def plane_in_6d():
    """Points on a 2-D plane in 6-D — intrinsic dim should be 2."""
    rng = np.random.default_rng(1)
    t1 = rng.uniform(0, 5, size=300)
    t2 = rng.uniform(0, 5, size=300)
    X = np.zeros((300, 6))
    X[:, 0] = t1
    X[:, 1] = t2
    return X


@pytest.fixture
def two_class_data():
    """Two Gaussian classes in 8-D for per-class tests."""
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((100, 8)) + np.array([5.0] + [0.0] * 7)
    X1 = rng.standard_normal((100, 8)) + np.array([-5.0] + [0.0] * 7)
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


# ---------------------------------------------------------------------------
# discover_dimensionality
# ---------------------------------------------------------------------------


class TestDiscoverDimensionality:
    def test_returns_dict_for_each_threshold(self, line_in_5d):
        thresholds = (0.95, 0.90, 0.85)
        result = discover_dimensionality(
            line_in_5d, n_samples=30, k=20, variance_thresholds=thresholds
        )
        assert set(result.keys()) == set(thresholds)

    def test_result_has_required_keys(self, line_in_5d):
        result = discover_dimensionality(line_in_5d, n_samples=20, k=15)
        for tau, stats in result.items():
            for key in ("mean", "std", "median", "min", "max"):
                assert key in stats, f"Missing key '{key}' for tau={tau}"

    def test_1d_line_has_low_intrinsic_dim(self, line_in_5d):
        result = discover_dimensionality(line_in_5d, n_samples=50, k=20)
        # At τ=0.95 the line should need only 1 component
        assert result[0.95]["max"] <= 2

    def test_2d_plane_has_intrinsic_dim_lte_3(self, plane_in_6d):
        result = discover_dimensionality(plane_in_6d, n_samples=50, k=20)
        assert result[0.95]["max"] <= 3

    def test_n_samples_capped_at_n_points(self):
        """n_samples larger than dataset → sample all points without error."""
        X = np.random.default_rng(7).standard_normal((30, 4))
        result = discover_dimensionality(X, n_samples=1000, k=10)
        assert result[0.90]["min"] >= 1

    def test_stats_are_consistent(self, line_in_5d):
        result = discover_dimensionality(line_in_5d, n_samples=40, k=15)
        for tau, stats in result.items():
            assert stats["min"] <= stats["median"] <= stats["max"]
            assert stats["std"] >= 0.0

    def test_single_threshold(self, line_in_5d):
        result = discover_dimensionality(
            line_in_5d, n_samples=20, k=15, variance_thresholds=(0.80,)
        )
        assert 0.80 in result


# ---------------------------------------------------------------------------
# discover_per_class_dimensionality
# ---------------------------------------------------------------------------


class TestDiscoverPerClassDimensionality:
    def test_returns_dict_per_class(self, two_class_data):
        X, y = two_class_data
        result = discover_per_class_dimensionality(X, y, k=20, tau=0.90, n_samples_per_class=20)
        assert set(result.keys()) == {0, 1}

    def test_each_class_has_required_keys(self, two_class_data):
        X, y = two_class_data
        result = discover_per_class_dimensionality(X, y, k=20, tau=0.90, n_samples_per_class=20)
        for cls, stats in result.items():
            for key in ("mean", "std", "min", "max"):
                assert key in stats, f"Missing key '{key}' for class {cls}"

    def test_stats_are_consistent(self, two_class_data):
        X, y = two_class_data
        result = discover_per_class_dimensionality(X, y, k=20, tau=0.90, n_samples_per_class=20)
        for cls, stats in result.items():
            assert stats["min"] <= stats["max"]
            assert stats["std"] >= 0.0

    def test_low_dim_data_per_class(self):
        """1-D lines per class → per-class dim should be small."""
        rng = np.random.default_rng(3)
        t0 = rng.uniform(0, 5, size=80)
        t1 = rng.uniform(10, 15, size=80)
        X0 = np.column_stack([t0, np.zeros((80, 4))])
        X1 = np.column_stack([t1, np.zeros((80, 4))])
        X = np.vstack([X0, X1])
        y = np.array([0] * 80 + [1] * 80)
        result = discover_per_class_dimensionality(X, y, k=15, tau=0.95, n_samples_per_class=20)
        for cls, stats in result.items():
            assert stats["max"] <= 2, f"Class {cls} max dim {stats['max']} unexpectedly high"
