"""
Tests for ManifoldModel: the manifold-as-model architecture.

Tests cover:
  - Exploration phase (fit)
  - Navigate phase (predict / classify)
  - Fly mode (interactive graph navigation)
  - Geometry diagnostics
  - Edge cases
"""

import numpy as np
import pytest

from waverider.manifold_model import ManifoldModel, NodeGeometry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_2class():
    """Two well-separated Gaussian blobs in 10D."""
    rng = np.random.default_rng(42)
    n = 80
    X0 = rng.standard_normal((n, 10)) + np.array([3.0] + [0.0] * 9)
    X1 = rng.standard_normal((n, 10)) + np.array([-3.0] + [0.0] * 9)
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


@pytest.fixture
def swiss_roll_2d():
    """2D manifold embedded in 3D (simplified Swiss roll)."""
    rng = np.random.default_rng(99)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=200))
    X = np.column_stack([t * np.cos(t), t * np.sin(t), rng.standard_normal(200) * 0.1])
    y = (t > np.median(t)).astype(int)
    return X, y


@pytest.fixture
def fitted_model(simple_2class):
    """A ManifoldModel already fitted on simple_2class."""
    X, y = simple_2class
    model = ManifoldModel(k_graph=10, k_pca=30, k_vote=5)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Phase 1: Explore (fit)
# ---------------------------------------------------------------------------


class TestExploration:
    def test_fit_returns_self(self, simple_2class):
        X, y = simple_2class
        model = ManifoldModel(k_graph=10, k_pca=30)
        result = model.fit(X, y)
        assert result is model

    def test_mode_after_fit(self, fitted_model):
        model, _, _ = fitted_model
        assert model.mode == ManifoldModel.NAVIGATE

    def test_graph_built(self, fitted_model):
        model, X, _ = fitted_model
        assert model.graph is not None
        assert model.n_nodes == len(X)

    def test_turtle_initialized(self, fitted_model):
        model, _, _ = fitted_model
        assert model.turtle is not None
        assert model.turtle.ndim == model.ndim

    def test_intrinsic_dim_discovered(self, fitted_model):
        model, _, _ = fitted_model
        assert model.intrinsic_dim is not None
        # Should discover that intrinsic dim < ambient dim
        assert model.intrinsic_dim < model.ndim

    def test_node_geometries_computed(self, fitted_model):
        model, X, _ = fitted_model
        for i in range(len(X)):
            geom = model.get_geometry(f"n{i}")
            assert isinstance(geom, NodeGeometry)
            assert geom.basis.shape == (geom.intrinsic_dim, model.ndim)
            assert geom.intrinsic_dim >= 1
            assert geom.intrinsic_dim <= model.ndim

    def test_edges_exist(self, fitted_model):
        model, _, _ = fitted_model
        summary = model.geometry_summary()
        assert summary["n_edges"] > 0

    def test_unsupervised_fit(self, simple_2class):
        X, _ = simple_2class
        model = ManifoldModel(k_graph=10, k_pca=30)
        model.fit(X)  # no labels
        assert model.mode == ManifoldModel.NAVIGATE
        assert model.n_nodes == len(X)

    def test_swiss_roll_intrinsic_dim(self, swiss_roll_2d):
        """Swiss roll is 2D manifold in 3D — should discover ~2 dimensions."""
        X, y = swiss_roll_2d
        model = ManifoldModel(k_graph=10, k_pca=30, variance_threshold=0.95)
        model.fit(X, y)
        # Mean intrinsic dim should be close to 2
        assert model.intrinsic_dim < 2.5


# ---------------------------------------------------------------------------
# Phase 2: Navigate (predict)
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_predict_shape(self, fitted_model):
        model, X, _ = fitted_model
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_accuracy_on_training(self, fitted_model):
        """Should get near-perfect accuracy on training data."""
        model, X, y = fitted_model
        acc = model.score(X, y)
        assert acc > 0.90

    def test_predict_on_test_data(self, simple_2class):
        """Test on held-out data from same distribution."""
        X, y = simple_2class
        rng = np.random.default_rng(123)
        n_test = 20
        X_test0 = rng.standard_normal((n_test, 10)) + np.array([3.0] + [0.0] * 9)
        X_test1 = rng.standard_normal((n_test, 10)) + np.array([-3.0] + [0.0] * 9)
        X_test = np.vstack([X_test0, X_test1])
        y_test = np.array([0] * n_test + [1] * n_test)

        model = ManifoldModel(k_graph=10, k_pca=30, k_vote=5)
        model.fit(X, y)
        acc = model.score(X_test, y_test)
        assert acc > 0.85

    def test_predict_without_fit_raises(self):
        model = ManifoldModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((5, 10)))

    def test_predict_without_labels_raises(self, simple_2class):
        X, _ = simple_2class
        model = ManifoldModel(k_graph=10, k_pca=30)
        model.fit(X)  # no labels
        with pytest.raises(RuntimeError, match="labels"):
            model.predict(X[:5])


# ---------------------------------------------------------------------------
# Phase 3: Fly
# ---------------------------------------------------------------------------


class TestFly:
    def test_fly_to(self, fitted_model):
        model, _, _ = fitted_model
        geom = model.fly_to("n0")
        assert model.current_node == "n0"
        assert isinstance(geom, NodeGeometry)

    def test_fly_to_nearest(self, fitted_model):
        model, X, _ = fitted_model
        model.fly_to_nearest(X[42])
        assert model.current_node == "n42"

    def test_fly_step(self, fitted_model):
        model, _, _ = fitted_model
        model.fly_to("n0")
        next_id = model.fly_step()
        assert next_id is not None
        assert next_id != "n0"
        assert len(model.flight_path) == 2

    def test_fly_toward(self, fitted_model):
        model, X, _ = fitted_model
        model.fly_to("n0")
        path = model.fly_toward(X[100])
        assert len(path) > 0

    def test_fly_without_position_raises(self, fitted_model):
        model, _, _ = fitted_model
        with pytest.raises(RuntimeError, match="fly_to"):
            model.fly_step()

    def test_fly_to_invalid_node(self, fitted_model):
        model, _, _ = fitted_model
        with pytest.raises(KeyError):
            model.fly_to("nonexistent")

    def test_reset_flight(self, fitted_model):
        model, _, _ = fitted_model
        model.fly_to("n0")
        model.fly_step()
        model.reset_flight()
        assert model.current_node is None
        assert model.flight_path == []

    def test_fly_oriented_to_local_geometry(self, fitted_model):
        """After fly_to, turtle frame first d rows should align with node's PCA basis."""
        model, _, _ = fitted_model
        geom = model.fly_to("n5")
        d = geom.intrinsic_dim
        # _pad_basis expands basis to (ndim, ndim); first d rows should match
        # original basis up to sign flip
        frame_top = model.turtle._frame[:d]
        np.testing.assert_allclose(np.abs(frame_top), np.abs(geom.basis), atol=1e-10)

    def test_fly_step_with_direction(self, fitted_model):
        """Fly step with explicit direction."""
        model, X, _ = fitted_model
        model.fly_to("n0")
        direction = X[50] - X[0]
        next_id = model.fly_step(direction)
        assert next_id is not None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_geometry_summary(self, fitted_model):
        model, _, _ = fitted_model
        summary = model.geometry_summary()
        assert "mean_intrinsic_dim" in summary
        assert "n_nodes" in summary
        assert "n_edges" in summary
        assert "ambient_dim" in summary
        assert summary["n_nodes"] == model.n_nodes
        assert summary["ambient_dim"] == model.ndim

    def test_get_neighbors(self, fitted_model):
        model, _, _ = fitted_model
        neighbors = model.get_neighbors("n0")
        assert len(neighbors) > 0
        # Sorted by weight descending
        weights = [w for _, w in neighbors]
        assert weights == sorted(weights, reverse=True)

    def test_repr(self, fitted_model):
        model, _, _ = fitted_model
        r = repr(model)
        assert "ManifoldModel" in r
        assert "navigate" in r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_small_dataset(self):
        """Should handle very small datasets gracefully."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model = ManifoldModel(k_graph=3, k_pca=5, k_vote=3)
        model.fit(X, y)
        preds = model.predict(X[:3])
        assert len(preds) == 3

    def test_high_dimensional(self):
        """Should work in high dimensions."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 100))
        y = (X[:, 0] > 0).astype(int)
        model = ManifoldModel(k_graph=5, k_pca=20, k_vote=3)
        model.fit(X, y)
        assert model.intrinsic_dim < 100

    def test_manifold_weight_zero(self, simple_2class):
        """manifold_weight=0 should fall back to Euclidean distances."""
        X, y = simple_2class
        model = ManifoldModel(k_graph=10, k_pca=30, k_vote=5, manifold_weight=0.0)
        model.fit(X, y)
        acc = model.score(X, y)
        assert acc > 0.85

    def test_manifold_weight_one(self, simple_2class):
        """manifold_weight=1 should use pure manifold distances."""
        X, y = simple_2class
        model = ManifoldModel(k_graph=10, k_pca=30, k_vote=5, manifold_weight=1.0)
        model.fit(X, y)
        acc = model.score(X, y)
        assert acc > 0.85
