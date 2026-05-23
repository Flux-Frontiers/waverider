"""Tests for ManifoldObserver: the (N+1)-D extrinsic observer over a ManifoldModel."""

import numpy as np
import pytest

from waverider.manifold_model import ManifoldModel
from waverider.manifold_observer import ManifoldObserver, ObservedGeometry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_subject():
    """Small ManifoldModel fitted on two separable Gaussian blobs in 5-D."""
    rng = np.random.default_rng(7)
    n = 60
    X0 = rng.standard_normal((n, 5)) + np.array([4.0, 0, 0, 0, 0])
    X1 = rng.standard_normal((n, 5)) + np.array([-4.0, 0, 0, 0, 0])
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    model = ManifoldModel(k_graph=8, k_pca=20, k_vote=5)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def observer(fitted_subject):
    """ManifoldObserver wrapping the fitted subject."""
    model, X, y = fitted_subject
    return ManifoldObserver(model), model, X, y


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestManifoldObserverConstruction:
    def test_ndim_is_subject_plus_one(self, fitted_subject):
        model, X, y = fitted_subject
        obs = ManifoldObserver(model)
        assert obs.ndim == model.ndim + 1

    def test_subject_property(self, fitted_subject):
        model, X, y = fitted_subject
        obs = ManifoldObserver(model)
        assert obs.subject is model

    def test_unfitted_subject_raises(self):
        unfitted = ManifoldModel(k_graph=5, k_pca=10)
        with pytest.raises(RuntimeError):
            ManifoldObserver(unfitted)

    def test_observer_turtle_ndim(self, observer):
        obs, model, X, y = observer
        assert obs.observer.ndim == obs.ndim

    def test_repr_before_observe(self, observer):
        obs, model, X, y = observer
        r = repr(obs)
        assert "ManifoldObserver" in r


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


class TestManifoldObserverObserve:
    def test_observe_returns_list(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        assert isinstance(field, list)

    def test_observe_length_matches_training(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        assert len(field) == len(X)

    def test_each_entry_is_observed_geometry(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        for entry in field:
            assert isinstance(entry, ObservedGeometry)

    def test_curvature_is_non_negative(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        for entry in field:
            assert entry.curvature >= 0.0

    def test_lifted_position_has_correct_dim(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        for entry in field:
            assert entry.position_lifted.shape == (obs.ndim,)

    def test_normal_vector_is_unit_like(self, observer):
        obs, model, X, y = observer
        field = obs.observe()
        # The normal should be non-zero
        for entry in field:
            assert np.linalg.norm(entry.normal) > 0.0

    def test_repr_after_observe(self, observer):
        obs, model, X, y = observer
        obs.observe()
        r = repr(obs)
        assert "ManifoldObserver" in r


# ---------------------------------------------------------------------------
# locate()
# ---------------------------------------------------------------------------


class TestManifoldObserverLocate:
    def test_locate_returns_dict(self, observer):
        obs, model, X, y = observer
        obs.observe()
        result = obs.locate(X[0])
        assert isinstance(result, dict)

    def test_locate_has_required_keys(self, observer):
        obs, model, X, y = observer
        obs.observe()
        result = obs.locate(X[0])
        for key in ("nearest_node", "distance", "height", "curvature"):
            assert key in result

    def test_locate_nearest_node_is_string(self, observer):
        obs, model, X, y = observer
        obs.observe()
        result = obs.locate(X[0])
        assert isinstance(result["nearest_node"], str)

    def test_locate_distance_non_negative(self, observer):
        obs, model, X, y = observer
        obs.observe()
        for pt in X[:5]:
            result = obs.locate(pt)
            assert result["distance"] >= 0.0

    def test_locate_training_point_near_zero_distance(self, observer):
        obs, model, X, y = observer
        obs.observe()
        # A training point should locate to very near itself
        result = obs.locate(X[0])
        assert result["distance"] < 2.0  # generous bound


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestManifoldObserverPredict:
    def test_predict_returns_array(self, observer):
        obs, model, X, y = observer
        obs.observe()
        preds = obs.predict(X[:10])
        assert hasattr(preds, "__len__")
        assert len(preds) == 10

    def test_predict_labels_are_valid(self, observer):
        obs, model, X, y = observer
        obs.observe()
        preds = obs.predict(X)
        valid_labels = set(np.unique(y))
        for p in preds:
            assert p in valid_labels

    def test_predict_accuracy_above_chance(self, observer):
        obs, model, X, y = observer
        obs.observe()
        preds = obs.predict(X)
        accuracy = np.mean(np.array(preds) == y)
        # Well-separated blobs → should classify above 50% chance
        assert accuracy > 0.5
