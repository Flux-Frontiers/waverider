"""Tests for ManifoldWalker - verifies manifold-aware navigation
of embedding spaces using local PCA and TurtleND."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from waverider.manifold_walker import ManifoldWalker


def _make_sphere_embeddings(n=500, ndim=10, intrinsic_dim=3, seed=42):
    """Create embeddings that live on a low-dimensional manifold in high-dim space.

    Points are sampled from a intrinsic_dim-dimensional sphere, then embedded
    in ndim dimensions via a random rotation. This simulates an LLM embedding
    space where the data manifold has lower intrinsic dimensionality.
    """
    rng = np.random.RandomState(seed)
    # Points on a low-dim sphere
    raw = rng.randn(n, intrinsic_dim)
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    # Pad to high dim
    padded = np.zeros((n, ndim))
    padded[:, :intrinsic_dim] = raw
    # Random rotation in ndim
    q, _ = np.linalg.qr(rng.randn(ndim, ndim))
    embeddings = padded @ q.T
    return embeddings, q


def _quadratic_objective(pos):
    """Simple quadratic: f(x) = ||x||^2. Minimum at origin."""
    return np.dot(pos, pos)


def _weighted_quadratic(pos):
    """Weighted quadratic where first 3 dims matter 100x more."""
    weights = np.ones(len(pos))
    weights[:3] = 100.0
    return np.dot(weights * pos, pos)


class TestManifoldWalkerInit(unittest.TestCase):
    def test_basic_init(self):
        emb = np.random.randn(100, 5)
        mw = ManifoldWalker(emb, _quadratic_objective)
        self.assertEqual(mw.ndim, 5)
        assert_allclose(mw.position, np.zeros(5))
        self.assertIsNone(mw.eigenvalues)
        self.assertIsNone(mw.intrinsic_dim)

    def test_set_position(self):
        emb = np.random.randn(100, 5)
        mw = ManifoldWalker(emb, _quadratic_objective)
        mw.position = np.ones(5)
        assert_allclose(mw.position, np.ones(5))


class TestManifoldWalkerOrient(unittest.TestCase):
    def test_orient_discovers_intrinsic_dim(self):
        """On a 3D manifold in 10D space, orient should find ~3 dimensions."""
        embeddings, _ = _make_sphere_embeddings(n=500, ndim=10, intrinsic_dim=3)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=50, variance_threshold=0.95)
        # Start at a point on the manifold
        mw.position = embeddings[0]
        dim = mw.orient()
        # Should discover approximately 3 intrinsic dimensions
        self.assertLessEqual(dim, 5)  # generous upper bound
        self.assertGreaterEqual(dim, 2)  # at least 2

    def test_orient_sets_eigenvalues(self):
        embeddings, _ = _make_sphere_embeddings()
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=50)
        mw.position = embeddings[0]
        mw.orient()
        self.assertIsNotNone(mw.eigenvalues)
        self.assertEqual(len(mw.eigenvalues), 10)
        # Eigenvalues should be in descending order
        for i in range(len(mw.eigenvalues) - 1):
            self.assertGreaterEqual(mw.eigenvalues[i], mw.eigenvalues[i + 1] - 1e-10)

    def test_frame_is_orthonormal_after_orient(self):
        embeddings, _ = _make_sphere_embeddings()
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=50)
        mw.position = embeddings[0]
        mw.orient()
        frame = mw.turtle.frame
        product = frame @ frame.T
        assert_allclose(product, np.eye(10), atol=1e-10)


class TestManifoldWalkerStep(unittest.TestCase):
    def test_step_reduces_objective(self):
        """A step should reduce the objective for a simple quadratic."""
        rng = np.random.RandomState(123)
        embeddings = rng.randn(200, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=30, learning_rate=0.01)
        mw.position = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        val_before = _quadratic_objective(mw.position)
        mw.step()
        val_after = _quadratic_objective(mw.position)
        self.assertLess(val_after, val_before)

    def test_step_with_provided_gradient(self):
        """Step with an explicit gradient should also reduce objective."""
        rng = np.random.RandomState(123)
        embeddings = rng.randn(200, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=30, learning_rate=0.01)
        start = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        mw.position = start
        grad = 2 * start  # analytic gradient of ||x||^2
        val_before = _quadratic_objective(mw.position)
        mw.step(gradient=grad)
        val_after = _quadratic_objective(mw.position)
        self.assertLess(val_after, val_before)

    def test_history_recorded(self):
        rng = np.random.RandomState(123)
        embeddings = rng.randn(200, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=30, learning_rate=0.01)
        mw.position = np.ones(5)
        mw.step()
        mw.step()
        self.assertEqual(len(mw.history), 2)


class TestManifoldWalkerWalk(unittest.TestCase):
    def test_walk_converges_toward_minimum(self):
        """Walking should move toward the objective minimum."""
        rng = np.random.RandomState(99)
        embeddings = rng.randn(300, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=40, learning_rate=0.05)
        mw.position = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        initial_obj = _quadratic_objective(mw.position)
        final_pos = mw.walk(n_steps=50)
        final_obj = _quadratic_objective(final_pos)
        self.assertLess(final_obj, initial_obj * 0.5)

    def test_walk_with_analytic_gradient(self):
        rng = np.random.RandomState(99)
        embeddings = rng.randn(300, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=40, learning_rate=0.05)
        mw.position = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        initial_obj = _quadratic_objective(mw.position)
        final_pos = mw.walk(n_steps=50, gradient_fn=lambda x: 2 * x)
        final_obj = _quadratic_objective(final_pos)
        self.assertLess(final_obj, initial_obj * 0.1)

    def test_walk_on_manifold(self):
        """Walking on a manifold-embedded objective should still converge."""
        embeddings, q = _make_sphere_embeddings(n=500, ndim=10, intrinsic_dim=3)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=50, learning_rate=0.01)
        mw.position = embeddings[0] * 5.0  # start away from origin
        initial_obj = _quadratic_objective(mw.position)
        final_pos = mw.walk(n_steps=30)
        final_obj = _quadratic_objective(final_pos)
        self.assertLess(final_obj, initial_obj)


class TestManifoldWalkerProbe(unittest.TestCase):
    def test_probe_returns_correct_shape(self):
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=30)
        mw.position = np.ones(5)
        mw.orient()
        distances = np.linspace(-1, 1, 11)
        values = mw.probe(0, distances)
        self.assertEqual(values.shape, (11,))

    def test_probe_minimum_near_origin(self):
        """For quadratic objective, probing from near origin should show a valley."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 5)
        mw = ManifoldWalker(embeddings, _quadratic_objective, k=30)
        mw.position = np.zeros(5)
        mw.orient()
        distances = np.linspace(-2, 2, 21)
        values = mw.probe(0, distances)
        # Minimum should be at or near distance=0 (index 10)
        min_idx = np.argmin(values)
        self.assertAlmostEqual(distances[min_idx], 0.0, places=0)


class TestManifoldWalkerRepr(unittest.TestCase):
    def test_repr(self):
        emb = np.random.randn(100, 5)
        mw = ManifoldWalker(emb, _quadratic_objective)
        s = repr(mw)
        self.assertIn("5D", s)
        self.assertIn("100 embeddings", s)


if __name__ == "__main__":
    unittest.main()
