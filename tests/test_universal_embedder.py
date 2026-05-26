"""Tests for UniversalEmbedder."""

import numpy as np
import pytest

from waverider.universal_embedder import UniversalEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _swiss_roll_2d(n=300, noise=0.05, seed=0):
    """Flat 2D manifold embedded in 8D space — d* should be ~2."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    X2 = np.column_stack([np.cos(t), np.sin(t)])  # (n, 2) on unit circle
    proj = rng.standard_normal((2, 8))
    proj, _ = np.linalg.qr(proj.T)
    proj = proj.T  # (2, 8) — orthonormal rows
    X8 = X2 @ proj + noise * rng.standard_normal((n, 8))
    return X8.astype("float64"), t


def _tabular(n=150, p=6, seed=1):
    """Generic tabular data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype("float32")


def _linear_data(n=200, p=10, d=2, seed=2):
    """Exactly linear: n samples on a d-dim affine subspace embedded in p-dim."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((p, d)))
    coeffs = rng.standard_normal((n, d))
    return (coeffs @ basis.T + 0.01 * rng.standard_normal((n, p))).astype("float64")


# ---------------------------------------------------------------------------
# Construction / repr
# ---------------------------------------------------------------------------


def test_repr_unfitted():
    ue = UniversalEmbedder()
    assert "unfitted" in repr(ue)


def test_repr_fitted():
    X, _ = _swiss_roll_2d(n=100)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, variance_threshold=0.90)
    ue.fit(X)
    r = repr(ue)
    assert "d_star=" in r
    assert "strategy=" in r
    assert "mli=" in r


def test_invalid_mode():
    with pytest.raises(ValueError, match="coordinate_mode"):
        UniversalEmbedder(coordinate_mode="banana")


def test_valid_modes_accepted():
    for mode in ("auto", "pca", "turtle", "tangent"):
        ue = UniversalEmbedder(coordinate_mode=mode)
        assert ue.coordinate_mode == mode


# ---------------------------------------------------------------------------
# Fit / transform output shape & dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tangent", "turtle", "pca", "auto"])
def test_output_shape_float32(mode):
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    Z_train = ue.fit_transform(X)
    assert Z_train.shape == (200, ue.d_star)
    assert Z_train.dtype == np.float32

    X_test, _ = _swiss_roll_2d(n=50, seed=99)
    Z_test = ue.transform(X_test)
    assert Z_test.shape == (50, ue.d_star)
    assert Z_test.dtype == np.float32


@pytest.mark.parametrize("mode", ["tangent", "turtle", "pca", "auto"])
def test_fit_transform_equals_fit_then_transform(mode):
    X, _ = _swiss_roll_2d(n=200, seed=7)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    Z1 = ue.fit_transform(X)

    ue2 = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    ue2.fit(X)
    Z2 = ue2.transform(X)

    np.testing.assert_array_equal(Z1, Z2)


def test_transform_before_fit_raises():
    ue = UniversalEmbedder()
    X = np.zeros((10, 5))
    with pytest.raises(RuntimeError):
        ue.transform(X)


# ---------------------------------------------------------------------------
# n_components override (benchmark parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tangent", "turtle", "pca", "auto"])
def test_n_components_truncates(mode):
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    ue.fit(X)
    d = ue.d_star
    if d < 2:
        pytest.skip("d* too small to truncate")

    ue2 = UniversalEmbedder(
        n_components=d - 1, k_pca=20, k_graph=8, coordinate_mode=mode
    )
    ue2.fit(X)
    Z = ue2.transform(X)
    assert Z.shape == (200, d - 1)


@pytest.mark.parametrize("mode", ["tangent", "turtle"])
def test_n_components_expands(mode):
    """n_components > d_star selects more anchors — all output columns are real."""
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    ue.fit(X)
    d = ue.d_star
    target = d + 3

    ue2 = UniversalEmbedder(
        n_components=target, k_pca=20, k_graph=8, coordinate_mode=mode
    )
    ue2.fit(X)
    Z = ue2.transform(X)
    assert Z.shape == (200, target)
    # All columns are anchor distances — non-negative, not uniformly zero
    assert np.all(Z >= 0.0)
    assert Z.any()


# ---------------------------------------------------------------------------
# d_star property
# ---------------------------------------------------------------------------


def test_d_star_is_positive():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    ue.fit(X)
    assert ue.d_star is not None
    assert ue.d_star >= 1


def test_d_star_none_before_fit():
    ue = UniversalEmbedder()
    assert ue.d_star is None


# ---------------------------------------------------------------------------
# strategy and mli properties
# ---------------------------------------------------------------------------


def test_strategy_none_before_fit():
    ue = UniversalEmbedder()
    assert ue.strategy is None


def test_mli_none_before_fit():
    ue = UniversalEmbedder()
    assert ue.mli is None


def test_strategy_is_set_after_fit():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    ue.fit(X)
    assert ue.strategy in ("pca", "turtle")


def test_mli_is_positive_float_after_fit():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    ue.fit(X)
    assert isinstance(ue.mli, float)
    assert ue.mli > 0.0


def test_explicit_pca_mode_sets_strategy_pca():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode="pca")
    ue.fit(X)
    assert ue.strategy == "pca"


def test_explicit_turtle_mode_sets_strategy_turtle():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode="turtle")
    ue.fit(X)
    assert ue.strategy == "turtle"


# ---------------------------------------------------------------------------
# Auto mode adaptive dispatch
# ---------------------------------------------------------------------------


def test_auto_picks_pca_for_linear_data():
    """Near-linear data should have MLI ≈ 1, triggering the PCA strategy."""
    X = _linear_data(n=200, p=10, d=2, seed=2)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode="auto", mli_threshold=3.0)
    ue.fit(X)
    # Linear data → MLI should be low → PCA chosen
    assert ue.strategy == "pca", (
        f"Expected 'pca' for near-linear data but got {ue.strategy!r} (MLI={ue.mli:.2f})"
    )


def test_auto_picks_turtle_for_curved_data():
    """Force auto to pick turtle by setting a very low mli_threshold."""
    X, _ = _swiss_roll_2d(n=200, noise=0.01)
    # With threshold=1.0 virtually any data has MLI > 1 → turtle
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode="auto", mli_threshold=1.0)
    ue.fit(X)
    assert ue.strategy == "turtle", (
        f"Expected 'turtle' at mli_threshold=1.0 but got {ue.strategy!r} (MLI={ue.mli:.2f})"
    )


def test_auto_mli_threshold_param():
    """mli_threshold is stored and available."""
    ue = UniversalEmbedder(mli_threshold=2.5)
    assert ue.mli_threshold == 2.5


# ---------------------------------------------------------------------------
# manifold_summary
# ---------------------------------------------------------------------------


def test_manifold_summary_empty_before_fit():
    ue = UniversalEmbedder()
    assert ue.manifold_summary == {}


def test_manifold_summary_keys():
    X, _ = _swiss_roll_2d(n=200)
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    ue.fit(X)
    s = ue.manifold_summary
    for key in (
        "d_star",
        "coordinate_mode",
        "ambient_dim",
        "noise_pct",
        "n_nodes",
        "strategy",
        "mli",
        "global_var_at_d_star",
        "mli_threshold",
    ):
        assert key in s, f"Missing key: {key}"
    assert s["ambient_dim"] == 8
    assert 0.0 <= s["noise_pct"] <= 100.0
    assert s["strategy"] in ("pca", "turtle")
    assert isinstance(s["mli"], float)
    assert 0.0 <= s["global_var_at_d_star"] <= 1.0


# ---------------------------------------------------------------------------
# Coordinate consistency: turtle should produce finite values everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tangent", "turtle", "pca", "auto"])
def test_embeddings_finite(mode):
    X, _ = _swiss_roll_2d(n=200, noise=0.1)
    ue = UniversalEmbedder(k_pca=20, k_graph=8, coordinate_mode=mode)
    Z = ue.fit_transform(X)
    assert np.all(np.isfinite(Z)), "Embedding contains non-finite values"


# ---------------------------------------------------------------------------
# Small / edge cases
# ---------------------------------------------------------------------------


def test_small_dataset():
    """Minimum viable: n=20 samples, p=3 features."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 3))
    ue = UniversalEmbedder(k_pca=10, k_graph=5)
    Z = ue.fit_transform(X)
    assert Z.shape[0] == 20
    assert Z.shape[1] >= 1


def test_float32_input():
    """float32 input should work without errors."""
    X = _tabular(n=100, p=6).astype("float32")
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    Z = ue.fit_transform(X)
    assert Z.dtype == np.float32


def test_supervised_fit_does_not_crash():
    """Passing labels to fit() should not raise."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8))
    y = (X[:, 0] > 0).astype(int)
    ue = UniversalEmbedder(k_pca=20, k_graph=8)
    Z = ue.fit_transform(X, y=y)
    assert Z.shape == (100, ue.d_star)


# ---------------------------------------------------------------------------
# sklearn PCA parity: drop-in usage pattern from benchmarks
# ---------------------------------------------------------------------------


def test_benchmark_parity_pattern():
    """Verify the exact fold-loop pattern used in disease benchmarks works."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data).astype("float32")
    y = data.target.astype(int)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    ue_global = UniversalEmbedder(k_pca=20, k_graph=8, variance_threshold=0.90)
    ue_global.fit(X, y)
    d_ue = ue_global.d_star
    assert d_ue >= 1

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]

        ue_fold = UniversalEmbedder(
            n_components=d_ue, k_pca=20, k_graph=8, variance_threshold=0.90
        )
        X_tr_ue = ue_fold.fit_transform(X_tr, y_tr).astype("float32")
        X_te_ue = ue_fold.transform(X_te).astype("float32")

        assert X_tr_ue.shape == (len(tr_idx), d_ue)
        assert X_te_ue.shape == (len(te_idx), d_ue)
        assert np.all(np.isfinite(X_tr_ue))
        assert np.all(np.isfinite(X_te_ue))
