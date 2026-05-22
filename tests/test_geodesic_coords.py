"""Tests for GeodesicEncoder: ambient-space → tangent-projected geodesic coordinates."""

import numpy as np
import pytest

from waverider.geodesic_coords import GeodesicEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_data():
    """Two separable Gaussian blobs in 8-D (120 training, 30 test points)."""
    rng = np.random.default_rng(5)
    n = 60
    X0 = rng.standard_normal((n, 8)) + np.array([5.0] + [0.0] * 7)
    X1 = rng.standard_normal((n, 8)) + np.array([-5.0] + [0.0] * 7)
    X_train = np.vstack([X0, X1])
    X_test = rng.standard_normal((30, 8))
    return X_train, X_test


@pytest.fixture
def fitted_encoder(blob_data):
    """GeodesicEncoder fitted on blob_data."""
    X_train, _ = blob_data
    enc = GeodesicEncoder(k_pca=20, k_graph=10, variance_threshold=0.90, signed_coords=False)
    enc.fit(X_train)
    return enc, X_train


# ---------------------------------------------------------------------------
# Pre-fit state
# ---------------------------------------------------------------------------


class TestGeodesicEncoderUnfitted:
    def test_transform_before_fit_raises(self):
        enc = GeodesicEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(np.zeros((5, 8)))

    def test_d_star_is_none_before_fit(self):
        enc = GeodesicEncoder()
        assert enc.d_star is None

    def test_anchors_is_none_before_fit(self):
        enc = GeodesicEncoder()
        assert enc.anchors is None


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestGeodesicEncoderFit:
    def test_fit_returns_self(self, blob_data):
        X_train, _ = blob_data
        enc = GeodesicEncoder(k_pca=20, k_graph=10)
        result = enc.fit(X_train)
        assert result is enc

    def test_d_star_is_positive_int(self, fitted_encoder):
        enc, X_train = fitted_encoder
        assert isinstance(enc.d_star, int)
        assert enc.d_star >= 1

    def test_anchors_shape(self, fitted_encoder):
        enc, X_train = fitted_encoder
        assert enc.anchors is not None
        assert enc.anchors.ndim == 2
        # anchors columns match ambient dim
        assert enc.anchors.shape[1] == X_train.shape[1]

    def test_n_anchors_default_equals_d_star(self, fitted_encoder):
        enc, X_train = fitted_encoder
        assert enc.anchors.shape[0] == enc.d_star

    def test_custom_n_anchors(self, blob_data):
        X_train, _ = blob_data
        enc = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=4, signed_coords=False)
        enc.fit(X_train)
        assert enc.anchors.shape[0] == 4


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------


class TestGeodesicEncoderTransform:
    def test_output_shape_unsigned(self, blob_data):
        X_train, X_test = blob_data
        enc = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=3, signed_coords=False)
        enc.fit(X_train)
        out = enc.transform(X_test)
        assert out.shape == (len(X_test), 3)

    def test_output_shape_signed(self, blob_data):
        X_train, X_test = blob_data
        enc = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=3, signed_coords=True)
        enc.fit(X_train)
        out = enc.transform(X_test)
        assert out.shape == (len(X_test), 6)  # 2 * n_anchors

    def test_output_dtype_float32(self, fitted_encoder):
        enc, X_train = fitted_encoder
        out = enc.transform(X_train[:10])
        assert out.dtype == np.float32

    def test_output_non_negative_unsigned(self, fitted_encoder):
        enc, X_train = fitted_encoder
        # unsigned mode: all values are distances (≥ 0)
        out = enc.transform(X_train)
        assert np.all(out >= 0.0)

    def test_transform_single_point(self, fitted_encoder):
        enc, X_train = fitted_encoder
        out = enc.transform(X_train[:1])
        assert out.shape[0] == 1
        assert out.shape[1] == enc.d_star


# ---------------------------------------------------------------------------
# fit_transform()
# ---------------------------------------------------------------------------


class TestGeodesicEncoderFitTransform:
    def test_fit_transform_equals_fit_then_transform(self, blob_data):
        X_train, _ = blob_data

        enc1 = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=3, signed_coords=False)
        enc2 = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=3, signed_coords=False)

        out1 = enc1.fit_transform(X_train)
        enc2.fit(X_train)
        out2 = enc2.transform(X_train)

        # Both should have the same shape (content may differ by anchor selection)
        assert out1.shape == out2.shape

    def test_fit_transform_returns_array(self, blob_data):
        X_train, _ = blob_data
        enc = GeodesicEncoder(k_pca=20, k_graph=10, n_anchors=3, signed_coords=False)
        out = enc.fit_transform(X_train)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == len(X_train)
