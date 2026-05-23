"""
ManifoldAdam — Adam optimizer with manifold-aware gradient projection.

Before each parameter update, gradients for weights whose input dimension
matches the data manifold are projected onto the top-d principal directions
discovered by local PCA.  Gradient components in the remaining (n_dims - d)
noise dimensions are zeroed out.

Projection: given basis V_d of shape (n_dims, d),
    g_proj = V_d @ (V_d.T @ g)          # (n_dims, units)
two small matmuls replace the full-rank update.

Usage
-----
    from waverider.manifold_optimizer import ManifoldAdam, make_basis

    V_d = make_basis(pca)                        # from a fitted sklearn PCA
    opt = ManifoldAdam(basis=V_d, learning_rate=0.001)
    model.compile(optimizer=opt, ...)

    Author: Eric G. Suchanek, PhD
    Last Revision: 2026-03-30 00:43:13
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA as skPCA


def make_basis(pca: skPCA) -> np.ndarray:
    """Extract the (n_dims, d) projection basis from a fitted sklearn PCA.

    :param pca: Fitted ``sklearn.decomposition.PCA`` instance.
    :returns: Float32 array of shape (n_dims, d) — columns are principal axes.
    """
    return pca.components_.T.astype(np.float32)  # (n_dims, d)


class ManifoldAdam(keras.optimizers.Adam):
    """Adam optimizer with gradient projection onto the manifold tangent space.

    Subclasses ``keras.optimizers.Adam``.  For any weight matrix whose first
    dimension equals ``n_dims`` (the ambient data dimensionality), the
    gradient is projected onto the top-d principal directions before the
    Adam moment updates are applied.  All other variables are updated
    normally.

    :param basis: Array of shape (n_dims, d) — top-d eigenvectors from
        global PCA.  Obtain via :func:`make_basis`.
    :param learning_rate: Adam learning rate.
    :param kwargs: Forwarded to ``keras.optimizers.Adam``.
    """

    def __init__(self, basis: np.ndarray, learning_rate: float = 0.001, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self._basis_np = np.array(basis, dtype=np.float32)  # keep for get_config
        self._basis_tf = tf.constant(self._basis_np)  # (n_dims, d)
        self._input_dim = int(self._basis_np.shape[0])

    def apply_gradients(self, grads_and_vars, **kwargs):
        projected = [
            (self._project(g, v), v) if g is not None else (g, v) for g, v in grads_and_vars
        ]
        return super().apply_gradients(projected, **kwargs)

    def _project(self, g, v):
        """Project gradient onto manifold basis if shapes align."""
        if len(g.shape) == 2 and int(g.shape[0]) == self._input_dim:
            # g: (n_dims, units)
            coords = tf.matmul(tf.transpose(self._basis_tf), g)  # (d, units)
            return tf.matmul(self._basis_tf, coords)  # (n_dims, units)
        return g

    def get_config(self):
        cfg = super().get_config()
        cfg["basis"] = self._basis_np.tolist()
        return cfg
