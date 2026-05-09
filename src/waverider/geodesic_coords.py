"""Geodesic coordinate encoder: ambient space → tangent-projected distances.

GeodesicEncoder transforms data from ambient space into a d*-dimensional
geodesic coordinate representation.  Each coordinate is the tangent-projected
distance from the query point to one of d* manifold anchor points:

    dist_geo(x, anchor_j) ≈ ‖B_d(near(x)) · (x − anchor_j)‖₂

where B_d(near(x)) is the local tangent basis (first d* principal directions,
shape (d*, ndim)) at the training point nearest to x.  This mirrors the
manifold distance computed in ManifoldModel._build_manifold_edges — the same
basis is rotated to the query's local frame, then applied to the chord from
anchor to query.

Geodesic coordinates are:
  - Noise-robust: tangent projection discards off-manifold dimensions
  - Curvature-aware: basis rotates with the manifold's tangent plane
  - Compact: d* << ambient dimension for real datasets

Typical use::

    enc = GeodesicEncoder(variance_threshold=0.90)
    enc.fit(X_train)
    X_geo = enc.transform(X_train)     # (n_train, d*)
    X_geo_test = enc.transform(X_test) # (n_test, d*)
"""

from __future__ import annotations

import numpy as np

from .manifold_model import ManifoldModel


class GeodesicEncoder:
    """Encode points as tangent-projected distances to d* manifold anchors.

    Parameters
    ----------
    k_pca : int
        Neighborhood size for local PCA at each node (default 50).
    k_graph : int
        Neighbors for graph construction in the underlying ManifoldModel
        (default 15).
    variance_threshold : float
        Cumulative variance fraction for intrinsic dimension d* (default 0.90).
    n_anchors : int or None
        Number of anchor points.  Defaults to d* (square encoding: one anchor
        per intrinsic dimension).
    manifold_weight : float
        Euclidean–manifold blend in ManifoldModel graph edges (default 0.8).

    Attributes
    ----------
    d_star : int
        Intrinsic dimension discovered during :meth:`fit`.
    anchors : np.ndarray
        Anchor points in ambient space, shape ``(n_anchors, ndim)``.
    """

    def __init__(
        self,
        k_pca: int = 50,
        k_graph: int = 15,
        variance_threshold: float = 0.90,
        n_anchors: int | None = None,
        manifold_weight: float = 0.8,
        signed_coords: bool = True,
    ) -> None:
        self.k_pca = k_pca
        self.k_graph = k_graph
        self.variance_threshold = variance_threshold
        self.n_anchors = n_anchors
        self.manifold_weight = manifold_weight
        self.signed_coords = signed_coords

        self._model: ManifoldModel | None = None
        self._anchors: np.ndarray | None = None
        self._d_star: int | None = None
        self._X_train: np.ndarray | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def d_star(self) -> int | None:
        """Intrinsic dimension discovered during fit."""
        return self._d_star

    @property
    def anchors(self) -> np.ndarray | None:
        """Anchor points in ambient space, shape (n_anchors, ndim)."""
        return self._anchors

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GeodesicEncoder":
        """Fit the encoder: discover manifold geometry and select anchors.

        :param X: Training data, shape (n_samples, n_features).
        :param y: Optional training labels for class-aware anchor selection.
        :return: self
        """
        X = np.asarray(X, dtype="d")

        self._model = ManifoldModel(
            k_pca=self.k_pca,
            k_graph=self.k_graph,
            variance_threshold=self.variance_threshold,
            manifold_weight=self.manifold_weight,
        )
        self._model.fit(X)
        self._X_train = self._model._X_train

        raw_d = self._model.intrinsic_dim  # float mean over training nodes
        self._d_star = max(1, int(round(raw_d))) if raw_d is not None else X.shape[1]

        n_anchors = self.n_anchors if self.n_anchors is not None else self._d_star
        self._anchors = self._select_anchors(X, n_anchors, y=y)

        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform points to geodesic distance coordinates.

        For each query point x:

        1. Find the nearest training node n in Euclidean space.
        2. Retrieve the local tangent basis B_d = geom.basis[:d*] at n.
        3. Compute geodesic-approximate distance to each anchor::

               diff_j = x − anchor_j          # chord in ambient space
               proj_j = B_d @ diff_j          # project into tangent space
               dist_j = ‖proj_j‖₂            # tangent-space distance

        :param X: Query points, shape (n_samples, n_features).
        :return: Geodesic coordinates, shape (n_samples, n_anchors).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before transform()")
        if self._anchors is None or self._X_train is None or self._d_star is None:
            raise RuntimeError("Encoder is not fully initialized; call fit() first")

        X = np.asarray(X, dtype="d")
        n_query = X.shape[0]
        n_anchors = len(self._anchors)
        out_dim = 2 * n_anchors if self.signed_coords else n_anchors
        out = np.zeros((n_query, out_dim), dtype="float32")

        X_train = self._X_train
        n_train = X_train.shape[0]
        anchors = self._anchors  # (n_anchors, ndim)

        # Precompute squared norms for BLAS distance computation
        train_sq = np.einsum("ij,ij->i", X_train, X_train)
        query_sq = np.einsum("ij,ij->i", X, X)

        # Chunk so one (chunk × n_train) block stays ≤ 256 MB
        chunk = max(1, min(n_query, (256 * 1024 * 1024) // (8 * max(n_train, 1))))
        chunk = max(chunk, 64)

        for cstart in range(0, n_query, chunk):
            cend = min(cstart + chunk, n_query)
            Xq = X[cstart:cend]  # (cq, ndim)

            # Nearest training node per query via batched squared Euclidean
            dist_sq = query_sq[cstart:cend, None] + train_sq[None, :] - 2.0 * (Xq @ X_train.T)
            np.maximum(dist_sq, 0.0, out=dist_sq)
            nearest_idx = np.argmin(dist_sq, axis=1)  # (cq,)

            for qi in range(cend - cstart):
                geom = self._model._geometries[f"n{nearest_idx[qi]}"]
                # Cap at the stored basis rank (geom.basis has d rows, not ndim)
                d = min(self._d_star, geom.intrinsic_dim)
                basis_d = geom.basis[:d]  # (d, ndim)

                # Chord from each anchor to query: (n_anchors, ndim)
                diff_matrix = Xq[qi][None, :] - anchors

                # Project each chord into tangent space: (d, n_anchors)
                proj_matrix = basis_d @ diff_matrix.T

                # Keep both signed and magnitude information per anchor.
                signed = proj_matrix[0] if proj_matrix.shape[0] > 0 else np.zeros(n_anchors)
                mag = np.linalg.norm(proj_matrix, axis=0)
                if self.signed_coords:
                    out[cstart + qi, :n_anchors] = signed
                    out[cstart + qi, n_anchors:] = mag
                else:
                    out[cstart + qi] = mag

        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call.

        :param X: Training data, shape (n_samples, n_features).
        :return: Geodesic coordinates, shape (n_samples, n_anchors).
        """
        return self.fit(X).transform(X)

    # ── Anchor selection ──────────────────────────────────────────────────────

    def _select_anchors(
        self,
        X: np.ndarray,
        n_anchors: int,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select n_anchors spread across the manifold via k-means++ init.

        Each subsequent anchor is chosen with probability proportional to its
        squared distance from the nearest already-chosen anchor, maximally
        spreading anchors across the data geometry.

        :param X: Training data.
        :param n_anchors: Number of anchors to select.
        :param y: Optional labels for class-aware balanced anchor selection.
        :return: Selected anchor points, shape (n_anchors, ndim).
        """
        n = len(X)
        n_anchors = min(n_anchors, n)
        rng = np.random.default_rng(42)

        if y is not None:
            y = np.asarray(y)
            classes = np.unique(y)
            per_class = max(1, n_anchors // len(classes))
            chosen_idx = []
            for cls in classes:
                cls_idx = np.flatnonzero(y == cls)
                if len(cls_idx) == 0:
                    continue
                local_picks = min(per_class, len(cls_idx))
                chosen_local = [int(rng.choice(cls_idx))]
                for _ in range(local_picks - 1):
                    min_d2 = np.full(len(cls_idx), np.inf)
                    for ci in chosen_local:
                        diff = X[cls_idx] - X[ci]
                        d2 = np.einsum("ij,ij->i", diff, diff)
                        np.minimum(min_d2, d2, out=min_d2)
                    probs = min_d2 / max(min_d2.sum(), 1e-12)
                    chosen_local.append(int(rng.choice(cls_idx, p=probs)))
                chosen_idx.extend(chosen_local)

            chosen_idx = list(dict.fromkeys(chosen_idx))
            while len(chosen_idx) < n_anchors:
                min_d2 = np.full(n, np.inf)
                for ci in chosen_idx:
                    diff = X - X[ci]
                    d2 = np.einsum("ij,ij->i", diff, diff)
                    np.minimum(min_d2, d2, out=min_d2)
                probs = min_d2 / max(min_d2.sum(), 1e-12)
                chosen_idx.append(int(rng.choice(n, p=probs)))

            return X[np.array(chosen_idx[:n_anchors])].copy()

        chosen_idx = [int(rng.integers(n))]

        for _ in range(n_anchors - 1):
            # Squared distance from each point to its nearest chosen anchor
            min_d2 = np.full(n, np.inf)
            for ci in chosen_idx:
                diff = X - X[ci]
                d2 = np.einsum("ij,ij->i", diff, diff)
                np.minimum(min_d2, d2, out=min_d2)

            probs = min_d2 / min_d2.sum()
            chosen_idx.append(int(rng.choice(n, p=probs)))

        return X[np.array(chosen_idx)].copy()

    def __repr__(self) -> str:
        state = (
            f"d_star={self._d_star}, n_anchors={len(self._anchors)}"
            if self._d_star is not None and self._anchors is not None
            else "unfitted"
        )
        return (
            f"GeodesicEncoder({state}, k_pca={self.k_pca}, "
            f"variance_threshold={self.variance_threshold})"
        )
