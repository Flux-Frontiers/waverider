"""
UniversalEmbedder: geometry-grounded, modality-agnostic dimensionality reduction.

Produces d*-dimensional coordinates for any dataset.  Drop-in replacement for
sklearn PCA in WaveRider benchmarks — same fit / transform / fit_transform API,
same numpy output, same d-sizing semantics.

Design
------
The embedder always discovers d* from local geometry (the WaveRider way: local PCA
on k-nearest neighbours, cumulative-variance threshold τ).  What it does with d*
depends on the **Manifold Linearity Index** (MLI):

    MLI = global_d_at_τ / d*

where ``global_d_at_τ`` is how many global-PCA components are needed to explain
fraction τ of *global* variance.

* MLI ≈ 1  →  the manifold is approximately linear.  Global PCA at d* is
  near-optimal; anchor-distance coordinates add noise.  Strategy: ``"pca"``.

* MLI > mli_threshold (default 3)  →  the manifold is genuinely curved.  Global
  PCA at d* loses structure that local frames preserve.  Strategy: ``"turtle"``.

``coordinate_mode="auto"`` (the default) measures MLI at fit time and dispatches
to the appropriate strategy.  You can also request a mode explicitly.

Coordinate strategies
---------------------
``"pca"``
    Global linear projection, same as sklearn PCA but sized to d*::

        z(x) = V_d* @ (x − μ)      shape (d*,)

    Globally comparable, lossless rotation, optimal for near-linear manifolds.

``"turtle"``
    Tangent-projected anchor distances using BFS Procrustes-transported frames::

        coord_j(x) = ‖ frame_nearest(x)[:d*] @ (x − anchor_j) ‖₂

    Optimal for genuinely curved manifolds (proteins, complex geometries).

``"tangent"``
    Same as ``"turtle"`` but with raw sign-corrected PCA frames instead of
    transported ones.  Faster fit; less consistent across the manifold.

``"auto"``  [default]
    Measures MLI, selects ``"pca"`` or ``"turtle"`` accordingly.

All strategies output shape ``(n, d*)`` or ``(n, n_components)``, dtype float32.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
License: Elastic 2.0
Last revised: 2026-05-25
"""

from __future__ import annotations

import numpy as np

from .manifold_model import ManifoldModel

_VALID_MODES = ("auto", "pca", "turtle", "tangent")


class UniversalEmbedder:
    """Geometry-grounded d*-dimensional embedding with adaptive strategy selection.

    Discovers d* from local manifold geometry, then chooses the best coordinate
    strategy based on the Manifold Linearity Index (MLI).  For near-linear data
    (most tabular / image datasets) this reduces to global PCA sized to d*.
    For curved manifolds (proteins, molecular conformations) it uses BFS
    Procrustes-transported TurtleND frames.

    Parameters
    ----------
    n_components : int or None
        Fixed output dimension.  If ``None`` (default), output has d* dimensions.
        Set this when a consistent dimension is required across CV folds.
    k_pca : int
        Neighbourhood size for local PCA at each node (default 50).
    k_graph : int
        Neighbours for KNN manifold graph (default 15).
    variance_threshold : float
        Cumulative variance fraction τ for d* selection (default 0.90).
    coordinate_mode : str
        ``"auto"`` (default) — measure MLI, then pick ``"pca"`` or ``"turtle"``.
        ``"pca"``    — always use global PCA projection.
        ``"turtle"`` — always use BFS-transported anchor distances.
        ``"tangent"`` — always use raw-PCA anchor distances.
    mli_threshold : float
        MLI above which ``"auto"`` switches from ``"pca"`` to ``"turtle"``
        (default 3.0).  Increase to prefer PCA more often.
    random_state : int
        Seed for anchor selection (default 42).

    Attributes
    ----------
    d_star : int or None
        Discovered intrinsic dimension (available after :meth:`fit`).
    strategy : str or None
        Coordinate strategy chosen at fit time (``"pca"`` or ``"turtle"``).
    mli : float or None
        Manifold Linearity Index measured at fit time.
    manifold_summary : dict
        Full geometry statistics from the most recent fit.

    Examples
    --------
    >>> import numpy as np
    >>> from waverider.universal_embedder import UniversalEmbedder
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 20))
    >>> ue = UniversalEmbedder()
    >>> Z = ue.fit_transform(X)
    >>> Z.shape[1] == ue.d_star
    True
    """

    def __init__(
        self,
        *,
        n_components: int | None = None,
        k_pca: int = 50,
        k_graph: int = 15,
        variance_threshold: float = 0.90,
        coordinate_mode: str = "auto",
        mli_threshold: float = 3.0,
        random_state: int = 42,
    ) -> None:
        if coordinate_mode not in _VALID_MODES:
            raise ValueError(
                f"coordinate_mode must be one of {_VALID_MODES}, got {coordinate_mode!r}"
            )
        self.n_components = n_components
        self.k_pca = k_pca
        self.k_graph = k_graph
        self.variance_threshold = variance_threshold
        self.coordinate_mode = coordinate_mode
        self.mli_threshold = mli_threshold
        self.random_state = random_state

        # Set after fit
        self._model: ManifoldModel | None = None
        self._d_star: int | None = None
        self._ndim: int | None = None
        self._X_train: np.ndarray | None = None
        self._strategy: str | None = None
        self._mli: float | None = None
        self._global_var_at_d_star: float | None = None

        # PCA strategy state
        self._pca_basis: np.ndarray | None = None   # (n_out, ndim)
        self._pca_mean: np.ndarray | None = None    # (ndim,)

        # Turtle/tangent strategy state
        self._node_frames: dict[str, np.ndarray] | None = None
        self._anchors: np.ndarray | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def d_star(self) -> int | None:
        """Discovered intrinsic dimension."""
        return self._d_star

    @property
    def strategy(self) -> str | None:
        """Coordinate strategy chosen at fit time: ``"pca"`` or ``"turtle"``."""
        return self._strategy

    @property
    def mli(self) -> float | None:
        """Manifold Linearity Index measured at fit time."""
        return self._mli

    @property
    def manifold_summary(self) -> dict:
        """Full geometry statistics from the most recent fit."""
        if self._d_star is None:
            return {}
        n_out = self.n_components if self.n_components is not None else self._d_star
        return {
            "d_star": self._d_star,
            "n_components": n_out,
            "strategy": self._strategy,
            "coordinate_mode": self.coordinate_mode,
            "mli": self._mli,
            "global_var_at_d_star": self._global_var_at_d_star,
            "variance_threshold": self.variance_threshold,
            "mli_threshold": self.mli_threshold,
            "ambient_dim": self._ndim,
            "noise_pct": (
                100.0 * (1.0 - self._d_star / self._ndim) if self._ndim else 0.0
            ),
            "n_nodes": self._model.n_nodes if self._model else 0,
            "mean_intrinsic_dim": float(self._model.intrinsic_dim or 0.0)
            if self._model
            else 0.0,
        }

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "UniversalEmbedder":
        """Discover manifold geometry and build the coordinate system.

        Phase 1 — local geometry:
            ManifoldModel explores the data, building a KNN graph and computing
            local PCA at each node to discover d*.

        Phase 2 — global PCA:
            Full global PCA via covariance eigendecomposition (numpy only).
            Used to compute MLI and — in ``"pca"`` strategy — for transform.

        Phase 3 — strategy selection:
            In ``"auto"`` mode: MLI < mli_threshold → ``"pca"``,
            else → ``"turtle"``.  Explicit modes bypass this.

        Phase 4 — turtle setup (only when strategy is ``"turtle"`` or ``"tangent"``):
            BFS Procrustes frame transport + k-means++ anchor selection.

        :param X: Training data, shape (n_samples, n_features).
        :param y: Optional labels — used for supervised graph construction and
            class-balanced anchor selection in turtle/tangent modes.
        :returns: self
        """
        X = np.asarray(X, dtype="d")
        n, p = X.shape
        self._ndim = p

        # ── Phase 1: local geometry via ManifoldModel ──────────────────────
        k_pca = min(self.k_pca, n - 1)
        k_graph = min(self.k_graph, n - 1)

        self._model = ManifoldModel(
            k_pca=k_pca,
            k_graph=k_graph,
            variance_threshold=self.variance_threshold,
        )
        self._model.fit(X, y)
        self._X_train = self._model._X_train

        raw_d = self._model.intrinsic_dim
        self._d_star = max(1, int(round(raw_d))) if raw_d is not None else p

        n_out = self.n_components if self.n_components is not None else self._d_star

        # ── Phase 2: global PCA + MLI ──────────────────────────────────────
        pca_basis, pca_mean, mli, global_var = self._fit_global_pca(X, n_out)
        self._pca_basis = pca_basis
        self._pca_mean = pca_mean
        self._mli = mli
        self._global_var_at_d_star = global_var

        # ── Phase 3: strategy selection ────────────────────────────────────
        if self.coordinate_mode == "auto":
            self._strategy = "turtle" if mli > self.mli_threshold else "pca"
        elif self.coordinate_mode == "pca":
            self._strategy = "pca"
        else:
            self._strategy = self.coordinate_mode  # "turtle" or "tangent"

        # ── Phase 4: turtle/tangent setup ──────────────────────────────────
        if self._strategy in ("turtle", "tangent"):
            if self._strategy == "turtle":
                self._node_frames = self._transport_frames()
            else:
                self._node_frames = self._build_tangent_frames()
            self._anchors = self._select_anchors(X, n_out, y=y)

        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed points using the strategy chosen at fit time.

        ``"pca"`` strategy:   z(x) = V_d* @ (x − μ)   [global linear projection]
        ``"turtle"`` strategy: coord_j = ‖frame_nearest @ (x − anchor_j)‖  [anchor dists]

        :param X: Query points, shape (n_samples, n_features).
        :returns: Embeddings, shape (n_samples, n_components or d*), float32.
        :raises RuntimeError: If called before :meth:`fit`.
        """
        if self._strategy is None:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X, dtype="d")

        if self._strategy == "pca":
            return self._transform_pca(X)
        return self._transform_anchor(X)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit and transform in one call.

        :param X: Training data, shape (n_samples, n_features).
        :param y: Optional labels.
        :returns: Embeddings, shape (n_samples, n_components or d*), float32.
        """
        return self.fit(X, y).transform(X)

    # ── Internal: global PCA ──────────────────────────────────────────────────

    def _fit_global_pca(
        self, X: np.ndarray, n_out: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute global PCA, MLI, and global variance retained at d*.

        Uses the covariance eigendecomposition when n ≥ p (efficient for tall
        matrices) and direct SVD when n < p.  Pure numpy — no sklearn dependency.

        :returns: (pca_basis, pca_mean, mli, global_var_at_d_star)
        """
        n, p = X.shape
        mean = X.mean(axis=0)
        Xc = (X - mean).astype("d")

        if n >= p:
            # Covariance matrix: (p, p) — efficient for n >> p
            cov = (Xc.T @ Xc) / max(n - 1, 1)
            eigenvalues, V = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues[::-1], 0.0)
            Vt = V[:, ::-1].T  # (p, p), rows = eigenvectors descending
        else:
            # Direct thin SVD — efficient for n < p
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            eigenvalues = np.maximum(s**2 / max(n - 1, 1), 0.0)

        total = eigenvalues.sum()
        if total <= 0:
            # Degenerate data (all constant)
            return (
                np.zeros((n_out, p), dtype="d"),
                mean,
                1.0,
                0.0,
            )

        cumvar = np.cumsum(eigenvalues) / total

        # Global d at τ: how many global PCs explain variance_threshold
        global_d_at_tau = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        global_d_at_tau = min(global_d_at_tau, len(cumvar))

        # Global variance retained at d*
        idx = min(self._d_star - 1, len(cumvar) - 1)
        global_var_at_d_star = float(cumvar[idx])

        mli = global_d_at_tau / max(self._d_star, 1)

        # Basis: take n_out top eigenvectors
        n_take = min(n_out, len(Vt))
        basis = Vt[:n_take].copy()
        if n_take < n_out:
            # Pad with zeros if data rank is lower than requested
            pad = np.zeros((n_out - n_take, p), dtype="d")
            basis = np.vstack([basis, pad])

        return basis, mean, float(mli), global_var_at_d_star

    # ── Internal: PCA transform ───────────────────────────────────────────────

    def _transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Global linear projection: z = V_d* @ (x − μ)."""
        Z = (X - self._pca_mean) @ self._pca_basis.T
        return Z.astype("float32")

    # ── Internal: anchor-distance transform ───────────────────────────────────

    def _transform_anchor(self, X: np.ndarray) -> np.ndarray:
        """Tangent-projected anchor distances (turtle or tangent strategy)."""
        n_query = X.shape[0]
        X_train = self._X_train
        n_train = X_train.shape[0]
        anchors = self._anchors        # (n_out, ndim)
        n_out = len(anchors)

        out = np.zeros((n_query, n_out), dtype="float32")

        train_sq = np.einsum("ij,ij->i", X_train, X_train)
        query_sq = np.einsum("ij,ij->i", X, X)
        max_block = 256 * 1024 * 1024
        chunk = max(64, min(n_query, max_block // (8 * max(n_train, 1))))

        for cstart in range(0, n_query, chunk):
            cend = min(cstart + chunk, n_query)
            Xq = X[cstart:cend]
            dist_sq = (
                query_sq[cstart:cend, None]
                + train_sq[None, :]
                - 2.0 * (Xq @ X_train.T)
            )
            np.maximum(dist_sq, 0.0, out=dist_sq)
            nearest = np.argmin(dist_sq, axis=1)

            for qi in range(cend - cstart):
                node_id = f"n{nearest[qi]}"
                frame_d = self._node_frames[node_id]       # (d*, ndim)
                diff_matrix = Xq[qi][None, :] - anchors   # (n_out, ndim)
                proj_matrix = frame_d @ diff_matrix.T      # (d*, n_out)
                out[cstart + qi] = np.linalg.norm(proj_matrix, axis=0).astype("float32")

        return out

    # ── Internal: anchor selection ────────────────────────────────────────────

    def _select_anchors(
        self, X: np.ndarray, n_anchors: int, y: np.ndarray | None = None
    ) -> np.ndarray:
        """k-means++ anchor selection, class-balanced when labels are provided."""
        n = len(X)
        n_anchors = min(n_anchors, n)
        rng = np.random.default_rng(self.random_state)

        if y is not None:
            y = np.asarray(y)
            classes = np.unique(y)
            per_class = max(1, n_anchors // len(classes))
            chosen: list[int] = []
            for cls in classes:
                cls_idx = np.flatnonzero(y == cls)
                if len(cls_idx) == 0:
                    continue
                picks = min(per_class, len(cls_idx))
                picked = [int(rng.choice(cls_idx))]
                for _ in range(picks - 1):
                    min_d2 = np.full(len(cls_idx), np.inf)
                    for ci in picked:
                        diff = X[cls_idx] - X[ci]
                        np.minimum(min_d2, np.einsum("ij,ij->i", diff, diff), out=min_d2)
                    probs = min_d2 / max(min_d2.sum(), 1e-12)
                    picked.append(int(rng.choice(cls_idx, p=probs)))
                chosen.extend(picked)
            chosen = list(dict.fromkeys(chosen))
            while len(chosen) < n_anchors:
                min_d2 = np.full(n, np.inf)
                for ci in chosen:
                    diff = X - X[ci]
                    np.minimum(min_d2, np.einsum("ij,ij->i", diff, diff), out=min_d2)
                probs = min_d2 / max(min_d2.sum(), 1e-12)
                chosen.append(int(rng.choice(n, p=probs)))
            return X[np.array(chosen[:n_anchors])].copy()

        chosen = [int(rng.integers(n))]
        for _ in range(n_anchors - 1):
            min_d2 = np.full(n, np.inf)
            for ci in chosen:
                diff = X - X[ci]
                np.minimum(min_d2, np.einsum("ij,ij->i", diff, diff), out=min_d2)
            probs = min_d2 / min_d2.sum()
            chosen.append(int(rng.choice(n, p=probs)))
        return X[np.array(chosen)].copy()

    # ── Internal: frame construction ──────────────────────────────────────────

    def _padded_basis(self, node_id: str) -> np.ndarray:
        """d*-row basis for a node, truncating or zero-padding to d*."""
        geom = self._model._geometries[node_id]
        basis = geom.basis
        d = self._d_star
        if len(basis) >= d:
            return basis[:d].copy()
        return np.vstack([basis, np.zeros((d - len(basis), self._ndim), dtype="d")])

    def _sign_correct(self, basis: np.ndarray) -> np.ndarray:
        """Flip each row so its max-abs component is positive."""
        for i in range(len(basis)):
            idx = int(np.argmax(np.abs(basis[i])))
            if basis[i][idx] < 0:
                basis[i] = -basis[i]
        return basis

    def _build_tangent_frames(self) -> dict[str, np.ndarray]:
        """Raw sign-corrected PCA frames at each node."""
        return {
            nid: self._sign_correct(self._padded_basis(nid))
            for nid in self._model._geometries
        }

    def _transport_frames(self) -> dict[str, np.ndarray]:
        """BFS Procrustes frame transport along the manifold graph.

        At each hop, R = Vt.T @ U.T from SVD(parent_frame @ neighbour_basis.T)
        rotates the parent frame to align with the neighbour's local PCA.
        Disconnected nodes fall back to sign-corrected tangent frames.
        """
        frames: dict[str, np.ndarray] = {}

        root_id = "n0"
        frames[root_id] = self._sign_correct(self._padded_basis(root_id))

        visited = {root_id}
        queue = [root_id]

        while queue:
            node_id = queue.pop(0)
            current = frames[node_id]

            for edge in self._model._graph._edges.get(node_id, []):
                nb_id = edge.target_id
                if nb_id in visited:
                    continue
                visited.add(nb_id)

                nb_basis = self._padded_basis(nb_id)
                d_act = min(self._model._geometries[nb_id].intrinsic_dim, self._d_star)

                try:
                    if d_act == self._d_star:
                        U, _, Vt = np.linalg.svd(current @ nb_basis.T, full_matrices=False)
                        transported = (Vt.T @ U.T) @ current
                    else:
                        U_k, _, Vt_k = np.linalg.svd(
                            current[:d_act] @ nb_basis[:d_act].T, full_matrices=False
                        )
                        transported = current.copy()
                        transported[:d_act] = (Vt_k.T @ U_k.T) @ current[:d_act]
                except np.linalg.LinAlgError:
                    transported = nb_basis

                frames[nb_id] = transported.astype("d")
                queue.append(nb_id)

        for node_id in self._model._geometries:
            if node_id not in frames:
                frames[node_id] = self._sign_correct(self._padded_basis(node_id))

        return frames

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._d_star is None:
            return (
                f"UniversalEmbedder(unfitted, mode={self.coordinate_mode!r}, "
                f"k_pca={self.k_pca}, tau={self.variance_threshold})"
            )
        n_out = self.n_components if self.n_components is not None else self._d_star
        mli_str = f"{self._mli:.2f}" if self._mli is not None else "?"
        return (
            f"UniversalEmbedder(d_star={self._d_star}, strategy={self._strategy!r}, "
            f"mli={mli_str}, n_out={n_out}, "
            f"k_pca={self.k_pca}, tau={self.variance_threshold})"
        )
