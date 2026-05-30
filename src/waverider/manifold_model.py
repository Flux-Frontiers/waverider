"""
ManifoldModel: The manifold IS the model.

A self-contained geometric classifier that discovers manifold structure
during an exploration phase, anchors it with a graph, and uses that
structure for fast classification during a navigate phase.

Architecture
------------
Phase 1 — Explore (fit):
    - Build a KNN graph over training data
    - At each node, compute local PCA to discover the tangent space
    - Store local basis (Frenet frame), intrinsic dimensionality, eigenvalues
    - Weight graph edges by manifold-aware (projected) distance

Phase 2 — Navigate (predict):
    - Project new point into nearest node's tangent space
    - Walk the graph to find manifold-aware neighbors
    - Classify by weighted vote in projected space

Phase 3 — Fly:
    - Start at any graph node
    - Navigate the manifold interactively using the TurtleND
    - Graph provides connectivity, local frames provide orientation

The "trained model" is:
    1. The graph (connectivity / topology)
    2. The local basis vectors at each node (geometry)
    3. The eigenvalue field (how geometry varies)

There are no learned weights. The model is literally a map you can walk through.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers, https://github.com/Flux-Frontiers
License: BSD
Last revised: 2026-03-23 -egs-
"""

__pdoc__ = {"__all__": True}

from dataclasses import dataclass, field

import numpy as np

from waverider.turtleND import TurtleND

# ---------------------------------------------------------------------------
# Minimal internal graph — no external graph_reasoner dependency
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    """Lightweight directed edge for the manifold graph."""

    target_id: str
    weight: float
    metadata: dict = field(default_factory=dict)


class _ManifoldGraph:
    """Simple adjacency-list graph storing node embeddings and weighted edges.

    Not exposed publicly — internal implementation detail of ManifoldModel.
    """

    def __init__(self):
        self._embeddings: dict[str, np.ndarray] = {}
        self._edges: dict[str, list[_Edge]] = {}

    def add_node(self, node_id: str, embedding: np.ndarray) -> None:
        self._embeddings[node_id] = embedding
        self._edges.setdefault(node_id, [])

    def get_embedding(self, node_id: str) -> np.ndarray:
        return self._embeddings[node_id]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        weight: float,
        metadata: dict | None = None,
    ) -> None:
        edge = _Edge(target_id=target_id, weight=weight, metadata=metadata or {})
        self._edges.setdefault(source_id, []).append(edge)

    def __len__(self) -> int:
        return len(self._embeddings)


# ---------------------------------------------------------------------------
# Node geometry — stored at each graph node during exploration
# ---------------------------------------------------------------------------


@dataclass
class NodeGeometry:
    """Local manifold geometry at a graph node.

    Attributes
    ----------
    basis : np.ndarray
        Orthonormal basis for the local tangent space, shape (d, ndim).
        Rows are principal directions in descending eigenvalue order.
    eigenvalues : np.ndarray
        PCA eigenvalues (descending), length ndim.
    intrinsic_dim : int
        Number of dimensions capturing ``variance_threshold`` of local variance.
    centroid : np.ndarray
        Mean of the local neighborhood used for PCA.
    label : optional
        Class label (if supervised).
    index : int
        Index into the original training array.
    """

    basis: np.ndarray
    eigenvalues: np.ndarray
    intrinsic_dim: int
    centroid: np.ndarray
    label: object = None
    index: int = 0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class ManifoldModel:
    """The manifold IS the model.

    A geometric classifier that discovers manifold structure during exploration,
    anchors it with a graph, and uses that graph for fast classification.

    Parameters
    ----------
    k_graph : int
        Number of neighbors for graph construction (default 15).
    k_pca : int
        Number of neighbors for local PCA at each node (default 50).
    k_vote : int
        Number of neighbors for voting during prediction (default 7).
    variance_threshold : float
        Cumulative variance fraction for intrinsic dimensionality (default 0.95).
    manifold_weight : float
        Blend between manifold distance (1.0) and Euclidean distance (0.0)
        during navigation. Default 0.8.

    Examples
    --------
    >>> import numpy as np
    >>> from waverider.manifold_model import ManifoldModel
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((100, 10))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = ManifoldModel(k_graph=10, k_pca=30)
    >>> model.fit(X, y)
    ManifoldModel(mode='navigate', nodes=100, ndim=10)
    >>> preds = model.predict(X[:5])
    >>> len(preds) == 5
    True
    """

    # Modes
    EXPLORE = "explore"
    NAVIGATE = "navigate"

    def __init__(
        self,
        k_graph: int = 15,
        k_pca: int = 50,
        k_vote: int = 7,
        variance_threshold: float = 0.95,
        manifold_weight: float = 0.8,
    ):
        self.k_graph = k_graph
        self.k_pca = k_pca
        self.k_vote = k_vote
        self.variance_threshold = variance_threshold
        self.manifold_weight = manifold_weight

        # State
        self._mode = self.EXPLORE
        self._graph: _ManifoldGraph | None = None
        self._turtle: TurtleND | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._ndim: int | None = None
        self._geometries: dict[str, NodeGeometry] = {}

        # Navigation state
        self._current_node: str | None = None
        self._flight_path: list[str] = []

        # Diagnostics
        self._global_intrinsic_dim: float | None = None

    # ----- Properties -----

    @property
    def mode(self) -> str:
        """Current mode: 'explore' or 'navigate'."""
        return self._mode

    @property
    def graph(self) -> _ManifoldGraph | None:
        """The underlying knowledge graph (available after fit)."""
        return self._graph

    @property
    def turtle(self) -> TurtleND | None:
        """The navigation turtle (available after fit)."""
        return self._turtle

    @property
    def ndim(self) -> int | None:
        """Ambient dimensionality."""
        return self._ndim

    @property
    def intrinsic_dim(self) -> float | None:
        """Mean intrinsic dimensionality discovered during exploration."""
        return self._global_intrinsic_dim

    @property
    def current_node(self) -> str | None:
        """Node ID the turtle is currently at (during fly mode)."""
        return self._current_node

    @property
    def flight_path(self) -> list[str]:
        """Sequence of node IDs visited during fly mode."""
        return list(self._flight_path)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._graph) if self._graph else 0

    # ----- Phase 1: Explore (fit) -----

    def fit(self, X, y=None):
        """Explore the data manifold and build the graph model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,), optional
            Labels for supervised classification.

        Returns
        -------
        self
        """
        self._mode = self.EXPLORE
        self._X_train = np.asarray(X, dtype="d")
        n_samples, self._ndim = self._X_train.shape

        if y is not None:
            self._y_train = np.asarray(y)
        else:
            self._y_train = None

        # Clamp neighborhood sizes
        k_pca = min(self.k_pca, n_samples - 1)
        k_graph = min(self.k_graph, n_samples - 1)

        # Build the graph
        self._graph = _ManifoldGraph()

        # Step 1: Add all points as nodes
        for i in range(n_samples):
            node_id = f"n{i}"
            self._graph.add_node(node_id, self._X_train[i])

        # Steps 2 & 3: vectorized kNN via chunked BLAS GEMM. The original
        # implementation called np.linalg.norm(X - point, axis=1) once per
        # node — O(N) Python-level calls, each with significant overhead.
        # We replace that with a single GEMM per chunk: ||a - b||² is
        # expanded as ||a||² + ||b||² - 2·a·b, so distance computation
        # becomes one (chunk × N) matrix multiply that Apple Accelerate
        # handles in one shot. SVD and edge construction are unchanged.
        intrinsic_dims = self._fit_vectorized(k_pca, k_graph)

        self._global_intrinsic_dim = float(np.mean(intrinsic_dims))

        # Step 4: Initialize turtle for navigation
        self._turtle = TurtleND(self._ndim, name="ManifoldModel_Turtle")

        # Switch to navigate mode
        self._mode = self.NAVIGATE

        return self

    def _fit_vectorized(self, k_pca: int, k_graph: int) -> list[int]:
        """Vectorized fit: chunked pairwise distances via BLAS GEMM.

        Replaces the per-node np.linalg.norm loop in the original fit().
        Computes ||a - b||² = ||a||² + ||b||² - 2·a·b for chunks of rows
        at a time, so the dominant cost flows through one matmul per chunk
        rather than N Python-level norm calls. SVD and edge construction
        per-node are unchanged.

        :param k_pca: Local-PCA neighborhood size (already clamped).
        :param k_graph: Manifold-graph degree (already clamped).
        :return: Per-node intrinsic dimensionality, in node order.
        """
        # Numpy 2.x + Apple Accelerate surfaces stale BLAS FP-env flags
        # (set by prior degenerate SVDs on zero-variance neighborhoods)
        # as matmul warnings even when the current call is clean. Inputs
        # are finite; the warnings are backend bookkeeping. Silence them
        # for the whole vectorized fit.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return self._fit_vectorized_inner(k_pca, k_graph)

    def _fit_vectorized_inner(self, k_pca: int, k_graph: int) -> list[int]:
        X = self._X_train
        n = X.shape[0]
        sq_norms = np.einsum("ij,ij->i", X, X)

        # Chunk sized to keep one distance block well under ~256 MB.
        # block bytes = chunk * n * 8.
        max_block_bytes = 256 * 1024 * 1024
        chunk = max(1, min(n, max_block_bytes // (8 * max(n, 1))))
        chunk = max(chunk, 64)  # don't go too small — Python-loop overhead

        intrinsic_dims: list[int] = []
        _log_interval = max(1, n // 10)

        # ----- Phase A: local geometry -----
        for cstart in range(0, n, chunk):
            cend = min(cstart + chunk, n)
            block = sq_norms[cstart:cend, None] + sq_norms[None, :] - 2.0 * (X[cstart:cend] @ X.T)
            np.maximum(block, 0.0, out=block)
            # Mask self-distance per row so argpartition skips it.
            row_idx = np.arange(cend - cstart)
            block[row_idx, np.arange(cstart, cend)] = np.inf

            nn_block = np.argpartition(block, k_pca, axis=1)[:, :k_pca]

            # Batched SVD: stack all (k_pca, ndim) neighbor matrices for the
            # chunk into one (chunk_size, k_pca, ndim) tensor and dispatch a
            # single LAPACK call instead of chunk_size separate ones.
            neighbors_batch = X[nn_block]  # (chunk_size, k_pca, ndim)
            centroids_batch = neighbors_batch.mean(axis=1)  # (chunk_size, ndim)
            centered_batch = neighbors_batch - centroids_batch[:, None, :]
            # Sanitize NaN/Inf (e.g. from missing omega values) and regularize
            # near-zero-variance columns (cos ω at ±180° has std ~0.002 in a
            # local neighbourhood, which makes LAPACK's batched SVD diverge).
            np.nan_to_num(centered_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            col_std = centered_batch.std(axis=1)  # (chunk_size, ndim)
            tiny = col_std < 1e-5  # (chunk_size, ndim)
            if tiny.any():
                noise = (
                    np.random.default_rng(0)
                    .standard_normal(centered_batch.shape)
                    .astype(centered_batch.dtype)
                )
                centered_batch = (
                    centered_batch + noise * np.where(tiny, 1e-4, 0.0)[:, np.newaxis, :]
                )
            _, s_batch, Vt_batch = np.linalg.svd(centered_batch, full_matrices=False)
            eig_batch = np.maximum((s_batch**2) / (k_pca - 1), 0.0)

            for r in range(cend - cstart):
                i = cstart + r
                eigenvalues = eig_batch[r]
                Vt = Vt_batch[r]
                centroid = centroids_batch[r]

                total = eigenvalues.sum()
                if total > 0:
                    cumulative = np.cumsum(eigenvalues) / total
                    intrinsic_dim = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
                else:
                    intrinsic_dim = self._ndim
                intrinsic_dim = max(1, min(intrinsic_dim, self._ndim))

                basis = Vt[:intrinsic_dim]
                if not np.isfinite(basis).all():
                    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)

                label = self._y_train[i] if self._y_train is not None else None

                self._geometries[f"n{i}"] = NodeGeometry(
                    basis=basis,
                    eigenvalues=eigenvalues,
                    intrinsic_dim=intrinsic_dim,
                    centroid=centroid,
                    label=label,
                    index=i,
                )
                intrinsic_dims.append(intrinsic_dim)

                if (i + 1) % _log_interval == 0:
                    end = "\n" if (i + 1) == n else "\r"
                    print(
                        f"  Explore: {i + 1}/{n} nodes (mean d={np.mean(intrinsic_dims):.1f})",
                        end=end,
                        flush=True,
                    )

        # ----- Phase B: manifold-aware edges -----
        w_blend = self.manifold_weight
        for cstart in range(0, n, chunk):
            cend = min(cstart + chunk, n)
            block = sq_norms[cstart:cend, None] + sq_norms[None, :] - 2.0 * (X[cstart:cend] @ X.T)
            np.maximum(block, 0.0, out=block)
            row_idx = np.arange(cend - cstart)
            block[row_idx, np.arange(cstart, cend)] = np.inf

            nn_block = np.argpartition(block, k_graph, axis=1)[:, :k_graph]

            for r in range(cend - cstart):
                i = cstart + r
                src_id = f"n{i}"
                src_geom = self._geometries[src_id]
                d = src_geom.intrinsic_dim
                nn_idx = nn_block[r]

                # Euclidean distances for the k_graph picked neighbors only
                # (sqrt of the small selected slice — cheap).
                sel_sq = block[r, nn_idx]
                sel_sq = np.maximum(sel_sq, 0.0)
                dists_euc_sel = np.sqrt(sel_sq)

                neighbor_pts = X[nn_idx]
                diffs = neighbor_pts - src_geom.centroid
                if not np.isfinite(diffs).all():
                    diffs = np.nan_to_num(diffs, nan=0.0, posinf=0.0, neginf=0.0)
                # Project onto the d tangent dims; one matmul for all neighbors.
                proj = diffs @ src_geom.basis[:d].T  # (k_graph, d)
                manifold_dists = np.linalg.norm(proj, axis=1)

                blended = w_blend * manifold_dists + (1.0 - w_blend) * dists_euc_sel
                weights = 1.0 / (1.0 + blended)

                for k_local, j in enumerate(nn_idx):
                    self._graph.add_edge(
                        src_id,
                        f"n{int(j)}",
                        float(weights[k_local]),
                        metadata={
                            "euclidean_dist": float(dists_euc_sel[k_local]),
                            "manifold_dist": float(manifold_dists[k_local]),
                            "blended_dist": float(blended[k_local]),
                        },
                    )

        return intrinsic_dims

    def _compute_local_geometry(self, idx: int, k_pca: int) -> NodeGeometry:
        """Compute local PCA at training point idx.

        Uses SVD on the centered data matrix instead of eigendecomposition
        on the covariance matrix. This is O(k²·ndim) vs O(ndim³), which
        is critical for high-dimensional data (e.g. CIFAR-10 at 3072D).

        Only the truncated basis (first d rows) is stored, reducing memory
        from O(ndim²) to O(d·ndim) per node.
        """
        point = self._X_train[idx]

        # Find k_pca nearest neighbors
        dists = np.linalg.norm(self._X_train - point, axis=1)
        dists[idx] = np.inf  # exclude self
        nn_idx = np.argpartition(dists, k_pca)[:k_pca]
        neighbors = self._X_train[nn_idx]

        # Local PCA via SVD (efficient for k << ndim)
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s**2) / (len(neighbors) - 1)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Intrinsic dimensionality
        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            intrinsic_dim = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        else:
            intrinsic_dim = self._ndim

        intrinsic_dim = max(1, min(intrinsic_dim, self._ndim))

        # Truncated basis: only store the d principal directions we need
        # Vt rows are principal directions in descending singular value order
        basis = Vt[:intrinsic_dim]  # shape (d, ndim) — NOT (ndim, ndim)
        if not np.isfinite(basis).all():
            basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)

        label = self._y_train[idx] if self._y_train is not None else None

        return NodeGeometry(
            basis=basis,
            eigenvalues=eigenvalues,
            intrinsic_dim=intrinsic_dim,
            centroid=centroid,
            label=label,
            index=idx,
        )

    def _build_manifold_edges(self, k_graph: int) -> None:
        """Build graph edges weighted by manifold-aware distance.

        For each node, finds k_graph nearest neighbors in Euclidean space,
        then recomputes distance by projecting into the source node's
        tangent space — the manifold distance.
        """
        n = len(self._X_train)

        for i in range(n):
            src_id = f"n{i}"
            src_geom = self._geometries[src_id]
            src_point = self._X_train[i]
            d = src_geom.intrinsic_dim

            # Euclidean nearest neighbors
            dists_euc = np.linalg.norm(self._X_train - src_point, axis=1)
            dists_euc[i] = np.inf
            nn_idx = np.argpartition(dists_euc, k_graph)[:k_graph]

            for j in nn_idx:
                tgt_id = f"n{j}"
                tgt_point = self._X_train[j]

                # Project difference into tangent space
                diff = tgt_point - src_geom.centroid
                if not np.isfinite(diff).all():
                    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
                proj = src_geom.basis[:d] @ diff  # project onto d tangent dims
                manifold_dist = float(np.linalg.norm(proj))
                euclidean_dist = float(dists_euc[j])

                # Blend: manifold-weighted distance
                w = self.manifold_weight
                blended_dist = w * manifold_dist + (1 - w) * euclidean_dist

                # Edge weight: inversely proportional to distance
                max(blended_dist, 1e-10)
                weight = 1.0 / (1.0 + blended_dist)

                self._graph.add_edge(
                    src_id,
                    tgt_id,
                    weight,
                    metadata={
                        "euclidean_dist": euclidean_dist,
                        "manifold_dist": manifold_dist,
                        "blended_dist": blended_dist,
                    },
                )

    # ----- Phase 2: Navigate (predict) -----

    def predict(self, X):
        """Classify new points using the manifold graph.

        For each query point:
        1. Find nearest graph node in Euclidean space
        2. Use that node's local geometry to compute manifold-aware distances
        3. Walk the graph to gather neighbor votes
        4. Majority vote

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Points to classify.

        Returns
        -------
        predictions : np.ndarray
            Predicted labels.
        """
        if self._mode != self.NAVIGATE:
            raise RuntimeError("Must call fit() before predict()")
        if self._y_train is None:
            raise RuntimeError("Model was fit without labels — cannot classify")

        X = np.asarray(X, dtype="d")

        # See _fit_vectorized: silence stale Accelerate FP-env flags from
        # degenerate per-query SVDs so predict doesn't spam.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return self._predict_vectorized(X)

    def _predict_vectorized(self, X: np.ndarray) -> np.ndarray:
        """Batched predict: chunked distances + batched local PCA + batched projection.

        Mirrors the fit-phase vectorization. Query→train distances flow
        through one BLAS GEMM per chunk, per-query local PCA runs as a
        single batched SVD across the chunk, and tangent projections are
        one matmul instead of n_queries separate ones. Only the final
        graph-walk + vote step is per-query, and at that point each query
        is cheap dict-level work.
        """
        X_train = self._X_train
        n_train, ndim = X_train.shape
        n_query = X.shape[0]

        k_pca = min(self.k_pca, n_train - 1)
        predictions = np.zeros(n_query, dtype=self._y_train.dtype)

        train_sq = np.einsum("ij,ij->i", X_train, X_train)
        query_sq = np.einsum("ij,ij->i", X, X)

        # Chunk sized so one (chunk × n_train) distance block stays <=256MB.
        max_block_bytes = 256 * 1024 * 1024
        chunk = max(1, min(n_query, max_block_bytes // (8 * max(n_train, 1))))
        chunk = max(chunk, 64)

        n_chunks = (n_query + chunk - 1) // chunk
        for chunk_i, cstart in enumerate(range(0, n_query, chunk)):
            if (chunk_i + 1) % max(1, n_chunks // 10) == 0 or chunk_i == 0:
                end = "\n" if (chunk_i + 1) == n_chunks else "\r"
                print(
                    f"  Predict: {cstart + chunk}/{n_query} ({chunk_i + 1}/{n_chunks})",
                    end=end,
                    flush=True,
                )
            cend = min(cstart + chunk, n_query)
            cq = cend - cstart
            Xq = X[cstart:cend]

            # --- Phase A: batched Euclidean distances (n_query_chunk, n_train)
            dist_sq = query_sq[cstart:cend, None] + train_sq[None, :] - 2.0 * (Xq @ X_train.T)
            np.maximum(dist_sq, 0.0, out=dist_sq)

            # --- Phase B: per-query k_pca candidate pool via batched argpartition
            pca_idx_batch = np.argpartition(dist_sq, k_pca, axis=1)[:, :k_pca]  # (cq, k_pca)

            # Gather neighbor coords: (cq, k_pca, ndim)
            neighbors_batch = X_train[pca_idx_batch]
            centroids_batch = neighbors_batch.mean(axis=1)  # (cq, ndim)
            centered_batch = neighbors_batch - centroids_batch[:, None, :]
            np.nan_to_num(centered_batch, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            col_std = centered_batch.std(axis=1)
            tiny = col_std < 1e-5
            if tiny.any():
                noise = (
                    np.random.default_rng(0)
                    .standard_normal(centered_batch.shape)
                    .astype(centered_batch.dtype)
                )
                centered_batch = (
                    centered_batch + noise * np.where(tiny, 1e-4, 0.0)[:, np.newaxis, :]
                )

            # --- Phase C: batched local PCA via one SVD call
            # SVD rank = min(k_pca, ndim); when k_pca > ndim, Vt has ndim rows.
            _, s_batch, Vt_batch = np.linalg.svd(centered_batch, full_matrices=False)
            svd_rank = s_batch.shape[1]  # min(k_pca, ndim)
            eig_batch = np.maximum((s_batch**2) / (k_pca - 1), 0.0)  # (cq, svd_rank)

            # Variance-threshold truncation per query
            totals = eig_batch.sum(axis=1)  # (cq,)
            safe = totals > 0
            cumulative = np.where(
                safe[:, None],
                np.cumsum(eig_batch, axis=1) / np.where(safe[:, None], totals[:, None], 1.0),
                1.0,
            )
            d_batch = np.where(
                safe,
                (cumulative < self.variance_threshold).sum(axis=1) + 1,
                svd_rank,
            )
            d_batch = np.clip(d_batch, 1, svd_rank).astype(int)  # (cq,)

            # --- Phase D: batched tangent projection
            query_diff = Xq - centroids_batch  # (cq, ndim)
            # Vt_batch[q]: (svd_rank, ndim). query_proj_full[q] = Vt[q] @ diff[q]
            query_proj_full = np.einsum("qrd,qd->qr", Vt_batch, query_diff)  # (cq, svd_rank)
            # Neighbor projection: (cq, k_pca_neigh, ndim) @ (cq, svd_rank, ndim).T
            neighbor_proj_full = np.einsum("qjd,qrd->qjr", centered_batch, Vt_batch)
            # shape: (cq, k_pca_neigh, svd_rank)

            # Per-query truncation mask indexed over svd_rank (not k_pca).
            r_idx = np.arange(svd_rank)
            mask = r_idx[None, :] < d_batch[:, None]  # (cq, svd_rank)

            diff_proj = neighbor_proj_full - query_proj_full[:, None, :]
            diff_proj_sq = diff_proj**2
            diff_proj_sq *= mask[:, None, :]  # (cq, 1, svd_rank) broadcast
            proj_dists_batch = np.sqrt(diff_proj_sq.sum(axis=-1))  # (cq, k_pca_neigh)

            # Graph entry point = Euclidean nearest
            entry_idx_batch = np.argmin(dist_sq, axis=1)  # (cq,)

            # --- Phase E: per-query vote (merge graph neighbors + majority)
            for r in range(cq):
                q_global = cstart + r
                pca_idx = pca_idx_batch[r]
                proj_dists = proj_dists_batch[r]
                d_q = int(d_batch[r])
                basis_q_T = Vt_batch[r, :d_q]  # (d_q, ndim)
                query_proj_d = query_proj_full[r, :d_q]  # (d_q,)
                centroid_q = centroids_batch[r]

                all_candidates: dict[int, float] = {}
                for local_i in range(k_pca):
                    all_candidates[int(pca_idx[local_i])] = float(proj_dists[local_i])

                entry_id = f"n{int(entry_idx_batch[r])}"
                graph_neighbors = self._gather_graph_neighbors(entry_id)

                for idx in graph_neighbors:
                    if idx in all_candidates:
                        continue
                    diff = X_train[idx] - centroid_q
                    proj = basis_q_T @ diff
                    all_candidates[idx] = float(np.linalg.norm(proj - query_proj_d))

                # Top-k vote
                sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])
                k = min(self.k_vote, len(sorted_candidates))
                label_scores: dict = {}
                for idx, dist in sorted_candidates[:k]:
                    label = self._y_train[idx]
                    w = 1.0 / (1.0 + dist)
                    label_scores[label] = label_scores.get(label, 0.0) + w
                predictions[q_global] = max(label_scores, key=label_scores.get)

        return predictions

    def _predict_single(self, query: np.ndarray):
        """Classify a single query point via query-local manifold projection.

        Strategy:
        1. Find k_pca Euclidean nearest neighbors (candidate pool)
        2. Compute local PCA at the query point to discover its tangent space
        3. Project query and candidates into the tangent space
        4. Vote among k_vote nearest in projected space
        5. Use graph connectivity as a tiebreaker / refinement
        """
        # Step 1: Find candidate neighbors in Euclidean space
        dists_euc = np.linalg.norm(self._X_train - query, axis=1)
        k_pca = min(self.k_pca, len(self._X_train) - 1)
        pca_idx = np.argpartition(dists_euc, k_pca)[:k_pca]
        pca_neighbors = self._X_train[pca_idx]

        # Step 2: Local PCA at the query point via SVD (efficient for k << ndim)
        centroid = pca_neighbors.mean(axis=0)
        centered = pca_neighbors - centroid
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s**2) / (len(pca_neighbors) - 1)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            d = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        else:
            d = self._ndim
        d = max(1, min(d, self._ndim))

        # Tangent space basis: Vt rows are principal directions
        V_d = Vt[:d].T  # shape (ndim, d) for projection

        # Step 3: Project into tangent space
        query_proj = V_d.T @ (query - centroid)
        neighbor_proj = centered @ V_d

        # Manifold distances in projected space
        proj_dists = np.linalg.norm(neighbor_proj - query_proj, axis=1)

        # Step 4: Also gather graph-connected neighbors for enrichment
        entry_idx = np.argmin(dists_euc)
        entry_id = f"n{entry_idx}"
        graph_neighbors = self._gather_graph_neighbors(entry_id)

        # Merge graph neighbors into candidate pool
        all_candidates = {}  # index -> manifold_dist
        for local_i, global_i in enumerate(pca_idx):
            all_candidates[global_i] = proj_dists[local_i]

        for idx in graph_neighbors:
            if idx not in all_candidates:
                # Project this graph neighbor into query's tangent space
                diff = self._X_train[idx] - centroid
                proj = V_d.T @ diff
                dist = float(np.linalg.norm(proj - query_proj))
                all_candidates[idx] = dist

        # Step 5: Vote among k_vote nearest in manifold space
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])
        k = min(self.k_vote, len(sorted_candidates))
        top_k = sorted_candidates[:k]

        # Distance-weighted vote
        label_scores = {}
        for idx, dist in top_k:
            label = self._y_train[idx]
            w = 1.0 / (1.0 + dist)
            label_scores[label] = label_scores.get(label, 0.0) + w

        return max(label_scores, key=label_scores.get)

    def _gather_graph_neighbors(self, entry_id: str, max_hops: int = 2) -> list[int]:
        """Walk the graph from entry_id, returning training indices of reachable nodes."""
        visited = set()
        result_indices = []
        frontier = {entry_id}

        for hop in range(max_hops):
            next_frontier = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                geom = self._geometries[node_id]
                result_indices.append(geom.index)

                # Expand via graph edges
                edges = self._graph._edges.get(node_id, [])
                for edge in edges:
                    if edge.target_id not in visited:
                        next_frontier.add(edge.target_id)

            frontier = next_frontier

        return result_indices

    def score(self, X, y):
        """Return classification accuracy.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test data.
        y : array-like, shape (n_samples,)
            True labels.

        Returns
        -------
        accuracy : float
        """
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))

    # ----- Phase 3: Fly -----

    def fly_to(self, node_id: str) -> NodeGeometry:
        """Position the turtle at a graph node and orient to local geometry.

        Parameters
        ----------
        node_id : str
            Target node ID (e.g. "n42").

        Returns
        -------
        NodeGeometry
            Local geometry at the node.
        """
        if self._mode != self.NAVIGATE:
            raise RuntimeError("Must call fit() before flying")
        if node_id not in self._geometries:
            raise KeyError(f"Node {node_id!r} not in graph")

        geom = self._geometries[node_id]
        emb = self._graph.get_embedding(node_id)

        # Position turtle
        self._turtle.position = emb.copy()

        # Orient turtle frame to local PCA basis (pad if truncated)
        self._turtle._frame = self._pad_basis(geom.basis)

        self._current_node = node_id
        self._flight_path.append(node_id)

        return geom

    def fly_to_nearest(self, point) -> NodeGeometry:
        """Fly to the graph node nearest to a given point.

        Parameters
        ----------
        point : array-like
            Query point in ambient space.

        Returns
        -------
        NodeGeometry
            Local geometry at the nearest node.
        """
        point = np.asarray(point, dtype="d")
        dists = np.linalg.norm(self._X_train - point, axis=1)
        nearest_idx = np.argmin(dists)
        return self.fly_to(f"n{nearest_idx}")

    def fly_step(
        self,
        direction: np.ndarray | None = None,
        excluded: set | None = None,
    ) -> str | None:
        """Take one step along the manifold graph.

        Moves to the best neighbor of the current node, choosing the
        one most aligned with the turtle's heading (or a specified direction).

        Parameters
        ----------
        direction : np.ndarray, optional
            Preferred direction in ambient space. If None, uses turtle heading.
        excluded : set of str, optional
            Node IDs to skip when scoring candidates.  Used by
            ``fly_toward`` to avoid revisiting nodes and break cycles.

        Returns
        -------
        str or None
            Node ID moved to, or None if dead end or all neighbours excluded.
        """
        if self._current_node is None:
            raise RuntimeError("Must call fly_to() first")

        if direction is not None:
            heading = np.asarray(direction, dtype="d")
            norm = np.linalg.norm(heading)
            if norm > 1e-10:
                heading = heading / norm
            else:
                heading = self._turtle.heading
        else:
            heading = self._turtle.heading

        current_emb = self._graph.get_embedding(self._current_node)
        edges = self._graph._edges.get(self._current_node, [])

        if not edges:
            return None

        # Score candidates by alignment with heading
        best_score = -np.inf
        best_id = None
        best_emb = None

        for edge in edges:
            if excluded and edge.target_id in excluded:
                continue
            target_emb = self._graph.get_embedding(edge.target_id)
            step_dir = target_emb - current_emb
            norm = np.linalg.norm(step_dir)
            if norm < 1e-10:
                continue
            alignment = float(np.dot(step_dir / norm, heading))
            score = 0.6 * alignment + 0.4 * edge.weight

            if score > best_score:
                best_score = score
                best_id = edge.target_id
                best_emb = target_emb

        if best_id is None:
            return None

        # Move turtle
        self._turtle.position = best_emb.copy()

        # Orient to new node's geometry (pad if truncated)
        geom = self._geometries[best_id]
        self._turtle._frame = self._pad_basis(geom.basis)

        self._current_node = best_id
        self._flight_path.append(best_id)

        return best_id

    def fly_toward(self, target, max_steps: int = 20, patience: int = 5) -> list[str]:
        """Fly toward a target point, following the graph.

        Parameters
        ----------
        target : array-like
            Target point in ambient space.
        max_steps : int
            Maximum number of graph hops.
        patience : int
            Number of consecutive non-improving hops allowed before stopping.
            With sparse graphs (small k) a greedy step occasionally moves
            sideways; patience lets the walker continue past local detours
            rather than terminating at the first non-improving step.

        Returns
        -------
        path : list[str]
            Sequence of node IDs visited.
        """
        target = np.asarray(target, dtype="d")
        path = []
        stall = 0
        visited: set[str] = set()
        if self._current_node is not None:
            visited.add(self._current_node)

        for _ in range(max_steps):
            if self._current_node is None:
                break

            current_emb = self._graph.get_embedding(self._current_node)
            direction = target - current_emb
            dist = np.linalg.norm(direction)

            if dist < 1e-8:
                break

            next_id = self.fly_step(direction, excluded=visited)
            if next_id is None:
                # All neighbours already visited — genuinely stuck
                break

            path.append(next_id)
            visited.add(next_id)

            new_emb = self._graph.get_embedding(next_id)
            new_dist = np.linalg.norm(target - new_emb)
            if new_dist >= dist:
                stall += 1
                if stall >= patience:
                    break
            else:
                stall = 0

        return path

    def get_geometry(self, node_id: str) -> NodeGeometry:
        """Return the local geometry at a node.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        NodeGeometry
        """
        return self._geometries[node_id]

    def get_neighbors(self, node_id: str) -> list[tuple[str, float]]:
        """Return graph neighbors of a node with edge weights.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list of (node_id, weight) tuples, sorted by weight descending.
        """
        edges = self._graph._edges.get(node_id, [])
        neighbors = [(e.target_id, e.weight) for e in edges]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors

    def reset_flight(self) -> None:
        """Reset flight state (current node, path)."""
        self._current_node = None
        self._flight_path = []

    # ----- Utilities -----

    def _pad_basis(self, basis: np.ndarray) -> np.ndarray:
        """Pad a truncated basis (d, ndim) to a full orthonormal frame (ndim, ndim).

        Used when setting the turtle's frame from a truncated PCA basis.
        The first d rows are the PCA directions; remaining rows are
        completed via QR decomposition to form a valid orthonormal basis.
        """
        d, ndim = basis.shape
        if d >= ndim:
            return basis[:ndim].copy()

        # Build full frame: start with the d PCA rows, fill with random
        # vectors, then orthonormalize via QR
        full = np.zeros((ndim, ndim), dtype=basis.dtype)
        full[:d] = basis
        # Random complement (seeded for reproducibility)
        rng = np.random.RandomState(0)
        full[d:] = rng.randn(ndim - d, ndim)
        # QR gives orthonormal rows
        Q, _ = np.linalg.qr(full.T)
        return Q.T

    # ----- Diagnostics -----

    def geometry_summary(self) -> dict:
        """Return summary statistics of the manifold geometry.

        Returns
        -------
        dict with keys:
            mean_intrinsic_dim, std_intrinsic_dim, min_intrinsic_dim,
            max_intrinsic_dim, n_nodes, ambient_dim, n_edges
        """
        dims = [g.intrinsic_dim for g in self._geometries.values()]
        n_edges = sum(len(edges) for edges in self._graph._edges.values())

        return {
            "mean_intrinsic_dim": float(np.mean(dims)),
            "std_intrinsic_dim": float(np.std(dims)),
            "min_intrinsic_dim": int(min(dims)),
            "max_intrinsic_dim": int(max(dims)),
            "n_nodes": len(self._geometries),
            "ambient_dim": self._ndim,
            "n_edges": n_edges,
        }

    # ----- Dunder -----

    def __repr__(self) -> str:
        n = self.n_nodes
        d = self._ndim or "?"
        return f"ManifoldModel(mode={self._mode!r}, nodes={n}, ndim={d})"


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "NodeGeometry",
    "ManifoldModel",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
