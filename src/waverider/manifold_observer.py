"""
ManifoldObserver: An (N+1)-dimensional observer that sees into an N-dimensional manifold.

The Concept
-----------
A :class:`~waverider.turtleND.TurtleND` walking an N-dim manifold is a
*flatland creature* — it can only discover neighbors by moving to them.
If we take its N-dim frame, extend it by one orthonormal dimension, and
embed the resulting (N+1)-dim turtle into a new space, we create an
**extrinsic observer** that can *see* the manifold surface from above.

The extra dimension is the **normal** to the manifold at each point.
From this vantage the observer can see:
  - **Topology**: the shape and connectivity of the surface
  - **Curvature**: how fast the tangent plane rotates from node to node
  - **Boundaries**: where the manifold thins or ends
  - **Class regions**: clusters, ridges, and valleys in the surface
  - **Any point's location**: project straight down to the surface — no search needed

Construction
------------
1. Take the subject's N-dim orthonormal frame at a node.
2. Append a zero column, then add the (N+1)-th basis vector ``[0,…,0,1]``.
3. Orthonormalize → valid (N+1)×(N+1) frame.
4. The observer's (N+1)-th basis vector is **perpendicular to the entire
   N-dim tangent space** — it points "straight up" off the surface.
5. Lift all training data to (N+1)-space by appending a normal-residual
   height coordinate.  Points on the manifold sit at h≈0; points off it
   rise above the surface.

The observer can then **see** the entire manifold as a relief map, and
locate any query point by geometric projection rather than graph search.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers, https://github.com/Flux-Frontiers
License: BSD
Last revised: 2026-03-23 -egs-

"""

from __future__ import annotations

__pdoc__ = {"__all__": True}

from dataclasses import dataclass

import numpy as np

from waverider.manifold_model import ManifoldModel, NodeGeometry
from waverider.turtleND import TurtleND

# ---------------------------------------------------------------------------
# Observer geometry — what the observer sees at each node
# ---------------------------------------------------------------------------


@dataclass
class ObservedGeometry:
    """What the observer sees at a single manifold node.

    Attributes
    ----------
    node_id : str
        Graph node identifier.
    position_lifted : np.ndarray
        Position in (N+1)-space, shape (N+1,).
    normal : np.ndarray
        Unit normal to the manifold at this node, shape (N+1,).
    curvature : float
        Scalar curvature estimate (rate of normal change).
    tangent_spread : float
        Frobenius norm of the tangent basis change to neighbors.
    height : float
        Reconstruction-error "height" above the ideal tangent plane.
    intrinsic_dim : int
        Local intrinsic dimensionality (from subject's PCA).
    """

    node_id: str
    position_lifted: np.ndarray
    normal: np.ndarray
    curvature: float = 0.0
    tangent_spread: float = 0.0
    height: float = 0.0
    intrinsic_dim: int = 0


# ---------------------------------------------------------------------------
# The observer
# ---------------------------------------------------------------------------


class ManifoldObserver:
    """An (N+1)-dimensional turtle that sees into an N-dimensional manifold.

    Wraps a fitted :class:`ManifoldModel` (the *subject*) and constructs
    an observer that operates in (N+1)-dimensional space.  The extra
    dimension is normal to the subject's manifold, giving the observer
    an extrinsic viewpoint from which the manifold's shape, curvature,
    and topology are directly visible.

    Parameters
    ----------
    subject : ManifoldModel
        A fitted ManifoldModel whose manifold we want to observe.
        Must be in ``'navigate'`` mode (i.e. ``fit()`` has been called).

    Examples
    --------
    >>> import numpy as np
    >>> from waverider.manifold_model import ManifoldModel
    >>> from waverider.manifold_observer import ManifoldObserver
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((80, 5))
    >>> model = ManifoldModel(k_graph=8, k_pca=20)
    >>> model.fit(X)  # doctest: +ELLIPSIS
    ManifoldModel(...)
    >>> obs = ManifoldObserver(model)
    >>> obs.ndim
    6
    >>> field = obs.observe()
    >>> len(field) == 80
    True
    >>> field[0].curvature >= 0.0
    True
    """

    def __init__(self, subject: ManifoldModel):
        if subject.mode != ManifoldModel.NAVIGATE:
            raise RuntimeError("Subject ManifoldModel must be fitted (call fit() first)")

        self._subject = subject
        N = subject.ndim
        self._ndim = N + 1  # observer lives in (N+1)-space

        # The observer turtle
        self._observer = TurtleND(self._ndim, name="ManifoldObserver")

        # Altitude above the manifold surface
        self._altitude = 0.0

        # Cached observation data
        self._lifted_data: np.ndarray | None = None  # shape (n, N+1)
        self._observed: dict[str, ObservedGeometry] = {}
        self._curvature_field: np.ndarray | None = None

    # ----- Properties -----

    @property
    def ndim(self) -> int:
        """Observer dimensionality (subject.ndim + 1)."""
        return self._ndim

    @property
    def subject(self) -> ManifoldModel:
        """The N-dim manifold model being observed."""
        return self._subject

    @property
    def observer(self) -> TurtleND:
        """The (N+1)-dim observer turtle."""
        return self._observer

    @property
    def altitude(self) -> float:
        """Observer's height above the manifold surface."""
        return self._altitude

    @property
    def lifted_data(self) -> np.ndarray | None:
        """Training data lifted to (N+1)-space, shape (n, N+1)."""
        return self._lifted_data

    # ----- Core: Lifting -----

    def _lift_point(self, point: np.ndarray, geom: NodeGeometry) -> np.ndarray:
        """Lift an N-dim point to (N+1)-dim using local tangent geometry.

        The (N+1)-th coordinate is the reconstruction error — how far
        the point lies from the local tangent plane.  Points exactly on
        the manifold have h=0; points off it have h>0.

        Parameters
        ----------
        point : np.ndarray
            Point in N-space, shape (N,).
        geom : NodeGeometry
            Local geometry (tangent basis + centroid) to measure against.

        Returns
        -------
        np.ndarray
            Lifted point in (N+1)-space, shape (N+1,).
        """
        diff = point - geom.centroid
        d = geom.intrinsic_dim
        projection = geom.basis[:d] @ diff  # project onto tangent (d,)
        reconstructed = geom.basis[:d].T @ projection  # back to N-space (N,)
        residual = diff - reconstructed
        height = float(np.linalg.norm(residual))

        lifted = np.zeros(self._ndim, dtype="d")
        lifted[: len(point)] = point
        lifted[-1] = height
        return lifted

    def lift_data(self) -> np.ndarray:
        """Lift all training data to (N+1)-space.

        Each point is lifted using the geometry of its nearest graph node.
        The (N+1)-th coordinate encodes reconstruction error (distance
        from the tangent plane).

        Returns
        -------
        np.ndarray
            Lifted data, shape (n_samples, N+1).
        """
        X = self._subject._X_train
        n = len(X)
        lifted = np.zeros((n, self._ndim), dtype="d")

        for i in range(n):
            node_id = f"n{i}"
            geom = self._subject.get_geometry(node_id)
            lifted[i] = self._lift_point(X[i], geom)

        self._lifted_data = lifted
        return lifted

    # ----- Core: Observing -----

    def _compute_curvature(self, node_id: str) -> tuple[float, float]:
        """Estimate curvature at a node from tangent-basis variation.

        Curvature is the rate at which the tangent plane rotates as we
        move along the surface.  Measured as the mean principal-angle
        between this node's tangent basis and its neighbors'.

        Returns
        -------
        tuple of (curvature, tangent_spread)
        """
        geom = self._subject.get_geometry(node_id)
        neighbors = self._subject.get_neighbors(node_id)

        if not neighbors:
            return 0.0, 0.0

        basis_src = geom.basis[: geom.intrinsic_dim]  # (d, N)
        angles = []

        for nbr_id, weight in neighbors:
            nbr_geom = self._subject.get_geometry(nbr_id)
            basis_tgt = nbr_geom.basis[: nbr_geom.intrinsic_dim]

            # Principal angles between the two tangent subspaces via SVD
            # of the cross-Gramian B_src @ B_tgt^T
            # Use errstate to silence Apple Accelerate FPU exceptions on subnormal
            # float32 patterns; sanitise result for degenerate tangent bases.
            d_min = min(basis_src.shape[0], basis_tgt.shape[0])
            b_src64 = np.nan_to_num(basis_src.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            b_tgt64 = np.nan_to_num(basis_tgt.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                cross = b_src64 @ b_tgt64.T  # (d_src, d_tgt)
            cross = np.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
            svs = np.linalg.svd(cross, compute_uv=False)
            # Singular values are cosines of principal angles
            svs = np.clip(svs[:d_min], -1.0, 1.0)
            principal_angles = np.arccos(svs)
            angles.append(float(np.mean(principal_angles)))

        curvature = float(np.mean(angles)) if angles else 0.0
        tangent_spread = float(np.std(angles)) if len(angles) > 1 else 0.0
        return curvature, tangent_spread

    def observe(self) -> list[ObservedGeometry]:
        """Observe the entire manifold from above.

        Lifts all data, computes curvature at every node, and builds
        the complete observation field.  This is the core operation:
        the observer *sees* the manifold in a single pass rather than
        searching node by node.

        Returns
        -------
        list of ObservedGeometry
            One entry per graph node, containing the lifted position,
            normal vector, curvature, and height.
        """
        # Step 1: Lift all data
        self.lift_data()

        X = self._subject._X_train
        n = len(X)
        curvatures = np.zeros(n, dtype="d")
        observations: list[ObservedGeometry] = []

        for i in range(n):
            node_id = f"n{i}"
            geom = self._subject.get_geometry(node_id)

            # Curvature from tangent-basis variation
            curvature, tangent_spread = self._compute_curvature(node_id)
            curvatures[i] = curvature

            # Normal vector in (N+1)-space: the last basis vector of the
            # observer's extended frame
            normal = np.zeros(self._ndim, dtype="d")
            normal[-1] = 1.0  # points "up" off the manifold

            obs = ObservedGeometry(
                node_id=node_id,
                position_lifted=self._lifted_data[i].copy(),
                normal=normal,
                curvature=curvature,
                tangent_spread=tangent_spread,
                height=float(self._lifted_data[i, -1]),
                intrinsic_dim=geom.intrinsic_dim,
            )
            observations.append(obs)
            self._observed[node_id] = obs

        self._curvature_field = curvatures
        return observations

    # ----- Core: Seeing without searching -----

    def locate(self, query: np.ndarray) -> dict:
        """Locate a query point on the manifold by direct projection.

        Instead of walking the graph (like the subject must), the
        observer projects the query straight down onto the manifold
        surface.  This is O(N) distance computation, not O(hops)
        graph traversal.

        Parameters
        ----------
        query : np.ndarray
            Point in N-space, shape (N,).

        Returns
        -------
        dict with keys:
            nearest_node : str
                Closest graph node ID.
            distance : float
                Euclidean distance to nearest node.
            height : float
                Reconstruction error (how far off the manifold).
            curvature : float
                Local curvature at the nearest node.
            label : optional
                Class label at the nearest node (if supervised).
            tangent_projection : np.ndarray
                Query projected into the local tangent space.
        """
        query = np.asarray(query, dtype="d")
        X = self._subject._X_train

        # Direct geometric lookup — no graph walking
        dists = np.linalg.norm(X - query, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_id = f"n{nearest_idx}"
        nearest_dist = float(dists[nearest_idx])

        geom = self._subject.get_geometry(nearest_id)

        # Compute height (reconstruction error from tangent plane)
        diff = np.nan_to_num(
            (query - geom.centroid).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        d = geom.intrinsic_dim
        basis_d = np.nan_to_num(geom.basis[:d].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tangent_proj = basis_d @ diff  # (d,)
            reconstructed = basis_d.T @ tangent_proj
        tangent_proj = np.nan_to_num(tangent_proj, nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        residual = diff - reconstructed
        height = float(np.linalg.norm(residual))

        # Curvature from cached observation
        curvature = 0.0
        if nearest_id in self._observed:
            curvature = self._observed[nearest_id].curvature

        label = None
        if self._subject._y_train is not None:
            label = self._subject._y_train[nearest_idx]

        return {
            "nearest_node": nearest_id,
            "distance": nearest_dist,
            "height": height,
            "curvature": curvature,
            "label": label,
            "tangent_projection": tangent_proj,
        }

    def locate_batch(self, X: np.ndarray) -> list[dict]:
        """Locate multiple points by direct projection.

        Parameters
        ----------
        X : np.ndarray
            Points in N-space, shape (n, N).

        Returns
        -------
        list of dict
            One :meth:`locate` result per query point.
        """
        X = np.asarray(X, dtype="d")
        return [self.locate(x) for x in X]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify by observation — no graph walking needed.

        The observer projects each query straight down to the manifold
        surface and reads the label at the landing point.  This is a
        direct geometric computation, not a search.

        Parameters
        ----------
        X : np.ndarray
            Query points, shape (n, N).

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if self._subject._y_train is None:
            raise RuntimeError("Subject model has no labels")

        X = np.asarray(X, dtype="d")
        predictions = np.empty(len(X), dtype=self._subject._y_train.dtype)

        for i, query in enumerate(X):
            result = self.locate(query)
            predictions[i] = result["label"]

        return predictions

    # ----- Observer movement -----

    def sync(self, node_id: str | None = None) -> None:
        """Synchronize the observer's frame with the subject's manifold.

        Extends the subject's N-dim frame at the given node (or the
        subject's current node) to an (N+1)-dim observer frame where
        the extra basis vector is the manifold normal.

        Parameters
        ----------
        node_id : str, optional
            Node to sync at.  If None, uses the subject's current node.
        """
        if node_id is None:
            node_id = self._subject.current_node
        if node_id is None:
            raise RuntimeError("No node to sync at — fly the subject first or specify node_id")

        N = self._subject.ndim
        geom = self._subject.get_geometry(node_id)
        emb = self._subject.graph.get_embedding(node_id)

        # Build the subject's full N×N frame (padded if truncated)
        subject_frame = self._subject._pad_basis(geom.basis)  # (N, N)

        # Extend to (N+1) × (N+1): pad each row with 0, add normal row
        frame_ext = np.zeros((self._ndim, self._ndim), dtype="d")
        frame_ext[:N, :N] = subject_frame
        frame_ext[N, N] = 1.0  # the normal — perpendicular to all of R^N

        # Position in (N+1)-space: subject position + altitude along normal
        pos_ext = np.zeros(self._ndim, dtype="d")
        pos_ext[:N] = emb
        pos_ext[N] = self._altitude

        self._observer._position = pos_ext
        self._observer._frame = frame_ext
        self._observer.orthonormalize()

    def lift_off(self, height: float) -> None:
        """Rise above (or descend toward) the manifold surface.

        Parameters
        ----------
        height : float
            Distance to move in the normal direction.
            Positive = rise above, negative = descend.
        """
        self._altitude += height
        # Move observer along its last basis vector (the normal)
        self._observer._position[-1] = self._altitude

    def look_down(self) -> np.ndarray:
        """Project the observer's position straight down to the manifold.

        Returns the N-dim point on the manifold surface directly below
        the observer.

        Returns
        -------
        np.ndarray
            Point in N-space, shape (N,).
        """
        return self._observer.position[: self._subject.ndim].copy()

    def pan(self, direction: np.ndarray, distance: float) -> None:
        """Move the observer laterally (parallel to the manifold surface).

        Parameters
        ----------
        direction : np.ndarray
            Direction in N-space to pan, shape (N,).
        distance : float
            Distance to move.
        """
        direction = np.asarray(direction, dtype="d")
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return
        direction = direction / norm

        # Move in (N+1)-space but only in the first N components
        move_vec = np.zeros(self._ndim, dtype="d")
        move_vec[: self._subject.ndim] = direction * distance
        self._observer._position += move_vec

    def orbit(self, angle: float, axis_i: int = 0) -> None:
        """Orbit around the look-down point.

        Rotates the observer in the plane defined by basis[axis_i] and
        the normal (last basis vector).

        Parameters
        ----------
        angle : float
            Orbit angle in degrees.
        axis_i : int
            Which tangent basis vector defines the orbit plane (default 0 = heading).
        """
        self._observer.rotate(angle, axis_i, self._ndim - 1)

    # ----- Curvature and topology -----

    def curvature_at(self, node_id: str) -> float:
        """Return the curvature at a specific node.

        If :meth:`observe` has been called, returns the cached value.
        Otherwise computes it on the fly.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        float
            Mean principal angle (radians) between the node's tangent
            space and its neighbors'.
        """
        if node_id in self._observed:
            return self._observed[node_id].curvature
        curvature, _ = self._compute_curvature(node_id)
        return curvature

    def curvature_field(self) -> np.ndarray:
        """Return curvature values at all nodes.

        Requires :meth:`observe` to have been called.

        Returns
        -------
        np.ndarray
            Curvature at each node, shape (n_nodes,).
        """
        if self._curvature_field is None:
            self.observe()
        return self._curvature_field.copy()

    def topology_summary(self) -> dict:
        """Summarize what the observer sees about the manifold.

        Returns
        -------
        dict with keys:
            n_nodes, ambient_dim, observer_dim, mean_curvature,
            max_curvature, mean_height, max_height,
            mean_intrinsic_dim, curvature_std, high_curvature_nodes
        """
        if not self._observed:
            self.observe()

        curvatures = [o.curvature for o in self._observed.values()]
        heights = [o.height for o in self._observed.values()]
        dims = [o.intrinsic_dim for o in self._observed.values()]

        mean_curv = float(np.mean(curvatures))
        std_curv = float(np.std(curvatures))
        threshold = mean_curv + 2.0 * std_curv

        high_curv_nodes = [o.node_id for o in self._observed.values() if o.curvature > threshold]

        return {
            "n_nodes": len(self._observed),
            "ambient_dim": self._subject.ndim,
            "observer_dim": self._ndim,
            "mean_curvature": mean_curv,
            "max_curvature": float(np.max(curvatures)) if curvatures else 0.0,
            "curvature_std": std_curv,
            "mean_height": float(np.mean(heights)) if heights else 0.0,
            "max_height": float(np.max(heights)) if heights else 0.0,
            "mean_intrinsic_dim": float(np.mean(dims)) if dims else 0.0,
            "high_curvature_nodes": high_curv_nodes,
        }

    # ----- Path tracing (pen-down view from above) -----

    def observe_path(
        self,
        path: list[str],
        pen_down: bool = True,
    ) -> dict:
        """See a walker's traced path from the observer's vantage above the surface.

        The classical turtle can drop a pen and draw on the plane.  In the
        manifold the walker traces a sequence of N-dim positions.  The observer,
        sitting one dimension higher, sees the complete trajectory as a relief
        curve: each hop is lifted to (N+1)-space, with the extra coordinate
        encoding height above the local tangent plane.

        This is the "pen-down" view: what Flatland's Sphere sees looking down
        at the curve the Flatlander drew while walking.

        Parameters
        ----------
        path : list of str
            Ordered list of node IDs visited by the walker (e.g.
            ``ManifoldModel.flight_path``).
        pen_down : bool
            If True (default), compute and return the full lifted trajectory.
            If False, returns only the per-hop metadata without lifting.

        Returns
        -------
        dict with keys:

        ``positions_N`` : np.ndarray, shape (n_hops, N)
            Walker positions in N-space (original manifold coordinates).
        ``positions_lifted`` : np.ndarray, shape (n_hops, N+1)
            Walker positions in (N+1)-space as seen by the observer.
            The last column is the reconstruction-error height at each hop.
        ``heights`` : np.ndarray, shape (n_hops,)
            Observer height coordinate at each hop (= reconstruction error
            from the local tangent plane).  Zero = on the manifold; positive =
            off the manifold surface (noise / curvature gap).
        ``curvatures`` : np.ndarray, shape (n_hops,)
            Curvature at each hop node (mean principal angle to neighbors).
        ``intrinsic_dims`` : np.ndarray of int, shape (n_hops,)
            Local intrinsic dimensionality at each hop.
        ``class_labels`` : list
            Class label at each hop node (None if model is unsupervised).
        ``path`` : list of str
            Input path, echoed back.
        ``n_hops`` : int
            Number of hops.
        ``boundary_crossings`` : list of int
            Hop indices where the class label changes.

        Examples
        --------
        >>> import numpy as np
        >>> from waverider.manifold_model import ManifoldModel
        >>> from waverider.manifold_observer import ManifoldObserver
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((60, 4))
        >>> y = (X[:, 0] > 0).astype(int)
        >>> model = ManifoldModel(k_graph=6, k_pca=15)
        >>> _ = model.fit(X, y)
        >>> _ = model.fly_to("n0")
        >>> path = model.fly_toward(X[30], max_steps=10)
        >>> obs = ManifoldObserver(model)
        >>> result = obs.observe_path(path)
        >>> result["n_hops"] == len(path)
        True
        >>> result["positions_lifted"].shape[1] == 5  # N+1 = 4+1
        True
        """
        if not path:
            return {
                "positions_N": np.empty((0, self._subject.ndim)),
                "positions_lifted": np.empty((0, self._ndim)),
                "heights": np.array([]),
                "curvatures": np.array([]),
                "intrinsic_dims": np.array([], dtype=int),
                "class_labels": [],
                "path": path,
                "n_hops": 0,
                "boundary_crossings": [],
            }

        N = self._subject.ndim
        n_hops = len(path)
        pos_N = np.zeros((n_hops, N), dtype="d")
        pos_lifted = np.zeros((n_hops, self._ndim), dtype="d")
        heights = np.zeros(n_hops, dtype="d")
        curvatures = np.zeros(n_hops, dtype="d")
        intrinsic_dims = np.zeros(n_hops, dtype=int)
        labels: list = []

        for i, node_id in enumerate(path):
            # Raw position in N-space
            emb = self._subject.graph.get_embedding(node_id)
            pos_N[i] = emb

            # Geometry at this node
            geom = self._subject.get_geometry(node_id)
            intrinsic_dims[i] = geom.intrinsic_dim

            # Curvature — use cached value if observe() was called, else compute
            if node_id in self._observed:
                curv = self._observed[node_id].curvature
            else:
                curv, _ = self._compute_curvature(node_id)
            curvatures[i] = curv

            # Lift to (N+1)-space
            if pen_down:
                lifted = self._lift_point(emb, geom)
            else:
                lifted = np.zeros(self._ndim, dtype="d")
                lifted[:N] = emb
            pos_lifted[i] = lifted
            heights[i] = float(lifted[-1])

            # Class label
            idx = int(node_id[1:])
            if self._subject._y_train is not None and idx < len(self._subject._y_train):
                labels.append(self._subject._y_train[idx])
            else:
                labels.append(None)

        # Boundary crossings — hops where class changes
        crossings = []
        for i in range(1, n_hops):
            if labels[i] is not None and labels[i - 1] is not None:
                if labels[i] != labels[i - 1]:
                    crossings.append(i)

        return {
            "positions_N": pos_N,
            "positions_lifted": pos_lifted,
            "heights": heights,
            "curvatures": curvatures,
            "intrinsic_dims": intrinsic_dims,
            "class_labels": labels,
            "path": path,
            "n_hops": n_hops,
            "boundary_crossings": crossings,
        }

    # ----- Dunder -----

    def __repr__(self) -> str:
        n = len(self._observed)
        N = self._subject.ndim
        return (
            f"ManifoldObserver(subject={N}D, observer={self._ndim}D, "
            f"observed={n} nodes, altitude={self._altitude:.2f})"
        )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ObservedGeometry",
    "ManifoldObserver",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
