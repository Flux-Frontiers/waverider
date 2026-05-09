"""
ManifoldWalker: Navigate embedding spaces along data manifolds using TurtleND.

Uses local PCA at each position to discover the tangent space of the data
manifold, then steers the turtle along manifold-aligned directions. This
replaces isotropic gradient descent with Riemannian-aware navigation that
respects the actual geometry of the embedding space.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers, https://github.com/Flux-Frontiers
License: BSD
Last revised: 2026-03-23 -egs-
"""

__pdoc__ = {"__all__": True}

from collections.abc import Callable

import numpy as np

from waverider.turtleND import TurtleND


class ManifoldWalker:
    """Navigate an N-dimensional embedding space along its data manifold.

    The walker maintains a TurtleND whose frame is continuously re-aligned
    to the local principal directions of the data via PCA on a neighborhood.
    Gradients are projected into this local frame, and components along
    low-variance (off-manifold) directions are suppressed.

    Parameters
    ----------
    embeddings : np.ndarray
        The embedding matrix of shape (n_points, ndim).
    objective : callable
        A function f(position) -> scalar to minimize. The walker computes
        numerical gradients of this function.
    k : int
        Number of nearest neighbors for local PCA (default 50).
    variance_threshold : float
        Fraction of cumulative variance to retain (default 0.95).
        Directions beyond this threshold are considered off-manifold noise.
    learning_rate : float
        Step size for each move (default 0.01).
    epsilon : float
        Perturbation size for numerical gradient estimation (default 1e-5).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        objective: Callable[[np.ndarray], float],
        k: int = 50,
        variance_threshold: float = 0.95,
        learning_rate: float = 0.01,
        epsilon: float = 1e-5,
    ):
        self._embeddings = np.asarray(embeddings, dtype="d")
        self._n_points, self._ndim = self._embeddings.shape
        self._objective = objective
        self._k = min(k, self._n_points - 1) if self._n_points > 1 else 1
        self._variance_threshold = variance_threshold
        self._lr = learning_rate
        self._epsilon = epsilon

        self._turtle = TurtleND(self._ndim, name="ManifoldWalker")

        # Diagnostics updated at each step
        self._eigenvalues = None
        self._intrinsic_dim = None
        self._history: list[tuple[np.ndarray, float]] = []

    # ----- Properties -----

    @property
    def turtle(self) -> TurtleND:
        """The underlying TurtleND."""
        return self._turtle

    @property
    def position(self) -> np.ndarray:
        return self._turtle.position

    @position.setter
    def position(self, pos):
        self._turtle.position = np.asarray(pos, dtype="d")

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def eigenvalues(self) -> np.ndarray | None:
        """Eigenvalues from the most recent local PCA (descending order)."""
        return self._eigenvalues

    @property
    def intrinsic_dim(self) -> int | None:
        """Estimated local intrinsic dimensionality from most recent PCA."""
        return self._intrinsic_dim

    @property
    def history(self) -> list:
        """List of (position, objective_value) tuples from each step."""
        return list(self._history)

    @property
    def learning_rate(self) -> float:
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._lr = value

    @property
    def variance_threshold(self) -> float:
        return self._variance_threshold

    @variance_threshold.setter
    def variance_threshold(self, value: float):
        self._variance_threshold = value

    # ----- Core algorithms -----

    def _knn(self, point: np.ndarray) -> np.ndarray:
        """Return the k nearest neighbors to point from the embedding matrix."""
        diffs = self._embeddings - point
        dists = np.linalg.norm(diffs, axis=1)
        indices = np.argpartition(dists, self._k)[: self._k]
        return self._embeddings[indices]

    def _local_pca(self, point: np.ndarray):
        """
        Compute local PCA around point.

        Returns
        -------
        eigenvalues : np.ndarray
            Eigenvalues in descending order.
        eigenvectors : np.ndarray
            Corresponding eigenvectors as columns, reordered descending.
        intrinsic_dim : int
            Number of dimensions capturing variance_threshold of variance.
        """
        neighbors = self._knn(point)
        centered = neighbors - neighbors.mean(axis=0)
        cov = (centered.T @ centered) / (len(neighbors) - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order; flip to descending
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Determine intrinsic dimensionality from cumulative variance
        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            intrinsic_dim = int(np.searchsorted(cumulative, self._variance_threshold) + 1)
        else:
            intrinsic_dim = self._ndim

        return eigenvalues, eigenvectors, intrinsic_dim

    def _numerical_gradient(self, point: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective via central differences."""
        grad = np.zeros(self._ndim, dtype="d")
        for i in range(self._ndim):
            perturb = np.zeros(self._ndim, dtype="d")
            perturb[i] = self._epsilon
            grad[i] = (self._objective(point + perturb) - self._objective(point - perturb)) / (
                2 * self._epsilon
            )
        return grad

    def orient(self) -> int:
        """
        Re-orient the turtle's frame to the local principal directions.

        Computes local PCA around the current position and aligns the
        turtle's basis vectors to the eigenvectors (descending eigenvalue).

        Returns
        -------
        intrinsic_dim : int
            The estimated local intrinsic dimensionality.
        """
        eigenvalues, eigenvectors, intrinsic_dim = self._local_pca(self._turtle._position)
        self._eigenvalues = eigenvalues
        self._intrinsic_dim = intrinsic_dim

        # Set the turtle's frame: rows are basis vectors = eigenvector columns transposed
        self._turtle._frame = eigenvectors.T.copy()

        return intrinsic_dim

    def step(self, gradient: np.ndarray | None = None) -> float:
        """
        Take one manifold-aware step.

        1. Compute local PCA and orient the turtle
        2. Compute or accept the objective gradient
        3. Project gradient into the local frame
        4. Zero out off-manifold components
        5. Step along the remaining manifold-aligned directions

        Parameters
        ----------
        gradient : np.ndarray, optional
            Pre-computed gradient in global coordinates. If None, the
            numerical gradient of the objective is computed.

        Returns
        -------
        objective_value : float
            The objective value at the new position.
        """
        pos = self._turtle._position.copy()

        # 1. Orient to local manifold
        intrinsic_dim = self.orient()

        # 2. Get gradient
        if gradient is None:
            raw_grad = self._numerical_gradient(pos)
        else:
            raw_grad = np.asarray(gradient, dtype="d")

        # 3. Project gradient into local (PCA-aligned) frame
        local_grad = self._turtle._frame @ raw_grad

        # 4. Zero out off-manifold components
        local_grad[intrinsic_dim:] = 0.0

        # 5. Weight by eigenvalue magnitude (natural gradient-style)
        # Directions with more variance get proportionally more trust
        if self._eigenvalues is not None and self._eigenvalues[0] > 0:
            weights = np.zeros(self._ndim, dtype="d")
            weights[:intrinsic_dim] = self._eigenvalues[:intrinsic_dim] / self._eigenvalues[0]
            local_grad *= weights

        # 6. Convert back to global and step (negative = descent)
        global_step = self._turtle._frame.T @ local_grad
        self._turtle._position = pos - self._lr * global_step

        obj_val = self._objective(self._turtle._position)
        self._history.append((self._turtle._position.copy(), obj_val))

        return obj_val

    def walk(
        self,
        n_steps: int = 100,
        gradient_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Walk along the manifold for n_steps, minimizing the objective.

        Parameters
        ----------
        n_steps : int
            Maximum number of steps.
        gradient_fn : callable, optional
            A function position -> gradient. If provided, used instead of
            numerical differentiation. Use this when you have analytic
            gradients (e.g. from autograd / torch).
        tol : float
            Stop if the objective changes by less than tol between steps.
        verbose : bool
            Print progress every 10 steps.

        Returns
        -------
        final_position : np.ndarray
            The position after walking.
        """
        prev_val = self._objective(self._turtle._position)

        for i in range(n_steps):
            grad = gradient_fn(self._turtle._position) if gradient_fn else None
            val = self.step(gradient=grad)

            if verbose and i % 10 == 0:
                print(
                    f"Step {i:4d}: obj={val:.6e}  intrinsic_dim={self._intrinsic_dim}/{self._ndim}"
                )

            if abs(val - prev_val) < tol:
                if verbose:
                    print(f"Converged at step {i} (delta={abs(val - prev_val):.2e})")
                break
            prev_val = val

        return self._turtle._position.copy()

    def probe(self, direction_index: int, distances: np.ndarray) -> np.ndarray:
        """
        Probe the objective along a specific basis direction.

        Useful for understanding the loss landscape along individual
        principal components without moving the turtle.

        Parameters
        ----------
        direction_index : int
            Which basis vector to probe along (0 = highest variance).
        distances : np.ndarray
            Array of signed distances to evaluate at.

        Returns
        -------
        values : np.ndarray
            Objective values at each probed point.
        """
        pos = self._turtle._position.copy()
        direction = self._turtle._frame[direction_index]
        values = np.array(
            [self._objective(pos + d * direction) for d in distances],
            dtype="d",
        )
        return values

    def __repr__(self):
        id_str = f"intrinsic_dim={self._intrinsic_dim}" if self._intrinsic_dim else "not oriented"
        return (
            f"<ManifoldWalker: {self._ndim}D, {self._n_points} embeddings, "
            f"k={self._k}, lr={self._lr}, {id_str}, "
            f"{len(self._history)} steps taken>"
        )


class ManifoldAdamWalker(ManifoldWalker):
    """ManifoldWalker with Adam-style momentum and adaptive LR in the projected subspace.

    The key insight: Adam's machinery (momentum + per-dimension adaptive LR)
    operates entirely within the d-dimensional manifold tangent space, not
    in the full N-dimensional ambient space.  Off-manifold gradient components
    are suppressed *before* the adaptive update, so momentum never accumulates
    noise and the adaptive denominator only tracks signal variance.

    This inverts the eigenvalue weighting of the base ManifoldWalker:
    instead of trusting high-variance directions more, the Adam denominator
    naturally takes smaller steps in steep (high-gradient-variance) directions
    and larger steps in flat directions — which is correct for optimization.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix (n_points, ndim).
    objective : callable
        f(position) -> scalar to minimize.
    k : int
        Neighborhood size for local PCA (default 50).
    variance_threshold : float
        Cumulative variance fraction for intrinsic-dim cutoff (default 0.95).
    learning_rate : float
        Step size (default 0.01).
    epsilon : float
        Perturbation for numerical gradients (default 1e-5).
    beta1 : float
        Exponential decay for first moment (default 0.9).
    beta2 : float
        Exponential decay for second moment (default 0.999).
    adam_eps : float
        Denominator epsilon for numerical stability (default 1e-7).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        objective: Callable[[np.ndarray], float],
        k: int = 50,
        variance_threshold: float = 0.95,
        learning_rate: float = 0.01,
        epsilon: float = 1e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-7,
    ):
        super().__init__(
            embeddings=embeddings,
            objective=objective,
            k=k,
            variance_threshold=variance_threshold,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )
        self._beta1 = beta1
        self._beta2 = beta2
        self._adam_eps = adam_eps

        # Adam state — lives in full N-dim space but only d dims are active.
        # Accumulated in *global* coordinates so the state persists across
        # frame re-orientations.
        self._m = np.zeros(self._ndim, dtype="d")  # first moment
        self._v = np.zeros(self._ndim, dtype="d")  # second moment
        self._t = 0

    def step(self, gradient: np.ndarray | None = None) -> float:
        """Take one manifold-aware Adam step.

        1. Orient turtle to local manifold via PCA
        2. Project gradient into local frame, zero off-manifold components
        3. Map projected gradient back to global coordinates
        4. Run Adam update on the manifold-projected gradient
        5. Step

        Parameters
        ----------
        gradient : np.ndarray, optional
            Pre-computed gradient in global coordinates.

        Returns
        -------
        objective_value : float
            Objective at the new position.
        """
        pos = self._turtle._position.copy()

        # 1. Orient to local manifold
        intrinsic_dim = self.orient()

        # 2. Get gradient
        if gradient is None:
            raw_grad = self._numerical_gradient(pos)
        else:
            raw_grad = np.asarray(gradient, dtype="d")

        # 3. Project into local frame and suppress off-manifold
        local_grad = self._turtle._frame @ raw_grad
        local_grad[intrinsic_dim:] = 0.0
        # No eigenvalue weighting — let Adam handle step-size adaptation

        # 4. Map back to global coordinates (manifold-projected gradient)
        projected_grad = self._turtle._frame.T @ local_grad

        # 5. Adam update on the projected gradient
        self._t += 1
        self._m = self._beta1 * self._m + (1 - self._beta1) * projected_grad
        self._v = self._beta2 * self._v + (1 - self._beta2) * (projected_grad**2)

        m_hat = self._m / (1 - self._beta1**self._t)
        v_hat = self._v / (1 - self._beta2**self._t)

        adam_step = m_hat / (np.sqrt(v_hat) + self._adam_eps)

        # 6. Descend
        self._turtle._position = pos - self._lr * adam_step

        obj_val = self._objective(self._turtle._position)
        self._history.append((self._turtle._position.copy(), obj_val))

        return obj_val

    def __repr__(self):
        id_str = f"intrinsic_dim={self._intrinsic_dim}" if self._intrinsic_dim else "not oriented"
        return (
            f"<ManifoldAdamWalker: {self._ndim}D, {self._n_points} embeddings, "
            f"k={self._k}, lr={self._lr}, {id_str}, "
            f"t={self._t}, {len(self._history)} steps taken>"
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
