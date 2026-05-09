#!/usr/bin/env python3
"""
Iris Benchmark: Adam vs ManifoldWalker vs ManifoldAdam in Weight Space
======================================================================

Compares three optimizers on an identical MLP (4→16→8→3 softmax, ~200 params)
trained on the Iris classification task (150 samples, 4 features, 3 classes,
80/20 stratified split).  Supports Apple Metal GPU via tensorflow-metal.

The key insight
---------------
A neural network's loss landscape lives in R^P (P = parameter count), but its
effective dimensionality is much lower — most gradient directions are dominated
by mini-batch sampling noise.  By collecting gradients from N different
mini-batches and treating each as a point in weight space, local PCA reveals
the "active subspace": the directions where the loss actually varies.

Three methods (--trials independent runs, default 10)
------------------------------------------------------
  Adam (canonical):
    Standard TensorFlow/Keras Adam on the full gradient.

  ManifoldWalker (raw):
    At each step, samples --n-samples gradients from different mini-batches,
    runs local PCA to discover the active subspace (τ=--variance-threshold),
    projects the full gradient onto that subspace, and takes a steepest-descent
    step.  Neighborhood is resampled every --resample-interval epochs.
    --steps-per-epoch ManifoldWalker steps are taken per epoch.
    A trajectory buffer of recent weight snapshots provides a second source of
    geometry (the optimization path itself).

  ManifoldAdam (hybrid):
    Same active-subspace projection as ManifoldWalker, but replaces the vanilla
    step with Adam momentum + adaptive learning rate running entirely within the
    discovered subspace.  Off-subspace gradient components are suppressed before
    the Adam update accumulates any momentum.

Results are saved to ``benchmarks/canonical_tests/iris_benchmark_results.json``
and an optional four-panel matplotlib figure is written alongside it.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/iris_adam_vs_manifold.py [--epochs 200] [--trials 10] [--plot]
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# TensorFlow setup — detect Metal / GPU before anything else
# ---------------------------------------------------------------------------


def _setup_tensorflow():
    """Import and configure TensorFlow, forced to CPU."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Force CPU — Metal GPU per-op sync overhead dominates for small MLPs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    device_info = {
        "tensorflow_version": tf.__version__,
        "device_used": "CPU (forced)",
    }

    print(f"TensorFlow {tf.__version__} | Device: {device_info['device_used']}")

    return tf, device_info


tf, DEVICE_INFO = _setup_tensorflow()

# Now safe to import tf-dependent modules
import keras  # noqa: E402

from waverider.manifold_walker import ManifoldWalker  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_iris():
    """Load Iris dataset, normalize features, one-hot encode labels.

    Returns (X_train, y_train, X_test, y_test) with a 80/20 split.
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X, y = data.data.astype("float32"), data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode
    y_onehot = np.zeros((len(y), 3), dtype="float32")
    y_onehot[np.arange(len(y)), y] = 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model factory — identical architecture for both optimizers
# ---------------------------------------------------------------------------


def build_model(hidden_units=(16, 8), activation="relu", learning_rate=0.01):
    """Build a small MLP for Iris classification.

    Architecture: 4 -> 16 -> 8 -> 3 (softmax)
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(hidden_units[0], activation=activation),
            keras.layers.Dense(hidden_units[1], activation=activation),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model):
    """Total trainable parameters."""
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Stores results from a single training run."""

    method: str
    trial: int
    epochs: int
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    test_loss: float = 0.0
    test_acc: float = 0.0
    wall_time: float = 0.0
    convergence_epoch: int | None = None  # epoch where acc first >= 0.95


# ---------------------------------------------------------------------------
# Benchmark 1: Adam (canonical baseline)
# ---------------------------------------------------------------------------


def benchmark_adam(
    X_train, y_train, X_test, y_test, epochs=200, lr=0.01, trial=0
) -> BenchmarkResult:
    """Train with standard Adam optimizer."""
    model = build_model(learning_rate=lr)
    result = BenchmarkResult(method="Adam", trial=trial, epochs=epochs)

    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        validation_data=(X_test, y_test),
    )
    result.wall_time = time.perf_counter() - t0

    result.train_loss = history.history["loss"]
    result.train_acc = history.history["accuracy"]
    result.test_loss, result.test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Find convergence epoch
    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.95:
            result.convergence_epoch = i
            break

    return result


# ---------------------------------------------------------------------------
# Benchmark 2: ManifoldWalker optimizer
# ---------------------------------------------------------------------------


class ManifoldOptimizer:
    """Use ManifoldWalker to optimize neural network weights.

    The key insight: the loss landscape of a neural network, while living in
    a high-dimensional weight space (R^P), has a much lower intrinsic
    dimensionality. Most gradient directions are dominated by noise from
    mini-batch sampling; only a small subspace captures true loss variation.

    We discover this subspace by computing gradients on DIFFERENT mini-batches
    and treating each gradient vector as a data point in R^P. PCA of these
    gradient samples reveals the "active subspace" — the directions where the
    loss actually changes. The ManifoldWalker then:

    1. Projects the full gradient onto this active subspace
    2. Suppresses off-subspace components (mini-batch noise)
    3. Weights on-subspace components by eigenvalue (natural gradient)

    This is mathematically equivalent to a low-rank natural gradient method
    but discovered nonparametrically via the manifold walker framework.

    Additionally, we maintain a trajectory buffer of recent weight snapshots,
    providing a second source of manifold geometry: the optimization path
    itself traces out a curve on the loss surface.
    """

    def __init__(
        self,
        model,
        X_train,
        y_train,
        k=30,
        variance_threshold=0.95,
        learning_rate=0.01,
        n_gradient_samples=40,
        mini_batch_size=16,
        trajectory_len=60,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.variance_threshold = variance_threshold
        self.lr = learning_rate
        self.n_gradient_samples = n_gradient_samples
        self.mini_batch_size = mini_batch_size

        # Flatten all weights into a single vector
        self._weight_shapes = [w.shape for w in model.trainable_weights]
        self._n_params = sum(int(np.prod(s)) for s in self._weight_shapes)

        # Trajectory buffer — stores recent weight snapshots
        self._trajectory = []
        self._trajectory_maxlen = trajectory_len

    def _get_flat_weights(self) -> np.ndarray:
        """Extract all model weights as a flat vector."""
        return np.concatenate([w.numpy().ravel() for w in self.model.trainable_weights])

    def _set_flat_weights(self, flat: np.ndarray):
        """Set model weights from a flat vector."""
        offset = 0
        new_weights = []
        for shape in self._weight_shapes:
            size = int(np.prod(shape))
            new_weights.append(flat[offset : offset + size].reshape(shape))
            offset += size
        self.model.set_weights([w.astype("float32") for w in new_weights])

    def _compute_loss(self, flat_weights: np.ndarray) -> float:
        """Compute training loss for a given weight configuration."""
        self._set_flat_weights(flat_weights)
        loss = self.model.evaluate(
            self.X_train,
            self.y_train,
            verbose=0,
            batch_size=len(self.X_train),
        )
        return float(loss[0])

    def _compute_gradient_full(self, flat_weights: np.ndarray) -> np.ndarray:
        """Compute full-batch gradient via TensorFlow's GradientTape."""
        self._set_flat_weights(flat_weights)
        with tf.GradientTape() as tape:
            predictions = self.model(self.X_train, training=True)
            loss = keras.losses.categorical_crossentropy(self.y_train, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _compute_gradient_minibatch(self, flat_weights, X_batch, y_batch):
        """Compute gradient on a single mini-batch."""
        self._set_flat_weights(flat_weights)
        with tf.GradientTape() as tape:
            predictions = self.model(X_batch, training=True)
            loss = keras.losses.categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _sample_gradient_neighborhood(self, center: np.ndarray) -> np.ndarray:
        """Build the embedding matrix from diverse gradient samples.

        Each row is a gradient vector computed on a different random
        mini-batch. PCA of this matrix reveals which directions in weight
        space actually carry signal (the active subspace) vs. which are
        just mini-batch noise.

        We also mix in trajectory points (recent weight snapshots) so the
        PCA captures the geometry of the optimization path itself.
        """
        n = self.n_gradient_samples
        n_train = len(self.X_train)
        bs = min(self.mini_batch_size, n_train)

        # Gradient samples from random mini-batches
        grad_samples = []
        for _ in range(n):
            idx = np.random.choice(n_train, size=bs, replace=False)
            g = self._compute_gradient_minibatch(center, self.X_train[idx], self.y_train[idx])
            # Represent as a point in weight space: center + scaled gradient
            # The scale makes the gradient samples comparable to trajectory spread
            grad_samples.append(center + 0.01 * g)

        embeddings = np.array(grad_samples)

        # Mix in trajectory points if available
        if len(self._trajectory) >= 5:
            traj_arr = np.array(self._trajectory[-self._trajectory_maxlen :])
            embeddings = np.vstack([embeddings, traj_arr])

        return embeddings

    def _record_trajectory(self, pos: np.ndarray):
        """Add a weight snapshot to the trajectory buffer."""
        self._trajectory.append(pos.copy())
        if len(self._trajectory) > self._trajectory_maxlen:
            self._trajectory = self._trajectory[-self._trajectory_maxlen :]

    def train_epoch(self, walker: ManifoldWalker, steps_per_epoch=4) -> float:
        """Run one epoch of ManifoldWalker optimization.

        Each epoch:
        1. Compute full-batch gradient (for the step direction)
        2. Let ManifoldWalker project it onto the discovered active subspace
        3. Step only along on-manifold directions
        """
        pos = self._get_flat_weights()

        for _ in range(steps_per_epoch):
            grad = self._compute_gradient_full(pos)
            val = walker.step(gradient=grad)
            pos = walker.position.copy()
            self._record_trajectory(pos)

        self._set_flat_weights(pos)
        return val

    def build_walker(self, center: np.ndarray) -> ManifoldWalker:
        """Build a ManifoldWalker with gradient-diversity embeddings."""
        self._record_trajectory(center)
        embeddings = self._sample_gradient_neighborhood(center)
        walker = ManifoldWalker(
            embeddings=embeddings,
            objective=self._compute_loss,
            k=min(self.k, len(embeddings)),
            variance_threshold=self.variance_threshold,
            learning_rate=self.lr,
        )
        walker.position = center.copy()
        return walker


def benchmark_manifold(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=200,
    lr=0.01,
    trial=0,
    k=30,
    variance_threshold=0.95,
    n_samples=100,
    steps_per_epoch=4,
    resample_interval=5,
) -> BenchmarkResult:
    """Train with ManifoldWalker optimizer."""
    model = build_model(learning_rate=lr)
    result = BenchmarkResult(method="ManifoldWalker", trial=trial, epochs=epochs)

    optimizer = ManifoldOptimizer(
        model,
        X_train,
        y_train,
        k=k,
        variance_threshold=variance_threshold,
        learning_rate=lr,
        n_gradient_samples=n_samples,
        mini_batch_size=16,
    )

    t0 = time.perf_counter()
    center = optimizer._get_flat_weights()
    walker = optimizer.build_walker(center)

    for epoch in range(epochs):
        # Periodically resample the neighborhood to track the manifold
        if epoch % resample_interval == 0:
            center = optimizer._get_flat_weights()
            walker = optimizer.build_walker(center)

        optimizer.train_epoch(walker, steps_per_epoch=steps_per_epoch)

        # Record metrics
        train_metrics = model.evaluate(X_train, y_train, verbose=0, batch_size=len(X_train))
        result.train_loss.append(float(train_metrics[0]))
        result.train_acc.append(float(train_metrics[1]))

        if result.convergence_epoch is None and train_metrics[1] >= 0.95:
            result.convergence_epoch = epoch

    result.wall_time = time.perf_counter() - t0
    result.test_loss, result.test_acc = model.evaluate(X_test, y_test, verbose=0)
    return result


# ---------------------------------------------------------------------------
# Benchmark 3: ManifoldAdam — Adam augmented with manifold-aware projection
# ---------------------------------------------------------------------------


class ManifoldAdamOptimizer:
    """Adam optimizer augmented with ManifoldWalker subspace projection.

    This is the key contribution: rather than replacing Adam, we AUGMENT it.

    At each step:
    1. Adam computes its standard update (with momentum and adaptive LR)
    2. We discover the active subspace of the loss landscape via
       ManifoldWalker's gradient-diversity PCA
    3. We project Adam's update onto this subspace
    4. Off-subspace components (noise) are suppressed
    5. On-subspace components are optionally eigenvalue-weighted

    This gives us Adam's fast convergence + ManifoldWalker's noise suppression.
    The hypothesis: by filtering out off-manifold gradient noise, we get
    cleaner updates that converge faster and generalize better.
    """

    def __init__(
        self,
        model,
        X_train,
        y_train,
        learning_rate=0.01,
        n_gradient_samples=40,
        mini_batch_size=16,
        variance_threshold=0.95,
        noise_suppression=0.8,  # how aggressively to suppress off-manifold (0=none, 1=full)
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.lr = learning_rate
        self.n_gradient_samples = n_gradient_samples
        self.mini_batch_size = mini_batch_size
        self.variance_threshold = variance_threshold
        self.noise_suppression = noise_suppression

        self._weight_shapes = [w.shape for w in model.trainable_weights]
        self._n_params = sum(int(np.prod(s)) for s in self._weight_shapes)

        # Adam state
        self._m = np.zeros(self._n_params, dtype="d")  # first moment
        self._v = np.zeros(self._n_params, dtype="d")  # second moment
        self._t = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._adam_eps = 1e-7

        # Trajectory buffer
        self._trajectory = []
        self._trajectory_maxlen = 60

        # Diagnostics
        self.intrinsic_dims = []

    def _get_flat_weights(self) -> np.ndarray:
        return np.concatenate([w.numpy().ravel() for w in self.model.trainable_weights])

    def _set_flat_weights(self, flat: np.ndarray):
        offset = 0
        new_weights = []
        for shape in self._weight_shapes:
            size = int(np.prod(shape))
            new_weights.append(flat[offset : offset + size].reshape(shape))
            offset += size
        self.model.set_weights([w.astype("float32") for w in new_weights])

    def _compute_gradient_minibatch(self, X_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = self.model(X_batch, training=True)
            loss = keras.losses.categorical_crossentropy(y_batch, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _compute_gradient_full(self):
        with tf.GradientTape() as tape:
            predictions = self.model(self.X_train, training=True)
            loss = keras.losses.categorical_crossentropy(self.y_train, predictions)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _discover_subspace(self):
        """Discover the active subspace via gradient-diversity PCA.

        Returns the projection matrix P that projects onto the on-manifold
        subspace, and the intrinsic dimensionality.
        """
        n_train = len(self.X_train)
        bs = min(self.mini_batch_size, n_train)
        self._get_flat_weights()

        # Collect gradient samples from different mini-batches
        grad_samples = []
        for _ in range(self.n_gradient_samples):
            idx = np.random.choice(n_train, size=bs, replace=False)
            g = self._compute_gradient_minibatch(self.X_train[idx], self.y_train[idx])
            grad_samples.append(g)

        G = np.array(grad_samples)  # (n_samples, n_params)
        G_centered = G - G.mean(axis=0)

        # PCA of gradient samples
        cov = (G_centered.T @ G_centered) / (len(G) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Flip to descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # Determine intrinsic dimensionality
        total = eigenvalues.sum()
        if total > 0:
            cumulative = np.cumsum(eigenvalues) / total
            intrinsic_dim = int(np.searchsorted(cumulative, self.variance_threshold) + 1)
        else:
            intrinsic_dim = self._n_params

        intrinsic_dim = max(1, min(intrinsic_dim, self._n_params))
        self.intrinsic_dims.append(intrinsic_dim)

        # Build projection matrix: P = V_d @ V_d^T
        # where V_d is the matrix of top-d eigenvectors
        V_d = eigenvectors[:, :intrinsic_dim]  # (n_params, d)

        return V_d, intrinsic_dim, eigenvalues

    def train_epoch(self, V_d, intrinsic_dim, eigenvalues, batch_size=32):
        """One ManifoldAdam epoch with mini-batch updates (matching Adam's regime).

        For each mini-batch:
        1. Compute mini-batch gradient
        2. Adam momentum/adaptive update
        3. Soft-project: scale off-manifold components by (1 - α) instead
           of hard-zeroing them. On-manifold components pass through at full
           strength.
        4. Apply the projected update
        """
        n_train = len(self.X_train)
        indices = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]
            X_batch = self.X_train[batch_idx]
            y_batch = self.y_train[batch_idx]

            grad = self._compute_gradient_minibatch(X_batch, y_batch)

            # Adam update
            self._t += 1
            self._m = self._beta1 * self._m + (1 - self._beta1) * grad
            self._v = self._beta2 * self._v + (1 - self._beta2) * (grad**2)
            m_hat = self._m / (1 - self._beta1**self._t)
            v_hat = self._v / (1 - self._beta2**self._t)

            adam_update = m_hat / (np.sqrt(v_hat) + self._adam_eps)

            # Soft manifold projection:
            # on_manifold component passes at full strength
            # off_manifold component is attenuated by (1 - α)
            on_manifold = V_d @ (V_d.T @ adam_update)
            off_manifold = adam_update - on_manifold
            α = self.noise_suppression
            blended_update = on_manifold + (1 - α) * off_manifold

            # Apply update
            pos = self._get_flat_weights()
            new_pos = pos - self.lr * blended_update
            self._set_flat_weights(new_pos)

        # Record trajectory once per epoch
        final_pos = self._get_flat_weights()
        self._trajectory.append(final_pos.copy())
        if len(self._trajectory) > self._trajectory_maxlen:
            self._trajectory = self._trajectory[-self._trajectory_maxlen :]


def benchmark_manifold_adam(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=200,
    lr=0.01,
    trial=0,
    n_gradient_samples=40,
    variance_threshold=0.95,
    noise_suppression=0.8,
    resample_interval=5,
) -> BenchmarkResult:
    """Train with ManifoldAdam (Adam + manifold-aware subspace projection)."""
    model = build_model(learning_rate=lr)
    result = BenchmarkResult(method="ManifoldAdam", trial=trial, epochs=epochs)

    optimizer = ManifoldAdamOptimizer(
        model,
        X_train,
        y_train,
        learning_rate=lr,
        n_gradient_samples=n_gradient_samples,
        variance_threshold=variance_threshold,
        noise_suppression=noise_suppression,
    )

    t0 = time.perf_counter()
    V_d, intrinsic_dim, eigenvalues = optimizer._discover_subspace()

    for epoch in range(epochs):
        if epoch % resample_interval == 0 and epoch > 0:
            V_d, intrinsic_dim, eigenvalues = optimizer._discover_subspace()

        optimizer.train_epoch(V_d, intrinsic_dim, eigenvalues)

        train_metrics = model.evaluate(X_train, y_train, verbose=0, batch_size=len(X_train))
        result.train_loss.append(float(train_metrics[0]))
        result.train_acc.append(float(train_metrics[1]))

        if result.convergence_epoch is None and train_metrics[1] >= 0.95:
            result.convergence_epoch = epoch

    result.wall_time = time.perf_counter() - t0
    result.test_loss, result.test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Report intrinsic dimensionality stats
    dims = optimizer.intrinsic_dims
    print(
        f"    Intrinsic dim: {np.mean(dims):.1f} ± {np.std(dims):.1f} "
        f"(of {optimizer._n_params} params)"
    )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    adam_results: list,
    manifold_results: list,
    manifold_adam_results: list = None,
    save_path: str = None,
):
    """Generate comparison plots for all methods."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    all_groups = [
        ("Adam", adam_results, "steelblue"),
        ("ManifoldWalker", manifold_results, "firebrick"),
    ]
    if manifold_adam_results:
        all_groups.append(("ManifoldAdam", manifold_adam_results, "forestgreen"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Iris Benchmark: Adam vs ManifoldWalker vs ManifoldAdam",
        fontsize=14,
        fontweight="bold",
    )

    # --- Loss curves (mean +/- std across trials) ---
    ax = axes[0, 0]
    for name, results, color in all_groups:
        losses = np.array([r.train_loss for r in results])
        epochs = np.arange(1, losses.shape[1] + 1)
        ax.plot(epochs, losses.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Loss Convergence")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Accuracy curves ---
    ax = axes[0, 1]
    for name, results, color in all_groups:
        accs = np.array([r.train_acc for r in results])
        epochs = np.arange(1, accs.shape[1] + 1)
        ax.plot(epochs, accs.mean(0), "-", label=name, linewidth=2, color=color)
        ax.fill_between(
            epochs,
            accs.mean(0) - accs.std(0),
            accs.mean(0) + accs.std(0),
            alpha=0.15,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")
    ax.set_title("Accuracy Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    # --- Convergence speed (bar chart) ---
    ax = axes[1, 0]
    labels, means, stds, colors = [], [], [], []
    for name, results, color, _ in all_groups:
        conv = [r.convergence_epoch for r in results if r.convergence_epoch is not None]
        if conv:
            labels.append(name)
            means.append(np.mean(conv))
            stds.append(np.std(conv))
            colors.append(color)
    if labels:
        bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
        ax.set_ylabel("Epochs to 95% Accuracy")
        ax.set_title("Convergence Speed (lower is better)")
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    ax.grid(True, alpha=0.3, axis="y")

    # --- Wall time comparison ---
    ax = axes[1, 1]
    labels, means, stds, colors = [], [], [], []
    for name, results, color, _ in all_groups:
        times = [r.wall_time for r in results]
        labels.append(name)
        means.append(np.mean(times))
        stds.append(np.std(times))
        colors.append(color)
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Training Time")
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{m:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = save_path or "benchmarks/iris_benchmark_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(
    adam_results: list,
    manifold_results: list,
    manifold_adam_results: list = None,
):
    """Print a formatted summary table."""
    print("\n" + "=" * 95)
    print("IRIS BENCHMARK RESULTS: Adam vs ManifoldWalker vs ManifoldAdam")
    print("=" * 95)
    print(f"Device: {DEVICE_INFO['device_used']}")
    print(f"TensorFlow: {DEVICE_INFO['tensorflow_version']}")
    print(f"Trials: {len(adam_results)}")
    print(f"Architecture: 4 -> 16 -> 8 -> 3 ({count_params(build_model())} params)")
    print("-" * 95)

    def _stats(results, key):
        vals = [getattr(r, key) for r in results]
        return np.mean(vals), np.std(vals)

    def _conv_stats(results):
        vals = [r.convergence_epoch for r in results if r.convergence_epoch is not None]
        if not vals:
            return float("inf"), 0.0, 0
        return np.mean(vals), np.std(vals), len(vals)

    groups = [("Adam", adam_results), ("ManifoldWalker", manifold_results)]
    if manifold_adam_results:
        groups.append(("ManifoldAdam", manifold_adam_results))

    col_w = 22
    header = f"{'Metric':<28}" + "".join(f"{name:>{col_w}}" for name, _ in groups)
    print(header)
    print("-" * 95)

    for label, key in [
        ("Final Test Loss", "test_loss"),
        ("Final Test Accuracy", "test_acc"),
        ("Wall Time (s)", "wall_time"),
    ]:
        fmt = ".4f" if "acc" in key or "loss" in key.lower() else ".2f"
        row = f"{label:<28}"
        for _, results in groups:
            m, s = _stats(results, key)
            row += f" {m:{fmt}} +/- {s:{fmt}}  "
        print(row)

    # Convergence
    row = f"{'Epochs to 95% Acc':<28}"
    for _, results in groups:
        cm, cs, cn = _conv_stats(results)
        if cm == float("inf"):
            row += f"{'N/A':>{col_w}}  "
        else:
            row += f" {cm:>5.1f} +/- {cs:>4.1f} ({cn}/{len(results)})  "
    print(row)

    # Intrinsic dimensionality insight
    print("-" * 95)
    total_params = count_params(build_model())
    print(f"Weight space dimensionality: {total_params}")
    print(
        "ManifoldWalker discovers the intrinsic dimensionality of the loss\n"
        "landscape and restricts gradient updates to on-manifold directions,\n"
        "suppressing noise in the remaining dimensions."
    )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------


def _results_to_dicts(results):
    return [
        {
            "trial": r.trial,
            "test_loss": r.test_loss,
            "test_acc": r.test_acc,
            "wall_time": r.wall_time,
            "convergence_epoch": r.convergence_epoch,
            "train_loss": r.train_loss,
            "train_acc": r.train_acc,
        }
        for r in results
    ]


def save_results(
    adam_results,
    manifold_results,
    manifold_adam_results=None,
    path="benchmarks/iris_benchmark_results.json",
):
    """Save results to JSON for reproducibility."""
    data = {
        "device": DEVICE_INFO,
        "adam": _results_to_dicts(adam_results),
        "manifold_walker": _results_to_dicts(manifold_results),
    }
    if manifold_adam_results:
        data["manifold_adam"] = _results_to_dicts(manifold_adam_results)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Iris Benchmark: Adam vs ManifoldWalker")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate plots")
    parser.add_argument("--k", type=int, default=30, help="ManifoldWalker neighborhood size")
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="ManifoldWalker variance threshold",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Weight-space samples for local PCA",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=4,
        help="ManifoldWalker steps per epoch",
    )
    parser.add_argument(
        "--resample-interval",
        type=int,
        default=5,
        help="Epochs between neighborhood resampling",
    )
    args = parser.parse_args()

    print("\nLoading Iris dataset...")
    X_train, y_train, X_test, y_test = load_iris()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Model: 4 → 16 → 8 → 3 ({count_params(build_model())} params)\n")

    adam_results = []
    manifold_results = []
    manifold_adam_results = []

    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials}")

        # Set random seed per trial for reproducibility
        np.random.seed(trial * 42)
        tf.random.set_seed(trial * 42)

        # Adam
        print("  Adam...", end=" ", flush=True)
        ar = benchmark_adam(X_train, y_train, X_test, y_test, args.epochs, args.lr, trial)
        print(f"loss={ar.test_loss:.4f}  acc={ar.test_acc:.4f}  time={ar.wall_time:.2f}s")
        adam_results.append(ar)

        # ManifoldWalker (raw)
        np.random.seed(trial * 42)
        tf.random.set_seed(trial * 42)

        print("  ManifoldWalker...", end=" ", flush=True)
        mr = benchmark_manifold(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=args.epochs,
            lr=args.lr,
            trial=trial,
            k=args.k,
            variance_threshold=args.variance_threshold,
            n_samples=args.n_samples,
            steps_per_epoch=args.steps_per_epoch,
            resample_interval=args.resample_interval,
        )
        print(f"loss={mr.test_loss:.4f}  acc={mr.test_acc:.4f}  time={mr.wall_time:.2f}s")
        manifold_results.append(mr)

        # ManifoldAdam (hybrid: Adam + manifold subspace projection)
        np.random.seed(trial * 42)
        tf.random.set_seed(trial * 42)

        print("  ManifoldAdam...", end=" ", flush=True)
        mar = benchmark_manifold_adam(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=args.epochs,
            lr=args.lr,
            trial=trial,
            n_gradient_samples=args.n_samples,
            variance_threshold=args.variance_threshold,
            resample_interval=args.resample_interval,
        )
        print(f"loss={mar.test_loss:.4f}  acc={mar.test_acc:.4f}  time={mar.wall_time:.2f}s")
        manifold_adam_results.append(mar)

    print_summary(adam_results, manifold_results, manifold_adam_results)
    save_results(adam_results, manifold_results, manifold_adam_results)

    if args.plot:
        plot_results(adam_results, manifold_results, manifold_adam_results)


if __name__ == "__main__":
    main()
