#!/usr/bin/env python3
"""
Iris Benchmark: Standalone ManifoldAdamWalker vs Adam
=====================================================

Can manifold-aware optimization with built-in Adam dynamics beat
canonical Adam on its home turf?

Trains an identical MLP (4→16→8→3 softmax) on Iris (150 samples, 4 features,
3 classes, 80/20 stratified split) using two optimizers over --trials
independent runs (default 10).  Supports Apple Metal GPU via tensorflow-metal.

The active-subspace approach
-----------------------------
At each optimization step, --n-samples gradients are collected from different
mini-batches and treated as points in the P-dimensional weight space.  Local
PCA (k=--k, τ=--variance-threshold) discovers the active subspace — directions
where the loss actually varies.  A trajectory buffer of recent weight snapshots
provides a second source of geometry.

Two methods
-----------
  Adam (canonical):
    Standard TensorFlow/Keras Adam on the full gradient.

  StandaloneManifoldAdam (ManifoldAdamWalker):
    Projects each gradient onto the active subspace, suppressing off-subspace
    components, then runs Adam momentum (β₁=--beta1, β₂=--beta2) and adaptive
    LR entirely within that subspace.  Effects:
      - Momentum never accumulates mini-batch noise
      - Adaptive denominator tracks only signal variance
      - Step sizes adapt to manifold curvature

Results are saved alongside the script as ``iris_maw_results.json`` and an
optional matplotlib figure is written as ``iris_maw_results.png``.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers

Usage
-----
    python benchmarks/canonical_tests/iris_manifold_adam_walker.py [--epochs 200] [--trials 10]
"""

import argparse
import json
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# TensorFlow setup
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow()
import keras  # noqa: E402
import numpy as np  # noqa: E402

from waverider.manifold_walker import ManifoldAdamWalker  # noqa: E402

# ---------------------------------------------------------------------------
# Import ManifoldAdamWalker
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_iris():
    """Load Iris, normalize, one-hot encode, 80/20 split."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X, y = data.data.astype("float32"), data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_onehot = np.zeros((len(y), 3), dtype="float32")
    y_onehot[np.arange(len(y)), y] = 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(hidden_units=(16, 8), activation="relu", learning_rate=0.01):
    """4 -> 16 -> 8 -> 3 (softmax)."""
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
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    method: str
    trial: int
    epochs: int
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    test_loss: float = 0.0
    test_acc: float = 0.0
    wall_time: float = 0.0
    convergence_epoch: int | None = None
    intrinsic_dims: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark 1: Adam (canonical baseline)
# ---------------------------------------------------------------------------


def benchmark_adam(X_train, y_train, X_test, y_test, epochs=200, lr=0.01, trial=0):
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

    for i, acc in enumerate(history.history["accuracy"]):
        if acc >= 0.95:
            result.convergence_epoch = i
            break

    return result


# ---------------------------------------------------------------------------
# Benchmark 2: Standalone ManifoldAdamWalker
# ---------------------------------------------------------------------------


class StandaloneManifoldAdam:
    """Drive ManifoldAdamWalker as the sole optimizer for a Keras model.

    Gradient-diversity PCA: we sample gradients from different mini-batches,
    treating each as a point in weight space.  PCA of these samples reveals
    the active subspace — directions where the loss actually varies vs
    mini-batch noise.  The ManifoldAdamWalker then runs Adam exclusively
    within this subspace.
    """

    def __init__(
        self,
        model,
        X_train,
        y_train,
        k=30,
        variance_threshold=0.90,
        learning_rate=0.01,
        n_gradient_samples=40,
        mini_batch_size=16,
        trajectory_len=60,
        beta1=0.9,
        beta2=0.999,
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.variance_threshold = variance_threshold
        self.lr = learning_rate
        self.n_gradient_samples = n_gradient_samples
        self.mini_batch_size = mini_batch_size
        self.beta1 = beta1
        self.beta2 = beta2

        self._weight_shapes = [w.shape for w in model.trainable_weights]
        self._n_params = sum(int(np.prod(s)) for s in self._weight_shapes)

        self._trajectory = []
        self._trajectory_maxlen = trajectory_len

    def _get_flat_weights(self):
        return np.concatenate([w.numpy().ravel() for w in self.model.trainable_weights])

    def _set_flat_weights(self, flat):
        offset = 0
        new_weights = []
        for shape in self._weight_shapes:
            size = int(np.prod(shape))
            new_weights.append(flat[offset : offset + size].reshape(shape))
            offset += size
        self.model.set_weights([w.astype("float32") for w in new_weights])

    def _compute_loss(self, flat_weights):
        self._set_flat_weights(flat_weights)
        loss = self.model.evaluate(
            self.X_train, self.y_train, verbose=0, batch_size=len(self.X_train)
        )
        return float(loss[0])

    def _compute_gradient_full(self, flat_weights):
        self._set_flat_weights(flat_weights)
        with tf.GradientTape() as tape:
            preds = self.model(self.X_train, training=True)
            loss = keras.losses.categorical_crossentropy(self.y_train, preds)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _compute_gradient_minibatch(self, flat_weights, X_batch, y_batch):
        self._set_flat_weights(flat_weights)
        with tf.GradientTape() as tape:
            preds = self.model(X_batch, training=True)
            loss = keras.losses.categorical_crossentropy(y_batch, preds)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.model.trainable_weights)
        return np.concatenate([g.numpy().ravel() for g in grads])

    def _sample_gradient_neighborhood(self, center):
        """Build embedding matrix from gradient-diversity samples + trajectory."""
        n_train = len(self.X_train)
        bs = min(self.mini_batch_size, n_train)

        grad_samples = []
        for _ in range(self.n_gradient_samples):
            idx = np.random.choice(n_train, size=bs, replace=False)
            g = self._compute_gradient_minibatch(center, self.X_train[idx], self.y_train[idx])
            grad_samples.append(center + 0.01 * g)

        embeddings = np.array(grad_samples)

        if len(self._trajectory) >= 5:
            traj_arr = np.array(self._trajectory[-self._trajectory_maxlen :])
            embeddings = np.vstack([embeddings, traj_arr])

        return embeddings

    def _record_trajectory(self, pos):
        self._trajectory.append(pos.copy())
        if len(self._trajectory) > self._trajectory_maxlen:
            self._trajectory = self._trajectory[-self._trajectory_maxlen :]

    def build_walker(self, center):
        """Build a ManifoldAdamWalker with gradient-diversity embeddings."""
        self._record_trajectory(center)
        embeddings = self._sample_gradient_neighborhood(center)
        walker = ManifoldAdamWalker(
            embeddings=embeddings,
            objective=self._compute_loss,
            k=min(self.k, len(embeddings)),
            variance_threshold=self.variance_threshold,
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
        )
        walker.position = center.copy()
        # Transfer Adam state from previous walker if we have one
        if hasattr(self, "_adam_m"):
            walker._m = self._adam_m.copy()
            walker._v = self._adam_v.copy()
            walker._t = self._adam_t
        return walker

    def _save_adam_state(self, walker):
        """Preserve Adam momentum across walker rebuilds."""
        self._adam_m = walker._m.copy()
        self._adam_v = walker._v.copy()
        self._adam_t = walker._t

    def train_epoch(self, walker, steps_per_epoch=4):
        """One epoch: multiple manifold-projected Adam steps."""
        pos = self._get_flat_weights()

        for _ in range(steps_per_epoch):
            grad = self._compute_gradient_full(pos)
            val = walker.step(gradient=grad)
            pos = walker.position.copy()
            self._record_trajectory(pos)

        self._set_flat_weights(pos)
        self._save_adam_state(walker)
        return val


def benchmark_manifold_adam_walker(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=200,
    lr=0.01,
    trial=0,
    k=30,
    variance_threshold=0.90,
    n_samples=40,
    steps_per_epoch=4,
    resample_interval=5,
    beta1=0.9,
    beta2=0.999,
):
    """Train with standalone ManifoldAdamWalker."""
    model = build_model(learning_rate=lr)
    result = BenchmarkResult(method="ManifoldAdamWalker", trial=trial, epochs=epochs)

    optimizer = StandaloneManifoldAdam(
        model,
        X_train,
        y_train,
        k=k,
        variance_threshold=variance_threshold,
        learning_rate=lr,
        n_gradient_samples=n_samples,
        mini_batch_size=16,
        beta1=beta1,
        beta2=beta2,
    )

    t0 = time.perf_counter()
    center = optimizer._get_flat_weights()
    walker = optimizer.build_walker(center)

    for epoch in range(epochs):
        if epoch % resample_interval == 0 and epoch > 0:
            center = optimizer._get_flat_weights()
            walker = optimizer.build_walker(center)

        optimizer.train_epoch(walker, steps_per_epoch=steps_per_epoch)

        train_metrics = model.evaluate(X_train, y_train, verbose=0, batch_size=len(X_train))
        result.train_loss.append(float(train_metrics[0]))
        result.train_acc.append(float(train_metrics[1]))

        if walker.intrinsic_dim is not None:
            result.intrinsic_dims.append(walker.intrinsic_dim)

        if result.convergence_epoch is None and train_metrics[1] >= 0.95:
            result.convergence_epoch = epoch

    result.wall_time = time.perf_counter() - t0
    result.test_loss, result.test_acc = model.evaluate(X_test, y_test, verbose=0)

    if result.intrinsic_dims:
        dims = result.intrinsic_dims
        print(
            f"    Intrinsic dim: {np.mean(dims):.1f} +/- {np.std(dims):.1f} "
            f"(of {optimizer._n_params} params)"
        )

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(adam_results, maw_results, save_path=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    groups = [
        ("Adam", adam_results, "steelblue"),
        ("ManifoldAdamWalker", maw_results, "firebrick"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Iris Benchmark: Adam vs Standalone ManifoldAdamWalker",
        fontsize=14,
        fontweight="bold",
    )

    # Loss curves
    ax = axes[0, 0]
    for name, results, color in groups:
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

    # Accuracy curves
    ax = axes[0, 1]
    for name, results, color in groups:
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

    # Convergence speed
    ax = axes[1, 0]
    labels, means, stds, colors = [], [], [], []
    for name, results, color in groups:
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

    # Test accuracy comparison
    ax = axes[1, 1]
    labels, means, stds, colors = [], [], [], []
    for name, results, color in groups:
        accs = [r.test_acc for r in results]
        labels.append(name)
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        colors.append(color)
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Final Test Accuracy (higher is better)")
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.8, 1.05)

    plt.tight_layout()
    save_path = save_path or "benchmarks/iris_maw_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(adam_results, maw_results):
    print("\n" + "=" * 80)
    print("IRIS BENCHMARK: Adam vs Standalone ManifoldAdamWalker")
    print("=" * 80)
    print(f"Device: {DEVICE_INFO['device_used']}")
    print(f"TensorFlow: {DEVICE_INFO['tensorflow_version']}")
    print(f"Trials: {len(adam_results)}")
    total_params = count_params(build_model())
    print(f"Architecture: 4 -> 16 -> 8 -> 3 ({total_params} params)")
    print("-" * 80)

    def _stats(results, key):
        vals = [getattr(r, key) for r in results]
        return np.mean(vals), np.std(vals)

    def _conv_stats(results):
        vals = [r.convergence_epoch for r in results if r.convergence_epoch is not None]
        if not vals:
            return float("inf"), 0.0, 0
        return np.mean(vals), np.std(vals), len(vals)

    groups = [("Adam", adam_results), ("ManifoldAdamWalker", maw_results)]

    col_w = 28
    header = f"{'Metric':<28}" + "".join(f"{name:>{col_w}}" for name, _ in groups)
    print(header)
    print("-" * 80)

    for label, key in [
        ("Final Test Loss", "test_loss"),
        ("Final Test Accuracy", "test_acc"),
        ("Wall Time (s)", "wall_time"),
    ]:
        fmt = ".4f" if "acc" in key or "loss" in key.lower() else ".2f"
        row = f"{label:<28}"
        for _, results in groups:
            m, s = _stats(results, key)
            row += f"  {m:{fmt}} +/- {s:{fmt}}  "
        print(row)

    row = f"{'Epochs to 95% Acc':<28}"
    for _, results in groups:
        cm, cs, cn = _conv_stats(results)
        if cm == float("inf"):
            row += f"{'N/A':>{col_w}}  "
        else:
            row += f"  {cm:>5.1f} +/- {cs:>4.1f} ({cn}/{len(results)})  "
    print(row)

    # Winner determination
    print("-" * 80)
    adam_acc_m, _ = _stats(adam_results, "test_acc")
    maw_acc_m, _ = _stats(maw_results, "test_acc")
    adam_conv, _, _ = _conv_stats(adam_results)
    maw_conv, _, _ = _conv_stats(maw_results)

    if maw_acc_m > adam_acc_m:
        print(
            f">> ManifoldAdamWalker WINS on test accuracy: "
            f"{maw_acc_m:.4f} vs {adam_acc_m:.4f} (+{maw_acc_m - adam_acc_m:.4f})"
        )
    elif adam_acc_m > maw_acc_m:
        print(f">> Adam wins on test accuracy: {adam_acc_m:.4f} vs {maw_acc_m:.4f}")
    else:
        print(f">> Tied on test accuracy: {adam_acc_m:.4f}")

    if maw_conv < adam_conv:
        print(
            f">> ManifoldAdamWalker WINS on convergence speed: "
            f"{maw_conv:.1f} vs {adam_conv:.1f} epochs"
        )
    elif adam_conv < maw_conv:
        print(f">> Adam wins on convergence speed: {adam_conv:.1f} vs {maw_conv:.1f} epochs")

    # Intrinsic dimensionality report
    all_dims = []
    for r in maw_results:
        all_dims.extend(r.intrinsic_dims)
    if all_dims:
        print("\nIntrinsic dimensionality of loss landscape:")
        print(f"  Mean: {np.mean(all_dims):.1f} / {total_params} parameters")
        print(f"  Range: [{min(all_dims)}, {max(all_dims)}]")
        print(
            f"  -> {100 * (1 - np.mean(all_dims) / total_params):.1f}% of gradient "
            f"dimensions are noise"
        )

    print("=" * 80)


def save_results(adam_results, maw_results, path="benchmarks/iris_maw_results.json"):
    def _to_dicts(results):
        return [
            {
                "trial": r.trial,
                "test_loss": r.test_loss,
                "test_acc": r.test_acc,
                "wall_time": r.wall_time,
                "convergence_epoch": r.convergence_epoch,
                "train_loss": r.train_loss,
                "train_acc": r.train_acc,
                "intrinsic_dims": r.intrinsic_dims,
            }
            for r in results
        ]

    data = {
        "device": DEVICE_INFO,
        "adam": _to_dicts(adam_results),
        "manifold_adam_walker": _to_dicts(maw_results),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Iris: Standalone ManifoldAdamWalker vs Adam")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--variance-threshold", type=float, default=0.90)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--resample-interval", type=int, default=5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--plot", action="store_true", default=True)
    args = parser.parse_args()

    print("\nLoading Iris dataset...")
    X_train, y_train, X_test, y_test = load_iris()
    total_params = count_params(build_model())
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Model: 4 -> 16 -> 8 -> 3 ({total_params} params)\n")

    adam_results = []
    maw_results = []

    for trial in range(args.trials):
        print(f"Trial {trial + 1}/{args.trials}")

        np.random.seed(trial * 42)
        tf.random.set_seed(trial * 42)

        print("  Adam...", end=" ", flush=True)
        ar = benchmark_adam(X_train, y_train, X_test, y_test, args.epochs, args.lr, trial)
        print(
            f"loss={ar.test_loss:.4f}  acc={ar.test_acc:.4f}  "
            f"conv={ar.convergence_epoch}  time={ar.wall_time:.2f}s"
        )
        adam_results.append(ar)

        np.random.seed(trial * 42)
        tf.random.set_seed(trial * 42)

        print("  ManifoldAdamWalker...", end=" ", flush=True)
        mr = benchmark_manifold_adam_walker(
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
            beta1=args.beta1,
            beta2=args.beta2,
        )
        print(
            f"loss={mr.test_loss:.4f}  acc={mr.test_acc:.4f}  "
            f"conv={mr.convergence_epoch}  time={mr.wall_time:.2f}s"
        )
        maw_results.append(mr)

    print_summary(adam_results, maw_results)
    save_results(adam_results, maw_results)

    if args.plot:
        plot_results(adam_results, maw_results)


if __name__ == "__main__":
    main()
