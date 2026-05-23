#!/usr/bin/env python3
"""
Synthetic Helix Benchmark: ManifoldObserver vs ManifoldModel subject
=====================================================================

Sanity-check experiment for the Normal Extension / ManifoldObserver
construction described in the TurtleND paper.

Setup
-----
A 1-dimensional helix curve in :math:`\\mathbb{R}^3`

    (cos(t), sin(t), t/6),  t uniform on [0, 6π]

is embedded in :math:`\\mathbb{R}^5` by appending two Gaussian-noise
coordinates (σ = 0.02).  Binary labels are assigned by dividing the
curve into two arcs (t < 3π  vs  t ≥ 3π), so the class boundary lies
on the manifold rather than off it.

For each random seed:

  1. Sample ``n_total`` points, split 80/20 into train/test.
  2. Fit a ``ManifoldModel`` on the training data (subject).
  3. Report the subject's local-PCA intrinsic dimensionality summary.
  4. Wrap the subject in a ``ManifoldObserver`` (extrinsic N+1 view).
  5. Observe the full manifold to populate curvature + height fields.
  6. Classify the held-out test set two ways:
       - ``subject.predict`` — graph-walk inside the N-dim manifold
       - ``observer.predict`` — direct (N+1)-dim projection from above
  7. Record accuracies, the agreement rate between the two predictors,
     mean/std curvature, and mean/max height.

Results are aggregated over ``n_trials`` seeds and written to
``helix_manifold_observer_results.json``.

The point of the experiment is *not* to achieve high accuracy — a
1-manifold with a clean mid-curve class boundary is trivially
separable.  The point is to verify that the extrinsic observer
reproduces the intrinsic subject's classification (Observer ↔ Subject
agreement near 100%) and that the observer's curvature / height
fields are well-defined and finite.

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD

Usage
-----
    python benchmarks/canonical_tests/helix_manifold_observer.py \\
        [--n-total 500] [--n-trials 10] [--noise-sigma 0.02]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from waverider.manifold_model import ManifoldModel
from waverider.manifold_observer import ManifoldObserver

# matplotlib is imported lazily inside make_plot() so the script still
# runs headless on CI boxes without a display backend.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def make_helix(n: int, noise_sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct an n-point helix in 3D, embedded into 5D with Gaussian noise.

    :param n: Number of points to sample along the helix curve.
    :param noise_sigma: Standard deviation of the Gaussian noise in the 2
        trailing dimensions.
    :param seed: RNG seed for reproducibility.
    :return: Tuple ``(X, y)`` where ``X`` has shape ``(n, 5)`` and ``y`` is a
        binary label array of shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0 * np.pi, n)

    # 3D helix
    x0 = np.cos(t)
    x1 = np.sin(t)
    x2 = t / 6.0

    # Embed in 5D by appending two noise coordinates
    x3 = rng.normal(0.0, noise_sigma, size=n)
    x4 = rng.normal(0.0, noise_sigma, size=n)

    X = np.column_stack([x0, x1, x2, x3, x4]).astype("d")
    y = (t >= 3.0 * np.pi).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_trial(
    seed: int,
    n_total: int,
    noise_sigma: float,
    k_graph: int,
    k_pca: int,
    k_vote: int,
    tau: float,
    keep_arrays: bool = False,
) -> dict:
    """Run one helix trial and return a dict of measurements.

    :param keep_arrays: If ``True``, attach the raw ``X`` / ``y``, the
        per-node curvature field, and the per-node heights to the
        returned dict.  Used for plotting a representative trial only;
        omit for normal aggregation to keep JSON output small.
    """
    X, y = make_helix(n_total, noise_sigma, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    t0 = time.perf_counter()
    subject = ManifoldModel(k_graph=k_graph, k_pca=k_pca, k_vote=k_vote, variance_threshold=tau)
    subject.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    # Subject-side (graph-walk) classification
    t0 = time.perf_counter()
    subject_preds = subject.predict(X_test)
    subject_time = time.perf_counter() - t0
    subject_acc = float(np.mean(subject_preds == y_test))

    # Observer-side (extrinsic projection) classification
    observer = ManifoldObserver(subject)
    observer.lift_data()
    field = observer.observe()  # populates curvature + height cache

    t0 = time.perf_counter()
    observer_preds = observer.predict(X_test)
    observer_time = time.perf_counter() - t0
    observer_acc = float(np.mean(observer_preds == y_test))

    agreement = float(np.mean(subject_preds == observer_preds))

    # Geometry summary
    curvatures = np.array([o.curvature for o in field], dtype="d")
    heights = np.array([o.height for o in field], dtype="d")

    # Intrinsic dim at τ — averaged over training nodes
    summary = subject.geometry_summary()

    result = {
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "ambient_dim": int(X.shape[1]),
        "mean_intrinsic_dim": float(summary["mean_intrinsic_dim"]),
        "subject_acc": subject_acc,
        "observer_acc": observer_acc,
        "agreement": agreement,
        "mean_height": float(np.mean(heights)),
        "max_height": float(np.max(heights)),
        "mean_curvature_rad": float(np.mean(curvatures)),
        "std_curvature_rad": float(np.std(curvatures)),
        "fit_time_s": fit_time,
        "subject_predict_time_s": subject_time,
        "observer_predict_time_s": observer_time,
    }

    if keep_arrays:
        result["_arrays"] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "subject_preds": subject_preds,
            "observer_preds": observer_preds,
            "curvatures": curvatures,
            "heights": heights,
        }

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_plot(trials: list[dict], agg: dict, arrays: dict, out_path: Path) -> None:
    """Render a 3-panel diagnostic figure for the helix benchmark.

    :param trials: Full list of trial dicts (used for per-trial bars + d*).
    :param agg: Aggregated summary dict from :func:`aggregate`.
    :param arrays: The ``_arrays`` sub-dict from a single representative
        trial (requires ``keep_arrays=True``).
    :param out_path: Destination ``.png`` path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

    X_train = arrays["X_train"]
    y_train = arrays["y_train"]

    fig = plt.figure(figsize=(15, 4.5))

    # Panel 1: raw helix in 3D, colored by label
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    for cls, color in [(0, "#1f77b4"), (1, "#d62728")]:
        mask = y_train == cls
        ax1.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            X_train[mask, 2],
            c=color,
            s=10,
            alpha=0.75,
            label=f"class {cls}",
        )
    ax1.set_xlabel("x0 = cos t")
    ax1.set_ylabel("x1 = sin t")
    ax1.set_zlabel("x2 = t/6")
    ax1.set_title("Helix (first 3 of 5 dims)")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: d* histogram across trials
    ax2 = fig.add_subplot(1, 3, 2)
    dims = np.array([t["mean_intrinsic_dim"] for t in trials], dtype="d")
    ax2.hist(dims, bins=min(10, len(dims)), color="#2ca02c", alpha=0.8, edgecolor="k")
    ax2.axvline(1.0, color="k", linestyle="--", linewidth=1.5, label="true d*=1")
    ax2.axvline(
        float(np.mean(dims)),
        color="#d62728",
        linestyle="-",
        linewidth=1.5,
        label=f"mean={np.mean(dims):.3f}",
    )
    ax2.set_xlabel("mean intrinsic dim (local PCA)")
    ax2.set_ylabel("trials")
    ax2.set_title("Recovered d* vs truth")
    ax2.legend(fontsize=8)

    # Panel 3: per-trial subject / observer / agreement bars
    ax3 = fig.add_subplot(1, 3, 3)
    n = len(trials)
    idx = np.arange(n)
    width = 0.27
    subj = np.array([t["subject_acc"] for t in trials], dtype="d")
    obs = np.array([t["observer_acc"] for t in trials], dtype="d")
    agr = np.array([t["agreement"] for t in trials], dtype="d")
    ax3.bar(idx - width, subj, width, label="subject acc", color="#1f77b4")
    ax3.bar(idx, obs, width, label="observer acc", color="#ff7f0e")
    ax3.bar(idx + width, agr, width, label="subj↔obs agreement", color="#2ca02c")
    ax3.set_ylim(0.0, 1.05)
    ax3.set_xticks(idx)
    ax3.set_xticklabels([str(t["seed"]) for t in trials], fontsize=7)
    ax3.set_xlabel("seed")
    ax3.set_ylabel("accuracy / agreement")
    ax3.set_title(f"Observer ↔ Subject  (mean agr = {agg['agreement']['mean']:.3f})")
    ax3.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def aggregate(trials: list[dict]) -> dict:
    """Mean ± std over a list of trial dicts for numeric fields."""
    keys = [
        "mean_intrinsic_dim",
        "subject_acc",
        "observer_acc",
        "agreement",
        "mean_height",
        "max_height",
        "mean_curvature_rad",
        "std_curvature_rad",
        "fit_time_s",
    ]
    out = {}
    for k in keys:
        vals = np.array([t[k] for t in trials], dtype="d")
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=500)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--k-graph", type=int, default=10)
    parser.add_argument("--k-pca", type=int, default=20)
    parser.add_argument("--k-vote", type=int, default=7)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render a 3-panel diagnostic PNG alongside the JSON output.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HELIX BENCHMARK: ManifoldObserver vs ManifoldModel subject")
    print(f"1-manifold in 3D, embedded in 5D with σ={args.noise_sigma:.3f} Gaussian noise")
    print("=" * 70)
    print(
        f"n_total={args.n_total}  n_trials={args.n_trials}  τ={args.tau}  "
        f"k_graph={args.k_graph}  k_pca={args.k_pca}  k_vote={args.k_vote}"
    )
    print()

    trials: list[dict] = []
    plot_arrays: dict | None = None
    for i in range(args.n_trials):
        seed = args.base_seed + i
        print(f"Trial {i + 1}/{args.n_trials}  (seed={seed}) ...", flush=True)
        keep = args.plot and i == 0
        result = run_trial(
            seed=seed,
            n_total=args.n_total,
            noise_sigma=args.noise_sigma,
            k_graph=args.k_graph,
            k_pca=args.k_pca,
            k_vote=args.k_vote,
            tau=args.tau,
            keep_arrays=keep,
        )
        if keep:
            plot_arrays = result.pop("_arrays")
        trials.append(result)
        print(
            f"  subject_acc={result['subject_acc']:.4f}  "
            f"observer_acc={result['observer_acc']:.4f}  "
            f"agreement={result['agreement']:.4f}  "
            f"d*={result['mean_intrinsic_dim']:.2f}  "
            f"mean_h={result['mean_height']:.4f}  "
            f"mean_κ={result['mean_curvature_rad']:.4f} rad"
        )

    agg = aggregate(trials)

    print()
    print("=" * 70)
    print(f"RESULTS (mean ± std over {args.n_trials} trials)")
    print("=" * 70)
    for k, v in agg.items():
        print(f"  {k:30s}  {v['mean']:.4f} ± {v['std']:.4f}")

    out = {
        "config": vars(args),
        "trials": trials,
        "aggregated": agg,
    }
    out_path = Path(__file__).with_name("helix_manifold_observer_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote {out_path}")

    if args.plot and plot_arrays is not None:
        png_path = out_path.with_suffix(".png")
        make_plot(trials, agg, plot_arrays, png_path)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
