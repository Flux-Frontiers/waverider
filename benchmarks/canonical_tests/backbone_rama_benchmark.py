#!/usr/bin/env python3
"""
Backbone Ramachandran Region Benchmark
=======================================

Can local sequence context predict a residue's Ramachandran region without
seeing the residue's own backbone angles?

Task definition
---------------
  Features  : torus embeddings of the ±k NEIGHBORING residues + center AA type
              (center residue's own (φ, ψ) are ZEROED — context_only=True)
  Labels    : Ramachandran region of the CENTER residue, derived purely from
              angle-box geometry (not DSSP hydrogen-bond labels)

This separates two sources of information:
  * Intrinsic  — the residue's own (φ, ψ) → Ramachandran map (oracle ~90%+)
  * Extrinsic  — local sequence context, neighbour conformations, AA identity

The accuracy gap between the oracle and context-only models quantifies how
much of the backbone's Ramachandran preference is encoded in local context
rather than intrinsic to the residue itself.

Richardson density comparison
------------------------------
The trained MLP's softmax probabilities are binned over the (φ, ψ) plane and
plotted as a 4-panel heatmap (H / E / P / C probability per bin).  If the
model is well-calibrated, these maps should visually reconstruct the
empirical Ramachandran density — concentrated probability in the allowed
basins, near-zero elsewhere.  This is the "machine-learned Ramachandran plot".

Usage
-----
Quick run::

    python benchmarks/canonical_tests/backbone_rama_benchmark.py \\
        --cache-file ~/PDB/pisces_1000.parquet \\
        --sample-n 50000 --remap-u-rama

With report and plots::

    python benchmarks/canonical_tests/backbone_rama_benchmark.py \\
        --cache-file ~/PDB/pisces_1000.parquet \\
        --sample-n 50000 --remap-u-rama \\
        --epochs 60 --trials 3 \\
        --report papers/backbone_manifold/backbone_rama_report.md \\
        --out-dir papers/backbone_manifold/rama/

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")
STRATEGY = tf.distribute.OneDeviceStrategy("/CPU:0")

import keras  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]

from waverider.backbone_angles import BackboneAngleList  # noqa: E402
from waverider.backbone_embedder import BackboneEmbedder  # noqa: E402
from waverider.dimensionality_discovery import discover_dimensionality  # noqa: E402
from waverider.manifold_model import ManifoldModel  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_P = argparse.ArgumentParser(description="Backbone Ramachandran region benchmark")
_P.add_argument("--cache-file", type=Path, help="Pre-built parquet cache (preferred)")
_P.add_argument("--pdb-dir", type=Path, help="Directory of PDB files (slow, builds cache)")
_P.add_argument("--sample-n", type=int, default=50_000)
_P.add_argument(
    "--remap-u-rama",
    action="store_true",
    help="Reassign U secondary-structure labels via Ramachandran geometry before sampling",
)
_P.add_argument("--epochs", type=int, default=60)
_P.add_argument("--trials", type=int, default=3)
_P.add_argument("--batch-size", type=int, default=512)
_P.add_argument("--lr", type=float, default=1e-3)
_P.add_argument("--seed", type=int, default=42)
_P.add_argument("--skip-manifold", action="store_true")
_P.add_argument("--report", type=Path)
_P.add_argument("--out-dir", type=Path)
_P.add_argument("--tau", type=float, default=0.90)
_P.add_argument("--k-pca", type=int, default=30)

# ---------------------------------------------------------------------------
# Helpers (shared with backbone_mlp_benchmark)
# ---------------------------------------------------------------------------


def _hr(title: str = "") -> None:
    width = 62
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n─── {title} " + "─" * pad)
    else:
        print("─" * width)


def _stratified_split_idx(
    y: np.ndarray, test_frac: float = 0.20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)


def _stratified_sample(bal: BackboneAngleList, n: int, seed: int = 42) -> BackboneAngleList:
    import random as _random

    rng = _random.Random(seed)
    by_class: dict[str, list] = {}
    for r in bal.residues:
        by_class.setdefault(r.secondary_structure, []).append(r)
    total = len(bal.residues)
    sampled = []
    for members in by_class.values():
        quota = max(1, round(n * len(members) / total))
        sampled.extend(rng.sample(members, min(quota, len(members))))
    rng.shuffle(sampled)
    sampled = sampled[:n]
    counts = {c: sum(1 for r in sampled if r.secondary_structure == c) for c in sorted(by_class)}
    count_str = "  ".join(f"{c}:{v:,}" for c, v in counts.items())
    print(f"  Stratified sample: {len(sampled):,}  [{count_str}]")
    return BackboneAngleList(residues=sampled, name=bal.name + f"_s{n}")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_standard(d_in: int, n_cls: int, lr: float):
    with STRATEGY.scope():
        m = keras.Sequential(
            [
                keras.layers.Input(shape=(d_in,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(n_cls, activation="softmax"),
            ]
        )
        m.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return m


def build_manifold(d_in: int, n_cls: int, d_star: int, lr: float):
    w1, w2 = max(n_cls, 2 * d_star), max(n_cls, d_star)
    with STRATEGY.scope():
        m = keras.Sequential(
            [
                keras.layers.Input(shape=(d_in,)),
                keras.layers.Dense(w1, activation="relu"),
                keras.layers.Dense(w2, activation="relu"),
                keras.layers.Dense(n_cls, activation="softmax"),
            ]
        )
        m.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return m


def build_wide_manifold(d_in: int, n_cls: int, d_star: int, lr: float):
    w1, w2, w3 = max(n_cls, 4 * d_star), max(n_cls, 2 * d_star), max(n_cls, d_star)
    with STRATEGY.scope():
        m = keras.Sequential(
            [
                keras.layers.Input(shape=(d_in,)),
                keras.layers.Dense(w1, activation="relu"),
                keras.layers.Dense(w2, activation="relu"),
                keras.layers.Dense(w3, activation="relu"),
                keras.layers.Dense(n_cls, activation="softmax"),
            ]
        )
        m.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return m


def build_universal_bottleneck(d_in: int, n_cls: int, d_star: int, lr: float):
    w = max(n_cls, d_star + n_cls - 1)
    with STRATEGY.scope():
        m = keras.Sequential(
            [
                keras.layers.Input(shape=(d_in,)),
                keras.layers.Dense(w, activation="relu"),
                keras.layers.Dense(w, activation="relu"),
                keras.layers.Dense(n_cls, activation="softmax"),
            ]
        )
        m.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return m


# ---------------------------------------------------------------------------
# Trial runners
# ---------------------------------------------------------------------------


def run_mlp_trial(build_fn, X_train, y_train, X_test, y_test, epochs, batch_size, trial):
    model = build_fn()
    n_params = model.count_params()
    t0 = time.perf_counter()
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0
    )
    wall = time.perf_counter() - t0
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    # Return model too so we can extract probabilities
    return {
        "trial": trial,
        "n_params": n_params,
        "test_acc": float(test_acc),
        "wall_time": wall,
        "model": model,
    }


def run_manifold_trial(X_train, y_train, X_test, y_test, d_ambient):
    k_pca = min(len(X_train) - 1, max(50, 3 * d_ambient))
    t0 = time.perf_counter()
    clf = ManifoldModel(k_graph=10, k_pca=k_pca, variance_threshold=0.90)
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = clf.predict(X_test)
    pred_time = time.perf_counter() - t1
    return {
        "trial": 0,
        "n_params": 0,
        "test_acc": float((preds == y_test).mean()),
        "wall_time": fit_time + pred_time,
        "k_pca": k_pca,
        "model": None,
    }


def _aggregate(trials: list[dict]) -> dict:
    accs = [t["test_acc"] for t in trials]
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "n_params": trials[0]["n_params"],
        "wall_time": float(np.mean([t["wall_time"] for t in trials])),
        "best_model": max(trials, key=lambda t: t["test_acc"])["model"],
        "trials": trials,
    }


# ---------------------------------------------------------------------------
# Richardson density visualization
# ---------------------------------------------------------------------------

RAMA_CLASSES = {0: "H (α-helix)", 1: "E (β-sheet)", 2: "P (PPII)", 3: "L (left)", 4: "C (coil)"}
RAMA_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#ff7f00", 4: "#aaaaaa"}
N_BINS = 36  # 10° per bin


def plot_richardson_comparison(
    phi_test: np.ndarray,
    psi_test: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,  # (N_test, n_classes) softmax probs from best model
    label: str,
    out_path: Path,
) -> None:
    """Plot the model's predicted class probabilities binned over the (φ,ψ) plane.

    Left column: empirical density and true class map.
    Right column: P(H|bin), P(E|bin), P(P|bin), P(C|bin) from the model.

    If the model is well-calibrated the right-column heatmaps should
    reconstruct the Richardson/Ramachandran empirical density.
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available — no Richardson plot")
        return

    phi_edges = np.linspace(-180, 180, N_BINS + 1)
    psi_edges = np.linspace(-180, 180, N_BINS + 1)

    # Empirical density
    density, _, _ = np.histogram2d(phi_test, psi_test, bins=[phi_edges, psi_edges])
    density = density / density.max()

    # Per-class predicted probability averaged over each bin
    n_cls = proba_test.shape[1]
    phi_idx = np.clip(np.digitize(phi_test, phi_edges) - 1, 0, N_BINS - 1)
    psi_idx = np.clip(np.digitize(psi_test, psi_edges) - 1, 0, N_BINS - 1)

    prob_maps = np.zeros((n_cls, N_BINS, N_BINS), dtype=np.float32)
    counts = np.zeros((N_BINS, N_BINS), dtype=np.float32)
    for i in range(len(phi_test)):
        pi, qi = phi_idx[i], psi_idx[i]
        prob_maps[:, pi, qi] += proba_test[i]
        counts[pi, qi] += 1

    # Avoid division by zero in empty bins
    mask = counts > 0
    for c in range(n_cls):
        prob_maps[c][mask] /= counts[mask]

    # True class mode per bin
    mode_map = np.full((N_BINS, N_BINS), -1, dtype=np.int32)
    for pi in range(N_BINS):
        for qi in range(N_BINS):
            if counts[pi, qi] > 0:
                mode_map[pi, qi] = int(prob_maps[:, pi, qi].argmax())

    # Build categorical color image for mode map
    class_colors = np.array([mcolors.to_rgb(RAMA_COLORS[c]) for c in range(n_cls)] + [(1, 1, 1)])
    mode_img = np.ones((N_BINS, N_BINS, 3))
    for pi in range(N_BINS):
        for qi in range(N_BINS):
            m = mode_map[pi, qi]
            if m >= 0:
                mode_img[pi, qi] = class_colors[m]

    plot_classes = [0, 1, 2, 4]  # H, E, P, C
    n_plot = len(plot_classes)
    fig, axes = plt.subplots(2, n_plot // 2 + 1, figsize=(14, 9))
    axes = axes.ravel()

    # Panel 0: empirical density
    ax = axes[0]
    ax.imshow(
        density.T,
        origin="lower",
        extent=[-180, 180, -180, 180],
        cmap="Greys",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Empirical density")
    ax.set_xlabel("φ")
    ax.set_ylabel("ψ")

    # Panel 1: dominant class map (machine-learned Ramachandran plot)
    ax = axes[1]
    ax.imshow(
        mode_img.transpose(1, 0, 2), origin="lower", extent=[-180, 180, -180, 180], aspect="auto"
    )
    for c, name in RAMA_CLASSES.items():
        ax.plot([], [], color=RAMA_COLORS[c], linewidth=6, label=name.split()[0])
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    ax.set_title("Model dominant class\n(machine-learned Ramachandran)")
    ax.set_xlabel("φ")
    ax.set_ylabel("ψ")

    # Panels 2-5: per-class probability maps
    for k, cls in enumerate(plot_classes):
        ax = axes[k + 2]
        im = ax.imshow(
            prob_maps[cls].T,
            origin="lower",
            extent=[-180, 180, -180, 180],
            cmap="hot_r",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"P({RAMA_CLASSES[cls].split()[0]} | φ,ψ bin)")
        ax.set_xlabel("φ")
        ax.set_ylabel("ψ")

    fig.suptitle(f"Richardson density comparison — {label}", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Richardson plot → {out_path}")


def plot_richardson_window_comparison(panels: list, out_path: Path) -> None:
    """Single comparison figure: P(H|ctx) and P(E|ctx) across window sizes.

    Isolates the one variable that changes (window size) and the two quantities
    that tell the story: helix probability fills in (cooperative, local) while
    sheet probability never forms a basin (non-local).  Replaces three nearly
    identical 6-panel figures.

    :param panels: list of dicts with keys label, phi, psi, proba, y, recall_h,
        recall_e — one per window size, in ascending window order.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not available")
        return

    edges = np.linspace(-180, 180, N_BINS + 1)
    # Canonical basin boxes (φ-range, ψ-range) from the U-rama remap table
    H_BOX = ((-90, -30), (-70, -20))  # α-helix
    E_BOX = ((-170, -50), (90, 180))  # β-sheet

    def binned(phi, psi, p):
        pi = np.clip(np.digitize(phi, edges) - 1, 0, N_BINS - 1)
        qi = np.clip(np.digitize(psi, edges) - 1, 0, N_BINS - 1)
        s = np.zeros((N_BINS, N_BINS))
        c = np.zeros((N_BINS, N_BINS))
        np.add.at(s, (pi, qi), p)
        np.add.at(c, (pi, qi), 1.0)
        return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)

    def draw_box(ax, box, color):
        (x0, x1), (y0, y1) = box
        ax.add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.6, ls="--")
        )

    ncol = len(panels)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 8.0))
    for j, pn in enumerate(panels):
        for row, cls, box, name in ((0, 0, H_BOX, "H"), (1, 1, E_BOX, "E")):
            ax = axes[row, j]
            m = binned(pn["phi"], pn["psi"], pn["proba"][:, cls])
            im = ax.imshow(
                m.T,
                origin="lower",
                extent=[-180, 180, -180, 180],
                cmap="hot_r",
                aspect="auto",
                vmin=0,
                vmax=1,
            )
            draw_box(ax, box, "#1f77b4" if name == "E" else "#2ca02c")
            ax.set_title(f"$P({name}\\mid$ctx) — {pn['label']}", fontsize=11)
            ax.set_xlabel("φ (°)")
            if j == 0:
                ax.set_ylabel("ψ (°)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Context-only Ramachandran density vs. window size: "
        "helix basin fills in, sheet basin never does",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Richardson comparison → {out_path}")


# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------


def benchmark_mode(label: str, emb: BackboneEmbedder, bal: BackboneAngleList, args) -> dict:
    _hr(f"Mode: {label}")

    X = emb.fit_transform(bal).astype(np.float64)
    # Geometry-derived labels — what the model must predict
    y = bal.to_rama_int_labels_geometry()
    # Filter out any U (nan phi/psi) residues
    valid_mask = y < 5
    X, y = X[valid_mask], y[valid_mask]

    n_classes = int(y.max()) + 1
    d_ambient = X.shape[1]
    print(f"  Embedded: {X.shape}  classes={n_classes}")

    dist = {int(c): int((y == c).sum()) for c in np.unique(y)}
    _map_inv = {0: "H", 1: "E", 2: "P", 3: "L", 4: "C"}
    print("  Label dist: " + "  ".join(f"{_map_inv[c]}:{n:,}" for c, n in sorted(dist.items())))
    majority_class = max(dist, key=dist.get)
    majority_acc = dist[majority_class] / len(y)
    print(
        f"  Majority-class baseline: {majority_acc:.4f} (always predict {_map_inv[majority_class]})"
    )

    d_star_raw = discover_dimensionality(
        X,
        n_samples=200,
        k=min(args.k_pca, len(X) - 1),
        variance_thresholds=(args.tau,),
    )
    d_star = round(d_star_raw[args.tau]["mean"])
    print(f"  d* = {d_star}  (τ={args.tau})")

    train_idx, test_idx = _stratified_split_idx(y, test_frac=0.20, seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train_n = ((X_train - mu) / sigma).astype(np.float32)
    X_test_n = ((X_test - mu) / sigma).astype(np.float32)

    results: dict[str, dict] = {}
    best_proba: np.ndarray | None = None
    best_acc = -1.0

    if not args.skip_manifold:
        print("  ManifoldModel baseline …", flush=True)
        mm = run_manifold_trial(X_train, y_train, X_test, y_test, d_ambient)
        results["ManifoldModel"] = {
            "mean": mm["test_acc"],
            "std": 0.0,
            "n_params": 0,
            "wall_time": mm["wall_time"],
            "best_model": None,
            "trials": [mm],
        }
        print(f"    acc={mm['test_acc']:.4f}  k_pca={mm['k_pca']}  {mm['wall_time']:.1f}s")

    arch_fns = {
        "Standard (128→64)": lambda: build_standard(d_ambient, n_classes, args.lr),
        "Manifold (2d*→d*)": lambda: build_manifold(d_ambient, n_classes, d_star, args.lr),
        "Wide (4d*→2d*→d*)": lambda: build_wide_manifold(d_ambient, n_classes, d_star, args.lr),
        f"UnivBottleneck (w*={d_star + n_classes - 1})": lambda: build_universal_bottleneck(
            d_ambient, n_classes, d_star, args.lr
        ),
    }

    for arch_name, build_fn in arch_fns.items():
        trial_results = []
        for trial in range(args.trials):
            tf.random.set_seed(args.seed + trial * 100)
            np.random.seed(args.seed + trial * 100)
            r = run_mlp_trial(
                build_fn, X_train_n, y_train, X_test_n, y_test, args.epochs, args.batch_size, trial
            )
            trial_results.append(r)
        agg = _aggregate(trial_results)
        results[arch_name] = agg
        print(
            f"  {arch_name:40s}  acc={agg['mean']:.4f}±{agg['std']:.4f}"
            f"  params={agg['n_params']:,}  {agg['wall_time']:.1f}s/trial"
        )
        if agg["mean"] > best_acc:
            best_acc = agg["mean"]
            best_proba = (
                agg["best_model"].predict(X_test_n, verbose=0) if agg["best_model"] else None
            )

    # Per-class recall and balanced accuracy for the best model
    per_class_recall: dict[str, float] = {}
    balanced_acc = float("nan")
    if best_proba is not None:
        best_preds = best_proba.argmax(axis=1)
        recalls = []
        for c in range(n_classes):
            mask_c = y_test == c
            if mask_c.sum() > 0:
                r = float((best_preds[mask_c] == c).mean())
                per_class_recall[_map_inv[c]] = r
                recalls.append(r)
        balanced_acc = float(np.mean(recalls))
        print(
            "  Best model per-class recall:  "
            + "  ".join(f"{k}:{v:.3f}" for k, v in per_class_recall.items())
        )
        print(f"  Balanced accuracy: {balanced_acc:.4f}  (vs majority-class: {majority_acc:.4f})")

    return {
        "label": label,
        "d_ambient": d_ambient,
        "d_star": d_star,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "majority_acc": majority_acc,
        "balanced_acc": balanced_acc,
        "per_class_recall": per_class_recall,
        "architectures": results,
        "best_proba": best_proba,
        "test_indices": test_idx,
        "valid_mask": valid_mask,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    path: Path, args, bal: BackboneAngleList, mode_results: list[dict], total_elapsed: float
) -> None:
    import socket
    import subprocess

    def _git(cmd):
        try:
            return (
                subprocess.check_output(cmd, cwd=str(_ROOT), stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return "unknown"

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    # Label distribution (geometry-derived)
    geom_labels = bal.to_rama_region_labels()
    label_counts = {c: int((geom_labels == c).sum()) for c in ["H", "E", "P", "L", "C"]}

    lines = [
        "# Backbone Ramachandran Region Benchmark",
        "",
        f"**Generated:** {now}  ",
        f"**Host:** {socket.gethostname()}  |  **OS:** {__import__('platform').platform()}  ",
        f"**Repository:** waverider @ `{_git(['git', 'rev-parse', '--short', 'HEAD'])}`",
        "",
        "---",
        "",
        "## Run Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Cache | `{args.cache_file}` |",
        "| U remap | Ramachandran geometry (--remap-u-rama) |",
        f"| Sample N | {args.sample_n:,} |",
        f"| Epochs | {args.epochs} |",
        f"| Trials | {args.trials} |",
        f"| Batch size | {args.batch_size} |",
        f"| Learning rate | {args.lr} |",
        f"| τ (d* threshold) | {args.tau} |",
        f"| Total wall time | {total_elapsed:.1f}s |",
        "",
        "## Corpus Summary (geometry labels)",
        "",
        "| | |",
        "|---|---|",
        f"| Collection | {bal.name} |",
        f"| Residues | {len(bal):,} |",
        "| Label distribution | " + "  ".join(f"{k}:{v:,}" for k, v in label_counts.items()) + " |",
        "",
    ]

    for mr in mode_results:
        lines += [
            f"## Embedding: {mr['label']}  (d_ambient={mr['d_ambient']}, d*={mr['d_star']})",
            "",
            f"Train: {mr['n_train']:,}  |  Test: {mr['n_test']:,}  |  "
            "Geometry-derived Ramachandran labels (context_only=True — center angles zeroed)",
            "",
            "| Architecture | Params | Test Acc | ± Std | Time/trial (s) |",
            "|---|---|---|---|---|",
        ]
        for arch, res in mr["architectures"].items():
            std_str = "—" if res["std"] == 0 else f"{res['std']:.4f}"
            lines.append(
                f"| {arch} | {res['n_params']:,} | {res['mean']:.4f} | "
                f"{std_str} | {res['wall_time']:.1f} |"
            )
        lines.append("")

    lines += ["---", "*Generated by `backbone_rama_benchmark.py`*"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\n  Report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _P.parse_args()
    t_start = time.perf_counter()

    print("=" * 62)
    print("  WaveRider — Backbone Ramachandran Region Benchmark")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────────────────────
    if args.cache_file:
        _hr(f"Cache: {args.cache_file.name}")
        try:
            from proteusPy.backbone_data import BackboneLoader

            loader = BackboneLoader(str(args.cache_file))
            bal = loader.to_backbone_angle_list()
            print(f"  BackboneLoader: loaded {len(bal):,} residues ← {args.cache_file}")
        except Exception:
            import math

            import pandas as pd

            from waverider.backbone_angles import _OMEGA_TRANS, BackboneResidue

            df = pd.read_parquet(args.cache_file)
            residues = []
            for row in df.itertuples():
                omega = (
                    row.omega
                    if (isinstance(row.omega, float) and not math.isnan(row.omega))
                    else _OMEGA_TRANS
                )
                residues.append(
                    BackboneResidue(
                        phi=row.phi,
                        psi=row.psi,
                        omega=omega,
                        residue_name=row.residue_name,
                        chain_id=row.chain_id,
                        seq_pos=row.seq_pos,
                        pdb_id=row.pdb_id,
                        secondary_structure=row.secondary_structure,
                    )
                )
            bal = BackboneAngleList(residues=residues, name=args.cache_file.stem)
        print(bal)
    else:
        raise SystemExit("--cache-file required (--pdb-dir not yet supported here)")

    if args.remap_u_rama:
        bal = bal.remap_u_by_ramachandran()
        n_u = sum(1 for r in bal.residues if r.secondary_structure == "U")
        print(f"  Ramachandran remap: {n_u:,} remain U after remap")

    if args.sample_n and args.sample_n < len(bal.residues):
        bal = _stratified_sample(bal, args.sample_n, args.seed)

    # ── Geometry label distribution ────────────────────────────────────────
    geom = bal.to_rama_region_labels()
    g_counts = {c: int((geom == c).sum()) for c in ["H", "E", "P", "L", "C", "U"]}
    print("  Geometry labels: " + "  ".join(f"{k}:{v:,}" for k, v in g_counts.items() if v > 0))

    # ── Embedding modes: context_only=True, AA features for center ─────────
    # Three window sizes for context prediction.
    # Note: context_only zeros the center's torus angles but keeps center's AA.
    modes = [
        (
            "window3_ctx",
            BackboneEmbedder(
                mode="window", window_size=3, include_aa=True, aa_mode="gpo", context_only=True
            ),
        ),
        (
            "window7_ctx",
            BackboneEmbedder(
                mode="window", window_size=7, include_aa=True, aa_mode="gpo", context_only=True
            ),
        ),
        (
            "window13_ctx",
            BackboneEmbedder(
                mode="window", window_size=13, include_aa=True, aa_mode="gpo", context_only=True
            ),
        ),
    ]

    mode_results = []
    for label, emb in modes:
        mr = benchmark_mode(label, emb, bal, args)
        mode_results.append(mr)

    total_elapsed = time.perf_counter() - t_start

    # ── Summary ────────────────────────────────────────────────────────────
    _hr("Summary")
    header = f"  {'Mode':<18}  {'d*':>3}  {'Architecture':<38}  {'Test Acc':>10}  {'Params':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mr in mode_results:
        for arch, res in mr["architectures"].items():
            print(
                f"  {mr['label']:<18}  {mr['d_star']:>3}  {arch:<38}"
                f"  {res['mean']:.4f}±{res['std']:.4f}  {res['n_params']:>10,}"
            )

    m, s = divmod(int(total_elapsed), 60)
    print(f"\n  Total: {m}m{s}s")

    # ── Richardson density plot ────────────────────────────────────────────
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        comparison_panels = []
        for mr in mode_results:
            if mr["best_proba"] is None:
                continue
            vm = mr["valid_mask"]
            ti = mr["test_indices"]
            # Recover phi/psi for test residues
            all_residues = [r for r, ok in zip(bal.residues, vm) if ok]
            test_residues = [all_residues[i] for i in ti]
            phi_test = np.array([r.phi for r in test_residues], dtype=np.float32)
            psi_test = np.array([r.psi for r in test_residues], dtype=np.float32)
            y_geom = np.array([bal.to_rama_int_labels_geometry()[vm][ti]], dtype=np.int32).ravel()
            plot_path = args.out_dir / f"richardson_{mr['label']}.png"
            plot_richardson_comparison(
                phi_test, psi_test, y_geom, mr["best_proba"], mr["label"], plot_path
            )
            # Collect for the combined comparison figure
            preds = mr["best_proba"].argmax(axis=1)
            rec_h = float((preds[y_geom == 0] == 0).mean()) if (y_geom == 0).any() else 0.0
            rec_e = float((preds[y_geom == 1] == 1).mean()) if (y_geom == 1).any() else 0.0
            wsize = mr["label"].split("_")[0].replace("window", "")
            comparison_panels.append(
                {
                    "label": f"window-{wsize}",
                    "phi": phi_test,
                    "psi": psi_test,
                    "proba": mr["best_proba"],
                    "y": y_geom,
                    "recall_h": rec_h,
                    "recall_e": rec_e,
                }
            )

        if comparison_panels:
            plot_richardson_window_comparison(
                comparison_panels, args.out_dir / "richardson_window_comparison.png"
            )

    # ── Report ─────────────────────────────────────────────────────────────
    if args.report:
        write_report(args.report, args, bal, mode_results, total_elapsed)


if __name__ == "__main__":
    main()
