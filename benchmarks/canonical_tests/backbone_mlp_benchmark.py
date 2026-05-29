#!/usr/bin/env python3
"""
Backbone MLP Benchmark
======================

Compares manifold-informed MLP architectures against the ManifoldModel
(geometric k-NN) baseline on protein secondary structure classification.

Motivation
----------
ManifoldModel achieves 85.3% test accuracy on the torus embedding of backbone
dihedral angles (φ, ψ) but underperforms on larger window embeddings because
it is a local-geometry estimator, not a sequence model.  This benchmark tests
whether a small MLP — with its bottleneck width set to the intrinsic dimension
d* discovered by WaveRider — can:

  1. Match or exceed ManifoldModel on the torus (the per-residue ceiling)
  2. Outperform ManifoldModel on window embeddings (where context adds value)
  3. Do so with far fewer parameters than a standard fixed-width MLP

Embedding modes
---------------
  torus     4-D  (cos φ, sin φ, cos ψ, sin ψ)        d*=2
  discrete 16-D  8-fold quantization → 16-D lookup    d*=2
  window7  28-D  7-residue sliding torus window        d*=9
  window13 52-D  13-residue sliding torus window       d*=16

Architecture families
---------------------
  Standard         input → 128 → 64 → C
  Manifold (2d→d)  input → 2d* → d* → C          (bottleneck at d*)
  Wide Manifold    input → 4d* → 2d* → d* → C
  Univ. Bottleneck input → w* → w* → C            w* = d* + C − 1
  ManifoldModel    geometric k-NN (non-parametric, no training)

Usage
-----
Quick run from cache::

    python benchmarks/canonical_tests/backbone_mlp_benchmark.py \\
        --cache-file ~/PDB/pisces_1000.parquet \\
        --sample-n 50000 \\
        --remap-u-rama

Full run with report::

    python benchmarks/canonical_tests/backbone_mlp_benchmark.py \\
        --cache-file ~/PDB/pisces_1000.parquet \\
        --sample-n 50000 \\
        --remap-u-rama \\
        --epochs 50 --trials 3 \\
        --report papers/backbone_manifold/backbone_mlp_report.md

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Force CPU — Metal gradient issues on M-series with small batches
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")
STRATEGY = tf.distribute.OneDeviceStrategy("/CPU:0")

import keras  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]

from waverider.backbone_angles import BackboneAngleList  # noqa: E402
from waverider.backbone_embedder import BackboneEmbedder  # noqa: E402
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
)
from waverider.manifold_model import ManifoldModel  # noqa: E402

try:
    from proteusPy.backbone_loader import BackboneLoader
except ImportError:
    BackboneLoader = None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_P = argparse.ArgumentParser(description="Backbone MLP vs ManifoldModel benchmark")
_P.add_argument("--cache-file", type=Path, help="Parquet cache (proteusPy BackboneLoader format)")
_P.add_argument("--pdb-dir", type=Path, help="Directory of pdb*.ent files")
_P.add_argument("--sample-n", type=int, default=50_000, help="Stratified subsample size")
_P.add_argument("--remap-u-rama", action="store_true", help="Reclassify U by Ramachandran geometry")
_P.add_argument("--remap-u-to-coil", action="store_true", help="Reclassify U → C")
_P.add_argument("--epochs", type=int, default=40, help="Training epochs per trial")
_P.add_argument("--trials", type=int, default=3, help="Independent random seeds per architecture")
_P.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
_P.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
_P.add_argument("--seed", type=int, default=42, help="Base random seed")
_P.add_argument("--skip-manifold", action="store_true", help="Skip ManifoldModel baseline (faster)")
_P.add_argument(
    "--include-aa", action="store_true", help="Append amino acid type features to every embedding"
)
_P.add_argument(
    "--aa-mode",
    choices=["gpo", "onehot", "phys"],
    default="gpo",
    help="AA encoding: 'gpo' = Gly/Pro/Other (3-D), 'phys' = GPO+hphob (4-D), 'onehot' = full 20-D",
)
_P.add_argument(
    "--include-omega",
    action="store_true",
    help="Append (cos ω, sin ω) peptide bond planarity to each residue embedding (+2 dims per position)",
)
_P.add_argument("--report", type=Path, help="Write Markdown report to this path")
_P.add_argument("--out-dir", type=Path, help="Directory for plot output (backbone_mlp_results.png)")
_P.add_argument("--tau", type=float, default=0.90, help="Variance threshold for d* (default 0.90)")
_P.add_argument("--k-pca", type=int, default=30, help="k for dimensionality discovery (default 30)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SS_NAMES = {
    0: "α-helix (H)",
    1: "β-sheet (E)",
    2: "PPII (P)",
    3: "left-helix (L)",
    4: "coil (C)",
    5: "unknown (U)",
}


def _hr(title: str = "") -> None:
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, 58 - len(title))}")
    else:
        print("─" * 62)


def _stratified_sample(bal: BackboneAngleList, n: int, seed: int) -> BackboneAngleList:
    import random as _r

    rng = _r.Random(seed)
    by_class: dict = {}
    for r in bal.residues:
        by_class.setdefault(r.secondary_structure, []).append(r)
    sampled = []
    total = len(bal.residues)
    for members in by_class.values():
        quota = max(1, round(n * len(members) / total))
        sampled.extend(rng.sample(members, min(quota, len(members))))
    rng.shuffle(sampled)
    sampled = sampled[:n]
    counts = {c: sum(1 for r in sampled if r.secondary_structure == c) for c in sorted(by_class)}
    print(
        "  Stratified sample: {:,}  [{}]".format(
            len(sampled), "  ".join(f"{c}:{v:,}" for c, v in counts.items())
        )
    )
    return BackboneAngleList(residues=sampled, name=bal.name + f"_s{n}")


def _stratified_split_idx(y: np.ndarray, test_frac: float = 0.20, seed: int = 42):
    import random as _r

    rng = _r.Random(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].tolist()
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def count_params(model) -> int:
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _compile(model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def build_standard(input_dim: int, n_classes: int, lr: float = 1e-3):
    """Standard baseline: input → 128 → 64 → C."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(n_classes),
        ]
    )
    return _compile(model, lr)


def build_manifold(input_dim: int, n_classes: int, d_star: int, lr: float = 1e-3):
    """Manifold-informed: input → 2d* → d* → C.

    Bottleneck width equals the discovered intrinsic dimension.  The wider
    preceding layer gives the network room to learn the projection.
    """
    d = max(d_star, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes),
        ]
    )
    return _compile(model, lr)


def build_wide_manifold(input_dim: int, n_classes: int, d_star: int, lr: float = 1e-3):
    """Wide manifold: input → 4d* → 2d* → d* → C."""
    d = max(d_star, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(4 * d, activation="relu"),
            keras.layers.Dense(2 * d, activation="relu"),
            keras.layers.Dense(d, activation="relu"),
            keras.layers.Dense(n_classes),
        ]
    )
    return _compile(model, lr)


def build_universal_bottleneck(input_dim: int, n_classes: int, d_star: int, lr: float = 1e-3):
    """Universal Bottleneck: input → w* → w* → C  where w* = d* + C − 1."""
    w = max(d_star + n_classes - 1, n_classes)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(w, activation="relu"),
            keras.layers.Dense(n_classes),
        ]
    )
    return _compile(model, lr)


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_mlp_trial(
    build_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    trial: int,
) -> dict:
    """Train one MLP and return metrics."""
    with STRATEGY.scope():
        model = build_fn()
    n_params = count_params(model)
    t0 = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    wall = time.perf_counter() - t0
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    conv = next((i for i, a in enumerate(history.history["accuracy"]) if a >= 0.90), None)
    return {
        "trial": trial,
        "n_params": n_params,
        "test_acc": float(test_acc),
        "wall_time": wall,
        "convergence_epoch": conv,
        "train_acc": [float(a) for a in history.history["accuracy"]],
        "val_acc": [float(a) for a in history.history["val_accuracy"]],
    }


def run_manifold_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    d_ambient: int,
) -> dict:
    """Fit ManifoldModel on train, evaluate on test."""
    k_pca = min(len(X_train) - 1, max(50, 3 * d_ambient))
    t0 = time.perf_counter()
    clf = ManifoldModel(k_graph=10, k_pca=k_pca, variance_threshold=0.90)
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    preds = clf.predict(X_test)
    pred_time = time.perf_counter() - t1
    acc = float((preds == y_test).mean())
    return {
        "trial": 0,
        "n_params": 0,
        "test_acc": acc,
        "wall_time": fit_time + pred_time,
        "k_pca": k_pca,
        "convergence_epoch": None,
    }


# ---------------------------------------------------------------------------
# Per-mode benchmark
# ---------------------------------------------------------------------------


def _aggregate(trials: list[dict]) -> dict:
    accs = [t["test_acc"] for t in trials]
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "n_params": trials[0]["n_params"],
        "wall_time": float(np.mean([t["wall_time"] for t in trials])),
        "trials": trials,
    }


def benchmark_mode(
    label: str,
    emb: BackboneEmbedder,
    bal: BackboneAngleList,
    args,
) -> dict:
    _hr(f"Mode: {label}")

    # Embed
    X = emb.fit_transform(bal).astype(np.float64)
    y = bal.to_ss_int_labels()
    n_classes = len(np.unique(y))
    d_ambient = X.shape[1]
    print(f"  Embedded: {X.shape}  classes={n_classes}")

    # Discover d*
    d_star_raw = discover_dimensionality(
        X,
        n_samples=200,
        k=min(args.k_pca, len(X) - 1),
        variance_thresholds=(args.tau,),
    )
    d_star = round(d_star_raw[args.tau]["mean"])
    print(f"  d* = {d_star}  (τ={args.tau})")

    # Stratified 80/20 split
    train_idx, test_idx = _stratified_split_idx(y, test_frac=0.20, seed=args.seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Normalize (fit on train only)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    X_train_n = ((X_train - mu) / sigma).astype(np.float32)
    X_test_n = ((X_test - mu) / sigma).astype(np.float32)

    results: dict[str, dict] = {}

    # ManifoldModel baseline
    if not args.skip_manifold:
        print("  ManifoldModel baseline …", flush=True)
        mm_result = run_manifold_trial(X_train, y_train, X_test, y_test, d_ambient)
        results["ManifoldModel"] = {
            "mean": mm_result["test_acc"],
            "std": 0.0,
            "n_params": 0,
            "wall_time": mm_result["wall_time"],
            "trials": [mm_result],
        }
        print(
            f"    acc={mm_result['test_acc']:.4f}  k_pca={mm_result['k_pca']}  {mm_result['wall_time']:.1f}s"
        )

    # MLP architectures
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
                build_fn,
                X_train_n,
                y_train,
                X_test_n,
                y_test,
                args.epochs,
                args.batch_size,
                trial,
            )
            trial_results.append(r)
        agg = _aggregate(trial_results)
        results[arch_name] = agg
        print(
            f"  {arch_name:35s}  acc={agg['mean']:.4f}±{agg['std']:.4f}"
            f"  params={agg['n_params']:,}  {agg['wall_time']:.1f}s/trial"
        )

    return {
        "label": label,
        "d_ambient": d_ambient,
        "d_star": d_star,
        "n_classes": n_classes,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "architectures": results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    path: Path, args, bal: BackboneAngleList, mode_results: list[dict], total_elapsed: float
) -> None:
    import platform
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
    ss_counts: dict = {}
    for r in bal.residues:
        ss_counts[r.secondary_structure] = ss_counts.get(r.secondary_structure, 0) + 1
    ss_str = "  ".join(f"{k}:{v:,}" for k, v in sorted(ss_counts.items()))

    remap = (
        "Ramachandran geometry (--remap-u-rama)"
        if args.remap_u_rama
        else "U→C (--remap-u-to-coil)"
        if args.remap_u_to_coil
        else "none"
    )

    lines = [
        "# Backbone MLP Benchmark",
        "",
        f"**Generated:** {now}  ",
        f"**Host:** {socket.gethostname()}  |  **OS:** {platform.platform()}  ",
        f"**Repository:** waverider @ `{_git(['git', 'rev-parse', '--short', 'HEAD'])}` "
        f"({_git(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])})  ",
        "",
        "---",
        "",
        "## Run Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Cache | `{args.cache_file}` |",
        f"| U remap | {remap} |",
        f"| Sample N | {args.sample_n:,} |",
        f"| Epochs | {args.epochs} |",
        f"| Trials | {args.trials} |",
        f"| Batch size | {args.batch_size} |",
        f"| Learning rate | {args.lr} |",
        f"| τ (d* threshold) | {args.tau} |",
        f"| Total wall time | {total_elapsed:.1f}s |",
        "",
        "## Corpus Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Collection | {bal.name} |",
        f"| Residues | {len(bal.residues):,} |",
        f"| SS distribution | {ss_str} |",
        "",
    ]

    for mr in mode_results:
        label = mr["label"]
        d_ambient = mr["d_ambient"]
        d_star = mr["d_star"]
        n_train = mr["n_train"]
        n_test = mr["n_test"]
        lines += [
            f"## Embedding: {label}  (d_ambient={d_ambient}, d*={d_star})",
            "",
            f"Train: {n_train:,}  |  Test: {n_test:,}  |  Eval: held-out test accuracy (80/20 stratified split)",
            "",
            "| Architecture | Params | Test Acc | ± Std | Time/trial (s) |",
            "|---|---|---|---|---|",
        ]
        for arch_name, agg in mr["architectures"].items():
            std_str = f"{agg['std']:.4f}" if agg["std"] > 0 else "—"
            lines.append(
                f"| {arch_name} | {agg['n_params']:,} | {agg['mean']:.4f} | {std_str} | {agg['wall_time']:.1f} |"
            )
        lines.append("")

    lines += ["---", "*Generated by `backbone_mlp_benchmark.py`*", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\n  Report → {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_results(mode_results: list[dict], save_dir: Path, args, elapsed: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Architecture family → (prefix, color, short label)
    FAMS = [
        ("ManifoldModel", "dimgray", "ManifoldModel"),
        ("Standard", "steelblue", "Standard"),
        ("Manifold", "firebrick", "Manifold"),
        ("Wide", "forestgreen", "Wide"),
        ("UnivBottleneck", "darkorchid", "UnivBN"),
    ]

    def _fam(name: str) -> tuple[str, str, str]:
        for prefix, col, short in FAMS:
            if name.startswith(prefix):
                return prefix, col, short
        return name, "gray", name[:12]

    # Ordered families (derived from first mode so order matches insertion)
    seen: set[str] = set()
    families: list[tuple[str, str, str]] = []
    for arch_name in mode_results[0]["architectures"]:
        prefix, col, short = _fam(arch_name)
        if prefix not in seen:
            families.append((prefix, col, short))
            seen.add(prefix)

    n_fams = len(families)
    n_modes = len(mode_results)
    mode_labels = [mr["label"] for mr in mode_results]
    MARKERS = {"torus": "o", "discrete": "s", "window7": "^", "window13": "D"}

    remap_str = "U→Rama" if args.remap_u_rama else "U→C" if args.remap_u_to_coil else "none"
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.44, wspace=0.30)
    ax_bars = fig.add_subplot(gs[0, :])  # top, full width
    ax_scatter = fig.add_subplot(gs[1, 0])  # bottom-left
    ax_curves = fig.add_subplot(gs[1, 1])  # bottom-right

    fig.suptitle(
        f"Backbone MLP Benchmark  |  N={args.sample_n:,}  "
        f"remap={remap_str}  epochs={args.epochs}  trials={args.trials}  "
        f"elapsed={elapsed / 60:.1f}m",
        fontsize=12,
        fontweight="bold",
    )

    # ── Panel 1: Grouped bar chart ────────────────────────────────────────────
    x = np.arange(n_modes)
    bar_width = 0.80 / n_fams

    fam_data: dict[str, dict] = {p: {"means": [], "stds": []} for p, _, _ in families}
    for mr in mode_results:
        for arch_name, agg in mr["architectures"].items():
            p, _, _ = _fam(arch_name)
            if p in fam_data:
                fam_data[p]["means"].append(agg["mean"])
                fam_data[p]["stds"].append(agg["std"])

    # Compute ylim before drawing so we can skip annotations outside the visible range
    all_accs = [agg["mean"] for mr in mode_results for agg in mr["architectures"].values()]
    y_lo = max(0.60, min(all_accs) - 0.03)
    y_hi = min(1.0, max(all_accs) + 0.04)

    for i, (prefix, col, short) in enumerate(families):
        means = fam_data[prefix]["means"]
        stds = fam_data[prefix]["stds"]
        if len(means) != n_modes:
            continue
        offset = (i - n_fams / 2 + 0.5) * bar_width
        bars = ax_bars.bar(
            x + offset,
            means,
            bar_width * 0.88,
            yerr=stds,
            capsize=3,
            color=col,
            alpha=0.82,
            label=short,
        )
        for bar, m in zip(bars, means):
            if m > y_lo + 0.005:  # only annotate bars visible within ylim
                ax_bars.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(m, y_hi - 0.005) + 0.002,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    fontweight="bold",
                )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(
        [f"{mr['label']}\n(d_amb={mr['d_ambient']}, d*={mr['d_star']})" for mr in mode_results],
        fontsize=10,
    )
    ax_bars.set_ylabel("Test Accuracy", fontsize=10)
    ax_bars.set_title("Test Accuracy: Embedding Mode × Architecture", fontsize=11)
    ax_bars.legend(loc="lower left", fontsize=9, ncol=n_fams)
    ax_bars.grid(True, alpha=0.3, axis="y")
    ax_bars.set_ylim(y_lo, y_hi)

    # ── Panel 2: Efficiency scatter (accuracy vs log₁₀ params) ───────────────
    for mr in mode_results:
        mode = mr["label"]
        marker = MARKERS.get(mode, "o")
        for arch_name, agg in mr["architectures"].items():
            _, col, _ = _fam(arch_name)
            params = agg["n_params"]
            acc = agg["mean"]
            std = agg["std"]
            if params > 0:
                ax_scatter.errorbar(
                    np.log10(params),
                    acc,
                    yerr=std,
                    fmt=marker,
                    color=col,
                    alpha=0.85,
                    markersize=8,
                    capsize=3,
                    elinewidth=1,
                    zorder=3,
                )
            else:
                # ManifoldModel — non-parametric; plot at x=-0.5 with star marker
                ax_scatter.errorbar(
                    -0.5,
                    acc,
                    yerr=std,
                    fmt="*",
                    color=col,
                    alpha=0.85,
                    markersize=12,
                    capsize=3,
                    elinewidth=1,
                    zorder=4,
                )

    arch_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=9, label=short)
        for _, col, short in families
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS.get(m, "o"),
            color="gray",
            markersize=8,
            linestyle="None",
            label=m,
        )
        for m in mode_labels
    ]
    ax_scatter.legend(handles=arch_handles + mode_handles, fontsize=8, loc="lower right", ncol=2)
    ax_scatter.set_xlabel("log₁₀(Parameters)  [★ = ManifoldModel, 0 params]", fontsize=9)
    ax_scatter.set_ylabel("Test Accuracy", fontsize=9)
    ax_scatter.set_title("Efficiency Frontier: Accuracy vs. Parameters", fontsize=10)
    ax_scatter.grid(True, alpha=0.3)

    # ── Panel 3: Val accuracy learning curves (window13 preferred) ────────────
    target_mr = next(
        (mr for mr in reversed(mode_results) if mr["label"] in ("window13", "window7")),
        mode_results[-1],
    )
    ax_curves.set_title(f"{target_mr['label']}: Validation Accuracy per Epoch", fontsize=10)
    for arch_name, agg in target_mr["architectures"].items():
        trials = agg.get("trials", [])
        if not trials or "val_acc" not in trials[0]:
            continue
        _, col, short = _fam(arch_name)
        accs = np.array([t["val_acc"] for t in trials])
        ep = np.arange(1, accs.shape[1] + 1)
        ax_curves.plot(ep, accs.mean(0), "-", label=short, linewidth=2, color=col)
        if len(trials) > 1:
            ax_curves.fill_between(
                ep,
                accs.mean(0) - accs.std(0),
                accs.mean(0) + accs.std(0),
                alpha=0.15,
                color=col,
            )
    ax_curves.set_xlabel("Epoch", fontsize=9)
    ax_curves.set_ylabel("Validation Accuracy", fontsize=9)
    ax_curves.legend(fontsize=9)
    ax_curves.grid(True, alpha=0.3)

    out_path = save_dir / "backbone_mlp_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _P.parse_args()
    t_start = time.perf_counter()

    print("=" * 62)
    print("  WaveRider — Backbone MLP Benchmark")
    print("=" * 62)

    # Load
    if args.cache_file and args.cache_file.exists():
        if BackboneLoader is None:
            raise SystemExit("proteusPy required for --cache-file")
        _hr(f"Cache: {args.cache_file.name}")
        residues = BackboneLoader.load_cache(args.cache_file)
        bal = BackboneAngleList.from_proteuspy(residues, name=args.cache_file.stem)
        print(bal)
        bal = bal.valid()
    elif args.pdb_dir:
        if BackboneLoader is None:
            raise SystemExit("proteusPy required for --pdb-dir")
        loader = BackboneLoader(pdb_dir=args.pdb_dir)
        residues = loader.load()
        bal = BackboneAngleList.from_proteuspy(residues, name=args.pdb_dir.name).valid()
    else:
        raise SystemExit("Provide --cache-file or --pdb-dir")

    # Remap U
    if args.remap_u_rama:
        n_u = sum(1 for r in bal.residues if r.secondary_structure == "U")
        bal = bal.remap_u_by_ramachandran()
        n_still = sum(1 for r in bal.residues if r.secondary_structure == "U")
        print(f"  Ramachandran remap: {n_u:,} U → geometry  ({n_still:,} remain U)")
    elif args.remap_u_to_coil:
        n = sum(1 for r in bal.residues if r.secondary_structure == "U")
        for r in bal.residues:
            if r.secondary_structure == "U":
                r.secondary_structure = "C"
        print(f"  Remapped {n:,} U → C")

    # Stratified sample
    if args.sample_n and args.sample_n < len(bal.residues):
        bal = _stratified_sample(bal, args.sample_n, args.seed)

    # Embedding modes — AA and ω features appended when flags are set
    aa_kw = {"include_aa": args.include_aa, "aa_mode": args.aa_mode}
    omega_kw = {"include_omega": args.include_omega}
    aa_suffix = f"+{args.aa_mode}" if args.include_aa else ""
    omega_suffix = "+omega" if args.include_omega else ""
    suffix = aa_suffix + omega_suffix
    modes = [
        (f"torus{suffix}", BackboneEmbedder(mode="torus", **aa_kw, **omega_kw)),
        (
            f"discrete{suffix}",
            BackboneEmbedder(mode="discrete", n_bins=8, embedding_dim=16, **aa_kw, **omega_kw),
        ),
        (f"window7{suffix}", BackboneEmbedder(mode="window", window_size=7, **aa_kw, **omega_kw)),
        (f"window13{suffix}", BackboneEmbedder(mode="window", window_size=13, **aa_kw, **omega_kw)),
    ]

    mode_results = []
    for label, emb in modes:
        mr = benchmark_mode(label, emb, bal, args)
        mode_results.append(mr)

    total_elapsed = time.perf_counter() - t_start

    # Summary table
    _hr("Summary")
    print(f"  {'Mode':10s}  {'d*':>4}  {'Architecture':35s}  {'Test Acc':>10}  {'Params':>10}")
    print(f"  {'-' * 10}  {'-' * 4}  {'-' * 35}  {'-' * 10}  {'-' * 10}")
    for mr in mode_results:
        for arch_name, agg in mr["architectures"].items():
            print(
                f"  {mr['label']:10s}  {mr['d_star']:>4}  {arch_name:35s}"
                f"  {agg['mean']:.4f}±{agg['std']:.4f}  {agg['n_params']:>10,}"
            )

    print(f"\n  Total: {total_elapsed / 60:.1f}m")

    if args.report:
        write_report(args.report, args, bal, mode_results, total_elapsed)

    if args.out_dir:
        plot_results(mode_results, args.out_dir, args, total_elapsed)


if __name__ == "__main__":
    main()
