#!/usr/bin/env python3
"""
Protein Backbone Latent-Space Discovery
========================================

Can WaveRider rediscover the Ramachandran plot from scratch?

The protein backbone dihedral angles (φ, ψ) live on a 2-torus T².  The
Ramachandran plot shows that only a few discrete regions on that torus are
sterically allowed — alpha-helix (H), beta-sheet (E), polyproline II (P),
and left-handed helix (L).  These clusters are known from first principles.

This benchmark asks: given nothing but the (φ, ψ) pairs and the WaveRider
manifold machinery, can we recover that cluster structure automatically?

Four experiments
-----------------
  Exp 1  Synthetic baseline
         Generate 2000 residues from known Ramachandran Gaussians, embed
         with all three modes, and verify that ManifoldModel discovers
         d* = 2-3 and the correct cluster topology.

  Exp 2  Embedding mode comparison
         Compare torus (4D), discrete (8-fold, 16D), and window (7-residue,
         28D) embeddings via ManifoldModel classification accuracy against
         ground-truth secondary structure labels.

  Exp 3  Dimensionality landscape
         Plot the distribution of local intrinsic dimension across residues
         for each secondary structure class.  Hypothesis: helix and sheet
         have lower local d* than coil.

  Exp 4  Latent space voxel visualisation
         Feed the embedded coordinates from Exp 2 (window-7) directly into
         the WaveRider voxel visualizer (fit_and_observe → voxelize →
         build_grid → render_multi).  Saves a 2×2 multi-scalar PNG
         (density / curvature / height / class).

Usage
-----
Synthetic baseline (no PDB data needed)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py

Single PDB file::

    python benchmarks/canonical_tests/protein_backbone_manifold.py --pdb /path/to/file.pdb

Build the Parquet cache from a PDB directory (run once)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py \
        --pdb-dir /data/pdb/good \
        --cache-file /data/pdb/good_backbone.parquet \
        --skip-viz

Quick timing run (cap files, no viz)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py \
        --pdb-dir /data/pdb/good --max-files 50 --skip-viz

Run from cache (fast, no PDB parsing)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py \
        --cache-file /data/pdb/good_backbone.parquet --skip-viz

Rebuild the cache (e.g. after adding new PDB files)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py \
        --pdb-dir /data/pdb/good \
        --cache-file /data/pdb/good_backbone.parquet \
        --rebuild-cache --skip-viz

Headless full run (PNG output only)::

    python benchmarks/canonical_tests/protein_backbone_manifold.py \
        --cache-file /data/pdb/good_backbone.parquet --off-screen

CLI flags
---------
--pdb PATH              Single PDB file to analyse.
--pdb-dir DIR           Directory of ``pdb*.ent`` files (proteusPy BackboneLoader).
--workers N             Parallel worker processes for --pdb-dir (default: cpu_count, max 12).
--max-files N           Cap number of PDB files loaded — useful for timing runs.
--max-residues N        Cap residues after loading (random sample).
--cache-file PATH       Parquet cache path.  Loaded if it exists; written after
                        a --pdb-dir parse.  Use --rebuild-cache to overwrite.
--rebuild-cache         Force re-parse from --pdb-dir even if --cache-file exists.
--skip-viz              Skip the voxel visualisation (Exp 4).
--off-screen            Headless PNG export only (no PyVista window).
--out-dir DIR           Output directory for figures (default: papers/backbone_manifold/).
--n N                   Synthetic residue count for Exp 1 (default: 2000).
--seed N                Random seed (default: 42).

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from waverider.backbone_angles import BackboneAngleList  # noqa: E402
from waverider.backbone_embedder import BackboneEmbedder  # noqa: E402
from waverider.backbone_manifold import fit_backbone_manifold  # noqa: E402
from waverider.manifold_model import ManifoldModel  # noqa: E402

try:
    from proteusPy.backbone_loader import BackboneLoader
except ImportError:
    BackboneLoader = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PARSER = argparse.ArgumentParser(description="Protein backbone manifold benchmark")
_PARSER.add_argument("--n", type=int, default=2000, help="Synthetic residue count")
_PARSER.add_argument("--seed", type=int, default=42)
_PARSER.add_argument("--off-screen", action="store_true", help="Headless PNG export only")
_PARSER.add_argument("--out-dir", type=Path, default=_ROOT / "papers" / "backbone_manifold")
_PARSER.add_argument("--pdb", type=Path, default=None, help="Single PDB file")
_PARSER.add_argument(
    "--pdb-dir",
    type=Path,
    default=None,
    help="Directory of PDB files — loaded via proteusPy BackboneLoader",
)
_PARSER.add_argument(
    "--workers", type=int, default=None, help="Parallel workers for --pdb-dir (default: cpu_count)"
)
_PARSER.add_argument(
    "--max-files",
    type=int,
    default=None,
    help="Cap PDB files loaded from --pdb-dir (e.g. --max-files 100)",
)
_PARSER.add_argument(
    "--max-residues",
    type=int,
    default=None,
    help="Cap residues loaded from --pdb-dir (useful for quick tests)",
)
_PARSER.add_argument(
    "--cache-file",
    type=Path,
    default=None,
    help="Parquet cache: read if it exists, write after --pdb-dir parse",
)
_PARSER.add_argument(
    "--rebuild-cache",
    action="store_true",
    help="Force re-parse from --pdb-dir even if --cache-file exists",
)
_PARSER.add_argument(
    "--cache-only",
    action="store_true",
    help="Build/rebuild the cache then exit — skip all experiments",
)
_PARSER.add_argument(
    "--batch-files",
    type=int,
    default=2000,
    help="Files per batch when building cache (default 2000, reduce to cut memory)",
)
_PARSER.add_argument(
    "--check-cache",
    action="store_true",
    help="Print cache statistics and exit — no experiments run",
)
_PARSER.add_argument(
    "--dssp",
    action="store_true",
    help="Re-annotate SS with pydssp after loading (requires --pdb-dir or --cache-file + --pdb-dir)",
)
_PARSER.add_argument(
    "--remap-u-to-coil",
    action="store_true",
    help="Reclassify unknown (U) secondary structure as coil (C) before fitting",
)
_PARSER.add_argument(
    "--remap-u-rama",
    action="store_true",
    help=(
        "Reclassify unknown (U) residues by Ramachandran angle-box geometry "
        "(H/L/P/E/C) rather than lumping all into coil. "
        "Mutually exclusive with --remap-u-to-coil; this flag takes precedence."
    ),
)
_PARSER.add_argument(
    "--sample-n",
    type=int,
    default=None,
    help="Stratified subsample to N residues by SS class before fitting",
)
_PARSER.add_argument("--skip-viz", action="store_true", help="Skip voxel visualisation")
_PARSER.add_argument(
    "--viz-only",
    action="store_true",
    help="Load saved grid/point-field from --out-dir and open interactive visualiser; skip all experiments",
)
_PARSER.add_argument(
    "--report", type=Path, default=None, help="Write a Markdown summary report to this path"
)

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


def _stratified_sample(bal: BackboneAngleList, n: int, seed: int) -> BackboneAngleList:
    """Return a new BackboneAngleList with at most *n* residues, sampled
    proportionally from each SS class so the class balance is preserved."""
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

    # trim any rounding overshoot
    rng.shuffle(sampled)
    sampled = sampled[:n]

    counts = {c: sum(1 for r in sampled if r.secondary_structure == c) for c in sorted(by_class)}
    count_str = "  ".join(f"{c}:{v:,}" for c, v in counts.items())
    print(f"  Stratified sample: {len(sampled):,} residues  [{count_str}]")
    return BackboneAngleList(residues=sampled, name=bal.name + f"_s{n}")


def _hr(title: str = "") -> None:
    if title:
        pad = max(0, 58 - len(title))
        print(f"\n{'─' * 3} {title} {'─' * pad}")
    else:
        print("─" * 62)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _stratified_split_idx(
    y: np.ndarray, test_frac: float = 0.20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) with class-proportional split."""
    import random as _random

    rng = _random.Random(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0].tolist()
        rng.shuffle(cls_idx)
        n_test = max(1, round(len(cls_idx) * test_frac))
        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# Exp 1 – Synthetic baseline
# ---------------------------------------------------------------------------


def exp1_synthetic_baseline(n: int, seed: int) -> BackboneAngleList:
    _hr("Exp 1: Synthetic baseline")
    bal = BackboneAngleList.from_synthetic(n=n, seed=seed)
    print(bal)

    arr = bal.to_phi_psi_array()
    codes = bal.to_combined_codes(n_bins=8)
    print(f"  φ range   : [{arr[:, 0].min():.1f}°, {arr[:, 0].max():.1f}°]")
    print(f"  ψ range   : [{arr[:, 1].min():.1f}°, {arr[:, 1].max():.1f}°]")
    print(f"  8-fold codes: {len(np.unique(codes))} of 64 bins populated")

    # Show which (phi_bin, psi_bin) pairs dominate each SS class
    print("\n  Dominant (φ_bin, ψ_bin) per class:")
    labels = bal.to_ss_int_labels()
    for int_label, name in _SS_NAMES.items():
        mask = labels == int_label
        if not mask.any():
            continue
        cls_codes = codes[mask]
        top_code = int(np.bincount(cls_codes).argmax())
        print(f"    {name:22s}: code={top_code:2d}  (φ_bin={top_code // 8}, ψ_bin={top_code % 8})")

    return bal.valid()


# ---------------------------------------------------------------------------
# Exp 2 – Embedding mode comparison
# ---------------------------------------------------------------------------


def exp2_embedding_modes(bal: BackboneAngleList) -> dict:
    """Fit ManifoldModel for each embedding mode; report held-out test accuracy.

    Uses a stratified 80/20 train/test split of the embedded data.  The full
    dataset is used for d* discovery (better statistics); a fresh ManifoldModel
    is then fit on the train split and evaluated on the held-out test split so
    that reported accuracy reflects genuine generalisation, not training recall.
    """
    _hr("Exp 2: Embedding mode comparison")

    rows = {}
    modes = [
        ("torus", BackboneEmbedder(mode="torus")),
        ("discrete", BackboneEmbedder(mode="discrete", n_bins=8, embedding_dim=16)),
        ("window7", BackboneEmbedder(mode="window", window_size=7)),
        ("window13", BackboneEmbedder(mode="window", window_size=13)),
    ]

    for label, emb in modes:
        t0 = time.perf_counter()
        # Full-data fit: d* discovery + geometry (result.X_embedded used by Exp 3/4)
        result = fit_backbone_manifold(
            bal,
            emb,
            k_pca=30,
            k_graph=10,
            variance_threshold=0.90,
            n_dim_samples=200,
            verbose=False,
        )

        # Stratified 80/20 split of the embedded matrix
        train_idx, test_idx = _stratified_split_idx(result.y, test_frac=0.20, seed=42)
        X_train = result.X_embedded[train_idx]
        X_test = result.X_embedded[test_idx]
        y_train = result.y[train_idx]
        y_test = result.y[test_idx]

        # Scale k_pca with ambient dimension so local PCA is overdetermined.
        # Rule: k_pca = max(50, 3 × d_ambient), capped at train set size − 1.
        d_ambient = result.X_embedded.shape[1]
        k_pca_clf = min(len(X_train) - 1, max(50, 3 * d_ambient))

        # Fresh ManifoldModel fit on train only, evaluated on held-out test
        clf = ManifoldModel(k_graph=10, k_pca=k_pca_clf, variance_threshold=0.90)
        clf.fit(X_train, y_train)
        acc = _accuracy(y_test, clf.predict(X_test))

        elapsed = time.perf_counter() - t0
        print(
            f"  {label:10s} | d_ambient={d_ambient:3d}"
            f" | d*={result.d_star}"
            f" | k_pca={k_pca_clf}"
            f" | test_acc={acc:.4f}"
            f" | train={len(train_idx):,}  test={len(test_idx):,}"
            f" | {elapsed:.1f}s"
        )
        rows[label] = {
            "result": result,
            "d_ambient": d_ambient,
            "d_star": result.d_star,
            "acc": acc,
            "k_pca_clf": k_pca_clf,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed": elapsed,
        }

    return rows


# ---------------------------------------------------------------------------
# Exp 3 – Dimensionality landscape
# ---------------------------------------------------------------------------


def exp3_dimensionality_landscape(result) -> tuple:
    """Return (dim_report, class_dims) for report generation."""
    _hr("Exp 3: Dimensionality landscape")
    print(result.summary())

    class_dims = {}
    try:
        from waverider.dimensionality_discovery import discover_per_class_dimensionality

        class_dims = discover_per_class_dimensionality(
            result.X_embedded,
            result.y,
            k=min(30, len(result.X_embedded) - 1),
            tau=0.90,
            n_samples_per_class=50,
        )
        print("\n  Per-class intrinsic dimension (τ=0.90):")
        for cls_int, stats in sorted(class_dims.items()):
            name = _SS_NAMES.get(cls_int, f"class-{cls_int}")
            print(f"    {name:22s}: d*={stats['mean']:.1f} ± {stats['std']:.1f}")
    except ImportError:
        print("  (scipy not available — skipping per-class analysis)")

    return result.dim_report, class_dims


# ---------------------------------------------------------------------------
# Exp 4 – Voxel visualisation
# ---------------------------------------------------------------------------


def exp4_voxel_viz(result, out_dir: Path, off_screen: bool) -> None:
    _hr("Exp 4: Voxel visualisation")

    try:
        from waverider.voxel_viz import build_grid, fit_and_observe, render_multi, voxelize
    except ImportError as exc:
        print(f"  Voxel viz skipped — missing dep: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # fit_and_observe expects (X, y) as separate args and handles
    # ManifoldModel + ManifoldObserver internally. We pass the pre-fitted
    # embedded matrix and labels so no re-fitting is needed.
    X = result.X_embedded
    y = result.y

    print(f"  Fitting voxel geometry (n={len(X)}, d={X.shape[1]})…")
    _, _, point_field, _ = fit_and_observe(X, y, k_graph=10, k_pca=30, k_vote=5, tau=0.90)

    print("  Voxelising…")
    grids = voxelize(point_field, resolution=32)
    grid = build_grid(grids)

    png_path = out_dir / "backbone_manifold_voxels.png"
    print(f"  Rendering → {png_path}")

    render_multi(
        grid,
        point_field,
        off_screen=off_screen,
        out_path=png_path,
    )
    print(f"  Saved: {png_path}")

    grid_path = out_dir / "backbone_manifold_grid.vti"
    pf_path = out_dir / "backbone_manifold_pointfield.npz"
    grid.save(str(grid_path))
    np.savez(
        pf_path,
        X3=point_field.X3,
        density_w=point_field.density_w,
        curvature=point_field.curvature,
        height=point_field.height,
        intrinsic_dim=point_field.intrinsic_dim,
        labels=point_field.labels,
    )
    print(f"  Saved: {grid_path}")
    print(f"  Saved: {pf_path}")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_pdb(path: Path) -> BackboneAngleList:
    _hr(f"PDB: {path.name}")
    bal = BackboneAngleList.from_pdb_file(str(path))
    print(bal)
    return bal.valid()


def load_pdb_dir(
    pdb_dir: Path,
    workers: int | None = None,
    max_files: int | None = None,
    max_residues: int | None = None,
) -> BackboneAngleList:
    """Load backbone angles from a directory of PDB files via proteusPy BackboneLoader."""
    if BackboneLoader is None:
        raise SystemExit(
            "proteusPy is required for --pdb-dir.  Install with: pip install proteusPy"
        )
    _hr(f"PDB dir: {pdb_dir}")

    kwargs = {}
    if workers is not None:
        kwargs["workers"] = workers

    loader = BackboneLoader(pdb_dir=pdb_dir, **kwargs)
    residues = loader.load(max_files=max_files)
    print(f"  Loaded {len(residues):,} residues from {pdb_dir}")

    if max_residues and len(residues) > max_residues:
        import random

        random.shuffle(residues)
        residues = residues[:max_residues]
        print(f"  Capped to {max_residues:,} residues (--max-residues)")

    bal = BackboneAngleList.from_proteuspy(residues, name=pdb_dir.name)
    print(bal)
    return bal.valid()


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    path: Path,
    args,
    bal: BackboneAngleList,
    mode_rows: dict,
    dim_report: dict,
    class_dims: dict,
    total_elapsed: float,
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

    git_hash = _git(["git", "rev-parse", "--short", "HEAD"])
    git_branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_date = _git(["git", "log", "-1", "--format=%ai"])
    git_msg = _git(["git", "log", "-1", "--format=%s"])

    try:
        import tensorflow as tf

        tf_ver = tf.__version__
    except ImportError:
        tf_ver = "n/a"

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    host = socket.gethostname()
    os_info = platform.platform()
    py_ver = platform.python_version()

    # SS counts
    ss_counts: dict = {}
    for r in bal.residues:
        ss_counts[r.secondary_structure] = ss_counts.get(r.secondary_structure, 0) + 1
    ss_lines = "  ".join(f"{k}:{v:,}" for k, v in sorted(ss_counts.items()))

    # Data provenance string
    if args.cache_file and args.cache_file.exists() and not getattr(args, "rebuild_cache", False):
        data_source = f"Parquet cache: `{args.cache_file}`"
    elif getattr(args, "pdb_dir", None):
        data_source = f"PDB directory: `{args.pdb_dir}`"
    elif getattr(args, "pdb", None):
        data_source = f"Single PDB: `{args.pdb}`"
    else:
        data_source = "Synthetic (Ramachandran Gaussians)"

    lines = [
        "# Protein Backbone Latent-Space Discovery",
        "",
        f"**Generated:** {now}  ",
        f"**Host:** {host}  |  **OS:** {os_info}  ",
        f"**Python:** {py_ver}  |  **TensorFlow:** {tf_ver}  ",
        f"**Repository:** waverider @ `{git_hash}` ({git_branch})  ",
        f"**Commit:** {git_date} — {git_msg}  ",
        "",
        "---",
        "",
        "## Run Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Data source | {data_source} |",
        f"| U remap | {'Ramachandran geometry (--remap-u-rama)' if getattr(args, 'remap_u_rama', False) else 'U→C (--remap-u-to-coil)' if getattr(args, 'remap_u_to_coil', False) else 'none'} |",
        f"| Sample N | {getattr(args, 'sample_n', None) or 'all'} |",
        f"| Max files | {getattr(args, 'max_files', None) or 'all'} |",
        "| k_pca | 30 |",
        "| k_graph | 10 |",
        "| Variance threshold (τ) | 0.90 |",
        "| Dim samples | 200 |",
        "| Eval split | 80% train / 20% test (stratified) |",
        f"| Random seed | {args.seed} |",
        f"| Total wall time | {total_elapsed:.1f}s |",
        "",
        "## Corpus Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Collection | {bal.name} |",
        f"| Residues (after filter) | {len(bal.residues):,} |",
        f"| SS distribution | {ss_lines} |",
        "",
        "## Exp 2: Embedding Mode Comparison",
        "",
        "Accuracy is held-out **test** accuracy (stratified 80/20 split; ManifoldModel fit on train only).",
        "",
        "| Mode | d_ambient | d* | k_pca | Test Acc | Train N | Test N | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for label, row in mode_rows.items():
        n_tr = f"{row['n_train']:,}" if "n_train" in row else "—"
        n_te = f"{row['n_test']:,}" if "n_test" in row else "—"
        k_pca = row.get("k_pca_clf", "—")
        lines.append(
            f"| {label} | {row['d_ambient']} | {row['d_star']} "
            f"| {k_pca} | {row['acc']:.4f} | {n_tr} | {n_te} "
            f"| {row['elapsed']:.1f} |"
        )

    lines += [
        "",
        "## Exp 3: Dimensionality Landscape",
        "",
        "Local PCA, k=30, 200 random samples.",
        "",
        "| τ | Mean d* | Std | Min | Max |",
        "|---|---|---|---|---|",
    ]

    for tau, stats in sorted(dim_report.items()):
        lines.append(
            f"| {tau:.2f} | {stats['mean']:.1f} | {stats['std']:.1f} "
            f"| {stats['min']} | {stats['max']} |"
        )

    if class_dims:
        lines += [
            "",
            "### Per-Class Intrinsic Dimension (τ=0.90)",
            "",
            "| Class | Mean d* | Std |",
            "|---|---|---|",
        ]
        for cls_int, stats in sorted(class_dims.items()):
            name = _SS_NAMES.get(cls_int, f"class-{cls_int}")
            lines.append(f"| {name} | {stats['mean']:.1f} | {stats['std']:.1f} |")

    lines += ["", "---", "*Generated by `protein_backbone_manifold.py`*", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\n  Report → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _PARSER.parse_args()

    t_start = time.perf_counter()

    print("=" * 62)
    print("  WaveRider — Protein Backbone Latent-Space Discovery")
    print("=" * 62)

    # ---- Viz-only: load saved artifacts and open interactive window ----
    if args.viz_only:
        try:
            import pyvista as pv  # noqa: PLC0415

            from waverider.voxel_viz import PointField, render_multi  # noqa: PLC0415
        except ImportError as exc:
            raise SystemExit(f"--viz-only requires pyvista: {exc}")
        grid_path = args.out_dir / "backbone_manifold_grid.vti"
        pf_path = args.out_dir / "backbone_manifold_pointfield.npz"
        if not grid_path.exists() or not pf_path.exists():
            raise SystemExit(
                f"Saved artifacts not found in {args.out_dir}.\n"
                "Run without --viz-only first to generate them."
            )
        _hr("Viz-only: loading saved artifacts")
        grid = pv.read(str(grid_path))
        d = np.load(pf_path)
        point_field = PointField(
            X3=d["X3"],
            density_w=d["density_w"],
            curvature=d["curvature"],
            height=d["height"],
            intrinsic_dim=d["intrinsic_dim"],
            labels=d["labels"],
        )
        print(f"  Grid : {grid_path}")
        print(f"  Field: {pf_path}  ({len(point_field.X3):,} points)")
        render_multi(grid, point_field, off_screen=False)
        return

    # ---- Check cache and exit ----
    if args.check_cache:
        if not args.cache_file:
            raise SystemExit("--check-cache requires --cache-file")
        if BackboneLoader is None:
            raise SystemExit(
                "proteusPy is required for --check-cache.  Install with: pip install proteusPy"
            )
        _hr(f"Cache: {args.cache_file.name}")
        if not args.cache_file.exists():
            print(f"  NOT FOUND: {args.cache_file}")
            return
        import os

        size_mb = os.path.getsize(args.cache_file) / 1024 / 1024
        residues = BackboneLoader.load_cache(args.cache_file)
        bal = BackboneAngleList.from_proteuspy(residues, name=args.cache_file.stem)
        ss_counts: dict = {}
        for r in bal.residues:
            ss_counts[r.secondary_structure] = ss_counts.get(r.secondary_structure, 0) + 1
        print(bal)
        print(f"  File size : {size_mb:.1f} MB")
        print("  SS breakdown:")
        total = len(bal.residues)
        for code, count in sorted(ss_counts.items()):
            print(f"    {code} : {count:>10,}  ({100 * count / total:.1f}%)")
        _hr()
        return

    # ---- Build cache (batched, memory-safe) ----
    if args.pdb_dir and args.cache_file and (args.rebuild_cache or not args.cache_file.exists()):
        if BackboneLoader is None:
            raise SystemExit(
                "proteusPy is required for --cache-file.  Install with: pip install proteusPy"
            )
        loader = BackboneLoader(pdb_dir=args.pdb_dir, workers=args.workers)
        loader.build_cache(
            args.cache_file,
            dssp=args.dssp,
            batch_files=args.batch_files,
            max_files=args.max_files,
        )
        if args.cache_only:
            _hr()
            print("Cache built. Exiting (--cache-only).")
            return

    # ---- Load data ----
    if args.cache_file and args.cache_file.exists():
        if BackboneLoader is None:
            raise SystemExit(
                "proteusPy is required for --cache-file.  Install with: pip install proteusPy"
            )
        _hr(f"Cache: {args.cache_file.name}")
        residues = BackboneLoader.load_cache(args.cache_file)
        if args.max_residues and len(residues) > args.max_residues:
            import random

            rng = random.Random(args.seed)
            rng.shuffle(residues)
            residues = residues[: args.max_residues]
            print(f"  Capped to {args.max_residues:,} residues (--max-residues)")
        bal = BackboneAngleList.from_proteuspy(residues, name=args.cache_file.stem)
        print(bal)
        bal = bal.valid()
    elif args.pdb_dir:
        bal = load_pdb_dir(args.pdb_dir, args.workers, args.max_files, args.max_residues)
    elif args.pdb:
        bal = load_pdb(args.pdb)
    else:
        bal = exp1_synthetic_baseline(args.n, args.seed)

    # ---- Remap U residues ----
    if args.remap_u_rama:
        n_u = sum(1 for r in bal.residues if r.secondary_structure == "U")
        bal = bal.remap_u_by_ramachandran()
        n_still_u = sum(1 for r in bal.residues if r.secondary_structure == "U")
        print(
            f"  Ramachandran remap: {n_u:,} U → geometry  ({n_still_u:,} remain U — NaN angles)  {bal}"
        )
    elif args.remap_u_to_coil:
        n_remapped = sum(1 for r in bal.residues if r.secondary_structure == "U")
        for r in bal.residues:
            if r.secondary_structure == "U":
                r.secondary_structure = "C"
        print(f"  Remapped {n_remapped:,} U → C  {bal}")

    # ---- Stratified sample ----
    if args.sample_n and args.sample_n < len(bal.residues):
        bal = _stratified_sample(bal, args.sample_n, args.seed)

    # ---- Exp 2 ----
    mode_rows = exp2_embedding_modes(bal)

    # Use the window-7 result for downstream experiments (best context)
    best_row = mode_rows.get("window7") or next(iter(mode_rows.values()))
    best_result = best_row["result"]

    # ---- Exp 3 ----
    dim_report, class_dims = exp3_dimensionality_landscape(best_result)

    # ---- Exp 4 ----
    if not args.skip_viz:
        exp4_voxel_viz(best_result, args.out_dir, args.off_screen)

    total_elapsed = time.perf_counter() - t_start

    # ---- Report ----
    if args.report:
        _write_report(args.report, args, bal, mode_rows, dim_report, class_dims, total_elapsed)

    _hr()
    print("Done.")


if __name__ == "__main__":
    main()
