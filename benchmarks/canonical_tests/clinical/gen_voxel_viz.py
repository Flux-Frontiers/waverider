#!/usr/bin/env python3
"""
Generate manifold voxel visualization PNGs for all 6 clinical datasets.

Runs the full WaveRider voxel pipeline (fit_and_observe → voxelize →
build_grid → render_multi) for each disease dataset and saves a 2×2
multi-scalar panel PNG (density / curvature / height / class_vote) to
papers/clinical_manifolds/.

Usage
-----
    cd /path/to/waverider
    python benchmarks/canonical_tests/clinical/gen_voxel_viz.py

Output
------
    papers/clinical_manifolds/{dataset}_manifold_voxel.png   (one per dataset)

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pyvista as pv  # noqa: E402

from waverider.voxel_viz import (  # noqa: E402
    build_grid,
    fit_and_observe,
    render_multi,
    voxelize,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = _ROOT / "papers" / "clinical_manifolds"
OUT_DIR.mkdir(exist_ok=True)

# Match benchmark parameters exactly
FIT_PARAMS = dict(k_graph=10, k_pca=20, k_vote=7, tau=0.9)
RESOLUTION = 32

# Large window → crisper PNG
pv.global_theme.window_size = [1600, 900]


# ---------------------------------------------------------------------------
# Dataset loaders (no TF dependency — loaders only)
# ---------------------------------------------------------------------------


def _scale(X: np.ndarray) -> np.ndarray:
    """Drop near-zero-variance cols, StandardScaler, nan→0."""
    col_std = X.std(axis=0)
    X = X[:, col_std > 1e-6]
    scaler = StandardScaler()
    return np.nan_to_num(scaler.fit_transform(X).astype("d"))


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer as _load

    bunch = _load()
    X = bunch.data.astype("d")
    y = bunch.target.astype(int)
    return _scale(X), y, bunch.target_names.tolist()


def load_heart():
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=45)
    X_df = repo.data.features.copy()
    y_df = repo.data.targets.copy()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df).astype("d")
    y = (y_df.iloc[:, 0].values > 0).astype(int)
    return _scale(X), y, ["No Disease", "Disease"]


def load_parkinsons():
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=174)
    X_df = repo.data.features.copy()
    y_df = repo.data.targets.copy()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df).astype("d")
    y_raw = y_df.iloc[:, 0].values.astype(int)
    uniq = sorted(np.unique(y_raw))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return _scale(X), y, ["Healthy", "Parkinson's"]


def load_dermatology():
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=33)
    X_df = repo.data.features.copy()
    y_df = repo.data.targets.copy()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df).astype("d")
    y_raw = y_df.iloc[:, 0].values.astype(int)
    uniq = sorted(np.unique(y_raw))
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    labels = ["Psoriasis", "Seb.Derm.", "Lichen Planus", "Pit.Rosea", "Chr.Derm.", "Pit.Rubra"]
    return _scale(X), y, labels


def load_alzheimers():
    import pandas as pd

    data_dir = _HERE.parent.parent / "data"
    for name in (
        "oasis_cross-sectional.xlsx",
        "oasis_cross-sectional.csv",
        "oasis_cross-sectional.tsv",
    ):
        p = data_dir / name
        if p.exists():
            if p.suffix == ".xlsx":
                df = pd.read_excel(str(p))
            elif p.suffix == ".tsv":
                df = pd.read_csv(str(p), sep="\t")
            else:
                df = pd.read_csv(str(p))
            break
    else:
        raise FileNotFoundError(f"OASIS data not found in {data_dir}")

    # BIDS TSV format
    if "cdr_global" in df.columns:
        df = df.rename(
            columns={"age_bl": "Age", "education_level": "Educ", "MMS": "MMSE", "cdr_global": "CDR"}
        )
        if "sex" in df.columns:
            df["Sex"] = (df["sex"].str.upper() == "M").astype(float)
    else:
        if "M/F" in df.columns:
            df["Sex"] = (df["M/F"] == "M").astype(float)

    df = df.dropna(subset=["CDR"])
    y = (df["CDR"] > 0).astype(int).values
    feat_cols = [
        c for c in ["Age", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "Sex"] if c in df.columns
    ]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df[feat_cols].astype("d"))
    return _scale(X), y, ["Non-demented", "Demented"]


def load_diabetes():
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=37, as_frame=True, parser="auto")
    X_df = bunch.data.copy()
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df.astype("d"))
    y_raw = bunch.target.astype(str).str.strip()
    y = (y_raw == "tested_positive").astype(int).values
    return _scale(X), y, ["Non-diabetic", "Diabetic"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

DATASETS: list[tuple[str, callable]] = [
    ("breast_cancer", load_breast_cancer),
    ("heart", load_heart),
    ("parkinsons", load_parkinsons),
    ("dermatology", load_dermatology),
    ("alzheimers", load_alzheimers),
    ("diabetes", load_diabetes),
]


_BREAST_CANCER_SCALARS = [
    ("density", "plasma", "Density"),
    ("curvature", "coolwarm", "Mean curvature"),
    ("height", "viridis", "Height above tangent"),
    ("class_vote", "Set1", "Majority class vote"),
]


def run_one(ds_name: str, loader_fn) -> Path | None:
    print(f"\n{'=' * 60}")
    print(f"  {ds_name.upper()}")
    print("=" * 60)
    try:
        X, y, class_names = loader_fn()
        print(f"  Loaded: X={X.shape}  n_classes={len(np.unique(y))}  labels={class_names}")

        subject, observer, pf, pca_info = fit_and_observe(X, y, **FIT_PARAMS)
        vox = voxelize(pf, resolution=RESOLUTION)
        grid = build_grid(vox)

        out_path = OUT_DIR / f"{ds_name}_manifold_voxel.png"
        scalars = _BREAST_CANCER_SCALARS if ds_name == "breast_cancer" else None
        render_multi(
            grid,
            pf,
            off_screen=True,
            out_path=out_path,
            show_volume=True,
            vol_opacity=0.10,
            vol_threshold=0.03,
            pca_info=pca_info,
            scalars=scalars,
        )
        return out_path
    except Exception as exc:
        print(f"  ERROR: {exc}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    successes = []
    failures = []
    for name, fn in DATASETS:
        result = run_one(name, fn)
        if result:
            successes.append(result)
        else:
            failures.append(name)

    print(f"\n{'=' * 60}")
    print(f"Done.  {len(successes)}/{len(DATASETS)} succeeded.")
    for p in successes:
        print(f"  ✓  {p}")
    for f in failures:
        print(f"  ✗  {f}")
