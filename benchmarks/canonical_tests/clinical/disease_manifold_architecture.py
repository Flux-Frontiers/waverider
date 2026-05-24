#!/usr/bin/env python3
"""
Disease Manifold Architecture Benchmark
========================================

Applies WaveRider's manifold-informed architecture suite to clinical / disease
datasets, asking the same question asked of MNIST and CIFAR-10:

    *What is the intrinsic dimensionality of this dataset, and does matching
    the network architecture to that geometry improve parameter efficiency?*

For tabular clinical data the story is even stronger than for images: biological
constraints compress hundreds of measurements onto manifolds of surprisingly low
dimension.  ManifoldModel — zero learned parameters, pure geometry — often
matches or exceeds small MLPs.

Supported datasets
------------------
  breast_cancer   569 × 30  2-class  sklearn built-in
  heart           303 × 13  2-class  Cleveland Heart Disease (ucimlrepo ID 45)
  diabetes        768 ×  8  2-class  Pima Diabetes (ucimlrepo ID 34)
  parkinsons      195 × 22  2-class  Voice measurements (ucimlrepo ID 174)
  dermatology     366 × 34  6-class  Skin disease classification (ucimlrepo ID 33)
  alzheimers     ~436 ×  8  2-class  OASIS cross-sectional (local CSV required)

Architecture suite
------------------
  Standard (scaled)          input → H₁ → H₂ → C    (H₁ = min(256, 8d))
  Manifold (2d→d)            input → 2d → d → C
  Wide Manifold (d+1)        input → d+1 → C
  Manifold+ManifoldAdam      input → 2d → d → C  (manifold-projected gradient)
  PCA→dD + MLP (2d→d)        d → 2d → d → C
  Intrinsic Dim (PCA→dD→C)   d → C
  ManifoldModel (τ=0.90)     0 learned parameters — pure geometry
  Euclidean KNN (k=7)        classic non-parametric baseline

Evaluation
----------
  5-fold stratified cross-validation throughout.
  Binary datasets (n_classes=2) additionally report AUC-ROC.

Output files
------------
  {dataset}_disease_architecture_results.json
  {dataset}_disease_architecture_results.png

Usage
-----
  python benchmarks/canonical_tests/disease_manifold_architecture.py --dataset breast_cancer
  python benchmarks/canonical_tests/disease_manifold_architecture.py --dataset alzheimers \\
      --alzheimers-csv /path/to/oasis_cross-sectional.csv
  python benchmarks/canonical_tests/disease_manifold_architecture.py --dataset heart \\
      --epochs 100 --trials 3

  # Alzheimer's CSV download (free, no account required):
  # https://github.com/uwescience/datasci_course_materials/raw/master/assignment6/oasis_cross-sectional.csv

Part of WaveRider, https://github.com/flux-frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
License: Elastic 2.0
Last Revision: 2026-04-15 09:29:55
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorFlow / CPU setup
# ---------------------------------------------------------------------------
from benchmarks.tf_setup import setup_tensorflow  # noqa: E402

tf, DEVICE_INFO = setup_tensorflow()
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA as skPCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Suppress retracing warnings: CV benchmarks rebuild models per fold intentionally.
tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import keras  # noqa: E402

# ---------------------------------------------------------------------------
# WaveRider imports
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
sys.path.insert(0, str(_HERE.parent.parent))

from model_builder import (  # noqa: E402
    build_manifold_model,
    build_pca_intrinsic_dim_model,
    build_pca_model,
    build_wide_manifold_model,
)
from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)
from waverider.manifold_model import ManifoldModel  # noqa: E402
from waverider.manifold_optimizer import ManifoldAdam, make_basis  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset loaders
# Each returns (X: float32 ndarray, y: int ndarray, meta: dict)
# ---------------------------------------------------------------------------


def _sklearn_meta(data, name, class_names=None):
    X = data.data.astype("float32")
    y = data.target.astype(int)
    meta = {
        "name": name,
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_names": list(class_names or [str(i) for i in np.unique(y)]),
        "source": "sklearn",
    }
    return X, y, meta


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer as _load

    data = _load()
    return _sklearn_meta(data, "Breast Cancer (Wisconsin)", data.target_names.tolist())


def _load_ucimlrepo(
    dataset_id,
    name,
    binary_threshold=None,
    drop_cols=None,
    target_col=None,
    class_names=None,
):
    """Generic ucimlrepo loader with optional binarization."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo not installed. Run:  pip install ucimlrepo\nor:  poetry add ucimlrepo"
        )
    repo = fetch_ucirepo(id=dataset_id)
    X_df = repo.data.features.copy()
    y_df = repo.data.targets.copy()

    if drop_cols:
        X_df = X_df.drop(columns=[c for c in drop_cols if c in X_df.columns])

    # Encode categoricals
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes

    # Impute missing values with column median
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df).astype("float32")

    # Target
    if target_col and target_col in y_df.columns:
        y_raw = y_df[target_col].values
    else:
        y_raw = y_df.iloc[:, 0].values

    if binary_threshold is not None:
        y = (y_raw > binary_threshold).astype(int)
        class_names = class_names or ["Negative", "Positive"]
    else:
        # Remap to 0-based contiguous integers
        uniq = sorted(np.unique(y_raw.astype(int)))
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[int(v)] for v in y_raw], dtype=int)

    n_classes = len(np.unique(y))
    meta = {
        "name": name,
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_classes": n_classes,
        "class_names": list(class_names or [str(i) for i in range(n_classes)]),
        "source": f"ucimlrepo (id={dataset_id})",
    }
    return X, y, meta


def load_heart():
    return _load_ucimlrepo(
        45,
        "Heart Disease (Cleveland)",
        binary_threshold=0,
        class_names=["No disease", "Disease"],
    )


def load_diabetes():
    """Load Pima Indians Diabetes via sklearn fetch_openml (UCI id=34 not API-importable)."""
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise ImportError("scikit-learn is required for the diabetes dataset.")

    bunch = fetch_openml(data_id=37, as_frame=True, parser="auto")
    X_df = bunch.data.copy()

    # Encode any remaining categoricals
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype("category").cat.codes

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df.astype("float32"))

    # Target: "tested_positive" / "tested_negative" → 1 / 0
    y_raw = bunch.target.astype(str).str.strip()
    y = (y_raw == "tested_positive").astype(int).values

    meta = {
        "name": "Pima Indians Diabetes",
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_classes": 2,
        "class_names": ["Non-diabetic", "Diabetic"],
        "source": "OpenML data_id=40715 (Pima Indians Diabetes)",
        "features": list(X_df.columns),
    }
    return X, y, meta


def load_parkinsons():
    return _load_ucimlrepo(
        174,
        "Parkinson's Disease (Voice)",
        class_names=["Healthy", "Parkinson's"],
    )


def load_dermatology():
    return _load_ucimlrepo(
        33,
        "Dermatology (Skin Disease)",
        class_names=[
            "Psoriasis",
            "Seb. Derm.",
            "Lichen Planus",
            "Pityriasis Rosea",
            "Chronic Derm.",
            "Pityriasis Rubra",
        ],
    )


_OASIS_RAW_URL = (
    "https://raw.githubusercontent.com/uwescience/"
    "datasci_course_materials/master/assignment6/oasis_cross-sectional.csv"
)

# Search order for the OASIS data file — xlsx preferred (full features)
_ALZHEIMERS_CANDIDATES = [
    "oasis_cross-sectional.xlsx",
    "oasis_cross-sectional.csv",
    "oasis_cross-sectional.tsv",
]


def _find_alzheimers_default() -> str:
    """Return the best available OASIS data file path, preferring xlsx > csv > tsv."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    for name in _ALZHEIMERS_CANDIDATES:
        p = data_dir / name
        if p.exists():
            return str(p)
    return str(data_dir / _ALZHEIMERS_CANDIDATES[0])


_DEFAULT_ALZHEIMERS_CSV = _find_alzheimers_default()


def load_alzheimers(csv_path):
    """Load OASIS cross-sectional data and return manifold-ready arrays.

    Accepts two formats (auto-detected by column names):

    **OASIS-1 original CSV** (OASIS standard)::

        ID, M/F, Hand, Age, Educ, SES, MMSE, CDR, eTIV, nWBV, ASF[, Delay]

    **OASIS-BIDS TSV** (14thibea/OASIS-1_dataset)::

        participant_id, session_id, alternative_id_1, sex, education_level,
        age_bl, cdr, diagnosis_bl, laterality, MMS, cdr_global, diagnosis

    Target: CDR binarized — 0 = Non-demented, 1 = Demented (CDR > 0).
    Missing values imputed with column median.

    BIDS TSV source::

        git clone https://github.com/14thibea/OASIS-1_dataset ../OASIS-1_dataset
        cp ../OASIS-1_dataset/tsv_files/lab_1/OASIS_BIDS.tsv benchmarks/data/oasis_cross-sectional.tsv

    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Alzheimer's CSV loading: pip install pandas")

    # Auto-resolve: if the given path doesn't exist, try sibling candidates
    p = Path(csv_path)
    if not p.exists():
        data_dir = p.parent
        for name in _ALZHEIMERS_CANDIDATES:
            candidate = data_dir / name
            if candidate.exists():
                csv_path = str(candidate)
                p = candidate
                break
        else:
            raise FileNotFoundError(
                f"OASIS data not found in '{data_dir}'. Expected one of: {_ALZHEIMERS_CANDIDATES}"
            )

    # Load based on file extension
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(csv_path)
        except ImportError:
            raise ImportError(
                "Reading .xlsx requires openpyxl: pip install openpyxl\n"
                "Alternatively, use the .tsv version:\n"
                f"  --alzheimers-csv {p.with_suffix('.tsv')}"
            )
    else:
        # Detect HTML masquerading as CSV (e.g. failed GitHub download)
        raw_head = p.read_bytes()[:16]
        if b"<" in raw_head or b"html" in raw_head.lower():
            raise ValueError(
                f"'{csv_path}' looks like an HTML page, not a CSV.\n"
                "See load_alzheimers docstring for valid data sources."
            )
        df = pd.read_csv(csv_path, sep=None, engine="python")

    # --- Detect and normalise BIDS TSV format ---
    if "cdr_global" in df.columns:
        # BIDS column mapping
        rename = {
            "age_bl": "Age",
            "education_level": "Educ",
            "MMS": "MMSE",
            "cdr_global": "CDR",
        }
        df = df.rename(columns=rename)
        if "sex" in df.columns:
            df["Sex"] = (df["sex"].str.upper() == "M").astype(int)
        # Drop BIDS-specific admin columns
        df = df.drop(
            columns=[
                "participant_id",
                "session_id",
                "alternative_id_1",
                "sex",
                "diagnosis_bl",
                "laterality",
                "cdr",
                "diagnosis",
            ],
            errors="ignore",
        )
    else:
        # Original OASIS-1 CSV format
        if "M/F" in df.columns:
            df["Sex"] = (df["M/F"] == "M").astype(int)
            df = df.drop(columns=["M/F"])
        df = df.drop(columns=["ID", "Hand", "Delay"], errors="ignore")

    # Drop rows with missing CDR (the target)
    df = df.dropna(subset=["CDR"])

    # Target
    y = (df["CDR"] > 0).astype(int).values
    df = df.drop(columns=["CDR"])

    # Impute remaining missing values
    for col in df.select_dtypes(include=["float64", "int64", "float32"]).columns:
        df[col] = df[col].fillna(df[col].median())

    X = df.values.astype("float32")

    n_demented = int(y.sum())
    meta = {
        "name": "Alzheimer's (OASIS Cross-Sectional)",
        "n_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "n_classes": 2,
        "class_names": ["Non-demented", "Demented"],
        "source": f"OASIS: {csv_path}",
        "class_balance": f"{n_demented} demented / {len(y) - n_demented} non-demented",
        "features": list(df.columns),
    }
    return X, y, meta


DATASET_LOADERS = {
    "breast_cancer": (load_breast_cancer, False),  # (loader, needs_csv_path)
    "heart": (load_heart, False),
    "diabetes": (load_diabetes, False),
    "parkinsons": (load_parkinsons, False),
    "dermatology": (load_dermatology, False),
    "alzheimers": (load_alzheimers, True),
}


# ---------------------------------------------------------------------------
# Clinical-scale standard model
# Sized proportionally to input_dim — not 1024→512 which is image-scale
# ---------------------------------------------------------------------------


def build_clinical_standard_model(input_dim, n_classes, lr=0.001):
    """Baseline MLP proportional to input dimensions.

    :param input_dim: Input feature count.
    :param n_classes: Number of output classes.
    :param lr: Adam learning rate.
    :returns: Compiled Keras model.
    """
    h1 = min(256, max(64, 8 * input_dim))
    h2 = min(128, max(32, 4 * input_dim))
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(h1, activation="relu"),
            keras.layers.Dense(h2, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------


def run_fold(
    build_fn,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs,
    batch_size,
    fold,
    is_sklearn=False,
    n_classes=2,
):
    """Run one CV fold for either a Keras model or a sklearn classifier.

    :param build_fn: Zero-arg callable returning a fresh model/classifier.
    :param X_train: Training features.
    :param y_train: Training integer labels.
    :param X_test: Test features.
    :param y_test: Test integer labels.
    :param epochs: Keras training epochs (ignored for sklearn).
    :param batch_size: Keras batch size (ignored for sklearn).
    :param fold: Fold index (recorded in result dict).
    :param is_sklearn: True for sklearn estimators.
    :param n_classes: Number of classes (used for AUC gating).
    :returns: Dict of per-fold metrics.
    """
    from sklearn.metrics import roc_auc_score

    is_binary = n_classes == 2

    if is_sklearn:
        clf = build_fn()
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        acc = float(clf.score(X_test, y_test))
        pred_time = time.perf_counter() - t1

        auc = None
        if is_binary and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            if proba.shape[1] > 1:
                try:
                    auc = float(roc_auc_score(y_test, proba[:, 1]))
                except Exception:
                    pass

        geometry = None
        if hasattr(clf, "geometry_summary"):
            geometry = clf.geometry_summary()

        return {
            "fold": fold,
            "n_params": 0,
            "test_loss": None,
            "test_acc": acc,
            "test_auc": auc,
            "wall_time": fit_time + pred_time,
            "fit_time": fit_time,
            "pred_time": pred_time,
            "convergence_epoch": None,
            "geometry": geometry,
        }
    else:
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
        wall_time = time.perf_counter() - t0
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        auc = None
        if is_binary:
            probs = model.predict(X_test, verbose=0)[:, 1]
            try:
                auc = float(roc_auc_score(y_test, probs))
            except Exception:
                pass

        conv_epoch = None
        for i, a in enumerate(history.history.get("accuracy", [])):
            if a >= 0.90:
                conv_epoch = i
                break

        return {
            "fold": fold,
            "n_params": n_params,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": auc,
            "wall_time": wall_time,
            "fit_time": wall_time,
            "pred_time": None,
            "convergence_epoch": conv_epoch,
            "geometry": None,
            "train_acc": [float(a) for a in history.history.get("accuracy", [])],
            "val_acc": [float(a) for a in history.history.get("val_accuracy", [])],
            "train_loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _draw_arch_schematics(ax, arch_layers, colors):
    """Draw schematic network diagrams (adapted from digits benchmark).

    :param ax: Matplotlib axes.
    :param arch_layers: Dict name → list[layer_sizes].
    :param colors: Dict name → colour string.
    """
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Network Architecture Schematics", fontsize=10, fontweight="bold")

    names = list(arch_layers.keys())
    n = len(names)
    col_w = 1.0 / n
    max_layers = max(len(v) for v in arch_layers.values())
    max_size = max(s for layers in arch_layers.values() for s in layers)

    box_h = 0.09
    y_top, y_bot = 0.82, 0.08
    y_step = (y_top - y_bot) / max(max_layers - 1, 1)

    for i, name in enumerate(names):
        layers = arch_layers[name]
        color = colors.get(name, "gray")
        x_ctr = (i + 0.5) * col_w
        max_box_w = col_w * 0.82

        ax.text(
            x_ctr,
            0.95,
            name.split("(")[0].strip(),
            ha="center",
            va="top",
            fontsize=7,
            fontweight="bold",
            color=color,
        )

        prev_y = None
        for j, size in enumerate(layers):
            yc = y_top - j * y_step
            w = max_box_w * math.log(size + 1) / math.log(max_size + 1)
            rect = FancyBboxPatch(
                (x_ctr - w / 2, yc - box_h / 2),
                w,
                box_h,
                boxstyle="round,pad=0.008",
                facecolor=color,
                alpha=0.78,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(rect)
            label = f"{size:,}" if size < 10000 else f"{size // 1000}K"
            ax.text(
                x_ctr,
                yc,
                label,
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                fontweight="bold",
            )
            if prev_y is not None:
                ax.annotate(
                    "",
                    xy=(x_ctr, yc + box_h / 2 + 0.005),
                    xytext=(x_ctr, prev_y - box_h / 2 - 0.005),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )
            prev_y = yc


def plot_results(all_results, meta, intrinsic_dim, save_path, elapsed=None):
    """Six-panel comparison figure.

    Panels: per-fold boxplot | training loss | accuracy bars |
            AUC bars (binary) or class count (multi) | param bars | arch schematics

    :param all_results: Dict name → list[fold_result].
    :param meta: Dataset metadata dict.
    :param intrinsic_dim: Discovered bottleneck d.
    :param save_path: PNG output path.
    :param elapsed: Total run time in seconds (optional).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    d = intrinsic_dim
    input_dim = meta["input_dim"]
    n_classes = meta["n_classes"]
    is_binary = n_classes == 2
    dataset_name = meta["name"]

    h1 = min(256, max(64, 8 * input_dim))
    h2 = min(128, max(32, 4 * input_dim))

    colors = {
        f"Standard ({h1}→{h2})": "steelblue",
        f"Manifold (2d→d, d={d})": "firebrick",
        f"Wide Manifold (d+1={d + 1})": "forestgreen",
        f"Manifold+ManifoldAdam (d={d})": "darkred",
        f"PCA→{d}D + MLP (2d→d)": "darkorchid",
        f"Intrinsic Dim (PCA→{d}D→C)": "darkorange",
        "ManifoldModel (τ=0.9)": "teal",
        "ManifoldModel": "teal",
        "Euclidean KNN (k=7)": "saddlebrown",
        "Euclidean KNN": "saddlebrown",
    }
    for name in all_results:
        if name not in colors:
            if "ManifoldModel" in name:
                colors[name] = "teal"
            elif "KNN" in name:
                colors[name] = "saddlebrown"

    arch_layers = {
        f"Standard ({h1}→{h2})": [input_dim, h1, h2, n_classes],
        f"Manifold (2d→d, d={d})": [input_dim, 2 * d, d, n_classes],
        f"Wide Manifold (d+1={d + 1})": [input_dim, d + 1, n_classes],
        f"Manifold+ManifoldAdam (d={d})": [input_dim, 2 * d, d, n_classes],
        f"PCA→{d}D + MLP (2d→d)": [d, 2 * d, d, n_classes],
        f"Intrinsic Dim (PCA→{d}D→C)": [d, n_classes],
    }

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    elapsed_str = f"  |  {elapsed:.0f}s" if elapsed else ""

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.4, wspace=0.3)
    ax_box = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_auc = fig.add_subplot(gs[1, 1])
    ax_par = fig.add_subplot(gs[2, 0])
    ax_arch = fig.add_subplot(gs[2, 1])

    fig.suptitle(
        f"{dataset_name}: Manifold Architecture Comparison (d={d})\n"
        f"{input_dim}D → manifold discovery → architecture  |  "
        f"{n_classes} classes  |  5-fold CV{elapsed_str}  |  {timestamp}",
        fontsize=13,
        fontweight="bold",
    )

    names = list(all_results.keys())
    means = [np.mean([r["test_acc"] for r in all_results[n]]) for n in names]
    stds = [np.std([r["test_acc"] for r in all_results[n]]) for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    short = [n.split("(")[0].strip() for n in names]

    # ---- per-fold boxplot ----
    bp = ax_box.boxplot(
        [[r["test_acc"] for r in all_results[n]] for n in names],
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], bar_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_box.set_xticks(range(1, len(names) + 1))
    ax_box.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax_box.set_ylabel("CV Fold Accuracy")
    ax_box.set_title("Per-Fold Accuracy Distribution (5-Fold CV)")
    ax_box.grid(True, alpha=0.3, axis="y")

    # ---- training loss (Keras only) ----
    keras_names = [n for n in names if any(r.get("train_loss") for r in all_results[n])]
    for name in keras_names:
        loss_curves = [r["train_loss"] for r in all_results[name] if r.get("train_loss")]
        if not loss_curves:
            continue
        max_len = max(len(c) for c in loss_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in loss_curves]
        losses = np.array(padded)
        ep = np.arange(1, losses.shape[1] + 1)
        c = colors.get(name, "gray")
        ax_loss.plot(
            ep,
            losses.mean(0),
            "-",
            label=name.split("(")[0].strip(),
            linewidth=2,
            color=c,
        )
        ax_loss.fill_between(
            ep,
            losses.mean(0) - losses.std(0),
            losses.mean(0) + losses.std(0),
            alpha=0.15,
            color=c,
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Loss (Keras models, mean ± std)")
    ax_loss.legend(fontsize=6)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3)

    # ---- accuracy bars ----
    bars = ax_acc.bar(short, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
    for bar, m in zip(bars, means):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8,
        )
    ax_acc.set_ylabel("Mean CV Test Accuracy")
    ax_acc.set_title("Final Test Accuracy (mean ± std)")
    ax_acc.set_ylim(0, min(1.0, float(max(means)) * 1.22))
    ax_acc.tick_params(axis="x", labelsize=7, rotation=45)
    for label in ax_acc.get_xticklabels():
        label.set_ha("right")
    ax_acc.grid(True, alpha=0.3, axis="y")

    # ---- AUC bars (binary) or parameter efficiency (multi-class) ----
    if is_binary:
        auc_means, auc_stds = [], []
        for n in names:
            aucs = [r["test_auc"] for r in all_results[n] if r.get("test_auc") is not None]
            auc_means.append(np.mean(aucs) if aucs else 0.0)
            auc_stds.append(np.std(aucs) if aucs else 0.0)
        bars_a = ax_auc.bar(short, auc_means, yerr=auc_stds, color=bar_colors, alpha=0.8, capsize=5)
        for bar, m in zip(bars_a, auc_means):
            if m > 0:
                ax_auc.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{m:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=8,
                )
        ax_auc.set_ylabel("Mean AUC-ROC")
        ax_auc.set_title("AUC-ROC (binary classification, mean ± std)")
        ax_auc.set_ylim(0.5, 1.05)
        ax_auc.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random")
        ax_auc.tick_params(axis="x", labelsize=7, rotation=45)
        for label in ax_auc.get_xticklabels():
            label.set_ha("right")
        ax_auc.grid(True, alpha=0.3, axis="y")
    else:
        eff = []
        for n in names:
            p = all_results[n][0]["n_params"]
            m = np.mean([r["test_acc"] for r in all_results[n]])
            eff.append(m / max(p, 1) * 1000)
        bars_e = ax_auc.bar(short, eff, color=bar_colors, alpha=0.8)
        for bar, e in zip(bars_e, eff):
            ax_auc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.05,
                f"{e:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax_auc.set_ylabel("Accuracy / Kparam")
        ax_auc.set_title("Parameter Efficiency (acc per 1K params)")
        ax_auc.tick_params(axis="x", labelsize=7, rotation=45)
        for label in ax_auc.get_xticklabels():
            label.set_ha("right")
        ax_auc.grid(True, alpha=0.3, axis="y")

    # ---- parameter bars ----
    param_counts = [max(all_results[n][0]["n_params"], 1) for n in names]
    bars_p = ax_par.bar(short, param_counts, color=bar_colors, alpha=0.8)
    for bar, n, p in zip(bars_p, names, param_counts):
        actual = all_results[n][0]["n_params"]
        lbl = "geometric" if actual == 0 else f"{actual:,}"
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            lbl,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax_par.set_ylabel("Parameters")
    ax_par.set_title("Parameter Count (log scale)")
    ax_par.set_yscale("log")
    ax_par.tick_params(axis="x", labelsize=7, rotation=45)
    for label in ax_par.get_xticklabels():
        label.set_ha("right")
    ax_par.grid(True, alpha=0.3, axis="y")

    # ---- architecture schematics ----
    _draw_arch_schematics(ax_arch, arch_layers, colors)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Disease Manifold Architecture Benchmark")
    parser.add_argument(
        "--dataset",
        default="breast_cancer",
        choices=list(DATASET_LOADERS.keys()),
        help="Clinical dataset to benchmark",
    )
    parser.add_argument(
        "--alzheimers-csv",
        default=_DEFAULT_ALZHEIMERS_CSV,
        metavar="PATH",
        help=(
            f"Path to OASIS cross-sectional CSV (default: {_DEFAULT_ALZHEIMERS_CSV}). "
            "Download with: "
            f"curl -L '{_OASIS_RAW_URL}' -o benchmarks/data/oasis_cross-sectional.csv"
        ),
    )
    parser.add_argument("--epochs", type=int, default=150, help="Keras training epochs")
    parser.add_argument("--trials", type=int, default=3, help="Keras trials per fold")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--k-pca", type=int, default=20, help="Local PCA neighborhood")
    parser.add_argument("--k-graph", type=int, default=10, help="ManifoldModel KNN graph")
    parser.add_argument("--k-vote", type=int, default=7, help="ManifoldModel voting k")
    parser.add_argument(
        "--discovery-samples",
        type=int,
        default=0,
        help="Points for dim discovery (0 = all)",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figure from existing results JSON without running any training",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all datasets sequentially (ignores --dataset)",
    )
    args = parser.parse_args()

    if args.plot_only:
        out_stem = f"{args.dataset}_disease_architecture_results"
        out_dir = Path(__file__).resolve().parent
        json_path = out_dir / f"{out_stem}.json"
        png_path = str(out_dir / f"{out_stem}.png")
        if not json_path.exists():
            print(f"No results file found: {json_path}")
            sys.exit(1)
        with open(json_path) as f:
            saved = json.load(f)
        plot_results(
            saved["results"],
            saved["meta"],
            saved["intrinsic_dim"],
            png_path,
            elapsed=saved.get("elapsed_s"),
        )
        sys.exit(0)

    if args.all:
        import copy

        datasets = list(DATASET_LOADERS.keys())
        base_argv = [a for a in sys.argv[1:] if a not in ("--all",)]
        print(f"\n{'=' * 70}")
        print(f"FULL SWEEP — {len(datasets)} datasets")
        print(f"{'=' * 70}")
        for ds in datasets:
            print(f"\n{'─' * 70}")
            sys.argv = [sys.argv[0], "--dataset", ds] + base_argv
            try:
                main()
            except SystemExit:
                pass
            except Exception as exc:
                print(f"[SKIP] {ds}: {exc}")
        sys.argv = copy.copy(sys.argv)  # restore
        return

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    loader_fn, needs_csv = DATASET_LOADERS[args.dataset]
    print(f"\nLoading dataset: {args.dataset}")

    if needs_csv:
        csv_path = args.alzheimers_csv
        if not Path(csv_path).exists():
            print(f"\nERROR: OASIS CSV not found at '{csv_path}'")
            print("Download with:")
            print("  mkdir -p benchmarks/data")
            print(f"  curl -L '{_OASIS_RAW_URL}' -o benchmarks/data/oasis_cross-sectional.csv")
            sys.exit(1)
        X, y, meta = loader_fn(csv_path)
    else:
        X, y, meta = loader_fn()

    input_dim = meta["input_dim"]
    n_classes = meta["n_classes"]
    is_binary = n_classes == 2

    # Drop near-zero-variance columns before scaling — StandardScaler produces
    # very large or inf values for near-constant features, which overflow
    # float32/float64 matmul in PCA.
    col_std = X.std(axis=0)
    zero_var = col_std < 1e-6
    if zero_var.any():
        dropped = [meta["features"][i] for i in np.where(zero_var)[0] if i < len(meta["features"])]
        print(f"  [drop] near-zero-variance features: {dropped}")
        X = X[:, ~zero_var]
        meta["features"] = [f for f, z in zip(meta["features"], zero_var) if not z]
        meta["input_dim"] = int(X.shape[1])

    scaler = StandardScaler()
    # Cast to float32 first, then nan_to_num — doing it after avoids re-introducing
    # inf when float64.max (~1.8e308) is silently truncated back to inf in float32.
    X = np.nan_to_num(scaler.fit_transform(X).astype("float32"))

    print(f"  {meta['name']}")
    print(f"  {X.shape[0]} samples  |  {input_dim} features  |  {n_classes} classes")
    print(f"  Classes: {meta['class_names']}")
    if "class_balance" in meta:
        print(f"  Balance: {meta['class_balance']}")
    print("  Evaluation: 5-fold stratified cross-validation")

    # -----------------------------------------------------------------------
    # Phase 1: Manifold discovery
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: MANIFOLD DISCOVERY")
    print("=" * 70)

    n_disc = args.discovery_samples if args.discovery_samples > 0 else X.shape[0]
    n_disc = min(n_disc, X.shape[0])
    k_disc = min(args.k_pca, n_disc - 1)

    print(f"\nUsing {n_disc} points, k={k_disc} neighbors (SVD-based local PCA)...")
    t0 = time.perf_counter()
    dim_report = discover_dimensionality(
        X,
        n_samples=n_disc,
        k=k_disc,
        variance_thresholds=(0.95, 0.90, 0.85, 0.80),
    )
    discovery_time = time.perf_counter() - t0
    print(f"Discovery time: {discovery_time:.1f}s\n")

    print(f"{'τ':>6} {'Mean d':>8} {'Std':>6} {'Min':>5} {'Max':>5} {'Noise %':>8}")
    print("-" * 45)
    for tau in sorted(dim_report.keys(), reverse=True):
        r = dim_report[tau]
        noise_pct = 100 * (1 - r["mean"] / input_dim)
        print(
            f"{tau:>6.2f} {r['mean']:>8.1f} {r['std']:>6.1f}"
            f" {r['min']:>5} {r['max']:>5} {noise_pct:>7.1f}%"
        )

    n_per_class = min(50, X.shape[0] // n_classes)
    print(f"\nPer-class intrinsic dimensionality (τ={args.tau}, {n_per_class} samples/class):")
    class_dims = discover_per_class_dimensionality(
        X,
        y,
        k=k_disc,
        tau=args.tau,
        n_samples_per_class=n_per_class,
    )
    for c in sorted(class_dims.keys()):
        cd = class_dims[c]
        label = meta["class_names"][c] if c < len(meta["class_names"]) else str(c)
        print(f"  {label:30s}: d = {cd['mean']:.1f} ± {cd['std']:.1f}  [{cd['min']}, {cd['max']}]")

    global_dim = int(round(dim_report[args.tau]["mean"]))
    intrinsic_dim = max(cd["max"] for cd in class_dims.values())
    d = intrinsic_dim
    noise_pct = 100 * (1 - d / input_dim)

    print(f"\n>> Global intrinsic dim (mean): {global_dim}")
    print(f">> Max per-class max:            {intrinsic_dim}  →  using d = {d}  (τ={args.tau})")
    print(
        f">> Noise dimensions suppressed:  {noise_pct:.1f}%  ({input_dim - d} of {input_dim} dims)"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Architecture summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE COMPARISON")
    print("=" * 70)

    pca = skPCA(n_components=d)
    pca.fit(X)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {d}D captures {var_explained * 100:.1f}% of global variance")

    V_d = make_basis(pca)

    h1 = min(256, max(64, 8 * input_dim))
    h2 = min(128, max(32, 4 * input_dim))

    _sample_builds = {
        f"Standard ({h1}→{h2})": lambda: build_clinical_standard_model(
            input_dim, n_classes, lr=args.lr
        ),
        f"Manifold (2d→d, d={d})": lambda: build_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"Wide Manifold (d+1={d + 1})": lambda: build_wide_manifold_model(
            input_dim, n_classes, d, lr=args.lr
        ),
        f"Manifold+ManifoldAdam (d={d})": lambda: build_manifold_model(
            input_dim,
            n_classes,
            d,
            lr=args.lr,
            optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
        ),
        f"PCA→{d}D + MLP (2d→d)": lambda: build_pca_model(n_classes, d, lr=args.lr),
        f"Intrinsic Dim (PCA→{d}D→C)": lambda: build_pca_intrinsic_dim_model(
            n_classes, d, lr=args.lr
        ),
    }

    for name, bfn in _sample_builds.items():
        model = bfn()
        n_params = count_params(model)
        print(f"  {name:<42} {n_params:>8,} params")

    print(f"  {'ManifoldModel (τ=' + str(args.tau) + ')':<42}        0 params  (pure geometry)")
    print(f"  {'Euclidean KNN (k=' + str(args.k_vote) + ')':<42}        0 params  (non-parametric)")

    # -----------------------------------------------------------------------
    # Phase 3: 5-fold CV
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    print(f"  Folds: 5  |  Keras trials/fold: {args.trials}  |  Epochs: {args.epochs}")
    print(
        f"  Batch: {args.batch_size}  |  LR: {args.lr}  |  τ: {args.tau}"
        + ("  |  AUC-ROC reported" if is_binary else "")
    )

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_splits = list(skf.split(X, y))

    architectures = {
        f"Euclidean KNN (k={args.k_vote})": (
            lambda: KNeighborsClassifier(n_neighbors=args.k_vote, metric="euclidean"),
            True,
            False,
        ),
        f"ManifoldModel (τ={args.tau})": (
            lambda: ManifoldModel(
                k_graph=args.k_graph,
                k_pca=args.k_pca,
                k_vote=args.k_vote,
                variance_threshold=args.tau,
            ),
            True,
            False,
        ),
        f"Standard ({h1}→{h2})": (
            lambda: build_clinical_standard_model(input_dim, n_classes, lr=args.lr),
            False,
            False,
        ),
        f"Manifold (2d→d, d={d})": (
            lambda: build_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"Wide Manifold (d+1={d + 1})": (
            lambda: build_wide_manifold_model(input_dim, n_classes, d, lr=args.lr),
            False,
            False,
        ),
        f"Manifold+ManifoldAdam (d={d})": (
            lambda: build_manifold_model(
                input_dim,
                n_classes,
                d,
                lr=args.lr,
                optimizer=ManifoldAdam(basis=V_d, learning_rate=args.lr),
            ),
            False,
            False,
        ),
        f"PCA→{d}D + MLP (2d→d)": (
            lambda: build_pca_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
        f"Intrinsic Dim (PCA→{d}D→C)": (
            lambda: build_pca_intrinsic_dim_model(n_classes, d, lr=args.lr),
            False,
            True,
        ),
    }

    all_results = {}

    for name, (build_fn, is_sklearn, needs_pca) in architectures.items():
        print(f"\n  {name}...")
        fold_results = []

        for fold_i, (tr_idx, te_idx) in enumerate(fold_splits):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            if needs_pca:
                pca_fold = skPCA(n_components=d)
                # Zero out any non-finite values (inf/nan) so PCA's X.T @ X
                # never overflows — posinf=0/neginf=0 treats the feature as
                # being at its mean rather than propagating a sentinel large value.
                X_tr_f = np.nan_to_num(X_tr.astype(np.float64), posinf=0.0, neginf=0.0)
                X_te_f = np.nan_to_num(X_te.astype(np.float64), posinf=0.0, neginf=0.0)
                X_tr = pca_fold.fit_transform(X_tr_f).astype("float32")
                X_te = pca_fold.transform(X_te_f).astype("float32")

            if is_sklearn:
                result = run_fold(
                    build_fn,
                    X_tr,
                    y_tr,
                    X_te,
                    y_te,
                    args.epochs,
                    args.batch_size,
                    fold_i,
                    is_sklearn=True,
                    n_classes=n_classes,
                )
                fold_results.append(result)
            else:
                for trial in range(args.trials):
                    np.random.seed(trial * 100 + fold_i)
                    tf.random.set_seed(trial * 100 + fold_i)
                    result = run_fold(
                        build_fn,
                        X_tr,
                        y_tr,
                        X_te,
                        y_te,
                        args.epochs,
                        args.batch_size,
                        fold_i,
                        is_sklearn=False,
                        n_classes=n_classes,
                    )
                    fold_results.append(result)

        all_results[name] = fold_results

        mean_acc = np.mean([r["test_acc"] for r in fold_results])
        std_acc = np.std([r["test_acc"] for r in fold_results])
        total_t = sum(r["wall_time"] for r in fold_results)
        extra = ""
        if is_binary:
            aucs = [r["test_auc"] for r in fold_results if r.get("test_auc") is not None]
            if aucs:
                extra = f"  AUC={np.mean(aucs):.4f}"
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in fold_results if r.get("geometry")]
            if geoms:
                mid = np.mean([g["mean_intrinsic_dim"] for g in geoms if g])
                extra += f"  id={mid:.1f}"
        print(f"    {mean_acc:.4f} ± {std_acc:.4f}  ({total_t:.1f}s){extra}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — 5-FOLD STRATIFIED CV")
    print("=" * 70)
    print(f"Dataset: {meta['name']}  ({input_dim}D, {n_classes} classes, {X.shape[0]} samples)")
    print(
        f"Intrinsic dim: d = {d}  (τ={args.tau})  |  "
        f"Noise: {noise_pct:.1f}%  ({input_dim - d}/{input_dim} dims)"
    )
    print(f"Epochs: {args.epochs}  |  Trials/fold: {args.trials}  |  Total time: {elapsed:.1f}s")
    print("-" * 70)

    best_acc = max(np.mean([r["test_acc"] for r in all_results[n]]) for n in all_results)
    auc_col = "    AUC" if is_binary else ""
    print(f"\n{'Architecture':<44} {'Acc':>8} {'±Std':>7}{auc_col} {'Params':>10} {'Time':>8}")
    print("-" * (78 + (7 if is_binary else 0)))

    for name in all_results:
        results = all_results[name]
        accs = [r["test_acc"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_t = np.mean([r["wall_time"] for r in results])
        n_params = results[0]["n_params"]
        params_str = "non-param" if n_params == 0 else f"{n_params:,}"
        marker = "  << BEST" if abs(mean_acc - best_acc) < 1e-9 else ""
        auc_str = ""
        if is_binary:
            aucs = [r["test_auc"] for r in results if r.get("test_auc") is not None]
            auc_str = f" {np.mean(aucs):>6.4f}" if aucs else "       "
        print(
            f"{name:<44} {mean_acc:>8.4f} {std_acc:>7.4f}{auc_str}"
            f" {params_str:>10} {mean_t:>7.1f}s{marker}"
        )

    print("\n" + "-" * 70)
    print("MANIFOLD GEOMETRY (ManifoldModel folds):")
    for name in all_results:
        if name.startswith("ManifoldModel"):
            geoms = [r["geometry"] for r in all_results[name] if r.get("geometry")]
            if geoms:
                mid = np.mean([g["mean_intrinsic_dim"] for g in geoms if g])
                ambient = geoms[0].get("ambient_dim", input_dim)
                print(
                    f"  mean intrinsic dim = {mid:.1f} / {ambient}  "
                    f"({100 * (1 - mid / ambient):.0f}% noise suppressed)"
                )

    std_key = [n for n in all_results if n.startswith("Standard")][0]
    best_key = max(all_results, key=lambda n: np.mean([r["test_acc"] for r in all_results[n]]))
    best_mean = np.mean([r["test_acc"] for r in all_results[best_key]])
    std_mean = np.mean([r["test_acc"] for r in all_results[std_key]])

    print("\n" + "-" * 70)
    if best_key != std_key:
        delta = best_mean - std_mean
        print(f">> WINNER: {best_key}")
        print(f"   {best_mean:.4f} vs {std_mean:.4f} (standard MLP)  Δ = +{delta:.4f}")
        n_p_best = all_results[best_key][0]["n_params"]
        n_p_std = all_results[std_key][0]["n_params"]
        if n_p_best == 0:
            print("   Zero learned parameters — pure manifold geometry")
        elif n_p_best < n_p_std:
            print(
                f"   {100 * (1 - n_p_best / n_p_std):.0f}% fewer parameters"
                f"  ({n_p_best:,} vs {n_p_std:,})"
            )
    else:
        print(f">> Standard architecture wins: {std_mean:.4f}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    out_stem = f"{args.dataset}_disease_architecture_results"
    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / f"{out_stem}.json"

    save_data = {
        "device": DEVICE_INFO,
        "dataset": args.dataset,
        "meta": meta,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "intrinsic_dim": d,
        "global_intrinsic_dim_mean": global_dim,
        "noise_pct": float(noise_pct),
        "tau": args.tau,
        "epochs": args.epochs,
        "trials": args.trials,
        "n_folds": n_folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "k_pca": args.k_pca,
        "k_graph": args.k_graph,
        "k_vote": args.k_vote,
        "elapsed_s": float(elapsed),
        "dimensionality_report": {str(k): v for k, v in dim_report.items()},
        "per_class_dims": {str(k): v for k, v in class_dims.items()},
        "results": {name: results for name, results in all_results.items()},
    }

    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults → {json_path}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    if args.plot:
        png_path = str(out_dir / f"{out_stem}.png")
        plot_results(all_results, meta, d, png_path, elapsed=elapsed)


if __name__ == "__main__":
    main()
