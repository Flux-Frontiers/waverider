#!/usr/bin/env python3
"""KAN classifier on heart disease manifold intrinsic coordinates.

Hypothesis: A KAN trained on the d*=9 PCA intrinsic coordinates of the
heart disease manifold can match or exceed ManifoldModel accuracy while
exposing a human-readable symbolic formula for the decision boundary.

Compares:
  - ManifoldModel   (0 params, pure geometry)
  - KAN-raw         (KAN on all 13 ambient features)
  - KAN-pca9        (KAN on 9 intrinsic PCA coordinates)

After cross-validation, runs symbolic regression on the full dataset to
extract a closed-form decision boundary in intrinsic coordinate space.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent / "src"))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from waverider.manifold_model import ManifoldModel  # noqa: E402

# ── Dataset loading ──────────────────────────────────────────────────────────


def load_heart() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Cleveland Heart Disease from UCI ML Repository.

    Preprocessing matches disease_manifold_architecture.py: impute → drop
    near-zero-variance → StandardScaler → nan_to_num.

    :return: X (303, 13) float32 standardized, y (303,) int, feature_names
    """
    from sklearn.preprocessing import StandardScaler
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(id=45)
    X_df = repo.data.features.copy()
    y_df = repo.data.targets.copy()

    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes

    feature_names = list(X_df.columns)
    X = SimpleImputer(strategy="median").fit_transform(X_df).astype("float32")

    # Drop near-zero-variance before scaling (StandardScaler produces inf on constants)
    stds = X.std(axis=0)
    keep = stds > 1e-6
    X = X[:, keep]
    feature_names = [f for f, k in zip(feature_names, keep) if k]

    X = np.nan_to_num(StandardScaler().fit_transform(X).astype("float32"))

    y_raw = y_df.iloc[:, 0].values
    y = (y_raw > 0).astype(int)

    return X, y, feature_names


# ── KAN helpers ──────────────────────────────────────────────────────────────


def _make_dataset(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    return {
        "train_input": torch.tensor(X_tr, dtype=torch.float32),
        "train_label": torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        "test_input": torch.tensor(X_te, dtype=torch.float32),
        "test_label": torch.tensor(y_te, dtype=torch.float32).unsqueeze(1),
    }


def _bce(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, tgt)


def _train_kan(
    dataset: dict,
    n_in: int,
    steps: int = 300,
    lamb: float = 0.005,
    hidden: int = 3,
    verbose: bool = False,
) -> object:
    from kan import KAN

    # grid_range must cover the actual data range; PCA coords reach ±5 on this dataset
    model = KAN(
        width=[n_in, hidden, 1],
        grid=5,
        k=3,
        seed=42,
        auto_save=False,
        grid_range=[-5, 5],
    )
    log_freq = steps if not verbose else 50
    model.fit(
        dataset,
        opt="Adam",
        lr=5e-3,
        steps=steps,
        lamb=lamb,
        loss_fn=_bce,
        log=log_freq,
        save_fig=False,
        update_grid=True,
    )
    return model


def _kan_predict(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return (logits.squeeze() > 0).numpy().astype(int)


def _kan_proba(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits.squeeze()).numpy()


# ── Cross-validation ─────────────────────────────────────────────────────────

D_STAR = 9  # intrinsic dimensionality at τ=0.90 (established in prior benchmarks)
N_FOLDS = 5
SEED = 42


def run_cv(X: np.ndarray, y: np.ndarray) -> dict:
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results: dict[str, list] = {
        "manifold_acc": [],
        "manifold_auc": [],
        "kan_raw_acc": [],
        "kan_raw_auc": [],
        "kan_pca_acc": [],
        "kan_pca_auc": [],
    }

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ── ManifoldModel (reference) ────────────────────────────────────────
        mm = ManifoldModel(k_graph=15, k_pca=50, k_vote=7, variance_threshold=0.90)
        mm.fit(X_tr, y_tr)
        mm_pred = mm.predict(X_te)
        results["manifold_acc"].append(float(accuracy_score(y_te, mm_pred)))
        results["manifold_auc"].append(float(roc_auc_score(y_te, mm_pred)))

        # ── KAN on raw 13-dim features ───────────────────────────────────────
        ds_raw = _make_dataset(X_tr, y_tr, X_te, y_te)
        kan_raw = _train_kan(ds_raw, n_in=X.shape[1])
        raw_pred = _kan_predict(kan_raw, X_te)
        raw_proba = _kan_proba(kan_raw, X_te)
        results["kan_raw_acc"].append(float(accuracy_score(y_te, raw_pred)))
        results["kan_raw_auc"].append(float(roc_auc_score(y_te, raw_proba)))

        # ── KAN on PCA-9 intrinsic coordinates ──────────────────────────────
        pca = PCA(n_components=D_STAR, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr).astype("float32")
        X_te_pca = pca.transform(X_te).astype("float32")
        ds_pca = _make_dataset(X_tr_pca, y_tr, X_te_pca, y_te)
        kan_pca = _train_kan(ds_pca, n_in=D_STAR)
        pca_pred = _kan_predict(kan_pca, X_te_pca)
        pca_proba = _kan_proba(kan_pca, X_te_pca)
        results["kan_pca_acc"].append(float(accuracy_score(y_te, pca_pred)))
        results["kan_pca_auc"].append(float(roc_auc_score(y_te, pca_proba)))

        print(
            f"  Fold {fold + 1}/{N_FOLDS}  "
            f"ManifoldModel={results['manifold_acc'][-1]:.3f}  "
            f"KAN-raw={results['kan_raw_acc'][-1]:.3f}  "
            f"KAN-pca9={results['kan_pca_acc'][-1]:.3f}"
        )

    return results


def print_results(results: dict) -> None:
    print("\n── 5-Fold CV Results ─────────────────────────────────────────────")
    header = f"{'Model':<20} {'Accuracy':>12} {'AUC-ROC':>12}  Params"
    print(header)
    print("─" * len(header))
    rows = [
        ("ManifoldModel", "manifold_acc", "manifold_auc", "0"),
        ("KAN-raw (13-D)", "kan_raw_acc", "kan_raw_auc", "learnable splines"),
        ("KAN-pca9 (9-D)", "kan_pca_acc", "kan_pca_auc", "learnable splines"),
    ]
    for label, acc_key, auc_key, params in rows:
        accs = results[acc_key]
        aucs = results[auc_key]
        mu_a, se_a = np.mean(accs), np.std(accs) / np.sqrt(len(accs))
        mu_r, se_r = np.mean(aucs), np.std(aucs) / np.sqrt(len(aucs))
        print(f"  {label:<18} {mu_a:.4f}±{se_a:.4f}  {mu_r:.4f}±{se_r:.4f}  {params}")


# ── Symbolic regression on full dataset ─────────────────────────────────────


def run_symbolic_regression(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    print("\n── Symbolic Regression (full dataset, PCA-9 coordinates) ─────────")

    import tempfile

    from kan import KAN

    pca_full = PCA(n_components=D_STAR, random_state=SEED)
    X_pca = pca_full.fit_transform(X).astype("float32")
    pc_names = [f"PC{i + 1}" for i in range(D_STAR)]

    ds_full = _make_dataset(X_pca, y, X_pca, y)
    # prune() needs a writable ckpt_path even with auto_save=True; use a temp dir
    _ckpt = tempfile.mkdtemp(prefix="kan_heart_")
    model = KAN(
        width=[D_STAR, 3, 1],
        grid=7,
        k=3,
        seed=SEED,
        auto_save=True,
        ckpt_path=_ckpt,
        grid_range=[-5, 5],
    )
    # Phase 1: learn the function with no regularization so splines are non-trivial
    print("  Phase 1 — fit without regularization (400 steps)...")
    model.fit(
        ds_full,
        opt="Adam",
        lr=1e-2,
        steps=400,
        lamb=0.0,
        loss_fn=_bce,
        log=400,
        update_grid=True,
    )
    # Phase 2: sparsify while keeping the learned shape
    print("  Phase 2 — sparsify (400 steps, lamb=5e-4)...")
    model.fit(
        ds_full,
        opt="Adam",
        lr=3e-3,
        steps=400,
        lamb=5e-4,
        loss_fn=_bce,
        log=400,
        update_grid=False,
    )

    # Prune dead edges / neurons
    model = model.prune(node_th=0.03, edge_th=0.03)

    print("\n  Active edges after pruning — suggesting symbolic functions...\n")
    try:
        model.auto_symbolic(verbose=True)
        formula = model.symbolic_formula(var=pc_names)
        print(f"\n  Symbolic formula:\n    {formula[0][0]}")
    except Exception as exc:
        print(f"  auto_symbolic failed ({exc})")

    # Save activation plot using correct API (folder=, not img_folder=)
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _HERE.parent.parent.parent / "papers" / "clinical_manifolds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kan_heart_activations.png"
    try:
        # pykan dumps sp_*.png into `folder`; use a temp dir to keep papers/ clean
        _plot_tmp = tempfile.mkdtemp(prefix="kan_plot_")
        model.plot(folder=_plot_tmp, beta=3, in_vars=pc_names, out_vars=["disease"])
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"\n  Activation plot → {out_path}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    # PCA component loadings (which original features drive each PC)
    print("\n── PCA-9 Component Loadings (top 3 features per PC) ─────────────")
    for i in range(D_STAR):
        loadings = np.abs(pca_full.components_[i])
        top3_idx = np.argsort(loadings)[::-1][:3]
        top3 = [(feature_names[j], float(loadings[j])) for j in top3_idx]
        print(f"  PC{i + 1}: " + ", ".join(f"{n}={v:.3f}" for n, v in top3))


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    print("═" * 60)
    print("  KAN on Heart Disease Manifold Intrinsic Coordinates")
    print("═" * 60)

    print("\nLoading heart disease dataset...")
    X, y, feature_names = load_heart()
    print(f"  Shape: {X.shape}  |  classes: {np.bincount(y)} (0=healthy, 1=disease)")
    print(f"  Intrinsic dimensionality d* = {D_STAR} (τ=0.90, established prior)")
    print(f"  Features: {feature_names}")

    print(f"\nRunning {N_FOLDS}-fold CV...")
    results = run_cv(X, y)
    print_results(results)

    out_json = _HERE / "kan_heart_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_json}")

    run_symbolic_regression(X, y, feature_names)

    print("\n" + "═" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
