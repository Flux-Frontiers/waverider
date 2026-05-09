#!/usr/bin/env python3
"""KAN classifiers on clinical manifold intrinsic coordinates — all datasets.

For each dataset, compares:
  - ManifoldModel   (0 params, pure geometry)
  - KAN-raw         (KAN on all ambient features)
  - KAN-pca         (KAN on d*-dimensional PCA intrinsic coordinates)

Then runs symbolic regression on the intrinsic coordinates of the full
training set to extract a closed-form decision boundary.

Usage:
    python kan_clinical.py               # all datasets
    python kan_clinical.py --dataset heart
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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
from sklearn.preprocessing import StandardScaler  # noqa: E402

from waverider.dimensionality_discovery import discover_dimensionality  # noqa: E402
from waverider.manifold_model import ManifoldModel  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

N_FOLDS = 5
SEED = 42
TAU = 0.90


# ── Dataset loaders ──────────────────────────────────────────────────────────


def _uci(dataset_id, retries=3, pause=2.0):
    """Fetch a UCI ML Repo dataset with simple retry on connection error."""
    from ucimlrepo import fetch_ucirepo

    for attempt in range(retries):
        try:
            return fetch_ucirepo(id=dataset_id)
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
            else:
                raise RuntimeError(f"UCI fetch failed for id={dataset_id}: {exc}") from exc


def _prep(X_df, y_raw, binary_threshold=None, remap_classes=True):
    """Common post-processing: impute → drop zero-var → scale → binarize."""
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = X_df[col].astype("category").cat.codes

    X = SimpleImputer(strategy="median").fit_transform(X_df).astype("float32")
    stds = X.std(axis=0)
    keep = stds > 1e-6
    X = X[:, keep]
    names = [c for c, k in zip(X_df.columns, keep) if k]

    X = np.nan_to_num(StandardScaler().fit_transform(X).astype("float32"))

    if binary_threshold is not None:
        y = (y_raw > binary_threshold).astype(int)
    elif remap_classes:
        uniq = sorted(np.unique(y_raw.astype(int)))
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[int(v)] for v in y_raw], dtype=int)
    else:
        y = y_raw.astype(int)

    return X, y, names


def load_heart():
    r = _uci(45)
    X, y, names = _prep(
        r.data.features.copy(), r.data.targets.iloc[:, 0].values, binary_threshold=0
    )
    return X, y, names, "Heart Disease (Cleveland)", ["No disease", "Heart disease"]


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer as _lbc

    d = _lbc()
    X = np.nan_to_num(StandardScaler().fit_transform(d.data).astype("float32"))
    stds = X.std(axis=0)
    X = X[:, stds > 1e-6]
    names = [d.feature_names[i] for i, k in enumerate(stds > 1e-6) if k]
    y = d.target.astype(int)
    return X, y, names, "Breast Cancer (Wisconsin)", d.target_names.tolist()


def load_diabetes():
    from sklearn.datasets import fetch_openml

    b = fetch_openml(data_id=37, as_frame=True, parser="auto")
    X_df = b.data.copy()
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        X_df[col] = X_df[col].astype("category").cat.codes
    X, y, names = _prep(
        X_df, (b.target.astype(str).str.strip() == "tested_positive").astype(int).values
    )
    return X, y, names, "Pima Indians Diabetes", ["Non-diabetic", "Diabetic"]


def load_parkinsons():
    r = _uci(174)
    X, y, names = _prep(r.data.features.copy(), r.data.targets.iloc[:, 0].values)
    return X, y, names, "Parkinson's Disease (Voice)", ["Healthy", "Parkinson's"]


def load_dermatology():
    r = _uci(33)
    X, y, names = _prep(r.data.features.copy(), r.data.targets.iloc[:, 0].values)
    class_names = [
        "Psoriasis",
        "Seb. Derm.",
        "Lichen Planus",
        "Pityriasis Rosea",
        "Chronic Derm.",
        "Pityriasis Rubra",
    ]
    return X, y, names, "Dermatology (Skin Disease)", class_names


LOADERS = {
    "heart": load_heart,
    "breast": load_breast_cancer,
    "diabetes": load_diabetes,
    "parkinsons": load_parkinsons,
    "dermatology": load_dermatology,
}


# ── KAN helpers ──────────────────────────────────────────────────────────────


def _make_dataset(X_tr, y_tr, X_te, y_te, n_classes):
    if n_classes == 2:
        label_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        label_te = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)
    else:
        label_tr = torch.tensor(y_tr, dtype=torch.long)
        label_te = torch.tensor(y_te, dtype=torch.long)
    return {
        "train_input": torch.tensor(X_tr, dtype=torch.float32),
        "train_label": label_tr,
        "test_input": torch.tensor(X_te, dtype=torch.float32),
        "test_label": label_te,
    }


def _loss_fn(n_classes):
    if n_classes == 2:
        return lambda p, t: F.binary_cross_entropy_with_logits(p, t)
    return lambda p, t: F.cross_entropy(p, t.squeeze().long())


def _train_kan(dataset, n_in, n_classes, steps=300, lamb=0.005, hidden=4):
    from kan import KAN

    out_width = 1 if n_classes == 2 else n_classes
    model = KAN(
        width=[n_in, hidden, out_width],
        grid=5,
        k=3,
        seed=SEED,
        auto_save=False,
        grid_range=[-5, 5],
    )
    model.fit(
        dataset,
        opt="Adam",
        lr=5e-3,
        steps=steps,
        lamb=lamb,
        loss_fn=_loss_fn(n_classes),
        log=steps,
        save_fig=False,
        update_grid=True,
    )
    return model


def _predict(model, X, n_classes):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    if n_classes == 2:
        return (out.squeeze() > 0).numpy().astype(int), torch.sigmoid(out.squeeze()).numpy()
    return out.argmax(dim=1).numpy(), F.softmax(out, dim=1).numpy()


def _auc(y_true, proba, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, proba))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


# ── Intrinsic dimensionality ─────────────────────────────────────────────────


def find_d_star(X, tau=TAU):
    """Compute global intrinsic dimensionality at variance threshold tau."""
    result = discover_dimensionality(X, variance_thresholds=(tau,))
    return int(round(result[tau]["mean"]))


# ── Cross-validation ─────────────────────────────────────────────────────────


def run_cv(X, y, d_star, n_classes):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    keys = [
        "manifold_acc",
        "manifold_auc",
        "kan_raw_acc",
        "kan_raw_auc",
        "kan_pca_acc",
        "kan_pca_auc",
    ]
    results = {k: [] for k in keys}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ManifoldModel
        mm = ManifoldModel(k_graph=15, k_pca=50, k_vote=7, variance_threshold=TAU)
        mm.fit(X_tr, y_tr)
        mm_pred = mm.predict(X_te)
        if n_classes == 2:
            mm_proba = mm_pred.astype(float)
        else:
            mm_proba = np.zeros((len(mm_pred), n_classes))
            mm_proba[np.arange(len(mm_pred)), mm_pred] = 1.0
        results["manifold_acc"].append(float(accuracy_score(y_te, mm_pred)))
        results["manifold_auc"].append(_auc(y_te, mm_proba, n_classes))

        # KAN-raw
        ds_raw = _make_dataset(X_tr, y_tr, X_te, y_te, n_classes)
        kan_r = _train_kan(ds_raw, X.shape[1], n_classes)
        r_pred, r_prob = _predict(kan_r, X_te, n_classes)
        results["kan_raw_acc"].append(float(accuracy_score(y_te, r_pred)))
        results["kan_raw_auc"].append(_auc(y_te, r_prob, n_classes))

        # KAN-pca
        pca = PCA(n_components=d_star, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr).astype("float32")
        X_te_pca = pca.transform(X_te).astype("float32")
        ds_pca = _make_dataset(X_tr_pca, y_tr, X_te_pca, y_te, n_classes)
        kan_p = _train_kan(ds_pca, d_star, n_classes)
        p_pred, p_prob = _predict(kan_p, X_te_pca, n_classes)
        results["kan_pca_acc"].append(float(accuracy_score(y_te, p_pred)))
        results["kan_pca_auc"].append(_auc(y_te, p_prob, n_classes))

        print(
            f"    Fold {fold + 1}/{N_FOLDS}  "
            f"ManifoldModel={results['manifold_acc'][-1]:.3f}  "
            f"KAN-raw={results['kan_raw_acc'][-1]:.3f}  "
            f"KAN-pca{d_star}={results['kan_pca_acc'][-1]:.3f}"
        )

    return results


# ── Symbolic regression ──────────────────────────────────────────────────────


def run_symbolic(X, y, d_star, n_classes, pc_names, class_names, out_dir: Path):
    """Two-phase KAN symbolic regression with per-edge R² gating.

    Returns a structured dict with fields:
        active_edges : list of (layer, in_node, out_node, symbol, r2)
        formula      : string formula or None
        n_active     : count of active edges after pruning

    Also returns the fitted PCA object as the second element (for caller
    compatibility with the existing ``run_dataset`` code).
    """
    import matplotlib
    from kan import KAN

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("  Symbolic regression (two-phase, full dataset)...")

    pca_full = PCA(n_components=d_star, random_state=SEED)
    X_pca = pca_full.fit_transform(X).astype("float32")

    ds_full = _make_dataset(X_pca, y, X_pca, y, n_classes)
    out_width = 1 if n_classes == 2 else n_classes
    ck = tempfile.mkdtemp(prefix="kan_sym_")
    lf = _loss_fn(n_classes)

    # ── Phase 1: learn without regularization ──────────────────────────────
    # Single-layer KAN [d*, out]: each PC maps directly to the output via one
    # spline — no hidden layer, no output-spline nesting problem. After
    # sparsification, symbolic_formula yields a clean additive formula.
    print("  Phase 1: 800 steps, λ=0, lr=1e-2 (learn shapes)")
    model = KAN(
        width=[d_star, out_width],
        grid=7,
        k=3,
        seed=SEED,
        auto_save=True,
        ckpt_path=ck,
        grid_range=[-5, 5],
    )
    model.fit(
        ds_full,
        opt="Adam",
        lr=1e-2,
        steps=800,
        lamb=0.0,
        loss_fn=lf,
        log=200,
        update_grid=True,
    )

    # ── Phase 2: aggressive sparsification ────────────────────────────────
    print("  Phase 2: 400 steps, λ=1e-3, lr=3e-3 (sparsification)")
    model.fit(
        ds_full,
        opt="Adam",
        lr=3e-3,
        steps=400,
        lamb=1e-3,
        loss_fn=lf,
        log=100,
        update_grid=False,
    )

    # ── Prune dead nodes and edges ─────────────────────────────────────────
    model = model.prune(node_th=0.03, edge_th=0.03)

    # Ensure activations are cached for auto_symbolic
    model.get_act(ds_full["train_input"])

    # ── Per-edge symbolic identification ──────────────────────────────────
    # We avoid auto_symbolic() because it assigns "0" (zeroing the mask) to
    # edges it can't fit at R²≥0.80, wiping the model before symbolic_formula.
    # Instead, suggest_symbolic with a relaxed R²≥0.60 threshold, fix only
    # good matches, and leave the rest as unfixed splines.
    active_edges = []
    act_funs = list(model.act_fun)
    n_layers = len(act_funs)
    print(f"\n  Symbolic identification on pruned model ({n_layers} layers)...")

    for layer_idx, layer in enumerate(act_funs):
        mask = layer.mask.data
        n_in, n_out = mask.shape
        for i in range(n_in):
            for j in range(n_out):
                if mask[i, j].item() < 0.5:
                    continue
                try:
                    result = model.suggest_symbolic(layer_idx, i, j, topk=1, verbose=False)
                    best_name, _, best_r2, _ = result
                    if isinstance(best_r2, (list, tuple)):
                        best_name, best_r2 = best_name[0], float(best_r2[0])
                    else:
                        best_name, best_r2 = str(best_name), float(best_r2)
                    if float(best_r2) >= 0.60:
                        model.fix_symbolic(layer_idx, i, j, best_name)
                    else:
                        best_name = "spline"
                except Exception:
                    best_name, best_r2 = "spline", 0.0
                active_edges.append((layer_idx, i, j, best_name, float(best_r2)))
                in_label = (
                    pc_names[i] if layer_idx == 0 and i < len(pc_names) else f"h{layer_idx}_{i}"
                )
                print(f"    L{layer_idx} {in_label} → {best_name}  R²={best_r2:.4f}")

    n_active = len(active_edges)
    n_symbolic = sum(1 for *_, sym, r2 in active_edges if sym not in ("0", "spline") and r2 >= 0.80)
    print(f"\n  {n_active} active edges  ({n_symbolic} identified with R²≥0.80)")

    # ── Full symbolic formula (includes Gaussian-expanded splines) ────────
    formula_str = None
    try:
        formula = model.symbolic_formula(var=pc_names)
        if formula is not None and formula[0]:
            formula_str = str(formula[0][0])
            print(f"  Formula: {formula_str[:120]}{'...' if len(formula_str) > 120 else ''}")
        else:
            print("  symbolic_formula returned empty result")
    except Exception as exc:
        print(f"  symbolic_formula failed: {exc}")

    # ── Simplified formula: zero out unidentified splines, re-extract ─────
    simplified_str = None
    if n_symbolic > 0:
        try:
            model.get_act(ds_full["train_input"])
            for li, ii, ji, sym, r2 in active_edges:
                if sym == "spline":
                    model.fix_symbolic(li, ii, ji, "0")
            model.get_act(ds_full["train_input"])
            sf = model.symbolic_formula(var=pc_names)
            if sf is not None and sf[0]:
                simplified_str = str(sf[0][0])
                print(f"  Simplified formula: {simplified_str}")
        except Exception as exc:
            print(f"  simplified_formula failed: {exc}")

    result_dict = {
        "active_edges": active_edges,
        "formula": formula_str,
        "simplified_formula": simplified_str,
        "n_active": n_active,
    }

    # ── Activation plot; pykan's per-edge sp_*.png go to a temp dir ────────
    try:
        _plot_tmp = tempfile.mkdtemp(prefix="kan_plot_")
        out_vars = [class_names[1]] if n_classes == 2 else list(class_names)
        # Scale label size down for wide networks; 0.4 works for binary, less for multiclass
        varscale = min(0.8, 4.0 / max(d_star, n_classes))
        model.plot(folder=_plot_tmp, beta=3, in_vars=pc_names, out_vars=out_vars, varscale=varscale)
        plot_path = out_dir / "activations.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Activation plot → {plot_path}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    return result_dict, pca_full


# ── Per-dataset entry point ──────────────────────────────────────────────────


def run_dataset(name: str) -> dict:
    print(f"\n{'═' * 60}")
    print(f"  {name.upper()}")
    print(f"{'═' * 60}")

    loader = LOADERS[name]
    X, y, feature_names, title, class_names = loader()
    n_classes = len(np.unique(y))
    n, d = X.shape

    print(f"  {title}")
    print(f"  {n} samples  |  {d} features  |  {n_classes} classes")
    print(f"  classes: {np.bincount(y)}")

    # Intrinsic dimensionality
    print("  Discovering intrinsic dimensionality...")
    d_star = find_d_star(X, tau=TAU)
    noise_frac = 1.0 - d_star / d
    print(f"  d* = {d_star}  (noise fraction = {noise_frac:.0%})")

    pc_names = [f"PC{i + 1}" for i in range(d_star)]

    print(f"\n  Running {N_FOLDS}-fold CV...")
    cv = run_cv(X, y, d_star, n_classes)

    def _summary(key_acc, key_auc):
        accs, aucs = cv[key_acc], cv[key_auc]
        return {
            "acc_mean": float(np.mean(accs)),
            "acc_se": float(np.std(accs) / np.sqrt(len(accs))),
            "auc_mean": float(np.mean(aucs)),
            "auc_se": float(np.std(aucs) / np.sqrt(len(aucs))),
        }

    manifold_s = _summary("manifold_acc", "manifold_auc")
    raw_s = _summary("kan_raw_acc", "kan_raw_auc")
    pca_s = _summary("kan_pca_acc", "kan_pca_auc")

    print("\n  ── Results ──────────────────────────────────────────────")
    print(f"  {'Model':<22} {'Accuracy':>14} {'AUC':>14}")
    for label, s in [
        ("ManifoldModel", manifold_s),
        (f"KAN-raw ({d}D)", raw_s),
        (f"KAN-pca ({d_star}D)", pca_s),
    ]:
        print(
            f"  {label:<22} {s['acc_mean']:.4f}±{s['acc_se']:.4f}  "
            f"{s['auc_mean']:.4f}±{s['auc_se']:.4f}"
        )

    # Symbolic regression
    out_dir = _HERE.parent.parent.parent / "papers" / "clinical_manifolds" / f"kan_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_result, pca_obj = run_symbolic(X, y, d_star, n_classes, pc_names, class_names, out_dir)

    # PCA loadings summary (top 3 per PC, first 5 PCs)
    loadings = {}
    for i in range(min(d_star, 5)):
        abs_l = np.abs(pca_obj.components_[i])
        top3 = sorted(zip(feature_names, abs_l.tolist()), key=lambda x: x[1], reverse=True)[:3]
        loadings[f"PC{i + 1}"] = top3

    # Serialize active_edges (tuples → lists) for JSON
    serializable_edges = [
        [li, i, j, sym, float(r2)] for (li, i, j, sym, r2) in sym_result["active_edges"]
    ]

    result = {
        "dataset": name,
        "title": title,
        "n_samples": n,
        "n_features": d,
        "n_classes": n_classes,
        "d_star": d_star,
        "noise_frac": float(noise_frac),
        "manifold": manifold_s,
        "kan_raw": raw_s,
        "kan_pca": pca_s,
        "kan_pca_label": f"KAN-pca{d_star}",
        "formula": sym_result["formula"],
        "simplified_formula": sym_result.get("simplified_formula"),
        "n_active_edges": sym_result["n_active"],
        "active_edges": serializable_edges,
        "pca_loadings": {k: [(f, round(v, 4)) for f, v in pairs] for k, pairs in loadings.items()},
        "cv_raw": cv,
    }

    out_json = _HERE / f"kan_{name}_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {out_json}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(LOADERS),
        default=None,
        help="Run a single dataset (default: all)",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(LOADERS)
    all_results = {}

    for ds in datasets:
        try:
            all_results[ds] = run_dataset(ds)
        except Exception as exc:
            print(f"\n  ERROR on {ds}: {exc}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n\n{'═' * 80}")
    print("  SUMMARY — KAN vs ManifoldModel across all datasets")
    print(f"{'═' * 80}")
    hdr = (
        f"  {'Dataset':<28} {'d*':>4}  {'MM Acc':>8}  {'MM AUC':>8}  "
        f"{'KAN-raw':>8}  {'KAN-pca':>8}  {'KAN-pca AUC':>11}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for ds, r in all_results.items():
        mm_a = r["manifold"]["acc_mean"]
        mm_u = r["manifold"]["auc_mean"]
        raw_a = r["kan_raw"]["acc_mean"]
        pca_a = r["kan_pca"]["acc_mean"]
        pca_u = r["kan_pca"]["auc_mean"]
        d_s = r["d_star"]
        print(
            f"  {r['title']:<28} {d_s:>4}  {mm_a:>8.4f}  {mm_u:>8.4f}  "
            f"{raw_a:>8.4f}  {pca_a:>8.4f}  {pca_u:>11.4f}"
        )

    summary_path = _HERE / "kan_clinical_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full summary → {summary_path}")


if __name__ == "__main__":
    main()
