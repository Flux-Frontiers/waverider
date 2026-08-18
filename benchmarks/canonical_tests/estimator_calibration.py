"""Calibrate the intrinsic-dimension estimator and test the w* prescription.

Five experiments, run as subcommands.  The first three are CPU-only and cheap;
the last two train networks and want a GPU.

    synthetic         Bias of local PCA against manifolds of KNOWN dimension.
                      CPU, ~30-45 min at the defaults (cost scales as
                      n_probe x n x ambient).  Answers: how wrong is the
                      estimator, and in which direction, as a function of
                      (k, tau)?

    profile           Scale-resolved dimension profile d(r) on a real dataset,
                      with plateau detection.  CPU, ~15-25 min.  Answers: is there
                      a scale at which the tangent estimate is stable, and what
                      dimension does it give?

    noise             Replication of Pope et al. (ICLR 2021) Table 3: inject a
                      d-dimensional uniform hypercube into CIFAR-10 and check
                      the estimate tracks the known increment.  CPU, ~30-60 min
                      (7 noise levels).  Ground-truth calibration on real data.

    prescription      THE DECISIVE ONE.  Sweep bottleneck width independently
                      of any estimator, find the empirically optimal w, then
                      ask which estimators' w = d + C - 1 point there.  GPU,
                      hours.  If the optimum is invariant to which estimator
                      you start from, the estimator choice is a calibration
                      constant we can name.  If it lands wherever the estimator
                      happens to point, the prescription is circular.  Cost is
                      (n_widths x n_trials x epochs) network trainings: about
                      45 runs of 60 epochs at the defaults.  Start with --quick
                      to confirm the pipeline before committing the hours.

    probe-convention  Re-run the dimension probe under BOTH aggregations
                      (global mean and per-class max) and test the identity
                      k_90 + n_extra = d* under each.  GPU, ~30 min.  The
                      published identity was computed under the global mean
                      while w* is derived from the per-class max; this settles
                      whether it survives the other convention.

Every run writes a JSON beside this script recording all results together with
the estimator parameters, seeds, dataset, and device, so the numbers are
reproducible from their own artifact.

Examples::

    python estimator_calibration.py synthetic
    python estimator_calibration.py profile --dataset cifar10
    python estimator_calibration.py noise --dataset cifar10
    python estimator_calibration.py prescription --dataset cifar10 --gpu
    python estimator_calibration.py probe-convention --dataset cifar10 --gpu
    python estimator_calibration.py prescription --dataset cifar10 --gpu --quick   # smoke test

Author: Eric G. Suchanek, PhD -- Flux-Frontiers
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from waverider.dimensionality_discovery import (  # noqa: E402
    discover_dimensionality,
    discover_per_class_dimensionality,
)
from waverider.dimensionality_profile import (  # noqa: E402
    dimension_profile,
    find_plateau,
    knn_radii,
    local_pca_dimension,
    mle_levina_bickel,
    reach_proxy,
    tau_corrected,
    twonn,
)

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260818

# Sweeps shared across experiments.
K_VALUES = (5, 10, 25, 50, 100, 200)
TAU_VALUES = (0.85, 0.90, 0.95)
MLE_K_VALUES = (3, 5, 10, 20)  # the k's Pope et al. report


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _write(payload, name, extra=None):
    """Write results JSON with full provenance."""
    payload = dict(payload)
    payload["provenance"] = {
        "script": Path(__file__).name,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
    }
    path = HERE / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Synthetic manifolds of known intrinsic dimension
# ---------------------------------------------------------------------------


def _embed(Z, ambient, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ Q.T


def make_manifold(kind, d, ambient, n, seed, noise=0.0):
    """Generate a manifold whose intrinsic dimension is d by construction.

    :param kind: ``cube`` (flat), ``sphere`` (constant positive curvature),
        ``swissroll`` (curved, non-uniform density) or ``torus``.
    :param d: True intrinsic dimension.
    :param ambient: Ambient dimension to embed into.
    :param n: Number of points.
    :param seed: RNG seed.
    :param noise: Standard deviation of additive ambient Gaussian noise.
    :returns: Array of shape (n, ambient).
    """
    rng = np.random.default_rng(seed)
    if kind == "cube":
        Z = rng.uniform(0, 1, (n, d))
    elif kind == "sphere":
        Z = rng.standard_normal((n, d + 1))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    elif kind == "torus":
        angles = rng.uniform(0, 2 * np.pi, (n, d))
        Z = np.hstack([np.cos(angles), np.sin(angles)])
    elif kind == "swissroll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
        rest = rng.uniform(0, 1, (n, max(d - 1, 0)))
        Z = np.hstack([(t * np.cos(t))[:, None], (t * np.sin(t))[:, None], rest])
        Z /= np.abs(Z).max()
    else:
        raise ValueError(f"unknown manifold kind: {kind}")
    if Z.shape[1] > ambient:
        # torus needs 2d coordinates and swissroll d+1; a parameterisation
        # wider than the ambient space cannot be embedded in it.
        raise ValueError(
            f"{kind} at d={d} needs {Z.shape[1]} coordinates, "
            f"more than the ambient dimension {ambient}"
        )
    X = _embed(Z, ambient, rng)
    if noise > 0:
        X = X + rng.normal(0, noise, X.shape)
    return X


def run_synthetic(args):
    """Bias of local PCA against known dimension, over (k, tau) and curvature."""
    print("=" * 76)
    print("SYNTHETIC CALIBRATION -- local PCA vs KNOWN intrinsic dimension")
    print("=" * 76)

    rows = []
    configs = [
        (kind, d, noise)
        for kind in ("cube", "sphere", "swissroll", "torus")
        for d in (5, 10, 20, 40)
        for noise in (0.0, 0.02)
    ]
    for kind, d, noise in configs:
        try:
            X = make_manifold(kind, d, args.ambient, args.n, DEFAULT_SEED, noise)
        except ValueError as exc:
            print(f"  {kind:10s} d={d:3d} noise={noise:.2f} | skipped: {exc}")
            continue
        mle = {
            k: mle_levina_bickel(X, k=k, n_probe=args.n_probe, seed=DEFAULT_SEED)
            for k in MLE_K_VALUES
        }
        tnn = twonn(X, n_probe=args.n_probe, seed=DEFAULT_SEED)
        prof = dimension_profile(
            X, k_values=K_VALUES, tau=0.90, n_probe=args.n_probe, seed=DEFAULT_SEED
        )
        plateau = find_plateau(prof)

        for tau in TAU_VALUES:
            for k in K_VALUES:
                est = local_pca_dimension(X, k=k, tau=tau, n_probe=args.n_probe, seed=DEFAULT_SEED)
                rows.append(
                    {
                        "manifold": kind,
                        "true_d": d,
                        "noise": noise,
                        "k": k,
                        "tau": tau,
                        "mean": est["mean"],
                        "median": est["median"],
                        "max": est["max"],
                        "radius_median": est["radius_median"],
                        "tau_corrected_median": tau_corrected(est["median"], tau),
                        "relative_error_median": (est["median"] - d) / d,
                    }
                )

        summary = {
            "manifold": kind,
            "true_d": d,
            "noise": noise,
            "mle": mle,
            "twonn": tnn,
            "profile": prof,
            "plateau": plateau,
            "plateau_tau_corrected": tau_corrected(plateau["dimension"], 0.90) if plateau else None,
            "reach_proxy": reach_proxy(prof, plateau),
        }
        rows.append({"_summary": summary})
        pl = f"{plateau['dimension']:.1f}" if plateau else "none"
        corr = f"{tau_corrected(plateau['dimension'], 0.90):.1f}" if plateau else "-"
        print(
            f"  {kind:10s} d={d:3d} noise={noise:.2f} | plateau={pl:>5s} "
            f"tau-corr={corr:>5s} | MLE(k=5)={mle[5]:5.1f} TwoNN={tnn:5.1f}"
        )

    _write(
        {"experiment": "synthetic", "config": vars(args), "rows": rows},
        "estimator_calibration_synthetic_results.json",
    )


# ---------------------------------------------------------------------------
# Real-data loading
# ---------------------------------------------------------------------------


def load_dataset(name, standardize=True):
    """Load a flattened image dataset, matching the benchmark preprocessing.

    :param name: ``cifar10``, ``cifar100``, ``mnist`` or ``fashion_mnist``.
    :param standardize: Apply StandardScaler, as every benchmark script does
        before dimension discovery.
    :returns: Tuple (X_train, y_train, X_test, y_test, n_classes, spatial_shape).
    """
    import keras  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    loaders = {
        "cifar10": (keras.datasets.cifar10, 10, (32, 32, 3)),
        "cifar100": (keras.datasets.cifar100, 100, (32, 32, 3)),
        "mnist": (keras.datasets.mnist, 10, (28, 28, 1)),
        "fashion_mnist": (keras.datasets.fashion_mnist, 10, (28, 28, 1)),
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset: {name}")
    mod, n_classes, shape = loaders[name]
    (X_train, y_train), (X_test, y_test) = mod.load_data()
    dim = int(np.prod(shape))
    X_train = X_train.reshape(-1, dim).astype("float32")
    X_test = X_test.reshape(-1, dim).astype("float32")
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test, n_classes, shape


def estimate_all(X, y, n_probe, seed, label=""):
    """Every estimator we can run, on one dataset, with parameters recorded.

    :returns: Dict of estimator name -> estimate, plus the profile and plateau.
    """
    out = {"label": label, "n_probe": n_probe, "seed": seed, "local_pca": {}}
    for tau in TAU_VALUES:
        for k in K_VALUES:
            est = local_pca_dimension(X, k=k, tau=tau, n_probe=n_probe, seed=seed)
            out["local_pca"][f"k{k}_tau{tau}"] = {
                "mean": est["mean"],
                "median": est["median"],
                "max": est["max"],
                "radius_median": est["radius_median"],
                "tau_corrected_median": tau_corrected(est["median"], tau),
            }
    if y is not None:
        out["per_class_max"] = {}
        for k in (25, 50):
            cls = discover_per_class_dimensionality(
                X, y, k=k, tau=0.90, n_samples_per_class=max(5, n_probe // len(set(y))), seed=seed
            )
            out["per_class_max"][f"k{k}_tau0.9"] = int(max(c["max"] for c in cls.values()))
    out["global_mean_shipped"] = {
        f"k{k}": int(
            round(
                discover_dimensionality(
                    X, n_samples=n_probe, k=k, variance_thresholds=(0.90,), seed=seed
                )[0.90]["mean"]
            )
        )
        for k in (25, 50)
    }
    out["mle"] = {
        f"k{k}": mle_levina_bickel(X, k=k, n_probe=n_probe, seed=seed) for k in MLE_K_VALUES
    }
    out["twonn"] = twonn(X, n_probe=n_probe, seed=seed)
    prof = dimension_profile(X, k_values=K_VALUES, tau=0.90, n_probe=n_probe, seed=seed)
    plateau = find_plateau(prof)
    out["profile"] = prof
    out["plateau"] = plateau
    out["plateau_tau_corrected"] = tau_corrected(plateau["dimension"], 0.90) if plateau else None
    out["reach_proxy"] = reach_proxy(prof, plateau)
    return out


def run_profile(args):
    """Scale-resolved profile and plateau on a real dataset."""
    print("=" * 76)
    print(f"DIMENSION PROFILE -- {args.dataset}")
    print("=" * 76)

    X, y, _, _, n_classes, _ = load_dataset(args.dataset)
    sub = np.random.default_rng(DEFAULT_SEED).choice(
        len(X), size=min(args.subsample, len(X)), replace=False
    )
    Xs, ys = X[sub], y[sub]

    est = estimate_all(Xs, ys, args.n_probe, DEFAULT_SEED, label=args.dataset)
    print(f"\n  {'k':>5s} {'median d':>9s} {'radius':>10s}")
    for p in est["profile"]:
        print(f"  {p['k']:5d} {p['median']:9.1f} {p['radius_median']:10.3f}")
    print(f"\n  plateau            : {est['plateau']}")
    print(f"  tau-corrected      : {est['plateau_tau_corrected']}")
    print(f"  reach proxy        : {est['reach_proxy']}")
    print(f"  MLE (k=3,5,10,20)  : {est['mle']}")
    print(f"  TwoNN              : {est['twonn']:.2f}")
    print(f"  shipped global mean: {est['global_mean_shipped']}")
    print(f"  shipped per-cls max: {est.get('per_class_max')}")

    radii = {k: knn_radii(Xs, k=k, n_probe=args.n_probe, seed=DEFAULT_SEED) for k in (25, 50)}
    _write(
        {
            "experiment": "profile",
            "dataset": args.dataset,
            "n_classes": n_classes,
            "subsample": len(sub),
            "config": vars(args),
            "estimates": est,
            "knn_radii": radii,
        },
        f"estimator_calibration_profile_{args.dataset}_results.json",
    )


def run_noise(args):
    """Pope et al. Table 3 replication: inject known-dimension noise."""
    print("=" * 76)
    print(f"NOISE INJECTION -- {args.dataset} (Pope et al. 2021, Table 3 protocol)")
    print("=" * 76)
    print("  Reference (their MLE k=3 on CIFAR-10): d=256 -> 19.7, 512 -> 30.9,")
    print("  1024 -> 57.1, 1536 -> 77.8, 2048 -> 110.0, 2560 -> 136.1\n")

    X, y, _, _, _, _ = load_dataset(args.dataset, standardize=False)
    rng = np.random.default_rng(DEFAULT_SEED)
    sub = rng.choice(len(X), size=min(args.subsample, len(X)), replace=False)
    X = X[sub] / 255.0
    ambient = X.shape[1]

    rows = []
    for d_noise in [0, 256, 512, 1024, 1536, 2048, 2560]:
        if d_noise == 0:
            Xn = X
        else:
            if d_noise > ambient:
                continue
            # A randomly oriented d-dimensional unit hypercube in pixel space:
            # guarantees the augmented data has dimension at least d_noise.
            basis, _ = np.linalg.qr(rng.standard_normal((ambient, d_noise)))
            Xn = X + rng.uniform(0, 1, (len(X), d_noise)) @ basis.T

        mle = {
            k: mle_levina_bickel(Xn, k=k, n_probe=args.n_probe, seed=DEFAULT_SEED)
            for k in (3, 4, 5)
        }
        prof = dimension_profile(
            Xn, k_values=K_VALUES, tau=0.90, n_probe=args.n_probe, seed=DEFAULT_SEED
        )
        plateau = find_plateau(prof)
        rows.append(
            {
                "d_noise": d_noise,
                "mle": mle,
                "profile": prof,
                "plateau": plateau,
                "plateau_tau_corrected": tau_corrected(plateau["dimension"], 0.90)
                if plateau
                else None,
            }
        )
        pl = f"{plateau['dimension']:.1f}" if plateau else "none"
        print(
            f"  d_noise={d_noise:5d} | MLE k=3: {mle[3]:6.1f}  k=5: {mle[5]:6.1f} | plateau: {pl}"
        )

    _write(
        {"experiment": "noise", "dataset": args.dataset, "config": vars(args), "rows": rows},
        f"estimator_calibration_noise_{args.dataset}_results.json",
    )


# ---------------------------------------------------------------------------
# GPU experiments
# ---------------------------------------------------------------------------


def _setup_tf():
    from benchmarks.tf_setup import setup_tensorflow  # noqa: PLC0415

    return setup_tensorflow(gpu_flag="--gpu")


def _train_at_width(build, X_train, y_train, X_test, y_test, width, args, tag):
    """Train n_trials models at one bottleneck width; return accuracy stats."""
    import keras  # noqa: PLC0415

    accs, params = [], None
    for trial in range(args.n_trials):
        keras.utils.set_random_seed(DEFAULT_SEED + trial)
        model = build(width)
        params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
        model.fit(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.0,
            verbose=0,
        )
        _, acc = model.evaluate(X_test, y_test, verbose=0)
        accs.append(float(acc))
        keras.backend.clear_session()
    return {
        "width": int(width),
        "tag": tag,
        "params": params,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "acc_trials": accs,
    }


def run_prescription(args):
    """E2: is the optimal bottleneck width invariant to the estimator?"""
    tf, device_info = _setup_tf()
    from model_builder import build_manifold_resnet  # noqa: PLC0415

    print("=" * 76)
    print(f"PRESCRIPTION TEST -- {args.dataset} -- {device_info['device_used']}")
    print("=" * 76)

    X_train, y_train, X_test, y_test, C, shape = load_dataset(args.dataset)
    input_dim = X_train.shape[1]

    # --- Step 1: what does each estimator prescribe? ------------------------
    print("\nStep 1: estimator sweep")
    sub = np.random.default_rng(DEFAULT_SEED).choice(
        len(X_train), size=min(args.subsample, len(X_train)), replace=False
    )
    est = estimate_all(X_train[sub], y_train[sub], args.n_probe, DEFAULT_SEED, args.dataset)

    prescriptions = {}
    for key, val in est["local_pca"].items():
        prescriptions[f"local_pca_{key}_median"] = int(round(val["median"])) + C - 1
    for key, val in est["global_mean_shipped"].items():
        prescriptions[f"shipped_global_{key}"] = int(val) + C - 1
    for key, val in est.get("per_class_max", {}).items():
        prescriptions[f"shipped_per_class_max_{key}"] = int(val) + C - 1
    for key, val in est["mle"].items():
        prescriptions[f"mle_{key}"] = int(round(val)) + C - 1
    prescriptions["twonn"] = int(round(est["twonn"])) + C - 1
    if est["plateau_tau_corrected"]:
        prescriptions["plateau_tau_corrected"] = int(round(est["plateau_tau_corrected"])) + C - 1

    for name, w in sorted(prescriptions.items(), key=lambda kv: kv[1]):
        print(f"  {name:38s} -> w = {w}")

    # --- Step 2: sweep width independently of every estimator ---------------
    widths = sorted({int(w) for w in prescriptions.values() if 2 <= w <= args.max_width})
    grid = list(range(args.min_width, args.max_width + 1, args.width_step))
    widths = sorted(set(widths) | set(grid))
    if args.quick:
        widths = widths[:: max(1, len(widths) // 4)]

    print(
        f"\nStep 2: training at {len(widths)} widths x {args.n_trials} trials "
        f"x {args.epochs} epochs -> {len(widths) * args.n_trials} runs"
    )
    print(f"  widths: {widths}\n")

    def build(width):
        return build_manifold_resnet(
            input_dim, C, width, lr=args.lr, spatial_shape=shape, dropout=args.dropout
        )

    results, t0 = [], time.perf_counter()
    for i, w in enumerate(widths, 1):
        r = _train_at_width(build, X_train, y_train, X_test, y_test, w, args, tag="sweep")
        results.append(r)
        elapsed = time.perf_counter() - t0
        eta = elapsed / i * (len(widths) - i)
        print(
            f"  [{i:2d}/{len(widths)}] w={w:4d}  acc={r['acc_mean']:.4f} "
            f"+/- {r['acc_std']:.4f}  params={r['params']:,}  eta={eta / 60:.0f}m"
        )

    # --- Step 3: the verdict ------------------------------------------------
    best = max(results, key=lambda r: r["acc_mean"])
    # Widths whose accuracy is statistically indistinguishable from the best.
    tol = best["acc_std"] if best["acc_std"] > 0 else 0.005
    plateau_widths = [r["width"] for r in results if r["acc_mean"] >= best["acc_mean"] - tol]

    verdict = {
        "best_width": best["width"],
        "best_acc": best["acc_mean"],
        "best_acc_std": best["acc_std"],
        "indistinguishable_widths": plateau_widths,
        "optimum_is_broad": len(plateau_widths) > max(3, len(results) // 3),
        "estimators_landing_in_optimum": sorted(
            name for name, w in prescriptions.items() if w in plateau_widths
        ),
        "estimators_missing_optimum": sorted(
            name for name, w in prescriptions.items() if w not in plateau_widths
        ),
    }
    print("\n" + "=" * 76)
    print(f"  empirical optimum   : w = {best['width']}  ({best['acc_mean']:.4f})")
    print(f"  indistinguishable   : {plateau_widths}")
    print(f"  estimators that hit : {verdict['estimators_landing_in_optimum']}")
    print(f"  estimators that miss: {verdict['estimators_missing_optimum']}")
    if verdict["optimum_is_broad"]:
        print("\n  NOTE: the optimum spans a wide range of widths.  A prescription")
        print("  cannot be credited with precision the accuracy curve does not have.")
    print("=" * 76)

    _write(
        {
            "experiment": "prescription",
            "dataset": args.dataset,
            "n_classes": C,
            "config": vars(args),
            "estimates": est,
            "prescriptions": prescriptions,
            "sweep": results,
            "verdict": verdict,
        },
        f"estimator_calibration_prescription_{args.dataset}_results.json",
        extra={"device": device_info},
    )


def run_probe_convention(args):
    """Re-run the dimension probe under both aggregation conventions."""
    tf, device_info = _setup_tf()
    import keras  # noqa: PLC0415

    from model_builder import build_manifold_resnet  # noqa: PLC0415

    print("=" * 76)
    print(f"PROBE CONVENTION TEST -- {args.dataset} -- {device_info['device_used']}")
    print("=" * 76)

    X_train, y_train, X_test, y_test, C, shape = load_dataset(args.dataset)
    input_dim = X_train.shape[1]

    sub = np.random.default_rng(DEFAULT_SEED).choice(
        len(X_train), size=min(args.subsample, len(X_train)), replace=False
    )
    report = discover_dimensionality(
        X_train[sub],
        n_samples=args.n_probe,
        k=args.k_pca,
        variance_thresholds=(0.90,),
        seed=DEFAULT_SEED,
    )
    global_mean = int(round(report[0.90]["mean"]))
    per_class = discover_per_class_dimensionality(
        X_train[sub],
        y_train[sub],
        k=args.k_pca,
        tau=0.90,
        n_samples_per_class=max(5, args.n_probe // C),
        seed=DEFAULT_SEED,
    )
    per_class_max = int(max(c["max"] for c in per_class.values()))
    print(f"\n  global mean d*   = {global_mean}  -> w* = {global_mean + C - 1}")
    print(f"  per-class max d* = {per_class_max}  -> w* = {per_class_max + C - 1}\n")

    conventions = []
    for name, d_star in (("global_mean", global_mean), ("per_class_max", per_class_max)):
        w_star = d_star + C - 1
        keras.utils.set_random_seed(DEFAULT_SEED)
        model = build_manifold_resnet(
            input_dim, C, w_star, lr=args.lr, spatial_shape=shape, dropout=args.dropout
        )
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=0)
        _, acc = model.evaluate(X_test, y_test, verbose=0)

        # Post-GAP activations: the layer feeding the classifier head.
        gap = [ly for ly in model.layers if "global_average_pooling" in ly.name.lower()][-1]
        extractor = keras.Model(inputs=model.inputs, outputs=gap.output)
        acts = extractor.predict(X_test, batch_size=512, verbose=0)

        from sklearn.decomposition import PCA as skPCA  # noqa: PLC0415

        pca = skPCA().fit(acts)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k_90 = int(np.searchsorted(cum, 0.90) + 1)
        n_extra = w_star - k_90
        conventions.append(
            {
                "convention": name,
                "d_star": d_star,
                "w_star": w_star,
                "test_acc": float(acc),
                "k_90": k_90,
                "n_extra": n_extra,
                "identity_k90_plus_extra": k_90 + n_extra,
                "identity_holds": (k_90 + n_extra) == d_star,
                "n_extra_equals_C_minus_1": n_extra == C - 1,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
        )
        print(
            f"  {name:15s}: d*={d_star:3d} w*={w_star:3d} acc={acc:.4f} "
            f"k_90={k_90:3d} n_extra={n_extra:3d} "
            f"| k_90+n_extra={k_90 + n_extra} {'==' if (k_90 + n_extra) == d_star else '!='} d* "
            f"| n_extra {'==' if n_extra == C - 1 else '!='} C-1"
        )
        keras.backend.clear_session()

    print("\n  NOTE: k_90 + n_extra = w* by construction, since n_extra is defined")
    print("  as w* - k_90.  The identity is therefore a restatement of w* = d* + C - 1,")
    print("  not independent evidence for it.  The substantive claim is n_extra == C-1,")
    print("  i.e. that exactly C-1 bottleneck directions fall outside the 90% variance")
    print("  shell.  That is what the two rows above test.")

    _write(
        {
            "experiment": "probe_convention",
            "dataset": args.dataset,
            "n_classes": C,
            "config": vars(args),
            "conventions": conventions,
        },
        f"estimator_calibration_probe_convention_{args.dataset}_results.json",
        extra={"device": device_info},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, gpu=False):
        p.add_argument(
            "--dataset",
            default="cifar10",
            choices=["cifar10", "cifar100", "mnist", "fashion_mnist"],
        )
        p.add_argument(
            "--n-probe", type=int, default=300, help="Probe points for dimension estimation"
        )
        p.add_argument(
            "--subsample",
            type=int,
            default=10000,
            help="Points used for dimension estimation (O(n) distance rows)",
        )
        if gpu:
            p.add_argument("--gpu", action="store_true", help="Use GPU (CUDA or Metal)")
            p.add_argument("--epochs", type=int, default=60)
            p.add_argument("--batch-size", type=int, default=512)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--dropout", type=float, default=0.3)
            p.add_argument(
                "--quick", action="store_true", help="Few widths, few epochs -- smoke test only"
            )
        return p

    p = sub.add_parser("synthetic", help="Bias vs manifolds of known dimension")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--ambient", type=int, default=256)
    p.add_argument("--n-probe", type=int, default=150)

    common(sub.add_parser("profile", help="Scale-resolved profile on real data"))
    common(sub.add_parser("noise", help="Pope et al. Table 3 replication"))

    p = common(sub.add_parser("prescription", help="Is optimal w estimator-invariant?"), gpu=True)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--min-width", type=int, default=8)
    p.add_argument("--max-width", type=int, default=64)
    p.add_argument("--width-step", type=int, default=4)

    p = common(sub.add_parser("probe-convention", help="Probe under both aggregations"), gpu=True)
    p.add_argument("--k-pca", type=int, default=25)
    p.add_argument("--n-trials", type=int, default=1)

    args = parser.parse_args()
    if getattr(args, "quick", False):
        args.epochs = min(args.epochs, 3)
        args.n_trials = 1
        args.subsample = min(args.subsample, 2000)

    {
        "synthetic": run_synthetic,
        "profile": run_profile,
        "noise": run_noise,
        "prescription": run_prescription,
        "probe-convention": run_probe_convention,
    }[args.command](args)


if __name__ == "__main__":
    main()
