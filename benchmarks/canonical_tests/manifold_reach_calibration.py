"""Validate the local-vs-global reconstruction-gap instrument on known geometry.

Phases 0 and 1 of ``MANIFOLD_REACH_QUANTIZATION_PLAN.md``. Pure NumPy, no
TensorFlow, no real embedding data -- this only asks whether
``waverider.manifold_reach`` can detect curvature it was built to detect,
before Phase 2 spends a day dumping real transformer activations to ask it a
question nobody has calibrated it to answer.

    phase0    d* = 5, 10.  The regime ``estimator_calibration_report.md``
              confirms local PCA saturates correctly in.  Gate: if the
              instrument cannot separate a curved manifold (sphere, torus)
              from a flat control (a linear subspace, infinite reach) here,
              the measurement does not work even where the estimator is known
              to.  Stop.

    phase1    d* = 20, 50, 100.  The same code, aimed at the documented
              failure regime: local-PCA dimension estimation never plateaus
              past d*=20, and published transformer intrinsic dimensions
              (25-100) sit inside exactly that range. This instrument does not
              fit a dimension, so it may or may not inherit that failure --
              that is the open question. Gate: if the corrected gap stops
              separating curved from flat here the way it does in phase0, the
              approach cannot be applied to real embeddings. Read the printed
              sweep, do not just check exit status: the report says which
              radii (if any) were informative, and the JSON output keeps the
              full sweep for every case.

Every run writes a JSON beside this script with the full by-radius sweep for
every (manifold kind, d*) case, plus provenance, so the numbers are
reproducible from their own artifact -- same convention as
``estimator_calibration.py``.

Examples::

    python manifold_reach_calibration.py phase0
    python manifold_reach_calibration.py phase1
    python manifold_reach_calibration.py phase1 --d-values 20 50 --n 2000

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

from waverider.dimensionality_profile import knn_radii  # noqa: E402
from waverider.manifold_reach import (  # noqa: E402
    corrected_gap,
    gaussian_null,
    reconstruction_gap,
)

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260825
RADIUS_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

# ---------------------------------------------------------------------------
# Provenance (same convention as estimator_calibration.py)
# ---------------------------------------------------------------------------


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _write(payload, name):
    payload = dict(payload)
    payload["provenance"] = {
        "script": Path(__file__).name,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = HERE / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Synthetic manifolds of known reach
# ---------------------------------------------------------------------------


def _embed(Z, ambient, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ q.T


def make_manifold(kind, d, ambient, n, seed, radius=1.0):
    """Generate a manifold of known reach, rigidly embedded in ``ambient`` dims.

    :param kind: ``linear`` (flat, infinite reach -- the control), ``sphere``
        (uniform d-sphere, reach = radius) or ``torus`` (flat d-torus, a
        product of d circles of the given radius, reach = radius).
    :param d: True intrinsic dimension.
    :param ambient: Ambient dimension to embed into (>= d+1 for sphere, >= 2d
        for torus).
    :param n: Number of points.
    :param seed: RNG seed.
    :param radius: Curvature scale for sphere/torus; unused for linear.
    :returns: Array, shape (n, ambient).
    """
    rng = np.random.default_rng(seed)
    if kind == "linear":
        Z = rng.standard_normal((n, d))
    elif kind == "sphere":
        Z = rng.standard_normal((n, d + 1))
        Z = radius * Z / np.linalg.norm(Z, axis=1, keepdims=True)
    elif kind == "torus":
        angles = rng.uniform(0, 2 * np.pi, (n, d))
        Z = radius * np.hstack([np.cos(angles), np.sin(angles)])
    else:
        raise ValueError(f"unknown manifold kind: {kind}")
    if Z.shape[1] > ambient:
        raise ValueError(f"{kind} at d={d} needs {Z.shape[1]} ambient dims, got {ambient}")
    return _embed(Z, ambient, seed + 1)


def min_ambient(kind, d):
    return {"linear": d, "sphere": d + 1, "torus": 2 * d}[kind] + 8


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def measure_case(kind, d, n, n_holdout, seed):
    """Full corrected-gap radius sweep for one (kind, d) case.

    :returns: Dict with the manifold parameters and the full sweep -- radius,
        raw real/null gap, corrected gap, and neighbour counts at every radius
        -- never collapsed to a single number.
    """
    ambient = min_ambient(kind, d)
    X = make_manifold(kind, d, ambient, n, seed)
    base_radius = knn_radii(X, k=max(d, 5), n_probe=min(200, n // 2), seed=seed)["median"]
    radii = [base_radius * m for m in RADIUS_MULTIPLIERS]

    real = reconstruction_gap(X, k=d, radii=radii, n_holdout=n_holdout, seed=seed)
    null_X = gaussian_null(X, seed=seed + 1)
    null = reconstruction_gap(null_X, k=d, radii=radii, n_holdout=n_holdout, seed=seed)
    corrected = corrected_gap(real, null)

    real_by_r = {e["radius"]: e for e in real["by_radius"]}
    null_by_r = {e["radius"]: e for e in null["by_radius"]}
    by_radius = [
        {
            **entry,
            "n_used_real": real_by_r[entry["radius"]]["n_used"],
            "n_used_null": null_by_r[entry["radius"]]["n_used"],
        }
        for entry in corrected
    ]
    return {
        "kind": kind,
        "d": d,
        "ambient": ambient,
        "n": n,
        "n_holdout": n_holdout,
        "base_radius_knn_median": base_radius,
        "mse_global_real": real["mse_global"],
        "mse_global_null": null["mse_global"],
        "by_radius": by_radius,
    }


def _usable_gap(case, min_used):
    """Best corrected gap among radii with enough neighbours on both sides."""
    usable = [
        e["gap_corrected"]
        for e in case["by_radius"]
        if e["n_used_real"] >= min_used and e["n_used_null"] >= min_used
    ]
    return max(usable) if usable else float("nan")


_KIND_SEED_OFFSET = {"linear": 0, "sphere": 1, "torus": 2}


def run_phase(phase_name, d_values, n, n_holdout, seed):
    print(f"\n{'=' * 70}\n{phase_name}: d* in {d_values}\n{'=' * 70}")
    cases = []
    for d in d_values:
        for kind in ("linear", "sphere", "torus"):
            t0 = time.time()
            # Python's hash() on str/tuple is randomized per process
            # (PYTHONHASHSEED) unless explicitly seeded -- using it here made
            # every run of this "reproducible" sweep produce different numbers.
            case_seed = seed + d * 10 + _KIND_SEED_OFFSET[kind]
            case = measure_case(kind, d, n, n_holdout, seed=case_seed)
            cases.append(case)
            gap = _usable_gap(case, n_holdout // 2)
            print(
                f"  d={d:>4} {kind:<7} best corrected gap = {gap:+.4g}  ({time.time() - t0:.1f}s)"
            )

    print(f"\n  {'d':>4}  {'sphere - linear':>18}  {'torus - linear':>16}   verdict")
    for d in d_values:
        by_kind = {c["kind"]: c for c in cases if c["d"] == d}
        linear_gap = _usable_gap(by_kind["linear"], n_holdout // 2)
        sphere_gap = _usable_gap(by_kind["sphere"], n_holdout // 2)
        torus_gap = _usable_gap(by_kind["torus"], n_holdout // 2)
        sphere_margin = sphere_gap - linear_gap
        torus_margin = torus_gap - linear_gap
        informative = sphere_margin > 0 and torus_margin > 0
        verdict = "separates curved/flat" if informative else "UNINFORMATIVE -- see plan gate"
        print(f"  {d:>4}  {sphere_margin:>18.4g}  {torus_margin:>16.4g}   {verdict}")

    _write(
        {
            "phase": phase_name,
            "d_values": list(d_values),
            "n": n,
            "n_holdout": n_holdout,
            "cases": cases,
        },
        f"manifold_reach_calibration_{phase_name}_results.json",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("phase0", help="d* = 5, 10 -- validate the instrument")
    p0.add_argument("--n", type=int, default=1500)
    p0.add_argument("--n-holdout", type=int, default=150)

    p1 = sub.add_parser("phase1", help="d* = 20, 50, 100 -- the documented failure regime")
    p1.add_argument("--d-values", type=int, nargs="+", default=[20, 50, 100])
    p1.add_argument("--n", type=int, default=2000)
    p1.add_argument("--n-holdout", type=int, default=150)

    args = parser.parse_args()
    if args.command == "phase0":
        run_phase("phase0", [5, 10], args.n, args.n_holdout, DEFAULT_SEED)
    elif args.command == "phase1":
        run_phase("phase1", args.d_values, args.n, args.n_holdout, DEFAULT_SEED)


if __name__ == "__main__":
    main()
