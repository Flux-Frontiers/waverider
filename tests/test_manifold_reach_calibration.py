"""Phase 0 gate: does the reconstruction-gap instrument recover known curvature?

Per ``MANIFOLD_REACH_QUANTIZATION_PLAN.md``, the instrument must be validated
on synthetic manifolds of known geometry, at d* = 5 and 10 -- the regime
``estimator_calibration_report.md`` confirms local PCA is well-behaved in --
before it is trusted on real embedding data where that estimator is already
known to fail. This file is that gate.

Each case compares a curved manifold (sphere, torus -- known finite reach)
against a flat control (a linear subspace -- infinite reach) at the same true
dimension, ambient size, and radius sweep. The claim under test is narrow: the
corrected gap distinguishes curved from flat. It does not pin down a specific
reach value -- that is read off the printed sweep by a human, not asserted
here, per the plan's "report the scale sweep, not a single scale."

Phase 1 (d* = 20, 50, 100, the failure regime the calibration report
documents) is deliberately NOT a hard assertion here: per the plan, whether
the instrument stays informative there is the open empirical question, and a
negative result is a valid, expected outcome, not a bug. That sweep runs as
``benchmarks/canonical_tests/manifold_reach_calibration.py``, which reports
numbers for a human to read the gate off of.
"""

import numpy as np
import pytest

from waverider.dimensionality_profile import knn_radii
from waverider.manifold_reach import corrected_gap, gaussian_null, reconstruction_gap

N_POINTS = 1500
N_HOLDOUT = 100
# A radius entry only counts toward the gate if at least this many held-out
# points had enough fit-pool neighbours to get a local fit at all -- an
# unusable radius (too tight for the data's actual density, which concentration
# of measure makes larger than intuition suggests at these dimensions) must not
# silently drop out as a NaN that happens to lose a max() comparison.
MIN_USABLE = N_HOLDOUT // 2


def _embed(Z, ambient, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((ambient, Z.shape[1])))
    return Z @ q.T


def linear_manifold(d, ambient, n=N_POINTS, seed=0):
    """Flat d-subspace, rigidly embedded. Reach is infinite: the flat control."""
    rng = np.random.default_rng(seed)
    return _embed(rng.standard_normal((n, d)), ambient, seed + 1)


def sphere_manifold(d, ambient, n=N_POINTS, radius=1.0, seed=0):
    """Uniform d-sphere of the given radius, rigidly embedded. Reach = radius."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d + 1))
    Z = radius * Z / np.linalg.norm(Z, axis=1, keepdims=True)
    return _embed(Z, ambient, seed + 1)


def torus_manifold(d, ambient, n=N_POINTS, radius=1.0, seed=0):
    """Flat d-torus (product of d circles of the given radius), rigidly embedded.

    Each circle factor has curvature radius ``radius``, so the torus's reach
    equals ``radius``, same as the sphere case -- but the curvature is
    concentrated in d independent 2-D planes rather than spread isotropically
    across d+1 dimensions, a structurally different way of being curved.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, (n, d))
    Z = radius * np.hstack([np.cos(angles), np.sin(angles)])
    return _embed(Z, ambient, seed + 1)


def radius_sweep(X, k, seed):
    """Radii scaled to this data's own neighbour density, not a fixed constant.

    Concentration of measure means the distance to a given neighbour count
    grows with ambient dimension; a radius picked by eye for one (d, ambient,
    n) is routinely too tight for another. Anchoring the sweep to the median
    k-th-neighbour distance keeps it meaningful across the parametrization.
    """
    base = knn_radii(X, k=max(k, 5), n_probe=200, seed=seed)["median"]
    return [base * m for m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)]


def max_corrected_gap(X, k, seed):
    """Best radius-sweep corrected gap: real gap minus a matching Gaussian null.

    Radii where too few held-out points had enough neighbours to fit at all are
    excluded before the max, so a NaN (or a max() tie-break against one) can
    never masquerade as either a positive or a negative finding.
    """
    radii = radius_sweep(X, k, seed)
    real = reconstruction_gap(X, k=k, radii=radii, n_holdout=N_HOLDOUT, seed=seed)
    null_X = gaussian_null(X, seed=seed + 1)
    null = reconstruction_gap(null_X, k=k, radii=radii, n_holdout=N_HOLDOUT, seed=seed)
    corrected = corrected_gap(real, null)
    real_by_r = {e["radius"]: e for e in real["by_radius"]}
    null_by_r = {e["radius"]: e for e in null["by_radius"]}
    usable = [
        e["gap_corrected"]
        for e in corrected
        if real_by_r[e["radius"]]["n_used"] >= MIN_USABLE
        and null_by_r[e["radius"]]["n_used"] >= MIN_USABLE
    ]
    assert usable, "no radius in the sweep had enough neighbours to evaluate -- widen the sweep"
    return max(usable)


@pytest.mark.parametrize("d", [5, 10])
class TestPhase0KnownReachRecovery:
    """The instrument must separate curved manifolds from a flat control."""

    def test_sphere_beats_flat_control(self, d):
        ambient = 3 * d + 8
        sphere_gap = max_corrected_gap(sphere_manifold(d, ambient, seed=10 * d), k=d, seed=10 * d)
        linear_gap = max_corrected_gap(
            linear_manifold(d, ambient, seed=10 * d + 1), k=d, seed=10 * d + 1
        )
        # The flat control's own gap is numerical noise (float SVD residual on
        # an exactly-flat manifold, order 1e-15) -- an absolute floor would have
        # to be re-tuned per d as the sphere's rank-d tangent fit absorbs more
        # of a (d+1)-sphere's curvature at higher d. Scaling the floor to that
        # noise instead keeps the same criterion meaningful at every d.
        floor = max(1e-6, 10 * abs(linear_gap))
        assert sphere_gap > floor, (
            f"d={d}: sphere corrected gap ({sphere_gap:.4g}) is not clearly above the "
            f"flat control's noise floor ({floor:.4g}) -- the instrument cannot tell "
            "curved from flat"
        )

    def test_torus_beats_flat_control(self, d):
        ambient = 2 * d + 8
        torus_gap = max_corrected_gap(torus_manifold(d, ambient, seed=20 * d), k=d, seed=20 * d)
        linear_gap = max_corrected_gap(
            linear_manifold(d, ambient, seed=20 * d + 1), k=d, seed=20 * d + 1
        )
        floor = max(1e-6, 10 * abs(linear_gap))
        assert torus_gap > floor, (
            f"d={d}: torus corrected gap ({torus_gap:.4g}) is not clearly above the "
            f"flat control's noise floor ({floor:.4g}) -- the instrument cannot tell "
            "curved from flat"
        )

    def test_flat_control_stays_near_zero(self, d):
        """No false positive: a manifold with no curvature to exploit should not
        register a large gap against its own null."""
        ambient = 3 * d + 8
        linear_gap = max_corrected_gap(linear_manifold(d, ambient, seed=30 * d), k=d, seed=30 * d)
        assert abs(linear_gap) < 0.1, (
            f"d={d}: flat control's corrected gap ({linear_gap:.4g}) is not near zero"
        )
