"""Rank-k local-versus-global reconstruction gap.

Answers a narrower question than an intrinsic-dimension estimate: at a fixed
rank budget ``k`` (the budget a quantizer's bit allocation actually supplies),
does a tangent frame fit *locally*, to a neighbourhood of radius ``r``,
reconstruct held-out points better than one global rank-``k`` fit to the whole
dataset? If curvature exists at a scale between the quantization noise floor
and the data's global extent, local frames win there; if it does not, they
cannot, and a single global subspace (as used by ResQ, OSCAR and KVTC) is
already the right shape.

This sidesteps the failure mode documented in
``benchmarks/canonical_tests/estimator_calibration_report.md``: the local-PCA
dimension estimate never plateaus past true dimension 20, which is inside the
range (25-100) reported for real transformer embeddings. Nothing here fits or
reads off a dimension. ``k`` is supplied externally as a design parameter, not
estimated, so the instrument stays meaningful exactly where dimension
estimation stops being trustworthy.

Two controls are load-bearing and built into every entry point rather than
left to the caller:

1. **Held-out evaluation.** A rank-``k`` fit to a neighbourhood of ``m``
   points with ``m`` close to ``k`` drives training residual to near zero and
   proves nothing. Every MSE here is measured on points excluded from the fit
   that produced the frame reconstructing them (:func:`held_out_split`).
2. **Centering per fit, not once globally.** Both the global and every local
   fit subtract their own mean before taking singular vectors
   (:func:`_fit_frame`), and evaluation subtracts that same mean before
   projecting. A strongly anisotropic cone -- the dominant shape of
   transformer embeddings -- is then absorbed identically by both the global
   and the local fit and contributes nothing to the gap. Only curvature that
   survives per-fit centering can produce ``Gap(r) != 0``.

A third control is not automatic and must be applied by the caller:
:func:`gaussian_null` on a matching-covariance Gaussian has zero curvature by
construction, so any ``Gap(r)`` it produces is a pure finite-sample overfitting
artefact; subtract it (see :func:`reconstruction_gap`'s ``gap`` field) before
treating a positive gap as evidence of manifold structure.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd

__all__ = [
    "held_out_split",
    "rank_k_reconstruction_mse",
    "local_rank_k_reconstruction_mse",
    "reconstruction_gap",
    "gaussian_null",
    "corrected_gap",
]

# Held-out points are matched against the fit pool in blocks of this many rows
# to bound peak memory at (block * n_fit) float64 rather than (n_holdout *
# n_fit).
_DIST_BLOCK = 64


def _rng(seed):
    """Return a Generator. ``seed=None`` yields a fresh, unseeded Generator."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def held_out_split(n_points, n_holdout, seed=None):
    """Partition point indices into a fit pool and a held-out probe set.

    :param n_points: Total number of points.
    :param n_holdout: Number of points to hold out. Clamped to ``n_points - 1``
        so the fit pool never empties.
    :param seed: Seed or Generator for the split.
    :returns: Tuple ``(fit_idx, holdout_idx)`` of disjoint index arrays.
    """
    n_holdout = min(n_holdout, n_points - 1)
    order = _rng(seed).permutation(n_points)
    return order[n_holdout:], order[:n_holdout]


def _fit_frame(points, k):
    """Mean and top-``k`` right singular vectors of mean-centered ``points``.

    :param points: Array, shape (n, d).
    :param k: Requested rank. Reduced to ``min(k, n - 1, d)`` when the fit set
        is too small or too narrow to support it.
    :returns: Tuple ``(mean, components)``; ``components`` has shape
        ``(k_eff, d)``, rows orthonormal, ``k_eff`` possibly 0.
    """
    mean = points.mean(axis=0)
    centered = points - mean
    k_eff = max(min(k, centered.shape[0] - 1, centered.shape[1]), 0)
    if k_eff == 0:
        return mean, np.zeros((0, points.shape[1]))
    _, _, vt = svd(centered, full_matrices=False, check_finite=False)
    return mean, vt[:k_eff]


def _reconstruction_sq_error(point, mean, components):
    """Squared error of projecting ``point`` onto a mean-centered frame."""
    centered = point - mean
    if components.shape[0] == 0:
        return float(np.dot(centered, centered))
    coeffs = components @ centered
    residual = centered - components.T @ coeffs
    return float(np.dot(residual, residual))


def _distances_to_pool(X, holdout_idx, fit_idx):
    """Yield (block_start, distances) of held-out points to the fit pool.

    Same expanded-norm trick as ``dimensionality_profile._distance_rows``, but
    rectangular: held-out points are never compared against each other or
    against themselves, only against the disjoint fit pool.
    """
    Xf = np.asarray(X, dtype=np.float64)
    pool = Xf[fit_idx]
    sq_pool = np.einsum("nd,nd->n", pool, pool)
    for start in range(0, len(holdout_idx), _DIST_BLOCK):
        block = holdout_idx[start : start + _DIST_BLOCK]
        P = Xf[block]
        sq_p = np.einsum("bd,bd->b", P, P)
        d2 = sq_p[:, None] - 2.0 * (P @ pool.T) + sq_pool[None, :]
        np.maximum(d2, 0.0, out=d2)
        yield start, np.sqrt(d2)


def rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k):
    """Global rank-``k`` PCA fit on ``fit_idx``, MSE on ``holdout_idx``.

    :param X: Data matrix, shape (n_points, n_dims).
    :param fit_idx: Indices used to fit the frame.
    :param holdout_idx: Indices reconstructed and scored; must be disjoint from
        ``fit_idx``.
    :param k: Rank budget.
    :returns: Mean squared reconstruction error over ``holdout_idx``.
    """
    X = np.asarray(X)
    mean, components = _fit_frame(X[fit_idx], k)
    errors = [_reconstruction_sq_error(X[i], mean, components) for i in holdout_idx]
    return float(np.mean(errors))


def local_rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k, radius, min_neighbors=None):
    """Per-point local rank-``k`` PCA, fit to fit-pool neighbours within ``radius``.

    Each held-out point gets its own frame, fit only to fit-pool points within
    ``radius`` of it, then reconstructs itself through that frame. A held-out
    point never contributes to its own frame or to any other held-out point's.

    :param X: Data matrix, shape (n_points, n_dims).
    :param fit_idx: Indices eligible to be neighbours.
    :param holdout_idx: Indices reconstructed and scored; must be disjoint from
        ``fit_idx``.
    :param k: Rank budget for each local frame.
    :param radius: Neighbourhood radius.
    :param min_neighbors: Minimum fit-pool neighbours required to fit a frame;
        defaults to ``k + 1`` so the frame is not rank-deficient by
        construction. Held-out points with fewer are skipped.
    :returns: Dict with ``mse`` (over used points only), ``n_used`` and
        ``n_skipped``.
    """
    X = np.asarray(X)
    min_neighbors = k + 1 if min_neighbors is None else min_neighbors
    errors = []
    n_skipped = 0
    for start, rows in _distances_to_pool(X, holdout_idx, fit_idx):
        for j, row in enumerate(rows):
            neighbor_pos = np.flatnonzero(row <= radius)
            if len(neighbor_pos) < min_neighbors:
                n_skipped += 1
                continue
            mean, components = _fit_frame(X[fit_idx[neighbor_pos]], k)
            errors.append(_reconstruction_sq_error(X[holdout_idx[start + j]], mean, components))
    return {
        "mse": float(np.mean(errors)) if errors else float("nan"),
        "n_used": int(len(errors)),
        "n_skipped": int(n_skipped),
    }


def reconstruction_gap(X, k, radii, n_holdout=200, min_neighbors=None, seed=None):
    """Rank-``k`` local-vs-global reconstruction gap across a radius sweep.

    Draws one fit/holdout split, shared across every radius so the sweep
    reflects scale rather than resampling noise, then compares one global
    rank-``k`` fit against a local rank-``k`` fit at each radius.

    :param X: Data matrix, shape (n_points, n_dims).
    :param k: Rank budget, shared between the global and every local fit.
    :param radii: Neighbourhood radii to sweep.
    :param n_holdout: Number of points held out for evaluation.
    :param min_neighbors: See :func:`local_rank_k_reconstruction_mse`.
    :param seed: Seed for the fit/holdout split.
    :returns: Dict with ``k``, ``mse_global``, ``n_holdout``, and
        ``by_radius``: a list of dicts (``radius``, ``mse_local``, ``gap``,
        ``n_used``, ``n_skipped``), ordered by radius. ``gap`` is
        ``mse_global - mse_local`` and is positive when the local fit
        reconstructs better.
    """
    X = np.asarray(X)
    fit_idx, holdout_idx = held_out_split(len(X), n_holdout, seed)
    mse_global = rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k)
    by_radius = []
    for r in sorted(radii):
        local = local_rank_k_reconstruction_mse(X, fit_idx, holdout_idx, k, r, min_neighbors)
        by_radius.append(
            {
                "radius": float(r),
                "mse_local": local["mse"],
                "gap": mse_global - local["mse"],
                "n_used": local["n_used"],
                "n_skipped": local["n_skipped"],
            }
        )
    return {
        "k": int(k),
        "mse_global": mse_global,
        "n_holdout": int(len(holdout_idx)),
        "by_radius": by_radius,
    }


def gaussian_null(X, seed=None):
    """Sample a Gaussian matching ``X``'s mean and covariance, same size as ``X``.

    Has zero curvature by construction. Running :func:`reconstruction_gap` on
    this and subtracting (:func:`corrected_gap`) isolates the part of a
    measured gap that is real geometric structure from the part that is a
    finite-sample artefact of fitting local frames to small neighbourhoods --
    which high-dimensional data manufactures regardless of the underlying
    truth.

    :param X: Data matrix, shape (n_points, n_dims).
    :param seed: Seed or Generator for sampling.
    :returns: Array, shape (n_points, n_dims).
    """
    X = np.asarray(X)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return _rng(seed).multivariate_normal(mean, cov, size=len(X))


def corrected_gap(gap_real, gap_null):
    """Subtract the Gaussian-null gap from the real gap, radius by radius.

    :param gap_real: Output of :func:`reconstruction_gap` on the real data.
    :param gap_null: Output of :func:`reconstruction_gap` on
        :func:`gaussian_null` of the same data, with the same ``k`` and radii.
    :returns: List of dicts (``radius``, ``gap_real``, ``gap_null``,
        ``gap_corrected``), ordered by radius.
    :raises ValueError: If the two sweeps used a different ``k`` or different
        radii.
    """
    if gap_real["k"] != gap_null["k"]:
        raise ValueError("gap_real and gap_null must share k")
    real_by_r = {e["radius"]: e["gap"] for e in gap_real["by_radius"]}
    null_by_r = {e["radius"]: e["gap"] for e in gap_null["by_radius"]}
    if real_by_r.keys() != null_by_r.keys():
        raise ValueError("gap_real and gap_null must sweep the same radii")
    return [
        {
            "radius": r,
            "gap_real": real_by_r[r],
            "gap_null": null_by_r[r],
            "gap_corrected": real_by_r[r] - null_by_r[r],
        }
        for r in sorted(real_by_r)
    ]
