"""Scale-resolved intrinsic dimensionality estimation.

``dimensionality_discovery`` answers "what is d*?" by running local PCA at one
neighbourhood size ``k`` and one variance threshold ``tau``.  That answer is not
a property of the data alone.  Local PCA recovers a tangent space only while the
neighbourhood radius stays small relative to the manifold's *reach* — the
largest ``r`` such that every point within distance ``r`` of the manifold has a
unique nearest point on it (Fefferman, Mitter & Narayanan, JAMS 2016).  Probe
wider than the reach and the local covariance mixes tangent directions with
curvature and with other sheets of the manifold, and the estimate inflates.
Probe too narrowly and finite-sample noise in the eigenvalues deflates it.

Two consequences drive this module:

1. **Report the scale, not just ``k``.**  A fixed ``k`` probes whatever radius
   the k-th neighbour happens to sit at, which varies with local density.  The
   same ``k`` therefore probes different physical scales in different regions of
   the same dataset.  :func:`knn_radii` and the ``radius`` option of
   :func:`local_pca_dimension` make that scale explicit and controllable.

2. **The profile is the measurement.**  Sweeping the neighbourhood and plotting
   ``d(r)`` is strictly more informative than any single value.  On data with a
   well-defined tangent structure the curve has a plateau; its value estimates
   the dimension and its extent brackets the usable scales.  Absence of a
   plateau is itself a finding — it says no scale in the probed range admits a
   stable tangent estimate.  :func:`dimension_profile` and :func:`find_plateau`
   compute this.

The threshold ``tau`` carries a bias that is worth stating explicitly.  For
locally isotropic data the ``d`` tangent eigenvalues are comparable, so
cumulative variance crosses ``tau`` at roughly ``ceil(tau * d)`` components.
Local PCA at threshold ``tau`` therefore has a ceiling near ``tau * d`` and
approaches it from below as ``k`` grows: it under-reports by construction.
:func:`tau_corrected` applies the first-order correction.

Two reference estimators are included so any local-PCA number can be reported
against the literature's: :func:`mle_levina_bickel` (Levina & Bickel, NIPS 2004
— the estimator used by Pope et al., ICLR 2021) and :func:`twonn` (Facco et
al., Sci. Rep. 2017).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd

__all__ = [
    "knn_radii",
    "local_pca_dimension",
    "dimension_profile",
    "find_plateau",
    "reach_proxy",
    "mle_levina_bickel",
    "twonn",
    "tau_corrected",
]

# Distance rows are computed in blocks of this many probe points to bound peak
# memory at (block * n_points) float64 rather than (n_probe * n_points).
_DIST_BLOCK = 64


def _rng(seed):
    """Return a Generator.  ``seed=None`` yields a fresh, unseeded Generator."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _probe_indices(n_points, n_probe, seed):
    n = min(n_probe, n_points)
    return _rng(seed).choice(n_points, size=n, replace=False)


def _distance_rows(X, probe_idx):
    """Yield (probe_position, distances_to_all_points) in memory-bounded blocks.

    Uses the expanded form |p - x|^2 = |p|^2 - 2 p.x + |x|^2 so peak memory is
    the (block, n_points) distance matrix rather than a (block, n_points,
    n_dims) difference tensor.  On CIFAR-10 that is the difference between
    ~25 MB and ~78 GB.
    """
    Xf = np.asarray(X, dtype=np.float64)
    sq_all = np.einsum("nd,nd->n", Xf, Xf)
    for start in range(0, len(probe_idx), _DIST_BLOCK):
        block = probe_idx[start : start + _DIST_BLOCK]
        P = Xf[block]
        sq_p = np.einsum("bd,bd->b", P, P)
        d2 = sq_p[:, None] - 2.0 * (P @ Xf.T) + sq_all[None, :]
        # Rounding can push near-zero entries slightly negative.
        np.maximum(d2, 0.0, out=d2)
        # The expanded form leaves a probe point's distance to itself at a tiny
        # positive residual rather than exactly zero.  Estimators that drop
        # zero distances to exclude the probe point (mle_levina_bickel, twonn)
        # would otherwise treat that residual as the nearest neighbour and
        # return garbage.  Pin it to exact zero.
        d2[np.arange(len(block)), block] = 0.0
        yield start, np.sqrt(d2)


def _eigenvalues(neighbors):
    """Local covariance eigenvalues, descending, via thin SVD."""
    centered = neighbors - neighbors.mean(axis=0)
    _, s, _ = svd(centered, full_matrices=False, check_finite=False)
    return (s**2) / max(len(neighbors) - 1, 1)


def _dim_at_tau(eigenvalues, tau):
    total = eigenvalues.sum()
    if total <= 0:
        return None
    return int(np.searchsorted(np.cumsum(eigenvalues) / total, tau) + 1)


def knn_radii(X, k=50, n_probe=500, seed=None):
    """Distance to the k-th nearest neighbour, over a random probe sample.

    This is the physical scale at which local PCA with neighbourhood size ``k``
    is operating.  Reporting it alongside a dimension estimate is what makes the
    estimate interpretable: two datasets probed at the same ``k`` are generally
    not probed at the same scale.

    :param X: Data matrix, shape (n_points, n_dims).
    :param k: Neighbourhood size.
    :param n_probe: Number of random probe points.
    :param seed: Seed or Generator for probe-point selection.
    :returns: Dict with keys ``median``, ``mean``, ``std``, ``p10``, ``p90``,
        ``min``, ``max``, ``k``, ``n_probe``.
    """
    X = np.asarray(X)
    idx = _probe_indices(len(X), n_probe, seed)
    k_use = min(k, len(X) - 1)

    radii = np.empty(len(idx), dtype=np.float64)
    for start, rows in _distance_rows(X, idx):
        for j, row in enumerate(rows):
            # k_use-th neighbour excluding the probe point itself (distance 0).
            radii[start + j] = np.partition(row, k_use)[k_use]

    return {
        "median": float(np.median(radii)),
        "mean": float(radii.mean()),
        "std": float(radii.std()),
        "p10": float(np.percentile(radii, 10)),
        "p90": float(np.percentile(radii, 90)),
        "min": float(radii.min()),
        "max": float(radii.max()),
        "k": int(k_use),
        "n_probe": int(len(idx)),
    }


def local_pca_dimension(X, k=None, radius=None, tau=0.90, n_probe=500, seed=None, min_neighbors=5):
    """Local-PCA intrinsic dimension at an explicit scale.

    Exactly one of ``k`` (fixed neighbour count) or ``radius`` (fixed metric
    scale) must be given.  ``radius`` is the option to prefer when comparing
    across datasets or across regions of differing density, because it holds the
    probing scale constant; ``k`` holds the sample size constant instead and
    lets the scale float.

    :param X: Data matrix, shape (n_points, n_dims).
    :param k: Neighbourhood size, or None to use ``radius``.
    :param radius: Neighbourhood radius, or None to use ``k``.
    :param tau: Cumulative-variance threshold.
    :param n_probe: Number of random probe points.
    :param seed: Seed or Generator for probe-point selection.
    :param min_neighbors: Probe points with fewer neighbours than this inside
        ``radius`` are skipped and counted in ``n_skipped``.
    :returns: Dict with the per-point ``dims`` array, its summary statistics,
        the realised neighbourhood ``radii``, and every parameter used.
    :raises ValueError: If neither or both of ``k`` and ``radius`` are given.
    """
    if (k is None) == (radius is None):
        raise ValueError("pass exactly one of k= or radius=")

    X = np.asarray(X)
    idx = _probe_indices(len(X), n_probe, seed)
    k_use = None if k is None else min(k, len(X) - 1)

    dims, used_radii, n_skipped = [], [], 0
    for _, rows in _distance_rows(X, idx):
        for row in rows:
            if k_use is not None:
                nn = np.argpartition(row, k_use)[:k_use]
                used_radii.append(float(np.partition(row, k_use)[k_use]))
            else:
                nn = np.flatnonzero(row <= radius)
                if len(nn) < min_neighbors:
                    n_skipped += 1
                    continue
                used_radii.append(float(radius))
            d = _dim_at_tau(_eigenvalues(X[nn]), tau)
            if d is not None:
                dims.append(d)

    dims = np.asarray(dims, dtype=int)
    radii = np.asarray(used_radii, dtype=float)
    stats = {
        "mean": float(dims.mean()) if dims.size else 0.0,
        "std": float(dims.std()) if dims.size else 0.0,
        "median": float(np.median(dims)) if dims.size else 0.0,
        "min": int(dims.min()) if dims.size else 0,
        "max": int(dims.max()) if dims.size else 0,
    }
    return {
        "dims": dims,
        **stats,
        "radius_median": float(np.median(radii)) if radii.size else 0.0,
        "radius_p10": float(np.percentile(radii, 10)) if radii.size else 0.0,
        "radius_p90": float(np.percentile(radii, 90)) if radii.size else 0.0,
        "k": k_use,
        "radius": radius,
        "tau": float(tau),
        "n_probe": int(len(idx)),
        "n_used": int(dims.size),
        "n_skipped": int(n_skipped),
        "seed": seed if not isinstance(seed, np.random.Generator) else None,
    }


def dimension_profile(X, k_values=(5, 10, 20, 40, 80, 160), tau=0.90, n_probe=300, seed=None):
    """Sweep the neighbourhood size and record dimension against realised scale.

    This is the measurement to report in place of a single ``d*``.  Each entry
    pairs a dimension estimate with the median radius it was measured at, so the
    curve can be read against scale rather than against an arbitrary ``k``.

    :param X: Data matrix, shape (n_points, n_dims).
    :param k_values: Neighbourhood sizes to sweep.
    :param tau: Cumulative-variance threshold.
    :param n_probe: Number of random probe points, shared across all ``k``.
    :param seed: Seed for probe-point selection.  The same probe points are used
        at every ``k`` so the curve reflects scale, not resampling noise.
    :returns: List of dicts, one per ``k``, ordered by ``k``.
    """
    X = np.asarray(X)
    # Draw the probe set once so points are held fixed across the sweep.
    fixed = _probe_indices(len(X), n_probe, seed)
    out = []
    for k in sorted(k_values):
        if k >= len(X):
            continue
        out.append(_profile_entry(X, fixed, k, tau))
    return out


def _profile_entry(X, probe_idx, k, tau):
    k_use = min(k, len(X) - 1)
    dims, radii = [], []
    for _, rows in _distance_rows(X, probe_idx):
        for row in rows:
            nn = np.argpartition(row, k_use)[:k_use]
            radii.append(float(np.partition(row, k_use)[k_use]))
            d = _dim_at_tau(_eigenvalues(X[nn]), tau)
            if d is not None:
                dims.append(d)
    dims = np.asarray(dims, dtype=int)
    return {
        "k": int(k_use),
        "mean": float(dims.mean()) if dims.size else 0.0,
        "median": float(np.median(dims)) if dims.size else 0.0,
        "std": float(dims.std()) if dims.size else 0.0,
        "max": int(dims.max()) if dims.size else 0,
        "radius_median": float(np.median(radii)),
        "tau": float(tau),
        "n_probe": int(len(probe_idx)),
    }


def find_plateau(profile, tol=1.0, min_points=3, statistic="median"):
    """Locate the flat region of a dimension profile.

    A plateau means a range of neighbourhood sizes agree on the dimension, which
    is the signature of a stable tangent estimate: the scale is small enough that
    curvature has not entered, and large enough that eigenvalue noise has
    settled.  Its absence means no probed scale gives a stable estimate, and any
    single-``k`` number quoted from such a profile is an artefact of the ``k``.

    Flatness is measured as the total spread (max minus min) across the run, not
    as a per-step change.  A per-step criterion cannot distinguish a plateau from
    a curve climbing steadily by one dimension per step, and a *relative* per-step
    criterion is unusable here because these statistics are near-integers: a
    one-dimension step is 20% at ``d=5`` but 2.5% at ``d=40``, so a single
    relative tolerance is simultaneously too tight at low dimension and too loose
    at high dimension.

    :param profile: Output of :func:`dimension_profile`.
    :param tol: Maximum spread, in dimensions, across the run.  The default of
        1.0 admits runs that wobble by a single dimension.
    :param min_points: Minimum number of consecutive entries to call a plateau.
    :param statistic: Which per-``k`` statistic to test (``median`` or ``mean``).
    :returns: Dict describing the longest plateau, or None if there is none.
    """
    if len(profile) < min_points:
        return None

    values = [p[statistic] for p in profile]
    best = (0, 0)  # half-open [start, end)
    for start in range(len(values)):
        for end in range(start + min_points, len(values) + 1):
            window = values[start:end]
            if max(window) - min(window) > tol:
                break
            if end - start > best[1] - best[0]:
                best = (start, end)

    if best[1] - best[0] < min_points:
        return None
    best = list(range(*best))

    entries = [profile[i] for i in best]
    vals = [e[statistic] for e in entries]
    return {
        "k_min": entries[0]["k"],
        "k_max": entries[-1]["k"],
        "radius_min": entries[0]["radius_median"],
        "radius_max": entries[-1]["radius_median"],
        "dimension": float(np.mean(vals)),
        "dimension_spread": float(max(vals) - min(vals)),
        "n_points": len(entries),
        "statistic": statistic,
    }


def reach_proxy(profile, plateau, tol=1.0, statistic="median"):
    """Scale at which the dimension estimate departs from its plateau.

    Beyond this radius the local covariance is picking up curvature or a second
    sheet, so the tangent approximation — and any dimension read off it — is no
    longer trustworthy.  This is an operational upper bound on the usable probing
    scale, *not* an estimate of the reach in the sense of Fefferman et al.:
    reach is a property of the manifold, this is a property of where our
    estimator stops being flat.  Use it to justify a choice of ``k``, not to
    make geometric claims.

    :param profile: Output of :func:`dimension_profile`.
    :param plateau: Output of :func:`find_plateau`, or None.
    :param tol: Departure from the plateau value, in dimensions, that counts as
        leaving the plateau.  Matches :func:`find_plateau`'s criterion.
    :param statistic: Which per-``k`` statistic to test.
    :returns: Dict with ``radius`` and ``k`` of the first departing entry, or
        None if the profile never departs (the plateau extends to the widest
        scale probed, so the usable range is not bounded above by this data).
    """
    if plateau is None:
        return None
    for entry in profile:
        if entry["k"] <= plateau["k_max"]:
            continue
        if abs(entry[statistic] - plateau["dimension"]) > tol:
            return {
                "radius": entry["radius_median"],
                "k": entry["k"],
                "dimension_at_departure": entry[statistic],
                "plateau_dimension": plateau["dimension"],
            }
    return None


def tau_corrected(d_hat, tau):
    """First-order correction for the threshold bias of local PCA at ``tau``.

    For locally isotropic data the ``d`` tangent eigenvalues are comparable, so
    cumulative variance reaches ``tau`` after about ``tau * d`` components and
    the raw estimate lands near ``tau * d`` rather than ``d``.  Dividing by
    ``tau`` removes that term.  The correction is exact only in the isotropic
    limit and with enough neighbours for the eigenvalues to be well resolved; on
    anisotropic data it over-corrects.  Report it alongside the raw value, never
    instead of it.

    :param d_hat: Raw local-PCA dimension estimate.
    :param tau: The threshold it was measured at, in (0, 1].
    :returns: Bias-corrected estimate as a float.
    """
    if not 0 < tau <= 1:
        raise ValueError("tau must lie in (0, 1]")
    return float(d_hat) / float(tau)


def mle_levina_bickel(X, k=5, n_probe=2000, seed=None):
    """Levina & Bickel maximum-likelihood intrinsic dimension estimate.

    The estimator Pope et al. (ICLR 2021) use for their published image-dataset
    dimensions, included here so local-PCA numbers can be reported against a
    literature-comparable baseline.  It is threshold-free, so it carries none of
    the ``tau`` bias, but it does inherit ``k`` dependence: Levina & Bickel
    recommend small ``k``, and the estimate rises with ``k``.

    :param X: Data matrix, shape (n_points, n_dims).
    :param k: Number of nearest neighbours (Pope et al. sweep 3, 5, 10, 20).
    :param n_probe: Number of random probe points.
    :param seed: Seed or Generator for probe-point selection.
    :returns: Scalar estimate, or ``nan`` if no probe point had ``k`` distinct
        non-zero neighbour distances.
    """
    X = np.asarray(X)
    idx = _probe_indices(len(X), n_probe, seed)
    inv = []
    for _, rows in _distance_rows(X, idx):
        for row in rows:
            d = np.sort(row[row > 0])[:k]
            if len(d) < k or d[-1] <= 0:
                continue
            inv.append(np.mean(np.log(d[-1] / d[:-1])))
    if not inv:
        return float("nan")
    return float(1.0 / np.mean(inv))


def twonn(X, n_probe=2000, discard_fraction=0.10, seed=None):
    """Facco et al. TwoNN intrinsic dimension estimate.

    Uses only the first two neighbour distances per point, which makes it the
    most local of the estimators here and the least sensitive to curvature — at
    the cost of sensitivity to noise and to duplicate points.

    :param X: Data matrix, shape (n_points, n_dims).
    :param n_probe: Number of random probe points.
    :param discard_fraction: Upper tail of the ratio distribution to drop before
        the fit, as recommended by the authors.
    :param seed: Seed or Generator for probe-point selection.
    :returns: Scalar estimate, or ``nan`` if too few valid ratios were found.
    """
    X = np.asarray(X)
    idx = _probe_indices(len(X), n_probe, seed)
    mu = []
    for _, rows in _distance_rows(X, idx):
        for row in rows:
            d = np.sort(row[row > 0])[:2]
            if len(d) < 2 or d[0] <= 0:
                continue
            mu.append(d[1] / d[0])
    mu = np.sort(np.asarray(mu))
    keep = int(len(mu) * (1.0 - discard_fraction))
    mu = mu[:keep]
    if len(mu) < 2:
        return float("nan")
    # F(mu) = 1 - mu^-d  =>  -log(1 - F) = d * log(mu); fit through the origin.
    f_emp = np.arange(1, len(mu) + 1) / len(mu)
    x = np.log(mu)
    y = -np.log(np.maximum(1.0 - f_emp, 1e-12))
    denom = float(np.dot(x, x))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x, y) / denom)
