"""Backbone latent-space discovery via the WaveRider manifold stack.

End-to-end orchestrator:

    BackboneAngleList  →  BackboneEmbedder  →  ManifoldModel
                                                     ↓
                                             ManifoldObserver
                                                     ↓
                                             GeodesicEncoder  →  (N, d*) latent coords

The discovered latent coordinates flow directly into the existing voxel
visualizer (``waverider.voxel_viz``) without any extra conversion.

Typical usage::

    from waverider.backbone_angles import BackboneAngleList
    from waverider.backbone_embedder import BackboneEmbedder
    from waverider.backbone_manifold import fit_backbone_manifold, BackboneManifoldResult

    bal = BackboneAngleList.from_synthetic(n=2000, seed=42).valid()
    emb = BackboneEmbedder(mode='window', window_size=7)

    result = fit_backbone_manifold(bal, emb)
    print(result)

    # Feed into voxel visualizer
    from waverider import fit_and_observe
    # result.X_latent is (N, d*); result.y is (N,) integer ss labels

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backbone_angles import SECONDARY_STRUCTURE_CODES, BackboneAngleList
from .backbone_embedder import BackboneEmbedder
from .dimensionality_discovery import discover_dimensionality
from .geodesic_coords import GeodesicEncoder
from .manifold_model import ManifoldModel
from .manifold_observer import ManifoldObserver

__all__ = [
    "BackboneManifoldResult",
    "fit_backbone_manifold",
]


@dataclass
class BackboneManifoldResult:
    """All artefacts from a single backbone manifold fit.

    Attributes
    ----------
    backbone_list : BackboneAngleList
        The (valid) input residues.
    embedder : BackboneEmbedder
        Fitted embedder used to produce X_embedded.
    X_embedded : np.ndarray
        Shape (N, d_ambient) — embedder output, input to ManifoldModel.
    model : ManifoldModel
        Fitted manifold graph + local PCA bases.
    observer : ManifoldObserver
        Extrinsic (d_ambient + 1)-D observer.
    encoder : GeodesicEncoder
        Fitted geodesic coordinate encoder.
    X_latent : np.ndarray
        Shape (N, d_star) — geodesic latent coordinates.
    d_star : int
        Discovered intrinsic dimensionality.
    y : np.ndarray
        Shape (N,) integer secondary-structure labels (0=H,1=E,2=P,3=L,4=C,5=U).
    dim_report : dict
        Output of :func:`~waverider.dimensionality_discovery.discover_dimensionality`.
    """

    backbone_list: BackboneAngleList
    embedder: BackboneEmbedder
    X_embedded: np.ndarray
    model: ManifoldModel
    observer: ManifoldObserver
    encoder: GeodesicEncoder
    X_latent: np.ndarray
    d_star: int
    y: np.ndarray
    dim_report: dict

    def __repr__(self) -> str:
        n = len(self.backbone_list)
        d_amb = self.X_embedded.shape[1]
        return (
            f"BackboneManifoldResult("
            f"n={n}, d_ambient={d_amb}, d_star={self.d_star}, "
            f"embedder={self.embedder.mode!r}, "
            f"ss_classes={len(set(self.y.tolist()))})"
        )

    def summary(self) -> str:
        """Human-readable summary suitable for printing."""
        lines = [
            "=" * 60,
            "Backbone Manifold Latent-Space Analysis",
            "=" * 60,
            f"  Collection    : {self.backbone_list.name}",
            f"  Residues      : {len(self.backbone_list)}",
            f"  Embed mode    : {self.embedder.mode}  →  d_ambient={self.X_embedded.shape[1]}",
            f"  Intrinsic dim : d* = {self.d_star}",
            "",
            "  Dimensionality report (variance thresholds):",
        ]
        for tau, stats in self.dim_report.items():
            lines.append(
                f"    τ={tau:.2f}  mean={stats['mean']:.1f} ± {stats['std']:.1f}"
                f"  range=[{stats['min']}, {stats['max']}]"
            )
        lines += [
            "",
            "  Secondary structure breakdown:",
        ]
        code_map = {0: "H", 1: "E", 2: "P", 3: "L", 4: "C", 5: "U"}
        for int_label, char_label in sorted(code_map.items()):
            count = int((self.y == int_label).sum())
            if count:
                name = SECONDARY_STRUCTURE_CODES.get(char_label, "?")
                lines.append(f"    {char_label} ({name:20s}): {count}")
        lines.append("=" * 60)
        return "\n".join(lines)


def fit_backbone_manifold(
    backbone_list: BackboneAngleList,
    embedder: BackboneEmbedder | None = None,
    *,
    k_pca: int = 50,
    k_graph: int = 15,
    variance_threshold: float = 0.90,
    manifold_weight: float = 0.8,
    n_dim_samples: int = 300,
    verbose: bool = True,
) -> BackboneManifoldResult:
    """Discover the latent space of protein backbone angles.

    Full pipeline:

    1. Embed (φ, ψ) angles via *embedder* → X_embedded (N, d_ambient)
    2. Estimate intrinsic dimension via local PCA
    3. Fit ManifoldModel (KNN graph + per-node tangent bases)
    4. Fit GeodesicEncoder → X_latent (N, d_star)

    Parameters
    ----------
    backbone_list : BackboneAngleList
        Input residues.  Automatically filtered to valid (finite φ, ψ) entries.
    embedder : BackboneEmbedder, optional
        Embedding strategy.  Defaults to ``BackboneEmbedder(mode='window', window_size=7)``.
    k_pca : int
        Neighborhood size for local PCA at each manifold node (default 50).
    k_graph : int
        Neighbors for KNN graph construction (default 15).
    variance_threshold : float
        Cumulative variance fraction used to determine d* (default 0.90).
    manifold_weight : float
        Blend between Euclidean and manifold-projected distance in graph edges
        (default 0.8).
    n_dim_samples : int
        Points sampled for the dimensionality discovery report (default 300).
    verbose : bool
        Print progress messages (default True).

    Returns
    -------
    BackboneManifoldResult
    """
    # -- 0. Validate input --------------------------------------------------
    bal = backbone_list.valid()
    if len(bal) == 0:
        raise ValueError("backbone_list contains no residues with finite φ and ψ.")
    if verbose:
        print(f"[backbone_manifold] {bal}")

    # -- 1. Embed -----------------------------------------------------------
    if embedder is None:
        embedder = BackboneEmbedder(mode="window", window_size=7)

    X = embedder.fit_transform(bal).astype(np.float64)
    if verbose:
        print(f"[backbone_manifold] Embedded: {X.shape}  mode={embedder.mode!r}")

    # -- 2. Dimensionality discovery ----------------------------------------
    if verbose:
        print("[backbone_manifold] Discovering intrinsic dimensionality…")

    dim_report = discover_dimensionality(
        X,
        n_samples=n_dim_samples,
        k=min(k_pca, len(bal) - 1),
        variance_thresholds=(0.95, variance_threshold, 0.85),
    )
    d_star = round(dim_report[variance_threshold]["mean"])
    if verbose:
        print(f"[backbone_manifold] d* = {d_star} (τ={variance_threshold})")

    # -- 3. ManifoldModel ---------------------------------------------------
    if verbose:
        print("[backbone_manifold] Fitting ManifoldModel…")

    y = bal.to_ss_int_labels()
    model = ManifoldModel(
        k_graph=k_graph,
        k_pca=k_pca,
        variance_threshold=variance_threshold,
        manifold_weight=manifold_weight,
    )
    model.fit(X, y)
    if verbose:
        print("[backbone_manifold] ManifoldModel fit done.", flush=True)

    # -- 4. ManifoldObserver ------------------------------------------------
    observer = ManifoldObserver(model)

    # -- 5. GeodesicEncoder -------------------------------------------------
    if verbose:
        print("[backbone_manifold] Fitting GeodesicEncoder…", flush=True)

    encoder = GeodesicEncoder(
        k_pca=k_pca,
        k_graph=k_graph,
        variance_threshold=variance_threshold,
        manifold_weight=manifold_weight,
    )
    encoder.fit(X)
    if verbose:
        print("[backbone_manifold] GeodesicEncoder fit done.", flush=True)
    X_latent = encoder.transform(X)

    if verbose:
        print(f"[backbone_manifold] Latent space: {X_latent.shape}", flush=True)

    return BackboneManifoldResult(
        backbone_list=bal,
        embedder=embedder,
        X_embedded=X,
        model=model,
        observer=observer,
        encoder=encoder,
        X_latent=X_latent,
        d_star=d_star,
        y=y,
        dim_report=dim_report,
    )
