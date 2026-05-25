# WaveRider — Manifold-Aware Geometric ML Stack
#
# U.S.S. WaveRider, NCC-7699 — Starfleet's first manifold exploration vessel.
#
# Core instruments:
#   TurtleND         — N-dimensional position + orthonormal frame
#   ManifoldWalker   — Riemannian gradient descent on data manifolds
#   ManifoldModel    — Zero-parameter geometric classifier
#   ManifoldObserver — (N+1)-dimensional extrinsic observer
#   GeodesicEncoder  — Ambient → geodesic distance coordinates (Phase 1: Riemannian KAN)
#
# Author: Eric G. Suchanek, PhD
# Affiliation: Flux-Frontiers
# "The only way I know to predict the future is to write it." — EGS

from .backbone_angles import BackboneAngleList, BackboneResidue, quantize_angle
from .backbone_embedder import BackboneEmbedder
from .backbone_manifold import BackboneManifoldResult, fit_backbone_manifold
from .geodesic_coords import GeodesicEncoder
from .manifold_model import ManifoldModel
from .manifold_observer import ManifoldObserver
from .manifold_walker import ManifoldWalker
from .turtleND import TurtleND
from .voxel_viz import (
    CMAP_MAP,
    PCAInfo,
    PointField,
    build_grid,
    fit_and_observe,
    load_dataset,
    render_multi,
    render_single,
    voxelize,
)

__version__ = "0.8.0"
__all__ = [
    # Protein backbone
    "BackboneResidue",
    "BackboneAngleList",
    "quantize_angle",
    "BackboneEmbedder",
    "BackboneManifoldResult",
    "fit_backbone_manifold",
    # Core manifold stack
    "TurtleND",
    "ManifoldWalker",
    "ManifoldModel",
    "ManifoldObserver",
    "GeodesicEncoder",
    # voxel visualizer
    "PointField",
    "PCAInfo",
    "CMAP_MAP",
    "fit_and_observe",
    "load_dataset",
    "voxelize",
    "build_grid",
    "render_single",
    "render_multi",
]
