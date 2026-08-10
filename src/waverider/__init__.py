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
#   UniversalEmbedder — Manifold-grounded drop-in for PCA in benchmark pipelines
#
# Author: Eric G. Suchanek, PhD
# Affiliation: Flux-Frontiers
# License: Elastic 2.0
# "The only way I know to predict the future is to write it." — EGS

# Holographic output lives in quiltwright, which was extracted from this
# package at 0.13.0.  Re-exported below so existing `from waverider import
# render_quilt` code keeps working; new code should import from quiltwright.
from quiltwright import (
    HLD_RESOLUTION,
    HLD_SAFE_MARGINS,
    QUILT_PRESETS,
    PovCamera,
    QuiltSpec,
    assemble_quilt,
    cast_quilt,
    focal_distance_for_range,
    pause_quilt,
    render_hld_video,
    render_pov_quilt,
    render_quilt,
    render_quilt_video,
    resume_quilt,
    save_quilt,
    stop_quilt,
    style_plotter_for_hld,
    view_disparity,
    view_offsets,
)

from .backbone_angles import BackboneAngleList, BackboneResidue, quantize_angle
from .backbone_embedder import BackboneEmbedder
from .backbone_manifold import BackboneManifoldResult, fit_backbone_manifold
from .geodesic_coords import GeodesicEncoder
from .manifold_model import ManifoldModel
from .manifold_observer import ManifoldObserver
from .manifold_walker import ManifoldWalker
from .turtleND import TurtleND
from .universal_embedder import UniversalEmbedder
from .voxel_viz import (
    CMAP_MAP,
    TVB_PRESETS,
    PCAInfo,
    PointField,
    build_grid,
    fit_and_observe,
    load_dataset,
    render_hld_single,
    render_multi,
    render_quilt_single,
    render_single,
    render_tvb_hld,
    render_tvb_quilt,
    render_tvb_still,
    render_tvb_viewer,
    voxelize,
)

__version__ = "0.14.0"
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
    "UniversalEmbedder",
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
    # Looking Glass holographic output
    "QuiltSpec",
    "QUILT_PRESETS",
    "render_quilt",
    "render_quilt_video",
    "render_quilt_single",
    "assemble_quilt",
    "view_offsets",
    "view_disparity",
    "focal_distance_for_range",
    "save_quilt",
    "cast_quilt",
    "pause_quilt",
    "resume_quilt",
    "stop_quilt",
    # POV-Ray holographic output
    "PovCamera",
    "render_pov_quilt",
    # Hololuminescent Display (HLD) output
    "HLD_RESOLUTION",
    "HLD_SAFE_MARGINS",
    "render_hld_video",
    "render_hld_single",
    "style_plotter_for_hld",
    # The Virtual Brain scenes.  The loaders themselves live in
    # quiltwright.tvb_data; only the scene presets and renderers are here.
    "TVB_PRESETS",
    "render_tvb_viewer",
    "render_tvb_quilt",
    "render_tvb_hld",
    "render_tvb_still",
]
