#!/usr/bin/env python3
"""
Manifold Voxel Visualizer — Streamlit Interface
================================================

Interactive web UI for the manifold voxel pipeline.

Three view modes available via tabs:

* **Interactive 3D** — Plotly WebGL viewer (rotate / zoom / pan).  Three
  sliders set the X / Y / Z slice plane positions; the density cloud appears
  as isosurfaces.  Runs entirely in the browser — no subprocess needed.
* **Static — single scalar** — fast off-screen PNG for a chosen field.
* **Static — all fields** — 2 × 2 PNG grid of density / curvature / height /
  class-vote.  Good for screenshots and export.

The model is fitted once and cached; changing any visual parameter (opacity,
background, point size …) re-renders without refitting.

Run with:
    streamlit run benchmarks/canonical_tests/manifold_voxel_app.py

Requires:
    poetry install --with benchmarks   # includes streamlit, stpyvista, pillow

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Last Revision: 2026-04-10 18:14:11
License: Elastic 2.0

"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
import streamlit as st

# ── Allow importing the sibling module ───────────────────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from manifold_voxel_viz import (  # noqa: E402
    CMAP_MAP,
    PCAInfo,
    PointField,
    _add_pca_arrows,
    _add_pca_axes,
    _add_voxel_cloud,
    build_grid,
    fit_and_observe,
    load_dataset,
    voxelize,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Manifold Voxel Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Manifold Voxel Visualizer")
st.caption(
    "Fit a ManifoldModel + ManifoldObserver, project geometric fields into a "
    "3-D PCA subspace, rasterize onto a voxel grid, and explore interactively."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

sb = st.sidebar
sb.header("Parameters")

with sb.expander("Dataset", expanded=True):
    dataset = st.selectbox(
        "Dataset",
        ["iris", "wine", "breast_cancer", "digits", "helix", "swiss_roll", "torus"],
        index=0,
    )
    n_points = st.slider("Max points", 100, 2000, 800, step=50)
    seed = st.number_input("Random seed", 0, 9999, 42, step=1)
    pre_pca = st.slider(
        "Pre-PCA dims  (0 = off)",
        0,
        100,
        0,
        step=5,
        help="Reduce to N dims before fitting ManifoldModel. Recommended ≥ 40 for MNIST / CIFAR.",
    )

with sb.expander("ManifoldModel", expanded=False):
    k_graph = st.slider("k_graph  (KNN graph degree)", 3, 30, 10)
    k_pca = st.slider("k_pca    (local PCA neighbours)", 5, 50, 20)
    k_vote = st.slider("k_vote   (classification vote)", 3, 20, 7)
    tau = st.slider("τ  (variance threshold for d*)", 0.70, 0.99, 0.90, step=0.01)

with sb.expander("Projection", expanded=False):
    pc1 = int(st.number_input("PC axis 1  (1-based)", 1, 20, 1))
    pc2 = int(st.number_input("PC axis 2  (1-based)", 1, 20, 2))
    pc3 = int(st.number_input("PC axis 3  (1-based)", 1, 20, 3))
    resolution = st.select_slider(
        "Voxel resolution  (N³ grid)",
        options=[16, 20, 24, 28, 32, 40, 48, 56, 64],
        value=32,
        help="Higher = finer detail but slower render. 32 is a good default.",
    )

with sb.expander("Scalar field", expanded=True):
    scalar = st.selectbox(
        "Field to display",
        ["density", "curvature", "height", "intrinsic_dim", "class_vote"],
        index=0,
        help=(
            "density: point concentration  ·  "
            "curvature: tangent-plane rotation rate  ·  "
            "height: reconstruction error  ·  "
            "intrinsic_dim: local d*  ·  "
            "class_vote: majority class per voxel"
        ),
    )

with sb.expander("Point cloud", expanded=True):
    show_points = st.toggle("Show scatter points", value=True)
    point_size = st.slider("Point size  (px)", 2, 24, 8)
    point_opacity = st.slider("Point opacity", 0.1, 1.0, 0.70, step=0.05)

with sb.expander("Voxel cloud & planes", expanded=True):
    show_volume = st.toggle("Show voxel cloud", value=True)
    vol_opacity = st.slider("Cloud opacity", 0.01, 0.60, 0.25, step=0.01)
    vol_threshold = st.slider(
        "Cloud density threshold  (fraction of max)",
        0.01,
        0.30,
        0.04,
        step=0.01,
        help="Voxels below this fraction of peak density are hidden from the cloud.",
    )
    plane_opacity = st.slider(
        "Slice plane opacity",
        0.20,
        1.00,
        0.60,
        step=0.05,
        help="Lower opacity lets the voxel cloud show through the planes.",
    )

with sb.expander("Display", expanded=True):
    bg_choice = st.radio(
        "Background",
        ["White", "Light grey", "Dark navy", "Black"],
        index=0,
        help="White / light grey make the voxel cloud much easier to read.",
    )
    show_arrows = st.toggle("Show PCA direction arrows", value=True)
    img_w = st.slider("Static image width  (px)", 600, 2400, 1400, step=100)
    img_h = st.slider("Static image height (px)", 400, 1600, 900, step=100)
    viewer_h = st.slider("Interactive viewer height (px)", 400, 1200, 700, step=50)

sb.divider()
fit_btn = sb.button("Fit & Render", type="primary", use_container_width=True)

# ── Constants ─────────────────────────────────────────────────────────────────

_BG_MAP = {
    "White": "white",
    "Light grey": "#e8e8e8",
    "Dark navy": "#0d1b2a",
    "Black": "black",
}

# ── Cached fitting pipeline ───────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def run_pipeline(
    dataset: str,
    n_points: int,
    seed: int,
    pre_pca: int,
    k_graph: int,
    k_pca: int,
    k_vote: int,
    tau: float,
    pca_components: tuple[int, int, int],
) -> tuple:
    """Load data, fit ManifoldModel + Observer, project to 3-D.

    Returns raw numpy arrays for clean Streamlit pickling.
    Heavy — only re-runs when fitting parameters change.
    """
    args = argparse.Namespace(
        dataset=dataset,
        n_points=n_points,
        seed=seed,
        X_file="X.npy",
        y_file=None,
    )
    X, y = load_dataset(args)

    if pre_pca > 0 and X.shape[1] > pre_pca:
        from sklearn.decomposition import PCA as _PCA

        X = _PCA(n_components=pre_pca, random_state=42).fit_transform(X).astype("d")

    _, _, pf, pca_info = fit_and_observe(
        X,
        y,
        k_graph=k_graph,
        k_pca=k_pca,
        k_vote=k_vote,
        tau=tau,
        pre_pca=0,
        pca_components=pca_components,
    )

    pca_dict: dict | None = None
    if pca_info is not None:
        pca_dict = {
            "evr": pca_info.explained_variance_ratio,
            "total": pca_info.total_explained,
            "ambient": pca_info.ambient_dim,
            "components": pca_info.components,
        }

    return (
        pf.X3,
        pf.density_w,
        pf.curvature,
        pf.height,
        pf.intrinsic_dim,
        pf.labels,
        pca_dict,
    )


def _unpack(raw: tuple) -> tuple[PointField, PCAInfo | None]:
    X3, dw, curv, ht, idim, labels, pca_dict = raw
    pf = PointField(
        X3=X3,
        density_w=dw,
        curvature=curv,
        height=ht,
        intrinsic_dim=idim,
        labels=labels,
    )
    pca_info = None
    if pca_dict is not None:
        pca_info = PCAInfo(
            explained_variance_ratio=pca_dict["evr"],
            total_explained=pca_dict["total"],
            ambient_dim=pca_dict["ambient"],
            components=pca_dict["components"],
        )
    return pf, pca_info


# ── Shared scene builder ──────────────────────────────────────────────────────


def _populate_scene(
    p,
    grid,
    pf: PointField,
    pca_info: PCAInfo | None,
    *,
    scalar: str,
    show_points: bool,
    point_size: int,
    point_opacity: float,
    show_volume: bool,
    vol_opacity: float,
    vol_threshold: float,
    plane_opacity: float,
    show_arrows: bool,
    background: str,
    dataset: str = "",
) -> None:
    """Add all meshes and overlays to an existing plotter.

    Called by both the static (off-screen) and interactive paths so that
    visual appearance is identical in both modes.
    """
    p.set_background(background)

    if show_volume:
        _add_voxel_cloud(p, grid, scalar, opacity=vol_opacity, threshold_frac=vol_threshold)

    p.add_mesh_slice_orthogonal(
        grid,
        scalars=scalar,
        cmap=CMAP_MAP.get(scalar, "plasma"),
        show_scalar_bar=True,
        opacity=plane_opacity,
    )

    if show_points:
        cloud = pv.PolyData(pf.X3.astype("f4"))
        cloud.point_data["label"] = pf.labels.astype("f4")
        p.add_points(
            cloud,
            scalars="label",
            cmap="Set1",
            point_size=point_size,
            opacity=point_opacity,
            show_scalar_bar=False,
            render_points_as_spheres=True,
        )

    _add_pca_axes(p, pca_info)
    if show_arrows:
        _add_pca_arrows(p, pf, pca_info)

    ds_prefix = f"{dataset}  ·  " if dataset else ""
    if pca_info is not None:
        title = (
            f"{ds_prefix}{scalar}   "
            f"[{pca_info.ambient_dim}D → 3D, "
            f"captured {pca_info.total_explained:.1%}]"
        )
    else:
        title = f"{ds_prefix}{scalar}"
    p.add_title(title, font_size=11)


# ── Static render (off-screen PNG) ────────────────────────────────────────────


def render_static(
    pf: PointField,
    pca_info: PCAInfo | None,
    *,
    scalar: str,
    resolution: int,
    width: int,
    height: int,
    **scene_kw,
) -> tuple[np.ndarray, dict]:
    """Render to a numpy RGB image; return (image, vox dict)."""
    vox = voxelize(pf, resolution=resolution)
    grid = build_grid(vox)

    p = pv.Plotter(off_screen=True, window_size=[width, height])
    _populate_scene(p, grid, pf, pca_info, scalar=scalar, **scene_kw)
    p.show(auto_close=False)
    img = p.screenshot(return_img=True)
    p.close()
    return img, vox


# ── Plotly colorscale map ─────────────────────────────────────────────────────

_PLOTLY_CMAP = {
    "density": "Plasma",
    "curvature": "RdBu",
    "height": "Viridis",
    "intrinsic_dim": "Turbo",
    "class_vote": "Rainbow",
}

# ── Interactive render (Plotly WebGL) ─────────────────────────────────────────


def render_plotly_3d(
    pf: PointField,
    pca_info: PCAInfo | None,
    *,
    scalar: str,
    resolution: int,
    show_points: bool,
    point_size: int,
    point_opacity: float,
    show_volume: bool,
    vol_opacity: float,
    vol_threshold: float,
    plane_opacity: float,
    slice_x_frac: float,
    slice_y_frac: float,
    slice_z_frac: float,
    show_arrows: bool,
    bg_color: str,
    height: int,
    dataset: str = "",
) -> dict:
    """Render an interactive 3-D Plotly figure with orthogonal slices.

    Uses ``go.Volume`` for the slice planes (positions controlled via the
    three fraction sliders) and ``go.Isosurface`` for the optional density
    cloud.  Runs entirely in the browser — no subprocess, no signal handlers.

    :param slice_x_frac: X-slice position as a fraction [0, 1] of the grid.
    :param slice_y_frac: Y-slice position as a fraction [0, 1] of the grid.
    :param slice_z_frac: Z-slice position as a fraction [0, 1] of the grid.
    """
    vox = voxelize(pf, resolution=resolution)
    res = resolution
    lo, sp = vox["origin"], vox["spacing"]

    xs = lo[0] + np.arange(res) * sp[0]
    ys = lo[1] + np.arange(res) * sp[1]
    zs = lo[2] + np.arange(res) * sp[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xf = X.ravel().astype("f4")
    yf = Y.ravel().astype("f4")
    zf = Z.ravel().astype("f4")

    values_3d = vox[scalar].reshape(res, res, res)
    vf = values_3d.ravel().astype("f4")
    v_min, v_max = float(vf.min()), float(vf.max())

    cmap = _PLOTLY_CMAP.get(scalar, "Plasma")

    # Slice plane world-space positions
    x_loc = float(xs[int(np.clip(slice_x_frac * (res - 1), 0, res - 1))])
    y_loc = float(ys[int(np.clip(slice_y_frac * (res - 1), 0, res - 1))])
    z_loc = float(zs[int(np.clip(slice_z_frac * (res - 1), 0, res - 1))])

    traces: list = []

    # ── Orthogonal slice planes ───────────────────────────────────────────────
    traces.append(
        go.Volume(
            x=xf,
            y=yf,
            z=zf,
            value=vf,
            isomin=v_min,
            isomax=v_max,
            opacity=float(plane_opacity),
            surface_count=1,
            colorscale=cmap,
            slices_x=dict(show=True, locations=[x_loc]),
            slices_y=dict(show=True, locations=[y_loc]),
            slices_z=dict(show=True, locations=[z_loc]),
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=True,
            name=scalar,
        )
    )

    # ── Density cloud (isosurfaces) ───────────────────────────────────────────
    if show_volume:
        density_3d = vox["density"].reshape(res, res, res)
        d_max = float(density_3d.max())
        d_thresh = d_max * vol_threshold
        if d_max > 0 and d_thresh < d_max:
            traces.append(
                go.Isosurface(
                    x=xf,
                    y=yf,
                    z=zf,
                    value=density_3d.ravel().astype("f4"),
                    isomin=d_thresh,
                    isomax=d_max,
                    surface=dict(count=3, fill=0.8),
                    opacity=float(vol_opacity),
                    colorscale="Plasma",
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    name="density cloud",
                )
            )

    # ── Point cloud ───────────────────────────────────────────────────────────
    if show_points:
        palette = px.colors.qualitative.Set1
        point_colors = [palette[int(lbl) % len(palette)] for lbl in pf.labels]
        traces.append(
            go.Scatter3d(
                x=pf.X3[:, 0].astype("f4"),
                y=pf.X3[:, 1].astype("f4"),
                z=pf.X3[:, 2].astype("f4"),
                mode="markers",
                marker=dict(
                    size=max(2, point_size // 2),
                    color=point_colors,
                    opacity=float(point_opacity),
                ),
                name="points",
                showlegend=False,
            )
        )

    # ── PCA direction arrows ──────────────────────────────────────────────────
    if show_arrows and pca_info is not None:
        centroid = pf.X3.mean(axis=0).astype("f4")
        span = (pf.X3.max(axis=0) - pf.X3.min(axis=0)).astype("f4")
        evr = pca_info.explained_variance_ratio
        colors = ["#e74c3c", "#2ecc71", "#3498db"]  # red, green, blue

        for i in range(3):
            direction = np.zeros(3, dtype="f4")
            direction[i] = 1.0
            length = float(span[i]) * 0.35 * max(float(np.sqrt(evr[i] / evr[0])), 0.30)
            tip = centroid + direction * length
            cone_size = length * 0.18  # arrowhead ≈ 18 % of shaft

            # Shaft
            traces.append(
                go.Scatter3d(
                    x=[float(centroid[0]), float(tip[0])],
                    y=[float(centroid[1]), float(tip[1])],
                    z=[float(centroid[2]), float(tip[2])],
                    mode="lines",
                    line=dict(color=colors[i], width=6),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Cone tip
            traces.append(
                go.Cone(
                    x=[float(tip[0])],
                    y=[float(tip[1])],
                    z=[float(tip[2])],
                    u=[float(direction[0]) * cone_size],
                    v=[float(direction[1]) * cone_size],
                    w=[float(direction[2]) * cone_size],
                    colorscale=[[0, colors[i]], [1, colors[i]]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=cone_size,
                    anchor="tail",
                    hoverinfo="skip",
                )
            )

    # ── Axis labels from PCA info ─────────────────────────────────────────────
    if pca_info is not None:
        evr, comp = pca_info.explained_variance_ratio, pca_info.components
        ax = dict(
            xaxis_title=f"PC{comp[0]} ({evr[0]:.1%})",
            yaxis_title=f"PC{comp[1]} ({evr[1]:.1%})",
            zaxis_title=f"PC{comp[2]} ({evr[2]:.1%})",
        )
        ds_prefix = f"{dataset}  ·  " if dataset else ""
        title_str = (
            f"{ds_prefix}{scalar}  [{pca_info.ambient_dim}D → 3D, {pca_info.total_explained:.1%}]"
        )
    else:
        ax = {}
        title_str = f"{dataset}  ·  {scalar}" if dataset else scalar

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(bgcolor=bg_color, **ax),
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
        title=dict(text=title_str, font=dict(size=13)),
        paper_bgcolor=bg_color,
    )

    st.plotly_chart(fig, use_container_width=True)
    return vox


# ── PNG helper ────────────────────────────────────────────────────────────────


def _to_png(img: np.ndarray) -> bytes:
    """Encode an RGB numpy array as PNG bytes (requires Pillow)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


# ── Validate PC selection ─────────────────────────────────────────────────────

pca_components = (pc1, pc2, pc3)
if len(set(pca_components)) < 3:
    st.error("PC axis 1, 2, and 3 must all be different.")
    st.stop()

# ── Fit (cached) ──────────────────────────────────────────────────────────────

fit_params = (
    dataset,
    n_points,
    seed,
    pre_pca,
    k_graph,
    k_pca,
    k_vote,
    tau,
    pca_components,
)

if "raw" not in st.session_state:
    st.session_state.raw = None
    st.session_state.fit_params = None

params_changed = st.session_state.fit_params != fit_params

if fit_btn or st.session_state.raw is None or params_changed:
    with st.spinner("Fitting ManifoldModel & Observer…  (cached after first run)"):
        st.session_state.raw = run_pipeline(*fit_params)
        st.session_state.fit_params = fit_params

pf, pca_info = _unpack(st.session_state.raw)

# ── Stats bar ─────────────────────────────────────────────────────────────────

n_classes = int(pf.labels.max()) + 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points", len(pf.X3))
c2.metric("Classes", n_classes)
if pca_info:
    c3.metric("Captured variance", f"{pca_info.total_explained:.1%}")
    c4.metric("Ambient dim", pca_info.ambient_dim)

if pca_info:
    evr = pca_info.explained_variance_ratio
    comp = pca_info.components
    st.caption(
        f"PC{comp[0]} = {evr[0]:.1%}  ·  PC{comp[1]} = {evr[1]:.1%}  ·  PC{comp[2]} = {evr[2]:.1%}"
    )

st.divider()

# ── Scene kwargs shared across all render calls ───────────────────────────────

_scene_kw = dict(
    show_points=show_points,
    point_size=point_size,
    point_opacity=point_opacity,
    show_volume=show_volume,
    vol_opacity=vol_opacity,
    vol_threshold=vol_threshold,
    plane_opacity=plane_opacity,
    show_arrows=show_arrows,
    background=_BG_MAP[bg_choice],
    dataset=dataset,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_3d, tab_single, tab_multi = st.tabs(
    [
        "Interactive 3D",
        "Static — single scalar",
        "Static — all fields  (2 × 2)",
    ]
)

# ── Tab 1: Interactive 3D via stpyvista ───────────────────────────────────────

with tab_3d:
    st.caption("Rotate: left-drag  ·  Zoom: scroll  ·  Pan: right-drag")

    sl_col1, sl_col2, sl_col3 = st.columns(3)
    slice_x = sl_col1.slider("X slice position", 0.0, 1.0, 0.5, 0.01, key="sx")
    slice_y = sl_col2.slider("Y slice position", 0.0, 1.0, 0.5, 0.01, key="sy")
    slice_z = sl_col3.slider("Z slice position", 0.0, 1.0, 0.5, 0.01, key="sz")

    with st.spinner("Rendering…"):
        vox_3d = render_plotly_3d(
            pf,
            pca_info,
            scalar=scalar,
            resolution=resolution,
            show_points=show_points,
            point_size=point_size,
            point_opacity=point_opacity,
            show_volume=show_volume,
            vol_opacity=vol_opacity,
            vol_threshold=vol_threshold,
            plane_opacity=plane_opacity,
            slice_x_frac=slice_x,
            slice_y_frac=slice_y,
            slice_z_frac=slice_z,
            show_arrows=show_arrows,
            bg_color=_BG_MAP[bg_choice],
            height=viewer_h,
            dataset=dataset,
        )

    occ = float((vox_3d["density"] > 0).sum()) / resolution**3 * 100
    st.caption(f"Voxel occupancy: {occ:.1f}%  ·  grid: {resolution}³")

# ── Tab 2: Static single scalar ───────────────────────────────────────────────

with tab_single:
    with st.spinner("Rendering…"):
        img, vox = render_static(
            pf,
            pca_info,
            scalar=scalar,
            resolution=resolution,
            width=img_w,
            height=img_h,
            **_scene_kw,
        )

    occ = float((vox["density"] > 0).sum()) / resolution**3 * 100
    st.caption(f"Voxel occupancy: {occ:.1f}%  ·  grid: {resolution}³")
    st.image(img, use_container_width=True)
    st.download_button(
        "Download PNG",
        data=_to_png(img),
        file_name=f"manifold_{dataset}_{scalar}.png",
        mime="image/png",
    )

# ── Tab 3: Static 2 × 2 all fields ───────────────────────────────────────────

with tab_multi:
    _fields = ["density", "curvature", "height", "class_vote"]
    _labels = [
        "Point density",
        "Mean curvature",
        "Height above tangent",
        "Majority class vote",
    ]

    half_w = max(500, img_w // 2)
    half_h = max(380, img_h // 2)

    field_imgs = []
    with st.spinner("Rendering 4 fields…"):
        for f_scalar in _fields:
            im, _ = render_static(
                pf,
                pca_info,
                scalar=f_scalar,
                resolution=resolution,
                width=half_w,
                height=half_h,
                **_scene_kw,
            )
            field_imgs.append(im)

    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, im, label, f_scalar in zip(row1 + row2, field_imgs, _labels, _fields):
        col.image(im, caption=label, use_container_width=True)
        col.download_button(
            f"Download  {label}",
            data=_to_png(im),
            file_name=f"manifold_{dataset}_{f_scalar}.png",
            mime="image/png",
            key=f"dl_{f_scalar}",
        )
