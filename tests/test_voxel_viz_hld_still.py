"""The manifold HLD path honors ``still``.

``waverider-voxel-viz --hld --still`` wrote the turntable video in manifold
mode while the CT and TVB demos wrote a PNG; the manifold dispatch never
looked at the flag.  These tests pin the still path on the function and on
the CLI dispatch.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("pyvista")
pytest.importorskip("PIL")

from quiltwright import hld  # noqa: E402

from waverider import voxel_viz  # noqa: E402


@pytest.fixture(scope="module")
def helix_scene():
    """A small helix fitted, observed and voxelized at low resolution."""
    X, y = voxel_viz._make_helix(n=150, seed=1)
    _, _, pf, pca_info = voxel_viz.fit_and_observe(X, y, k_graph=8, k_pca=10, k_vote=5, tau=0.9)
    vox = voxel_viz.voxelize(pf, resolution=12, padding=0.05)
    return voxel_viz.build_grid(vox), pf, pca_info


def test_render_hld_single_still_writes_png(tmp_path, helix_scene, monkeypatch):
    grid, pf, pca_info = helix_scene
    # Render the still small; the default is 3840x2160.
    orig = hld.render_hld_still
    monkeypatch.setattr(
        hld, "render_hld_still", lambda p, stem, **kw: orig(p, stem, resolution=(192, 108))
    )

    out = voxel_viz.render_hld_single(
        grid,
        pf,
        scalar="density",
        out_path=tmp_path / "helix",
        pca_info=pca_info,
        still=True,
    )

    assert out == tmp_path / "helix_hld.png"
    assert out.exists()
    assert not (tmp_path / "helix_hld.mp4").exists()


def test_cli_dispatch_passes_still(monkeypatch, tmp_path):
    """``--hld --still`` in manifold mode reaches ``render_hld_single(still=True)``."""
    seen: dict = {}

    def fake_render(grid, pf, **kw):
        seen.update(kw)
        return tmp_path / "x_hld.png"

    monkeypatch.setattr(voxel_viz, "render_hld_single", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "waverider-voxel-viz",
            "--dataset",
            "helix",
            "--n-points",
            "120",
            "--resolution",
            "10",
            "--hld",
            "--still",
            "--out",
            str(tmp_path / "x.png"),
        ],
    )
    voxel_viz.main()

    assert seen.get("still") is True
