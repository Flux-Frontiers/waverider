"""
Tests for the TVB scene presets in :mod:`waverider.voxel_viz`.

The loaders themselves live in :mod:`quiltwright.tvb_data` and are tested
there.  What matters on this side is that every preset names a dataset that
actually exists in those registries, and that its colours survive the HLD's
white-is-transparent rule — both checkable without downloading anything.
"""

from __future__ import annotations

import pytest
from quiltwright import tvb_data

from waverider.voxel_viz import TVB_PRESETS


def test_presets_reference_registered_datasets():
    """Every dataset a preset names must exist in the loader registries."""
    for name, preset in TVB_PRESETS.items():
        if preset["kind"] == "surface":
            assert preset["surface"] in tvb_data.SURFACES, name
            if preset.get("region_mapping"):
                assert preset["region_mapping"] in tvb_data.REGION_MAPPINGS, name
        elif preset["kind"] == "connectome":
            assert preset["connectivity"] in tvb_data.CONNECTIVITIES, name
            assert preset["surface"] in tvb_data.SURFACES, name
        elif preset["kind"] == "layers":
            for layer in preset["layers"]:
                assert layer["surface"] in tvb_data.SURFACES, name
        else:  # pragma: no cover - guards against a new kind slipping through
            pytest.fail(f"preset '{name}' has unhandled kind '{preset['kind']}'")


def test_presets_avoid_pure_white():
    """White is transparent on an HLD, so no preset colour may be #ffffff."""
    for name, preset in TVB_PRESETS.items():
        colors = [preset.get("color"), preset.get("node_color"), preset.get("shell_color")]
        colors += [layer["color"] for layer in preset.get("layers", [])]
        for color in filter(None, colors):
            assert color.lower() not in {"#ffffff", "#fff", "white"}, f"{name}: {color}"


def test_parcellated_presets_pair_matching_vertex_counts():
    """A region mapping only makes sense on a surface of the same size.

    Caught at preset-definition time rather than at render time, since the
    mismatch would otherwise surface only after a 337 MB download.
    """
    vertex_counts = {
        "cortex_16384": 16384,
        "cortex_80k": 81924,
        "cortex_2x120k": 283380,
        "macaque_147k": 147460,
    }
    mapping_lengths = {
        "regionMapping_16k_76": 16384,
        "regionMapping_80k_80": 81924,
        "regionMapping_147k_84": 147460,
        "regionMapping_16k_192": 16500,
    }
    for name, preset in TVB_PRESETS.items():
        mapping = preset.get("region_mapping")
        if preset["kind"] != "surface" or not mapping:
            continue
        surface = preset["surface"]
        assert vertex_counts[surface] == mapping_lengths[mapping], (
            f"preset '{name}' pairs {surface} ({vertex_counts[surface]} pts) "
            f"with {mapping} ({mapping_lengths[mapping]} labels)"
        )


@pytest.mark.parametrize("name,preset", sorted(TVB_PRESETS.items()))
def test_every_preset_has_a_description(name, preset):
    assert preset.get("description"), name
    assert preset["kind"] in {"surface", "connectome", "layers"}, name


def test_connectome_presets_declare_a_sane_threshold():
    """A percentile outside 0-100 would silently drop every edge, or none."""
    for name, preset in TVB_PRESETS.items():
        if preset["kind"] != "connectome":
            continue
        assert 0.0 <= preset["percentile"] < 100.0, name


def test_decimation_fractions_are_in_range():
    for name, preset in TVB_PRESETS.items():
        assert 0.0 <= preset.get("decimate", 0.0) < 1.0, name
