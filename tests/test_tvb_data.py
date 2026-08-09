"""
Tests for :mod:`waverider.tvb_data`.

These never touch the network.  A synthetic ``tvb_data.zip`` is built in a
temporary directory and ``$WAVERIDER_TVB_CACHE`` is pointed at it, so the
parsing, registry and cache logic is exercised against the same member
layout as the real 337 MB Zenodo archive — including its quirks: split
hemispheres, 1-based triangle indices, folder-nested members, bz2-compressed
members and an empty vertex-normals stub.
"""

from __future__ import annotations

import bz2
import io
import zipfile

import numpy as np
import pytest

from waverider import tvb_data

# ---------------------------------------------------------------------------
# Fixtures — a miniature stand-in for the real archive
# ---------------------------------------------------------------------------

# A unit tetrahedron: small enough to assert on exactly.
_TETRA_VERTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETRA_TRIS = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def _txt(array: np.ndarray, fmt: str = "%.6f") -> bytes:
    """Serialise an array the way the TVB archive stores them."""
    buffer = io.StringIO()
    np.savetxt(buffer, array, fmt=fmt)
    return buffer.getvalue().encode()


def _surface_zip(
    *, one_based: bool = False, prefix: str = "", normals: bytes | None = None
) -> bytes:
    """Build a single-surface member zip."""
    tris = _TETRA_TRIS + 1 if one_based else _TETRA_TRIS
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"{prefix}vertices.txt", _txt(_TETRA_VERTS))
        archive.writestr(f"{prefix}triangles.txt", _txt(tris, fmt="%d"))
        if normals is not None:
            archive.writestr(f"{prefix}vertex_normals.txt", normals)
    return payload.getvalue()


def _split_surface_zip() -> bytes:
    """Build a split-hemisphere member zip with 1-based indices, as TVB ships."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("verticesl.txt", _txt(_TETRA_VERTS))
        archive.writestr("verticesr.txt", _txt(_TETRA_VERTS + 10.0))
        archive.writestr("trianglesl.txt", _txt(_TETRA_TRIS + 1, fmt="%d"))
        archive.writestr("trianglesr.txt", _txt(_TETRA_TRIS + 1, fmt="%d"))
    return payload.getvalue()


def _connectivity_zip(*, compressed: bool = False) -> bytes:
    """Build a connectome member zip, optionally with bz2-compressed members."""
    weights = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 2.0], [5.0, 2.0, 0.0]])
    lengths = np.array([[0.0, 10.0, 20.0], [10.0, 0.0, 30.0], [20.0, 30.0, 0.0]])
    centres = b"rA 1.0 2.0 3.0\nrB 4.0 5.0 6.0\nrC 7.0 8.0 9.0\n"

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        if compressed:
            archive.writestr("weights.txt.bz2", bz2.compress(_txt(weights)))
            archive.writestr("tract_lengths.txt.bz2", bz2.compress(_txt(lengths)))
            archive.writestr("centres.txt.bz2", bz2.compress(centres))
        else:
            archive.writestr("weights.txt", _txt(weights))
            archive.writestr("tract_lengths.txt", _txt(lengths))
            archive.writestr("centres.txt", centres)
    return payload.getvalue()


@pytest.fixture
def fake_archive(tmp_path, monkeypatch):
    """Point the cache at a synthetic archive mirroring the real layout."""
    cache = tmp_path / "cache"
    cache.mkdir()

    with zipfile.ZipFile(cache / "tvb_data.zip", "w") as archive:
        archive.writestr(
            tvb_data.SURFACES["cortex_16384"], _surface_zip(normals=_txt(_TETRA_VERTS))
        )
        archive.writestr(tvb_data.SURFACES["cortex_2x120k"], _split_surface_zip())
        # face_8614 ships a 1-byte normals stub rather than omitting the file.
        archive.writestr(tvb_data.SURFACES["face_8614"], _surface_zip(normals=b"\n"))
        # The macaque surface nests its arrays under a folder.
        archive.writestr(tvb_data.SURFACES["macaque_147k"], _surface_zip(prefix="Surface/"))
        archive.writestr(tvb_data.CONNECTIVITIES["connectivity_76"], _connectivity_zip())
        archive.writestr(
            tvb_data.CONNECTIVITIES["connectivity_68"], _connectivity_zip(compressed=True)
        )
        archive.writestr(tvb_data.REGION_MAPPINGS["regionMapping_16k_76"], b"0\n1\n2\n1\n")
        archive.writestr(tvb_data.SENSORS["eeg_63"], b"Fp1 1.0 2.0 3.0\nFp2 4.0 5.0 6.0\n")
        archive.writestr(tvb_data.SENSORS["seeg_39"], bz2.compress(b"A1 1.0 2.0 3.0\n"))

    monkeypatch.setenv(tvb_data._CACHE_ENV, str(cache))
    return cache


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything tries to download during a test."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("test attempted a network download")

    monkeypatch.setattr(tvb_data.urllib.request, "urlopen", _boom)


# ---------------------------------------------------------------------------
# Cache resolution
# ---------------------------------------------------------------------------


def test_cache_dir_prefers_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv(tvb_data._CACHE_ENV, str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert tvb_data.cache_dir() == tmp_path / "explicit"
    assert tvb_data.cache_dir().is_dir()


def test_cache_dir_uses_platform_native_root(tmp_path, monkeypatch):
    """Without an override, the archive lands under the platform cache root."""
    monkeypatch.delenv(tvb_data._CACHE_ENV, raising=False)
    monkeypatch.setattr(tvb_data, "_platform_cache_root", lambda: tmp_path / "native")
    assert tvb_data.cache_dir() == tmp_path / "native" / "tvb"


def test_platform_cache_root_prefers_platformdirs(monkeypatch):
    """platformdirs decides the root, so macOS/Windows get native paths."""
    import platformdirs

    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda app: f"/somewhere/{app}")
    assert tvb_data._platform_cache_root() == tvb_data.Path("/somewhere/waverider")


def test_platform_cache_root_falls_back_to_xdg(tmp_path, monkeypatch):
    """A core install without platformdirs still resolves somewhere sane."""
    real_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == "platformdirs":
            raise ImportError("platformdirs disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert tvb_data._platform_cache_root() == tmp_path / "xdg" / "waverider"


def test_platform_cache_root_falls_back_to_home(tmp_path, monkeypatch):
    real_import = __import__

    def _blocked(name, *args, **kwargs):
        if name == "platformdirs":
            raise ImportError("platformdirs disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(tvb_data.Path, "home", classmethod(lambda _cls: tmp_path))
    assert tvb_data._platform_cache_root() == tmp_path / ".cache" / "waverider"


def test_fetch_archive_uses_cache_without_network(fake_archive, no_network):
    assert tvb_data.fetch_archive() == fake_archive / "tvb_data.zip"


def test_clear_cache_removes_archive(fake_archive):
    assert tvb_data.archive_path().exists()
    tvb_data.clear_cache()
    assert not tvb_data.archive_path().exists()


# ---------------------------------------------------------------------------
# Index-base normalisation — the quirk that silently corrupts meshes
# ---------------------------------------------------------------------------


def test_zero_based_indices_pass_through():
    tris = np.array([[0, 1, 2]])
    assert np.array_equal(tvb_data._to_zero_based(tris, 3), tris)


def test_one_based_indices_are_shifted():
    tris = np.array([[1, 2, 3]])
    assert np.array_equal(tvb_data._to_zero_based(tris, 3), np.array([[0, 1, 2]]))


def test_out_of_range_indices_raise():
    with pytest.raises(ValueError, match="fits neither"):
        tvb_data._to_zero_based(np.array([[0, 1, 9]]), 3)


def test_float_encoded_indices_are_parsed():
    raw = b"1.0000000e+00 2.0000000e+00 3.0000000e+00\n4.0000000e+00 5.0000000e+00 6.0000000e+00\n"
    parsed = tvb_data._load_indices(raw)
    assert np.array_equal(parsed, np.array([[1, 2, 3], [4, 5, 6]]))
    assert parsed.dtype == np.int64


def test_non_integer_indices_raise():
    with pytest.raises(ValueError, match="non-integer"):
        tvb_data._load_indices(b"1.5 2.0 3.0\n")


def test_empty_normals_stub_reads_as_absent():
    assert tvb_data._load_normals(b"\n") is None
    assert tvb_data._load_normals(b"") is None


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def test_load_surface_single(fake_archive, no_network):
    verts, tris, normals = tvb_data.load_surface("cortex_16384")
    assert verts.shape == (4, 3)
    assert tris.shape == (4, 3)
    assert normals is not None and normals.shape == (4, 3)
    assert tris.min() >= 0 and tris.max() < len(verts)


def test_load_surface_split_hemispheres_offsets_and_rebases(fake_archive, no_network):
    verts, tris, _ = tvb_data.load_surface("cortex_2x120k")
    assert len(verts) == 8
    # Every index must be addressable, and the right hemisphere must have been
    # offset past the left rather than aliasing onto it.
    assert tris.min() == 0
    assert tris.max() == len(verts) - 1
    assert tris[len(_TETRA_TRIS) :].min() >= len(_TETRA_VERTS)


def test_load_surface_handles_nested_members(fake_archive, no_network):
    verts, tris, _ = tvb_data.load_surface("macaque_147k")
    assert verts.shape == (4, 3) and tris.shape == (4, 3)


def test_load_surface_tolerates_empty_normals(fake_archive, no_network):
    _, _, normals = tvb_data.load_surface("face_8614")
    assert normals is None


def test_unknown_surface_name_lists_alternatives(fake_archive, no_network):
    with pytest.raises(ValueError, match="cortex_16384"):
        tvb_data.load_surface("not_a_surface")


# ---------------------------------------------------------------------------
# Connectivity, region mappings, sensors
# ---------------------------------------------------------------------------


def test_load_connectivity(fake_archive, no_network):
    conn = tvb_data.load_connectivity("connectivity_76")
    assert conn.weights.shape == (3, 3)
    assert conn.tract_lengths.shape == (3, 3)
    assert conn.centres.shape == (3, 3)
    assert list(conn.labels) == ["rA", "rB", "rC"]
    assert conn.n_regions == 3
    np.testing.assert_allclose(conn.degree, [6.0, 3.0, 7.0])


def test_load_connectivity_handles_bz2_members(fake_archive, no_network):
    conn = tvb_data.load_connectivity("connectivity_68")
    assert conn.weights.shape == (3, 3)
    assert list(conn.labels) == ["rA", "rB", "rC"]


def test_load_region_mapping(fake_archive, no_network):
    labels = tvb_data.load_region_mapping("regionMapping_16k_76")
    assert labels.tolist() == [0, 1, 2, 1]
    assert labels.dtype == np.int64


def test_load_sensors(fake_archive, no_network):
    names, positions = tvb_data.load_sensors("eeg_63")
    assert list(names) == ["Fp1", "Fp2"]
    assert positions.shape == (2, 3)


def test_load_sensors_handles_top_level_bz2(fake_archive, no_network):
    names, positions = tvb_data.load_sensors("seeg_39")
    assert list(names) == ["A1"]
    assert positions.shape == (1, 3)


def test_missing_member_error_mentions_the_cache(fake_archive, no_network):
    with pytest.raises(KeyError, match="clear_cache"):
        tvb_data.load_surface("scalp_1082")  # registered but absent from the stub


# ---------------------------------------------------------------------------
# Categorical label transfer used when decimating a parcellated surface
# ---------------------------------------------------------------------------


def test_nearest_labels_recovers_exact_assignment():
    rng = np.random.default_rng(0)
    source = rng.random((200, 3))
    labels = rng.integers(0, 12, 200)
    jittered = source + rng.normal(0.0, 1e-5, source.shape)
    assert np.array_equal(tvb_data._nearest_labels(source, labels, jittered), labels)


def test_nearest_labels_matches_bruteforce_without_scipy(monkeypatch):
    """The SciPy-free fallback must agree with the cKDTree path."""
    rng = np.random.default_rng(1)
    source = rng.random((150, 3))
    labels = rng.integers(0, 7, 150)
    targets = rng.random((900, 3))

    reference = tvb_data._nearest_labels(source, labels, targets)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _blocked)
    fallback = tvb_data._nearest_labels(source, labels, targets)

    assert np.array_equal(fallback, reference)


def test_nearest_labels_returns_one_label_per_target():
    source = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    labels = np.array([5, 9])
    targets = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert tvb_data._nearest_labels(source, labels, targets).tolist() == [5, 9, 5]


# ---------------------------------------------------------------------------
# Registry consistency — catches typos without downloading anything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registry",
    [tvb_data.SURFACES, tvb_data.CONNECTIVITIES, tvb_data.REGION_MAPPINGS, tvb_data.SENSORS],
)
def test_registry_members_are_archive_relative_paths(registry):
    for name, member in registry.items():
        assert member.startswith("tvb_data/"), f"{name} -> {member}"
        assert not member.startswith("/"), f"{name} -> {member}"


def test_presets_reference_registered_datasets():
    """Every dataset a preset names must exist in the loader registries."""
    from waverider.voxel_viz import TVB_PRESETS

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
    from waverider.voxel_viz import TVB_PRESETS

    for name, preset in TVB_PRESETS.items():
        colors = [preset.get("color"), preset.get("node_color"), preset.get("shell_color")]
        colors += [layer["color"] for layer in preset.get("layers", [])]
        for color in filter(None, colors):
            assert color.lower() not in {"#ffffff", "#fff", "white"}, f"{name}: {color}"
