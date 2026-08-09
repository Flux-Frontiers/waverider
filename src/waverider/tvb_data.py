"""
TVB Dataset Loader
==================

Downloads and parses demonstration data from `The Virtual Brain
<https://www.thevirtualbrain.org/>`_ (TVB) so that real human, macaque and
mouse brain geometry can be pushed through the WaveRider holographic
pipeline — Looking Glass light-field quilts (:mod:`quiltwright.lfd`) and
Hololuminescent Display video (:mod:`quiltwright.hld`).

Where the data comes from
-------------------------
The `tvb-root <https://github.com/the-virtual-brain/tvb-root>`_ source tree
contains **no data**.  All demonstration datasets ship separately as
``tvb-data``, published on Zenodo as a single ~337 MB archive
(:data:`TVB_DATA_DOI`).  The old ``tvb-data`` GitHub repository is
deprecated and the PyPI package carries a reduced file set, so Zenodo is
the canonical source and the only one this module uses.

Nothing is vendored into WaveRider.  The archive is fetched on first use and
cached on disk, exactly as :mod:`pyvista.examples` does for its own
downloads — which also keeps TVB's GPL-3.0 licensing out of this source
tree.  See :func:`cache_dir` for cache placement and overrides.

Licensing and citation
----------------------
``tvb-data`` is distributed under **GPL-3.0**.  The data is downloaded at
runtime and never redistributed as part of WaveRider.  TVB asks that
scientific publications cite the platform; see :data:`TVB_CITATION`.

What the archive contains
-------------------------
Cortical surfaces are the useful part for holography, and they are stored in
the friendliest possible format: nested zips of plain-text ``vertices.txt`` /
``triangles.txt`` / ``vertex_normals.txt``.  No ``tvb`` package, no HDF5 and
no NIfTI reader is needed for any of the geometry this module exposes.

Surfaces (:func:`load_surface`)
    ``cortex_16384`` (16 384 pts), ``cortex_80k`` (81 924 pts),
    ``cortex_2x120k`` (283 380 pts, split hemispheres), plus the
    ``inner_skull_4096`` / ``outer_skull_4096`` / ``outer_skin_4096`` /
    ``scalp_1082`` / ``face_8614`` head shells and the macaque
    ``surface_147k``.

Connectivity (:func:`load_connectivity`)
    Region-level connectomes at 66/68/76/80/96/192/998 nodes — a weights
    matrix, a tract-length matrix and named 3-D region centres, which is
    everything needed to draw a connectome as nodes-and-tubes in space.

Region mappings (:func:`load_region_mapping`)
    Per-vertex integer parcellation labels that colour a surface by region.

Sensors (:func:`load_sensors`)
    EEG (63/65), MEG (248/276) and sEEG (588/960) electrode positions.

Quick start
-----------
::

    from waverider.tvb_data import load_surface, load_connectivity

    verts, tris, normals = load_surface("cortex_16384")   # downloads once
    conn = load_connectivity("connectivity_76")
    print(conn.weights.shape, conn.centres.shape, conn.labels[:3])

To build PyVista meshes directly (requires the ``viz`` extras)::

    from waverider.tvb_data import surface_polydata, connectome_polydata

    cortex = surface_polydata("cortex_16384")
    nodes, edges = connectome_polydata("connectivity_76", percentile=90.0)

For rendering these to a display, see the ``--tvb-demo`` mode of
:mod:`waverider.voxel_viz`.

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import bz2
import hashlib
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

__all__ = [
    "TVB_DATA_URL",
    "TVB_DATA_DOI",
    "TVB_DATA_MD5",
    "TVB_DATA_VERSION",
    "TVB_LICENSE",
    "TVB_CITATION",
    "SURFACES",
    "CONNECTIVITIES",
    "REGION_MAPPINGS",
    "SENSORS",
    "Connectome",
    "cache_dir",
    "archive_path",
    "fetch_archive",
    "clear_cache",
    "load_surface",
    "load_connectivity",
    "load_region_mapping",
    "load_sensors",
    "surface_polydata",
    "connectome_polydata",
]

# ---------------------------------------------------------------------------
# Source of record
# ---------------------------------------------------------------------------

#: Zenodo record for ``tvb-data`` 2.8.1 — the canonical distribution.
TVB_DATA_DOI = "10.5281/zenodo.10128131"

#: Direct download URL for the ``tvb_data.zip`` archive (~337 MB).
TVB_DATA_URL = "https://zenodo.org/records/10128131/files/tvb_data.zip?download=1"

#: MD5 published alongside the Zenodo record, verified after download.
TVB_DATA_MD5 = "08ae19833ba8ac158c91fbcb988b9bf0"

#: Version string of the pinned Zenodo record.
TVB_DATA_VERSION = "2.8.1"

#: Expected archive size in bytes — used only for download progress display.
_TVB_DATA_BYTES = 337_115_643

#: License of the downloaded data.  Not vendored; fetched at runtime.
TVB_LICENSE = "GPL-3.0 (The Virtual Brain) — downloaded at runtime, not redistributed"

#: Citation requested by the TVB project for scientific publications.
TVB_CITATION = (
    "Sanz Leon P, Knock SA, Woodman MM, Domide L, Mersmann J, McIntosh AR, "
    "Jirsa VK (2013). The Virtual Brain: a simulator of primate brain network "
    "dynamics. Front. Neuroinform. 7:10. doi:10.3389/fninf.2013.00010"
)

_CACHE_ENV = "WAVERIDER_TVB_CACHE"
_ARCHIVE_NAME = "tvb_data.zip"

# ---------------------------------------------------------------------------
# Dataset registries — logical name -> path inside the archive
# ---------------------------------------------------------------------------

#: Triangulated surfaces.  Each value is the member path inside
#: ``tvb_data.zip``; every one is itself a zip of plain-text arrays.
SURFACES: dict[str, str] = {
    "cortex_16384": "tvb_data/surfaceData/cortex_16384.zip",
    "cortex_80k": "tvb_data/surfaceData/cortex_80k.zip",
    "cortex_2x120k": "tvb_data/surfaceData/cortex_2x120k.zip",
    "inner_skull_4096": "tvb_data/surfaceData/inner_skull_4096.zip",
    "inner_skull_642": "tvb_data/surfaceData/inner_skull_642.zip",
    "outer_skull_4096": "tvb_data/surfaceData/outer_skull_4096.zip",
    "outer_skull_642": "tvb_data/surfaceData/outer_skull_642.zip",
    "outer_skin_4096": "tvb_data/surfaceData/outer_skin_4096.zip",
    "scalp_1082": "tvb_data/surfaceData/scalp_1082.zip",
    "face_8614": "tvb_data/surfaceData/face_8614.zip",
    "macaque_147k": "tvb_data/macaque_v3/surface_147k.zip",
}

#: Region-level connectomes (weights + tract lengths + named centres).
CONNECTIVITIES: dict[str, str] = {
    "connectivity_66": "tvb_data/connectivity/connectivity_66.zip",
    "connectivity_68": "tvb_data/connectivity/connectivity_68.zip",
    "connectivity_76": "tvb_data/connectivity/connectivity_76.zip",
    "connectivity_80": "tvb_data/connectivity/connectivity_80.zip",
    "connectivity_96": "tvb_data/connectivity/connectivity_96.zip",
    "connectivity_192": "tvb_data/connectivity/connectivity_192.zip",
    "connectivity_998": "tvb_data/connectivity/connectivity_998.zip",
    "macaque_84": "tvb_data/macaque_v3/connectivity_84.zip",
}

#: Per-vertex parcellation labels.  Keys pair with a surface of matching
#: vertex count — ``regionMapping_16k_76`` goes with ``cortex_16384``.
REGION_MAPPINGS: dict[str, str] = {
    "regionMapping_16k_76": "tvb_data/regionMapping/regionMapping_16k_76.txt",
    "regionMapping_16k_192": "tvb_data/regionMapping/regionMapping_16k_192.txt",
    "regionMapping_80k_80": "tvb_data/regionMapping/regionMapping_80k_80.txt",
    "regionMapping_147k_84": "tvb_data/macaque_v3/regionMapping_147k_84.txt",
}

#: Electrode / sensor position files.  Columns are ``label x y z``.
SENSORS: dict[str, str] = {
    "eeg_63": "tvb_data/sensors/eeg_63.txt",
    "eeg_brainstorm_65": "tvb_data/sensors/eeg_brainstorm_65.txt",
    "eeg_unitvector_62": "tvb_data/sensors/eeg_unitvector_62.txt.bz2",
    "meg_151": "tvb_data/sensors/meg_151.txt.bz2",
    "meg_248": "tvb_data/sensors/meg_248.txt",
    "meg_brainstorm_276": "tvb_data/sensors/meg_brainstorm_276.txt",
    "seeg_39": "tvb_data/sensors/seeg_39.txt.bz2",
    "seeg_588": "tvb_data/sensors/seeg_588.txt",
    "seeg_brainstorm_960": "tvb_data/sensors/seeg_brainstorm_960.txt",
}


class Connectome(NamedTuple):
    """A TVB region-level connectome.

    :param weights: ``(n, n)`` structural connection strengths.
    :param tract_lengths: ``(n, n)`` fibre-tract lengths in mm.
    :param centres: ``(n, 3)`` region centre coordinates in mm.
    :param labels: ``(n,)`` region names, aligned with the matrices.
    """

    weights: np.ndarray
    tract_lengths: np.ndarray
    centres: np.ndarray
    labels: np.ndarray

    @property
    def n_regions(self) -> int:
        """Number of regions in the connectome."""
        return len(self.labels)

    @property
    def degree(self) -> np.ndarray:
        """Weighted degree (row sum of *weights*) per region."""
        return self.weights.sum(axis=1)


@dataclass(frozen=True)
class _SurfaceArrays:
    """Internal container for a parsed surface."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray | None


# ---------------------------------------------------------------------------
# Cache management and download
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    """Return the directory where the TVB archive is cached.

    Resolution order:

    1. ``$WAVERIDER_TVB_CACHE`` if set — useful for shared or read-only
       installs, CI, putting a 337 MB archive on a different volume, and
       pointing several checkouts at one copy.
    2. The platform's native per-user cache directory, plus ``waverider/tvb``
       — see :func:`_platform_cache_root`.

    The directory is created if it does not exist.

    :return: Path to the cache directory.
    """
    override = os.environ.get(_CACHE_ENV)
    if override:
        path = Path(override).expanduser()
    else:
        path = _platform_cache_root() / "tvb"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _platform_cache_root() -> Path:
    """Return the per-user cache root for WaveRider on this OS.

    Uses :mod:`platformdirs` when available so the archive lands where the
    platform expects it — ``~/Library/Caches/waverider`` on macOS,
    ``%LOCALAPPDATA%\\waverider\\Cache`` on Windows, and
    ``$XDG_CACHE_HOME/waverider`` (default ``~/.cache/waverider``) on Linux.
    This matches PyVista, which caches its own downloads via
    ``pooch.os_cache``; hard-coding ``~/.cache`` would scatter a 337 MB file
    into a non-native location on two of the three platforms.

    ``platformdirs`` arrives transitively with the ``viz`` extras (pyvista →
    pooch), so the XDG-style fallback below only runs on a core install that
    could not render this data anyway.
    """
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("waverider"))
    except ImportError:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
        return base / "waverider"


def archive_path() -> Path:
    """Return the cached archive's path, downloaded or not.

    :return: Path to ``tvb_data.zip`` inside :func:`cache_dir`.
    """
    return cache_dir() / _ARCHIVE_NAME


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of *path*, streaming in *chunk*-byte blocks."""
    digest = hashlib.md5()  # noqa: S324  # integrity check against Zenodo, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_progress(seen: int, total: int) -> None:
    """Print a single-line download progress indicator to stderr."""
    if not sys.stderr.isatty():
        return
    mb = seen / 1e6
    if total > 0:
        pct = 100.0 * seen / total
        bar_len = 28
        filled = int(bar_len * seen / total)
        bar = "=" * filled + "-" * (bar_len - filled)
        msg = f"\r    [{bar}] {pct:5.1f}%  {mb:7.1f} / {total / 1e6:.1f} MB"
    else:
        msg = f"\r    {mb:7.1f} MB"
    sys.stderr.write(msg)
    sys.stderr.flush()


def fetch_archive(*, force: bool = False, verify: bool = True, quiet: bool = False) -> Path:
    """Download the TVB data archive to the local cache, if not already there.

    The download streams to a temporary file in the cache directory and is
    only moved into place once complete, so an interrupted transfer can
    never leave a truncated archive behind.

    :param force: Re-download even if a cached copy exists.
    :param verify: Check the download against :data:`TVB_DATA_MD5`.  A
        mismatch raises and removes the bad file.
    :param quiet: Suppress progress output.
    :return: Path to the cached ``tvb_data.zip``.
    :raises RuntimeError: If the download fails or the checksum mismatches.
    """
    target = archive_path()

    if target.exists() and not force:
        return target

    if not quiet:
        print(
            f"  Downloading tvb-data {TVB_DATA_VERSION} "
            f"(~{_TVB_DATA_BYTES / 1e6:.0f} MB, one time)\n"
            f"    from {TVB_DATA_URL}\n"
            f"    to   {target}",
            flush=True,
        )

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        with urllib.request.urlopen(TVB_DATA_URL) as response:  # noqa: S310  # fixed https URL
            declared = int(response.headers.get("Content-Length") or 0)
            total = declared or _TVB_DATA_BYTES
            seen = 0
            with open(tmp_path, "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    seen += len(block)
                    if not quiet:
                        _report_progress(seen, total)
        if not quiet and sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()
    except Exception as exc:  # noqa: BLE001  # re-raised with actionable context
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download tvb-data from {TVB_DATA_URL}: {exc}\n"
            f"You can download it manually and place it at {target}"
        ) from exc

    if verify:
        actual = _md5(tmp_path)
        if actual != TVB_DATA_MD5:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for tvb_data.zip: expected {TVB_DATA_MD5}, got {actual}.\n"
                "The download may be corrupt or the Zenodo record changed."
            )

    shutil.move(str(tmp_path), str(target))
    if not quiet:
        print(f"  Cached {target} ({target.stat().st_size / 1e6:.0f} MB)", flush=True)
    return target


def clear_cache() -> None:
    """Delete the cached TVB archive, if present."""
    target = archive_path()
    if target.exists():
        target.unlink()
        print(f"  Removed {target}")


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def _read_member(member: str, *, quiet: bool = False) -> bytes:
    """Read one member out of the cached archive, downloading it if needed.

    Members are read straight from the zip; the 337 MB archive is never
    expanded to disk.
    """
    path = fetch_archive(quiet=quiet)
    try:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read(member)
    except KeyError as exc:
        raise KeyError(
            f"'{member}' is not present in {path}. The cached archive may be "
            f"from a different tvb-data release; delete it (waverider.tvb_data."
            f"clear_cache()) and retry."
        ) from exc

    return bz2.decompress(raw) if member.endswith(".bz2") else raw


def _resolve(name: str, registry: dict[str, str], kind: str) -> str:
    """Look *name* up in *registry*, raising a helpful error if absent."""
    if name not in registry:
        raise ValueError(f"Unknown TVB {kind} '{name}'.  Available: {', '.join(sorted(registry))}")
    return registry[name]


def _loadtxt(raw: bytes, dtype=float) -> np.ndarray:
    """Parse a whitespace-delimited text blob into an array."""
    return np.loadtxt(io.BytesIO(raw), dtype=dtype)


def _load_normals(raw: bytes) -> np.ndarray | None:
    """Parse a vertex-normals blob, tolerating the empty stubs in the archive.

    ``face_8614`` ships a 1-byte ``vertex_normals.txt`` placeholder rather
    than omitting the file, so an empty or malformed parse is reported as
    "no normals" instead of an unusable array.  PyVista computes its own
    normals when none are supplied.
    """
    text = raw.strip()
    if not text:
        return None
    values = np.loadtxt(io.BytesIO(raw))
    if values.ndim != 2 or values.shape[1] != 3:
        return None
    return values


def _member_reader(archive: zipfile.ZipFile):
    """Build a basename-keyed reader over an open zip.

    Two quirks of the archive are absorbed here:

    * Some surfaces nest their arrays in a folder (``Surface/vertices.txt``),
      so members are keyed by basename rather than full path.
    * Some connectomes store bz2-compressed members
      (``weights.txt.bz2``), which are decompressed transparently and keyed
      under the uncompressed name.

    :param archive: An open :class:`zipfile.ZipFile`.
    :return: ``(read, available)`` — a callable taking an uncompressed
        basename and returning bytes, and the set of available basenames.
    """
    mapping: dict[str, str] = {}
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        base = name.rsplit("/", 1)[-1]
        mapping[base.removesuffix(".bz2")] = name

    def read(base: str) -> bytes:
        member = mapping[base]
        raw = archive.read(member)
        if member.endswith(".bz2"):
            return bz2.decompress(raw)
        return raw

    return read, set(mapping)


def _load_indices(raw: bytes) -> np.ndarray:
    """Parse an integer index array that may be written in float notation.

    The macaque surface stores its triangles as ``1.0000000e+00`` rather
    than ``1``, so reading straight to ``int64`` fails.  Parse as float and
    round, rejecting anything that is not actually integral.
    """
    values = np.loadtxt(io.BytesIO(raw), dtype=float)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0, atol=1e-6):
        raise ValueError("Index array contains non-integer values.")
    return rounded.astype(np.int64)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _to_zero_based(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    """Normalise triangle indices to 0-based.

    The archive mixes conventions: the single-surface files
    (``cortex_16384``, the head shells) index vertices from 0, but the
    split-hemisphere ``cortex_2x120k`` files index from 1.  Loading the
    latter as-is yields an index one past the last vertex, which produces a
    silently corrupt mesh rather than an error.

    A file is treated as 1-based only when it is unambiguous — the smallest
    index is 1 *and* the largest is exactly ``n_vertices``.

    :param triangles: ``(m, 3)`` vertex indices as stored.
    :param n_vertices: Vertex count of the surface these index into.
    :return: ``(m, 3)`` 0-based indices.
    :raises ValueError: If indices are out of range for either convention.
    """
    if triangles.size == 0:
        return triangles

    low, high = int(triangles.min()), int(triangles.max())

    if low >= 1 and high == n_vertices:
        return triangles - 1
    if low >= 0 and high <= n_vertices - 1:
        return triangles
    raise ValueError(
        f"Triangle indices span {low}..{high}, which fits neither 0-based nor "
        f"1-based addressing of {n_vertices} vertices."
    )


def _parse_surface_zip(raw: bytes) -> _SurfaceArrays:
    """Parse a TVB surface zip into vertices / triangles / normals.

    Handles both layouts found in the archive: a single surface
    (``vertices.txt``) and split hemispheres (``verticesl.txt`` +
    ``verticesr.txt``), which are concatenated with the right-hemisphere
    triangle indices offset by the left vertex count.  Index base is
    normalised per hemisphere by :func:`_to_zero_based` before the offset is
    applied, because the two layouts do not agree on it.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        read, by_base = _member_reader(inner)

        if "vertices.txt" in by_base:
            vertices = _loadtxt(read("vertices.txt"))
            triangles = _to_zero_based(_load_indices(read("triangles.txt")), len(vertices))
            normals = (
                _load_normals(read("vertex_normals.txt"))
                if "vertex_normals.txt" in by_base
                else None
            )
        elif "verticesl.txt" in by_base:
            left = _loadtxt(read("verticesl.txt"))
            right = _loadtxt(read("verticesr.txt"))
            tri_l = _to_zero_based(_load_indices(read("trianglesl.txt")), len(left))
            tri_r = _to_zero_based(_load_indices(read("trianglesr.txt")), len(right))
            vertices = np.vstack([left, right])
            triangles = np.vstack([tri_l, tri_r + len(left)])
            normals = None
            if "normalsl.txt" in by_base and "normalsr.txt" in by_base:
                left_n = _load_normals(read("normalsl.txt"))
                right_n = _load_normals(read("normalsr.txt"))
                if left_n is not None and right_n is not None:
                    normals = np.vstack([left_n, right_n])
        else:
            raise ValueError(
                f"Unrecognised TVB surface layout; expected vertices.txt or "
                f"verticesl.txt, found: {sorted(by_base)}"
            )

    return _SurfaceArrays(vertices=vertices, triangles=triangles, normals=normals)


def load_surface(
    name: str, *, quiet: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a triangulated TVB surface.

    :param name: Key from :data:`SURFACES`, e.g. ``"cortex_16384"``.
    :param quiet: Suppress download progress output.
    :return: ``(vertices, triangles, normals)`` where *vertices* is
        ``(n, 3)`` float mm coordinates, *triangles* is ``(m, 3)`` int
        vertex indices, and *normals* is ``(n, 3)`` or ``None``.
    """
    member = _resolve(name, SURFACES, "surface")
    surface = _parse_surface_zip(_read_member(member, quiet=quiet))
    return surface.vertices, surface.triangles, surface.normals


def load_connectivity(name: str, *, quiet: bool = False) -> Connectome:
    """Load a TVB region-level connectome.

    :param name: Key from :data:`CONNECTIVITIES`, e.g. ``"connectivity_76"``.
    :param quiet: Suppress download progress output.
    :return: A :class:`Connectome` with weights, tract lengths, region
        centres and region labels.
    """
    member = _resolve(name, CONNECTIVITIES, "connectivity")
    raw = _read_member(member, quiet=quiet)

    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        read, by_base = _member_reader(inner)
        missing = {"weights.txt", "centres.txt"} - by_base
        if missing:
            raise ValueError(
                f"Connectome '{name}' is missing {sorted(missing)}; found {sorted(by_base)}"
            )
        weights = _loadtxt(read("weights.txt"))
        tract_lengths = (
            _loadtxt(read("tract_lengths.txt"))
            if "tract_lengths.txt" in by_base
            else np.zeros_like(weights)
        )
        centres_raw = read("centres.txt").decode("utf-8")

    labels: list[str] = []
    coords: list[list[float]] = []
    for line in centres_raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        labels.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])

    return Connectome(
        weights=weights,
        tract_lengths=tract_lengths,
        centres=np.asarray(coords, dtype=float),
        labels=np.asarray(labels, dtype=object),
    )


def load_region_mapping(name: str, *, quiet: bool = False) -> np.ndarray:
    """Load per-vertex parcellation labels for a surface.

    :param name: Key from :data:`REGION_MAPPINGS`, e.g.
        ``"regionMapping_16k_76"``.
    :param quiet: Suppress download progress output.
    :return: ``(n_vertices,)`` integer region indices.
    """
    member = _resolve(name, REGION_MAPPINGS, "region mapping")
    return _loadtxt(_read_member(member, quiet=quiet)).astype(np.int64)


def load_sensors(name: str, *, quiet: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load electrode / sensor positions.

    :param name: Key from :data:`SENSORS`, e.g. ``"eeg_63"``.
    :param quiet: Suppress download progress output.
    :return: ``(labels, positions)`` with ``(n,)`` names and ``(n, 3)``
        millimetre coordinates.
    """
    member = _resolve(name, SENSORS, "sensor set")
    text = _read_member(member, quiet=quiet).decode("utf-8")

    labels: list[str] = []
    coords: list[list[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        labels.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])

    return np.asarray(labels, dtype=object), np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# PyVista bridge — requires the 'viz' extras
# ---------------------------------------------------------------------------


def _require_pyvista(fn_name: str):
    """Import and return PyVista, with a clear error if it is missing."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - exercised only without viz extras
        raise ImportError(
            f"{fn_name}() requires PyVista.\nInstall with:  poetry install --with viz"
        ) from exc
    return pv


def _faces_array(triangles: np.ndarray) -> np.ndarray:
    """Convert ``(m, 3)`` triangle indices to PyVista's flat faces format."""
    padding = np.full((len(triangles), 1), 3, dtype=np.int64)
    return np.hstack([padding, triangles]).ravel()


def _nearest_labels(
    source_points: np.ndarray, labels: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Transfer categorical labels to *target_points* by nearest neighbour.

    Uses :class:`scipy.spatial.cKDTree` when SciPy is available (it ships in
    the ``viz`` extras) and falls back to a chunked brute-force search
    otherwise, so decimation never silently drops a parcellation.

    :param source_points: ``(n, 3)`` points that *labels* belong to.
    :param labels: ``(n,)`` categorical values.
    :param target_points: ``(m, 3)`` points to label.
    :return: ``(m,)`` labels, one per target point.
    """
    try:
        from scipy.spatial import cKDTree

        _, index = cKDTree(source_points).query(target_points, k=1)
        return labels[index]
    except ImportError:
        pass

    # Chunked so peak memory stays bounded regardless of mesh size.
    out = np.empty(len(target_points), dtype=labels.dtype)
    chunk = max(1, 2_000_000 // max(len(source_points), 1))
    for start in range(0, len(target_points), chunk):
        block = target_points[start : start + chunk]
        dist = ((block[:, None, :] - source_points[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = labels[dist.argmin(axis=1)]
    return out


def surface_polydata(
    name: str,
    *,
    region_mapping: str | None = None,
    smooth_iters: int = 0,
    decimate: float = 0.0,
    quiet: bool = False,
):
    """Build a :class:`pyvista.PolyData` mesh from a TVB surface.

    :param name: Key from :data:`SURFACES`.
    :param region_mapping: Optional key from :data:`REGION_MAPPINGS`.  When
        given, the labels are attached as a ``"region"`` point scalar so the
        mesh can be coloured by parcellation.  The vertex counts must match.
    :param smooth_iters: Laplacian smoothing iterations (0 disables).
        Applied *after* the region scalars are attached, so labels survive.
    :param decimate: Fraction of triangles to remove, 0.0–1.0.  Useful for
        ``cortex_2x120k`` (566 752 triangles), which is heavy to sweep
        across 48 quilt views.
    :param quiet: Suppress download progress output.
    :return: The surface as a PyVista mesh.
    :raises ValueError: If *region_mapping* length does not match the mesh.
    """
    pv = _require_pyvista("surface_polydata")

    vertices, triangles, normals = load_surface(name, quiet=quiet)
    mesh = pv.PolyData(vertices, _faces_array(triangles))

    if normals is not None and len(normals) == mesh.n_points:
        mesh.point_data["Normals"] = normals

    if region_mapping is not None:
        labels = load_region_mapping(region_mapping, quiet=quiet)
        if len(labels) != mesh.n_points:
            raise ValueError(
                f"Region mapping '{region_mapping}' has {len(labels)} labels but "
                f"surface '{name}' has {mesh.n_points} vertices — they do not pair."
            )
        mesh.point_data["region"] = labels

    if decimate > 0.0:
        reduced = mesh.decimate(decimate)
        # Quadric decimation builds new vertices and discards point data, so
        # parcellation labels have to be carried across explicitly.  Nearest
        # neighbour is the right rule here: the labels are categorical, and
        # interpolating between region 3 and region 70 is meaningless.
        if region_mapping is not None:
            reduced.point_data["region"] = _nearest_labels(
                mesh.points, mesh.point_data["region"], reduced.points
            )
        mesh = reduced

    if smooth_iters > 0:
        mesh = mesh.smooth(n_iter=smooth_iters, relaxation_factor=0.1)

    return mesh


def connectome_polydata(
    name: str,
    *,
    percentile: float = 90.0,
    tube_radius: float = 1.1,
    node_radius: float = 4.5,
    quiet: bool = False,
):
    """Build node and edge meshes for a TVB connectome.

    Edges below the *percentile* of non-zero connection weights are dropped —
    a full connectome is far too dense to read as a hologram, and the
    strongest tracts are what carry the structure.

    :param name: Key from :data:`CONNECTIVITIES`.
    :param percentile: Keep only edges at or above this percentile of the
        non-zero weights (0 keeps every non-zero edge).
    :param tube_radius: Edge tube radius in mm.
    :param node_radius: Radius in mm of the largest region sphere; spheres
        are scaled by weighted degree.
    :param quiet: Suppress download progress output.
    :return: ``(nodes, edges)`` PyVista meshes.  *nodes* carries a
        ``"degree"`` scalar, *edges* carries a ``"weight"`` scalar.
    """
    pv = _require_pyvista("connectome_polydata")

    conn = load_connectivity(name, quiet=quiet)
    weights = conn.weights
    centres = conn.centres

    nonzero = weights[weights > 0]
    if nonzero.size == 0:
        raise ValueError(f"Connectome '{name}' has no non-zero weights.")
    threshold = np.percentile(nonzero, percentile) if percentile > 0 else nonzero.min()

    # Upper triangle only — TVB weight matrices are effectively symmetric for
    # display purposes, and drawing both directions doubles the tube count.
    rows, cols = np.nonzero(np.triu(weights >= threshold, k=1))
    if len(rows) == 0:
        raise ValueError(f"No edges survived the {percentile}th-percentile threshold for '{name}'.")

    lines = np.hstack([np.column_stack([np.full(len(rows), 2), rows, cols])]).ravel()
    edge_net = pv.PolyData(centres, lines=lines)
    edge_net.cell_data["weight"] = weights[rows, cols]
    edges = edge_net.tube(radius=tube_radius)

    degree = conn.degree
    node_cloud = pv.PolyData(centres)
    node_cloud.point_data["degree"] = degree
    scale = node_radius / degree.max() if degree.max() > 0 else node_radius
    # orient=False: the cloud carries no vectors, only the degree scalar.
    nodes = node_cloud.glyph(geom=pv.Sphere(radius=1.0), scale="degree", factor=scale, orient=False)

    return nodes, edges
