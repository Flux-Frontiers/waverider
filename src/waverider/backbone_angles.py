"""Protein backbone dihedral angle representation for manifold analysis.

BackboneResidue holds the (φ, ψ, ω) angles for one residue.
BackboneAngleList is the collection class — analogous to DisulfideList in
proteusPy — and is the primary input to BackboneEmbedder and
fit_backbone_manifold.

Angle conventions (standard biochemistry):
    φ (phi)   : C(i-1) – N(i)  – Cα(i) – C(i)    range (−180, +180]
    ψ (psi)   : N(i)   – Cα(i) – C(i)  – N(i+1)  range (−180, +180]
    ω (omega) : Cα(i-1)– C(i-1)– N(i)  – Cα(i)   range (−180, +180], near ±180°

Known Ramachandran clusters (used by from_synthetic):
    'H'  alpha-helix        φ ≈ −60°, ψ ≈ −40°
    'E'  beta-sheet         φ ≈ −120°, ψ ≈ +130°
    'P'  polyproline II     φ ≈ −75°,  ψ ≈ +150°
    'L'  left-handed helix  φ ≈ +60°,  ψ ≈ +40°
    'C'  coil               mixture of H / E / P with wider spread

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import copy
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .vector3D import Vector3D, calc_dihedral

if TYPE_CHECKING:
    pass

__all__ = [
    "BackboneResidue",
    "BackboneAngleList",
    "quantize_angle",
    "SECONDARY_STRUCTURE_CODES",
    "RAMA_REGIONS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECONDARY_STRUCTURE_CODES: dict[str, str] = {
    "H": "alpha-helix",
    "E": "beta-sheet",
    "P": "polyproline-II",
    "L": "left-handed helix",
    "C": "coil",
    "U": "unknown",
}

# (phi_mean, phi_std, psi_mean, psi_std) in degrees
RAMA_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "H": (-60.0, 8.0, -40.0, 8.0),
    "E": (-120.0, 12.0, 130.0, 12.0),
    "P": (-75.0, 8.0, 150.0, 8.0),
    "L": (60.0, 8.0, 40.0, 8.0),
}

_OMEGA_TRANS = 180.0  # degrees; standard trans peptide bond

# Standard 20 amino acids, alphabetical by 3-letter code.
# GLY is index 7, PRO is index 14 — the two Ramachandran outliers.
_AA20 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
_AA20_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA20)}

# Kyte-Doolittle (1982) hydrophobicity, normalized to [−1, 1] by dividing by 4.5.
# Unknown / non-standard residues map to 0.0 (neutral).
_KD_SCALE: dict[str, float] = {
    "ILE": 1.000,
    "VAL": 0.933,
    "LEU": 0.844,
    "PHE": 0.622,
    "CYS": 0.556,
    "MET": 0.422,
    "ALA": 0.400,
    "GLY": -0.089,
    "THR": -0.156,
    "SER": -0.178,
    "TRP": -0.200,
    "TYR": -0.289,
    "PRO": -0.356,
    "HIS": -0.711,
    "GLU": -0.778,
    "GLN": -0.778,
    "ASP": -0.778,
    "ASN": -0.778,
    "LYS": -0.867,
    "ARG": -1.000,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def quantize_angle(angle: float, n_bins: int = 8) -> int:
    """Map an angle in (−180, +180] to a bin index in [0, n_bins).

    Bins are equal-width (360 / n_bins degrees) starting at −180°.

    Parameters
    ----------
    angle : float
        Angle in degrees.  Will be wrapped to (−180, +180] automatically.
    n_bins : int
        Number of bins (default 8 → 45° per bin).

    Returns
    -------
    int
        Bin index in [0, n_bins).

    Examples
    --------
    >>> quantize_angle(-60.0)    # alpha-helix phi → bin 2
    2
    >>> quantize_angle(130.0)    # beta-sheet psi → bin 6
    6
    >>> quantize_angle(-180.0)   # lower boundary
    0
    >>> quantize_angle(180.0)    # upper boundary wraps to bin 0
    0
    """
    # Wrap to (−180, +180]
    angle = angle % 360.0
    if angle > 180.0:
        angle -= 360.0
    # Shift to [0, 360) and discretize
    shifted = (angle + 180.0) % 360.0
    bin_width = 360.0 / n_bins
    return int(shifted // bin_width) % n_bins


# ---------------------------------------------------------------------------
# BackboneResidue
# ---------------------------------------------------------------------------


@dataclass
class BackboneResidue:
    """Dihedral angles and identity for one protein backbone residue.

    Parameters
    ----------
    phi, psi, omega : float
        Backbone dihedral angles in degrees, range (−180, +180].
        Use ``math.nan`` for terminal residues where the angle is undefined.
    residue_name : str
        Three-letter amino acid code (e.g. 'ALA', 'GLY').
    chain_id : str
        PDB chain identifier.
    seq_pos : int
        Residue sequence number from the PDB ATOM record.
    pdb_id : str
        Four-character PDB identifier or synthetic label.
    secondary_structure : str
        Single-character secondary structure code (H/E/P/L/C/U).
    """

    phi: float
    psi: float
    omega: float = _OMEGA_TRANS
    residue_name: str = "UNK"
    chain_id: str = "A"
    seq_pos: int = 0
    pdb_id: str = "SYN"
    secondary_structure: str = "U"

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def phi_bin(self, n_bins: int = 8) -> int:
        """Bin index for φ in [0, n_bins)."""
        return quantize_angle(self.phi, n_bins)

    def psi_bin(self, n_bins: int = 8) -> int:
        """Bin index for ψ in [0, n_bins)."""
        return quantize_angle(self.psi, n_bins)

    def combined_code(self, n_bins: int = 8) -> int:
        """Joint (φ, ψ) discrete code in [0, n_bins²).

        Encoding: ``phi_bin * n_bins + psi_bin``.  For the default n_bins=8
        this gives 64 possible states, analogous to the 8-class torsion
        binning used for disulfide bonds in proteusPy.
        """
        return self.phi_bin(n_bins) * n_bins + self.psi_bin(n_bins)

    # ------------------------------------------------------------------
    # Continuous embeddings
    # ------------------------------------------------------------------

    def torus_coords(self) -> np.ndarray:
        """Embed (φ, ψ) as a point on T² ↪ ℝ⁴.

        Returns (cos φ, sin φ, cos ψ, sin ψ) — the canonical isometric
        embedding of the 2-torus that preserves circular topology without
        the wrap-around discontinuity of raw degree values.
        """
        phi_r = math.radians(self.phi)
        psi_r = math.radians(self.psi)
        return np.array(
            [math.cos(phi_r), math.sin(phi_r), math.cos(psi_r), math.sin(psi_r)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Amino acid type encoding
    # ------------------------------------------------------------------

    def aa_gpo_class(self) -> int:
        """Return Gly/Pro/Other class index (0, 1, 2).

        Glycine (0) has no Cβ and a symmetric Ramachandran plot.
        Proline (1) has a cyclic ring that constrains φ to ≈ −60°.
        All other residues (2) share the standard generalized Ramachandran map.
        """
        name = self.residue_name.upper()[:3]
        if name == "GLY":
            return 0
        if name == "PRO":
            return 1
        return 2

    def aa_idx(self) -> int:
        """Return index into the standard 20-AA alphabet, or -1 for UNK/non-standard."""
        return _AA20_IDX.get(self.residue_name.upper()[:3], -1)

    def aa_hphob(self) -> float:
        """Kyte-Doolittle hydrophobicity normalized to [−1, 1]; 0.0 for unknowns."""
        return _KD_SCALE.get(self.residue_name.upper()[:3], 0.0)

    def __repr__(self) -> str:
        return (
            f"BackboneResidue({self.pdb_id}:{self.chain_id}:{self.seq_pos} "
            f"{self.residue_name} φ={self.phi:.1f}° ψ={self.psi:.1f}° "
            f"ss={self.secondary_structure})"
        )


# ---------------------------------------------------------------------------
# BackboneAngleList
# ---------------------------------------------------------------------------


@dataclass
class BackboneAngleList:
    """An ordered collection of BackboneResidue objects.

    Analogous to DisulfideList in proteusPy.  Provides bulk conversions to
    numpy arrays suitable for WaveRider manifold analysis.

    Parameters
    ----------
    residues : list[BackboneResidue]
        The backbone residues.
    name : str
        Human-readable label for this collection.
    """

    residues: list[BackboneResidue] = field(default_factory=list)
    name: str = "BackboneAngleList"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_synthetic(
        cls,
        n: int = 2000,
        fractions: dict[str, float] | None = None,
        seed: int | None = 42,
        name: str = "synthetic",
    ) -> BackboneAngleList:
        """Generate angles from known Ramachandran distributions.

        Each secondary structure type is drawn from a 2D Gaussian centred on
        the canonical (φ, ψ) values.  Useful for baseline benchmarks and unit
        tests where the ground-truth clusters are known.

        Parameters
        ----------
        n : int
            Total number of residues.
        fractions : dict, optional
            Fraction of each secondary structure type.  Keys are single-letter
            codes from SECONDARY_STRUCTURE_CODES.  Must sum to ≤ 1; any
            remainder is assigned to coil ('C').  Defaults to a protein-like
            mixture: H=0.33, E=0.27, P=0.15, L=0.05, C=0.20.
        seed : int or None
            Random seed for reproducibility.
        name : str
            Label for the returned list.

        Returns
        -------
        BackboneAngleList
        """
        rng = np.random.default_rng(seed)

        if fractions is None:
            fractions = {"H": 0.33, "E": 0.27, "P": 0.15, "L": 0.05, "C": 0.20}

        residues: list[BackboneResidue] = []
        seq = 0

        for ss_code, frac in fractions.items():
            count = round(n * frac)
            if ss_code == "C":
                # Coil: draw from a weighted mixture of the four structured regions
                # with increased spread (models the diffuse outer allowed area).
                mix_codes = ["H", "E", "P", "L"]
                mix_counts = [count // 4 + (1 if i < count % 4 else 0) for i in range(4)]
                for mc, mc_n in zip(mix_codes, mix_counts):
                    phi_m, phi_s, psi_m, psi_s = RAMA_REGIONS[mc]
                    phis = rng.normal(phi_m, phi_s * 2.5, mc_n)
                    psis = rng.normal(psi_m, psi_s * 2.5, mc_n)
                    for phi, psi in zip(phis, psis):
                        residues.append(
                            BackboneResidue(
                                phi=float(phi),
                                psi=float(psi),
                                omega=_OMEGA_TRANS + float(rng.normal(0, 3)),
                                residue_name="GLY",
                                chain_id="A",
                                seq_pos=seq,
                                pdb_id=name,
                                secondary_structure="C",
                            )
                        )
                        seq += 1
            else:
                phi_m, phi_s, psi_m, psi_s = RAMA_REGIONS[ss_code]
                phis = rng.normal(phi_m, phi_s, count)
                psis = rng.normal(psi_m, psi_s, count)
                for phi, psi in zip(phis, psis):
                    residues.append(
                        BackboneResidue(
                            phi=float(phi),
                            psi=float(psi),
                            omega=_OMEGA_TRANS + float(rng.normal(0, 3)),
                            residue_name="ALA",
                            chain_id="A",
                            seq_pos=seq,
                            pdb_id=name,
                            secondary_structure=ss_code,
                        )
                    )
                    seq += 1

        return cls(residues=residues, name=name)

    @classmethod
    def from_pdb_file(
        cls,
        path: str,
        pdb_id: str | None = None,
        chain_ids: list[str] | None = None,
    ) -> BackboneAngleList:
        """Parse backbone dihedrals directly from a PDB ATOM record file.

        Uses the Vector3D / calc_dihedral machinery already in WaveRider;
        no external parser required.

        Parameters
        ----------
        path : str
            Path to a .pdb (or .ent) file.
        pdb_id : str, optional
            Label to tag each residue; defaults to the filename stem.
        chain_ids : list[str], optional
            If given, only these chain IDs are processed.

        Returns
        -------
        BackboneAngleList

        Notes
        -----
        Only the first alternate-conformation location ('A' or blank) is used.
        Residues missing N, Cα, or C atoms are silently skipped.
        Terminal residues have undefined φ or ψ stored as math.nan.
        """
        if pdb_id is None:
            pdb_id = os.path.splitext(os.path.basename(path))[0].upper()

        # -- Parse ATOM records -------------------------------------------------
        # key: (chain_id, seq_num, ins_code) → {atom_name: Vector3D}
        residue_atoms: dict[tuple[str, int, str], dict[str, Vector3D]] = {}
        residue_names: dict[tuple[str, int, str], str] = {}

        with open(path) as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec not in ("ATOM", "HETATM"):
                    continue
                alt_loc = line[16].strip()
                if alt_loc and alt_loc != "A":
                    continue
                atom_name = line[12:16].strip()
                if atom_name not in ("N", "CA", "C"):
                    continue
                res_name = line[17:20].strip()
                chain = line[21].strip() or "A"
                if chain_ids and chain not in chain_ids:
                    continue
                try:
                    seq_num = int(line[22:26])
                except ValueError:
                    continue
                ins_code = line[26].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue

                key = (chain, seq_num, ins_code)
                residue_atoms.setdefault(key, {})[atom_name] = Vector3D(x, y, z)
                residue_names[key] = res_name

        # -- Sort residues per chain in sequence order --------------------------
        chains: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for chain, seq_num, ins in residue_atoms:
            chains[chain].append((chain, seq_num, ins))
        for ch in chains:
            chains[ch].sort(key=lambda k: (k[1], k[2]))

        # -- Compute dihedrals -------------------------------------------------
        residues: list[BackboneResidue] = []

        for ch_keys in chains.values():
            for i, key in enumerate(ch_keys):
                atoms = residue_atoms[key]
                if not {"N", "CA", "C"}.issubset(atoms):
                    continue

                # phi: C(i-1) – N – CA – C
                phi = math.nan
                if i > 0:
                    prev_atoms = residue_atoms.get(ch_keys[i - 1])
                    if prev_atoms and "C" in prev_atoms:
                        phi = calc_dihedral(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"])

                # psi: N – CA – C – N(i+1)
                psi = math.nan
                if i < len(ch_keys) - 1:
                    next_atoms = residue_atoms.get(ch_keys[i + 1])
                    if next_atoms and "N" in next_atoms:
                        psi = calc_dihedral(atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"])

                # omega: CA(i-1) – C(i-1) – N – CA
                omega = math.nan
                if i > 0:
                    prev_atoms = residue_atoms.get(ch_keys[i - 1])
                    if prev_atoms and {"CA", "C"}.issubset(prev_atoms):
                        omega = calc_dihedral(
                            prev_atoms["CA"],
                            prev_atoms["C"],
                            atoms["N"],
                            atoms["CA"],
                        )

                residues.append(
                    BackboneResidue(
                        phi=phi,
                        psi=psi,
                        omega=omega if not math.isnan(omega) else _OMEGA_TRANS,
                        residue_name=residue_names[key],
                        chain_id=key[0],
                        seq_pos=key[1],
                        pdb_id=pdb_id,
                        secondary_structure="U",
                    )
                )

        return cls(residues=residues, name=pdb_id)

    @classmethod
    def from_proteuspy(cls, source, name: str = "proteuspy") -> BackboneAngleList:
        """Load backbone angles from a proteusPy structure object.

        Parameters
        ----------
        source : proteusPy.protein.Protein or list of residue-like objects
            Any object that yields items with ``.phi``, ``.psi`` attributes
            and optionally ``.residue_name``, ``.chain_id``, ``.seq_pos``,
            ``.pdb_id``.

        Returns
        -------
        BackboneAngleList
        """
        residues = []
        for i, r in enumerate(source):
            residues.append(
                BackboneResidue(
                    phi=float(getattr(r, "phi", math.nan)),
                    psi=float(getattr(r, "psi", math.nan)),
                    omega=float(getattr(r, "omega", _OMEGA_TRANS)),
                    residue_name=getattr(r, "residue_name", "UNK"),
                    chain_id=getattr(r, "chain_id", "A"),
                    seq_pos=int(getattr(r, "seq_pos", i)),
                    pdb_id=getattr(r, "pdb_id", name),
                    secondary_structure=getattr(r, "secondary_structure", "U"),
                )
            )
        return cls(residues=residues, name=name)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def valid(self) -> BackboneAngleList:
        """Return a new list containing only residues with finite φ and ψ."""
        kept = [r for r in self.residues if not (math.isnan(r.phi) or math.isnan(r.psi))]
        return BackboneAngleList(residues=kept, name=self.name + "_valid")

    def filter_ss(self, *codes: str) -> BackboneAngleList:
        """Return residues whose secondary_structure matches any of *codes*."""
        kept = [r for r in self.residues if r.secondary_structure in codes]
        return BackboneAngleList(residues=kept, name=self.name + f"_{''.join(codes)}")

    def remap_u_by_ramachandran(self) -> BackboneAngleList:
        """Re-classify U (unknown) residues by Ramachandran angle-box geometry.

        Instead of lumping all U residues into coil, assigns each one to the
        correct structural class based on where its (φ, ψ) actually falls on
        the Ramachandran plot.  Boxes are checked in priority order to resolve
        overlaps (P before E; H before P).

        Boxes used (degrees):

        ============  =================  =================
        Class         φ range            ψ range
        ============  =================  =================
        H (α-helix)   [−90, −30]         [−70, −20]
        L (left)      [+30, +90]         [+20, +80]
        P (PPII)      [−90, −45]         [+110, +180]
        E (β-sheet)   [−170, −50]        [+90, +180] or [−180, −150]
        C (coil)      all other valid (φ, ψ)
        ============  =================  =================

        Residues whose φ or ψ is NaN remain U (terminal residues).

        Returns a new :class:`BackboneAngleList`; does not modify in place.
        """

        def _assign(phi: float, psi: float) -> str:
            if math.isnan(phi) or math.isnan(psi):
                return "U"
            if -90 <= phi <= -30 and -70 <= psi <= -20:
                return "H"
            if 30 <= phi <= 90 and 20 <= psi <= 80:
                return "L"
            if -90 <= phi <= -45 and 110 <= psi <= 180:
                return "P"
            if -170 <= phi <= -50 and (90 <= psi <= 180 or -180 <= psi <= -150):
                return "E"
            return "C"

        remapped = []
        for r in self.residues:
            if r.secondary_structure == "U":
                new_r = copy.copy(r)
                new_r.secondary_structure = _assign(r.phi, r.psi)
                remapped.append(new_r)
            else:
                remapped.append(r)
        return BackboneAngleList(residues=remapped, name=self.name + "_rama")

    # ------------------------------------------------------------------
    # Bulk conversions
    # ------------------------------------------------------------------

    def to_phi_psi_array(self) -> np.ndarray:
        """Return (N, 2) float32 array of raw (φ, ψ) degrees."""
        return np.array([[r.phi, r.psi] for r in self.residues], dtype=np.float32)

    def to_torus_array(self) -> np.ndarray:
        """Return (N, 4) float32 array of T² embeddings (cos φ, sin φ, cos ψ, sin ψ).

        This is the canonical input for ManifoldModel / GeodesicEncoder.
        The embedding is isometric on the torus — Euclidean distance in ℝ⁴
        approximates geodesic distance on T² for nearby points.
        """
        return np.array([r.torus_coords() for r in self.residues], dtype=np.float32)

    def to_combined_codes(self, n_bins: int = 8) -> np.ndarray:
        """Return (N,) int32 array of joint (φ, ψ) bin codes in [0, n_bins²)."""
        return np.array([r.combined_code(n_bins) for r in self.residues], dtype=np.int32)

    def to_ss_labels(self) -> np.ndarray:
        """Return (N,) object array of single-character secondary structure codes."""
        return np.array([r.secondary_structure for r in self.residues])

    def to_ss_int_labels(self) -> np.ndarray:
        """Return (N,) int32 array mapping ss code → integer for ManifoldModel.

        Mapping: H=0, E=1, P=2, L=3, C=4, U=5.
        """
        _map = {"H": 0, "E": 1, "P": 2, "L": 3, "C": 4, "U": 5}
        return np.array([_map.get(r.secondary_structure, 5) for r in self.residues], dtype=np.int32)

    def to_rama_region_labels(self) -> np.ndarray:
        """Return (N,) geometry-derived Ramachandran region codes for all residues.

        Labels are assigned from (φ, ψ) angle boxes regardless of stored DSSP
        secondary_structure.  Boxes match those used in remap_u_by_ramachandran.
        Residues with NaN angles return 'U'.
        """

        def _region(phi: float, psi: float) -> str:
            if math.isnan(phi) or math.isnan(psi):
                return "U"
            if -90 <= phi <= -30 and -70 <= psi <= -20:
                return "H"
            if 30 <= phi <= 90 and 20 <= psi <= 80:
                return "L"
            if -90 <= phi <= -45 and 110 <= psi <= 180:
                return "P"
            if -170 <= phi <= -50 and (90 <= psi <= 180 or -180 <= psi <= -150):
                return "E"
            return "C"

        return np.array([_region(r.phi, r.psi) for r in self.residues])

    def to_rama_int_labels_geometry(self) -> np.ndarray:
        """Return (N,) int32 geometry-derived Ramachandran labels: H=0 E=1 P=2 L=3 C=4 U=5."""
        _map = {"H": 0, "E": 1, "P": 2, "L": 3, "C": 4, "U": 5}
        return np.array([_map[c] for c in self.to_rama_region_labels()], dtype=np.int32)

    def to_aa_class_array(self) -> np.ndarray:
        """Return (N, 3) float32 one-hot encoding of Gly / Pro / Other.

        Column 0 = Glycine, column 1 = Proline, column 2 = all other residues.
        These three classes have fundamentally different Ramachandran maps:
        Gly is symmetric (no Cβ), Pro is constrained (cyclic ring), all others
        share the standard generalized map.
        """
        out = np.zeros((len(self.residues), 3), dtype=np.float32)
        for i, r in enumerate(self.residues):
            out[i, r.aa_gpo_class()] = 1.0
        return out

    def to_omega_torus_array(self) -> np.ndarray:
        """Return (N, 2) float32: (cos ω, sin ω) for each residue.

        ω is the peptide bond planarity angle (Cα(i-1)–C(i-1)–N–Cα).
        Trans peptides cluster near ±180°; cis-Pro near 0°.
        Residues with NaN ω (rare parquet gaps) fall back to _OMEGA_TRANS.
        """
        out = []
        for r in self.residues:
            omega = r.omega if math.isfinite(r.omega) else _OMEGA_TRANS
            rad = math.radians(omega)
            out.append([math.cos(rad), math.sin(rad)])
        return np.array(out, dtype=np.float32)

    def to_aa_phys_array(self) -> np.ndarray:
        """Return (N, 4) float32: GPO one-hot (3) + normalized Kyte-Doolittle (1).

        Combines the Ramachandran-critical Gly/Pro/Other distinction with a
        continuous hydrophobicity signal (range [−1, 1]).  Unknown residues get
        hphob=0.0 (neutral).
        """
        gpo = self.to_aa_class_array()  # (N, 3)
        hphob = np.array([[r.aa_hphob()] for r in self.residues], dtype=np.float32)  # (N, 1)
        return np.concatenate([gpo, hphob], axis=1)

    def to_aa_onehot_array(self) -> np.ndarray:
        """Return (N, 20) float32 one-hot over the standard 20-AA alphabet.

        Non-standard / unknown residues produce an all-zero row.
        Column order: ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE
                      LEU LYS MET PHE PRO SER THR TRP TYR VAL
        """
        out = np.zeros((len(self.residues), 20), dtype=np.float32)
        for i, r in enumerate(self.residues):
            idx = r.aa_idx()
            if idx >= 0:
                out[i, idx] = 1.0
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.residues)

    def __repr__(self) -> str:
        valid = sum(1 for r in self.residues if not (math.isnan(r.phi) or math.isnan(r.psi)))
        ss_counts: dict[str, int] = {}
        for r in self.residues:
            ss_counts[r.secondary_structure] = ss_counts.get(r.secondary_structure, 0) + 1
        ss_str = " ".join(f"{k}:{v}" for k, v in sorted(ss_counts.items()))
        return f"BackboneAngleList('{self.name}', n={len(self)}, valid={valid}, ss=[{ss_str}])"

    def ramachandran_plot(self, title: str | None = None, ax=None):
        """Scatter plot of (φ, ψ) pairs coloured by secondary structure.

        Requires matplotlib.  Returns the Axes object.

        Parameters
        ----------
        title : str, optional
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into; creates a new figure if None.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        ss_colours = {
            "H": "#e41a1c",
            "E": "#377eb8",
            "P": "#4daf4a",
            "L": "#ff7f00",
            "C": "#aaaaaa",
            "U": "#dddddd",
        }

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        arr = self.to_phi_psi_array()
        ss = self.to_ss_labels()

        for code, colour in ss_colours.items():
            mask = ss == code
            if mask.any():
                ax.scatter(
                    arr[mask, 0],
                    arr[mask, 1],
                    c=colour,
                    s=4,
                    alpha=0.5,
                    label=f"{code} ({SECONDARY_STRUCTURE_CODES[code]})",
                )

        ax.set_xlabel("φ (degrees)")
        ax.set_ylabel("ψ (degrees)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_title(title or self.name)
        ax.legend(markerscale=3, fontsize=8)
        return ax


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Quick sanity check
    bal = BackboneAngleList.from_synthetic(n=500, seed=0)
    print(bal)
    print("Torus array shape:", bal.to_torus_array().shape)
    print("Combined codes sample:", bal.to_combined_codes()[:10])
