"""Backbone angle embedder: maps (φ, ψ) pairs to vectors for manifold analysis.

Three embedding strategies are provided:

    'torus'
        The four-dimensional T² ↪ ℝ⁴ embedding (cos φ, sin φ, cos ψ, sin ψ).
        Exact — no information loss.  Zero trainable parameters.  Best for
        ManifoldModel and GeodesicEncoder when the dataset is not too large.

    'discrete'
        8-fold quantization of each angle → 64 joint codes → lookup table of
        shape (n_bins², embedding_dim).  Analogous to the 8-class torsion
        binning used for disulfide bonds in proteusPy.  Codebook is fit via
        k-means on torus coordinates so each centroid is geometrically
        meaningful.

    'window'
        Sliding window of *window_size* consecutive torus embeddings,
        zero-padded at chain boundaries.  Output shape (N, 4 · window_size).
        Captures local secondary-structure context — a window of seven
        residues is wide enough to cover one α-helix turn or one β-strand
        step.

The three modes are unified behind a single sklearn-style fit / transform API
so they can be swapped without changing downstream code.

Part of WaveRider — https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
License: Elastic 2.0

"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .backbone_angles import BackboneAngleList

__all__ = ["BackboneEmbedder"]

EmbedMode = Literal["torus", "discrete", "window"]
AAMode = Literal["gpo", "onehot", "phys"]


class BackboneEmbedder:
    """Embed protein backbone (φ, ψ) angles into fixed-length vectors.

    Parameters
    ----------
    mode : {'torus', 'discrete', 'window'}
        Embedding strategy.  See module docstring.
    n_bins : int
        Number of angle bins per dimension for 'discrete' mode (default 8).
        Ignored for 'torus' and 'window'.
    embedding_dim : int
        Lookup-table width for 'discrete' mode (default 16).
        Ignored for 'torus' and 'window'.
    window_size : int
        Number of residues per window for 'window' mode (default 7).
        Ignored for 'torus' and 'discrete'.
    include_aa : bool
        If True, append amino acid type features to the angle embedding.
        For 'window' mode each residue position in the window gets its own
        AA features.  Default False.
    aa_mode : {'gpo', 'onehot'}
        AA encoding when include_aa=True.
        'gpo'    — 3-D one-hot: Glycine / Proline / Other.  Captures the
                   two residues with fundamentally different Ramachandran maps.
        'onehot' — 20-D one-hot over the standard 20-AA alphabet.

    Attributes
    ----------
    output_dim : int
        Length of each output vector.  Set after :meth:`fit`.
    codebook_ : np.ndarray or None
        Shape ``(n_bins², embedding_dim)`` lookup table.  Only set after
        :meth:`fit` in 'discrete' mode.
    """

    def __init__(
        self,
        mode: EmbedMode = "window",
        n_bins: int = 8,
        embedding_dim: int = 16,
        window_size: int = 7,
        include_aa: bool = False,
        aa_mode: AAMode = "gpo",
        include_omega: bool = False,
        context_only: bool = False,
    ) -> None:
        if mode not in ("torus", "discrete", "window"):
            raise ValueError(f"mode must be 'torus', 'discrete', or 'window'; got {mode!r}")
        if aa_mode not in ("gpo", "onehot", "phys"):
            raise ValueError(f"aa_mode must be 'gpo', 'onehot', or 'phys'; got {aa_mode!r}")
        self.mode = mode
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.include_aa = include_aa
        self.aa_mode = aa_mode
        self.include_omega = include_omega
        self.context_only = context_only

        self.codebook_: np.ndarray | None = None
        self._fitted = False

        aa_dim = self._aa_dim
        omega_dim = 2 if include_omega else 0
        if mode == "torus":
            self.output_dim = 4 + aa_dim + omega_dim
        elif mode == "window":
            self.output_dim = (4 + aa_dim + omega_dim) * window_size
        else:
            self.output_dim = embedding_dim + aa_dim + omega_dim

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------

    def fit(self, backbone_list: BackboneAngleList) -> BackboneEmbedder:
        """Fit the embedder on *backbone_list*.

        For 'torus' and 'window' modes this is a no-op.  For 'discrete' mode
        the codebook is initialised by computing the centroid of torus
        coordinates within each quantization bin.  Bins with no training
        samples fall back to a random unit vector on T².

        Parameters
        ----------
        backbone_list : BackboneAngleList
            Training collection.  Should contain only residues with finite
            φ and ψ (i.e. call ``.valid()`` beforehand).

        Returns
        -------
        self
        """
        if self.mode == "discrete":
            torus = backbone_list.to_torus_array()  # (N, 4)
            codes = backbone_list.to_combined_codes(self.n_bins)  # (N,)
            n_codes = self.n_bins * self.n_bins
            codebook = np.zeros((n_codes, self.embedding_dim), dtype=np.float32)

            for code in range(n_codes):
                mask = codes == code
                if mask.any():
                    # Project centroid of torus coords into embedding_dim via
                    # random linear projection (fixed seed per code).
                    centroid = torus[mask].mean(axis=0)  # (4,)
                    # Expand 4D centroid to embedding_dim via random projection
                    rng = np.random.default_rng(seed=code)
                    proj = rng.standard_normal((4, self.embedding_dim)).astype(np.float32)
                    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
                    codebook[code] = centroid @ proj
                else:
                    # Empty bin: unit vector in direction of bin centre
                    phi_bin = code // self.n_bins
                    psi_bin = code % self.n_bins
                    bin_width = 360.0 / self.n_bins
                    phi_c = math.radians(-180.0 + (phi_bin + 0.5) * bin_width)
                    psi_c = math.radians(-180.0 + (psi_bin + 0.5) * bin_width)
                    centre = np.array(
                        [
                            math.cos(phi_c),
                            math.sin(phi_c),
                            math.cos(psi_c),
                            math.sin(psi_c),
                        ],
                        dtype=np.float32,
                    )
                    rng = np.random.default_rng(seed=code)
                    proj = rng.standard_normal((4, self.embedding_dim)).astype(np.float32)
                    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
                    codebook[code] = centre @ proj

            self.codebook_ = codebook

        self._fitted = True
        return self

    def transform(self, backbone_list: BackboneAngleList) -> np.ndarray:
        """Embed *backbone_list* into a (N, output_dim) float32 array.

        Parameters
        ----------
        backbone_list : BackboneAngleList
            Collection to embed.  Use ``.valid()`` to remove terminal residues
            with NaN angles before calling.

        Returns
        -------
        np.ndarray
            Shape ``(N, output_dim)``, float32.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called for 'discrete' mode.
        """
        if self.mode == "torus":
            return self._embed_torus(backbone_list)
        if self.mode == "discrete":
            if self.codebook_ is None:
                raise RuntimeError("Call fit() before transform() for mode='discrete'.")
            return self._embed_discrete(backbone_list)
        return self._embed_window(backbone_list)

    def fit_transform(self, backbone_list: BackboneAngleList) -> np.ndarray:
        """Fit and immediately transform *backbone_list*."""
        return self.fit(backbone_list).transform(backbone_list)

    # ------------------------------------------------------------------
    # AA helpers
    # ------------------------------------------------------------------

    @property
    def _aa_dim(self) -> int:
        """Dimension contributed by the AA feature block (0 if include_aa=False)."""
        if not self.include_aa:
            return 0
        return {"gpo": 3, "phys": 4, "onehot": 20}[self.aa_mode]

    def _aa_features(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, _aa_dim) AA feature matrix."""
        if self.aa_mode == "gpo":
            return bal.to_aa_class_array()
        if self.aa_mode == "phys":
            return bal.to_aa_phys_array()
        return bal.to_aa_onehot_array()

    # ------------------------------------------------------------------
    # Private embedding implementations
    # ------------------------------------------------------------------

    def _embed_torus(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, 4+aa_dim+omega_dim) torus coordinates with optional extras."""
        parts = [bal.to_torus_array()]
        if self.include_aa:
            parts.append(self._aa_features(bal))
        if self.include_omega:
            parts.append(bal.to_omega_torus_array())
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _embed_discrete(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, embedding_dim+aa_dim+omega_dim) via codebook lookup."""
        parts = [self.codebook_[bal.to_combined_codes(self.n_bins)]]  # type: ignore[index]
        if self.include_aa:
            parts.append(self._aa_features(bal))
        if self.include_omega:
            parts.append(bal.to_omega_torus_array())
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _embed_window(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, (4+aa_dim+omega_dim)*window_size) via sliding-window concatenation.

        The window is centred on each residue.  Positions outside the chain
        boundaries are zero-padded.  Each position in the window carries its
        own AA and ω features when the respective flags are set.

        When context_only=True the center position's torus (and ω) features are
        zeroed so the model cannot see the center's own dihedral angles.  AA
        features at the center position are preserved — the model knows the
        residue identity but not its conformation.  This is the correct setup
        for predicting Ramachandran region from neighborhood context alone.
        """
        parts = [bal.to_torus_array()]
        if self.include_aa:
            parts.append(self._aa_features(bal))
        if self.include_omega:
            parts.append(bal.to_omega_torus_array())
        feat = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

        n = len(feat)
        w = self.window_size
        half = w // 2
        feat_dim = feat.shape[1]

        padded = np.zeros((n + 2 * half, feat_dim), dtype=np.float32)
        padded[half : half + n] = feat

        result = np.empty((n, feat_dim * w), dtype=np.float32)
        for i in range(n):
            result[i] = padded[i : i + w].ravel()

        if self.context_only:
            # Zero center position's torus (0:4) and omega (after AA) features.
            # Per-position layout: torus(4) | aa(_aa_dim) | omega(2 or 0)
            omega_dims = 2 if self.include_omega else 0
            center_off = half * feat_dim
            result[:, center_off : center_off + 4] = 0.0
            if omega_dims:
                aa_end = center_off + 4 + self._aa_dim
                result[:, aa_end : aa_end + omega_dims] = 0.0

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        aa_tag = f", aa={self.aa_mode}" if self.include_aa else ""
        omega_tag = ", omega" if self.include_omega else ""
        if self.mode == "torus":
            return (
                f"BackboneEmbedder(mode='torus', output_dim={self.output_dim}{aa_tag}{omega_tag})"
            )
        if self.mode == "discrete":
            fitted = "fitted" if self._fitted else "unfitted"
            return (
                f"BackboneEmbedder(mode='discrete', n_bins={self.n_bins}, "
                f"embedding_dim={self.embedding_dim}{aa_tag}{omega_tag}, {fitted})"
            )
        return (
            f"BackboneEmbedder(mode='window', window_size={self.window_size}, "
            f"output_dim={self.output_dim}{aa_tag}{omega_tag})"
        )
