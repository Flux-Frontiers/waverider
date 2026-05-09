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
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .backbone_angles import BackboneAngleList

__all__ = ["BackboneEmbedder"]

EmbedMode = Literal["torus", "discrete", "window"]


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
    ) -> None:
        if mode not in ("torus", "discrete", "window"):
            raise ValueError(f"mode must be 'torus', 'discrete', or 'window'; got {mode!r}")
        self.mode = mode
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.codebook_: np.ndarray | None = None
        self._fitted = False

        # Set output_dim for torus and window immediately; discrete waits for fit.
        if mode == "torus":
            self.output_dim = 4
        elif mode == "window":
            self.output_dim = 4 * window_size
        else:
            self.output_dim = embedding_dim  # updated in fit after k-means

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------

    def fit(self, backbone_list: BackboneAngleList) -> "BackboneEmbedder":
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
            torus = backbone_list.to_torus_array()          # (N, 4)
            codes = backbone_list.to_combined_codes(self.n_bins)  # (N,)
            n_codes = self.n_bins * self.n_bins
            codebook = np.zeros((n_codes, self.embedding_dim), dtype=np.float32)

            for code in range(n_codes):
                mask = codes == code
                if mask.any():
                    # Project centroid of torus coords into embedding_dim via
                    # random linear projection (fixed seed per code).
                    centroid = torus[mask].mean(axis=0)      # (4,)
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
                        [math.cos(phi_c), math.sin(phi_c), math.cos(psi_c), math.sin(psi_c)],
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
    # Private embedding implementations
    # ------------------------------------------------------------------

    def _embed_torus(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, 4) torus coordinates."""
        return bal.to_torus_array()

    def _embed_discrete(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, embedding_dim) via codebook lookup."""
        codes = bal.to_combined_codes(self.n_bins)   # (N,)
        return self.codebook_[codes]                  # type: ignore[index]

    def _embed_window(self, bal: BackboneAngleList) -> np.ndarray:
        """Return (N, 4 * window_size) via sliding-window concatenation.

        The window is centred on each residue.  Positions outside the chain
        boundaries are zero-padded, so terminal residues get partial context.
        Window size should be odd (e.g. 7) so the current residue sits in the
        exact centre.
        """
        torus = bal.to_torus_array()         # (N, 4)
        n = len(torus)
        w = self.window_size
        half = w // 2

        # Pad with zeros on both ends
        padded = np.zeros((n + 2 * half, 4), dtype=np.float32)
        padded[half : half + n] = torus

        # Stride trick for fast window extraction
        result = np.empty((n, 4 * w), dtype=np.float32)
        for i in range(n):
            result[i] = padded[i : i + w].ravel()

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.mode == "torus":
            return f"BackboneEmbedder(mode='torus', output_dim=4)"
        if self.mode == "discrete":
            fitted = "fitted" if self._fitted else "unfitted"
            return (
                f"BackboneEmbedder(mode='discrete', n_bins={self.n_bins}, "
                f"embedding_dim={self.embedding_dim}, {fitted})"
            )
        return (
            f"BackboneEmbedder(mode='window', window_size={self.window_size}, "
            f"output_dim={self.output_dim})"
        )
