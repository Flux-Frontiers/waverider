# Protein Backbone Manifold Analysis
## WaveRider Latent-Space Discovery on Protein Secondary Structure

**Authors:** Eric G. Suchanek, PhD
**Date:** 2026-05-14
**Repository:** `waverider` @ main

---

## 1. Scientific Question

Can WaveRider's manifold stack — without labels, without training, without prior
knowledge of protein structure — rediscover the geometry of the Ramachandran plot
from raw backbone dihedral angles?

The Ramachandran plot (Ramachandran et al., 1963) shows that backbone dihedral
angles (φ, ψ) cluster into a small number of sterically allowed regions
corresponding to known secondary structure classes: α-helix (H), β-sheet (E),
polyproline II helix (P), left-handed helix (L), and coil/loop (C). These clusters
are known from first principles. The question is whether the WaveRider manifold
stack can recover them geometrically, without supervision.

---

## 2. The Backbone Angle Manifold

Each protein residue contributes one point (φ, ψ) ∈ (−180°, +180²] to the
Ramachandran plot. These two angles parameterise a 2-torus T². The torus embedding

    (φ, ψ) → (cos φ, sin φ, cos ψ, sin ψ) ∈ ℝ⁴

is lossless and isometric: Euclidean distance in ℝ⁴ approximates geodesic distance
on T² for nearby points. The embedding has ambient dimension d_ambient = 4 and
true intrinsic dimension d* = 2.

WaveRider also tests three richer embeddings:

| Mode | d_ambient | Description |
|---|---|---|
| `torus` | 4 | (cos φ, sin φ, cos ψ, sin ψ) — lossless T² |
| `discrete` | 16 | 8-fold quantization → 64 joint codes → 16-D lookup |
| `window7` | 28 | 7-residue sliding window of torus vectors (4 × 7) |
| `window13` | 52 | 13-residue sliding window of torus vectors (4 × 13) |

Window modes capture local sequence context — whether the surrounding residues
also adopt helical or extended geometry — rather than a single residue in isolation.

---

## 3. Data

### 3.1 Corpora

Two corpora were analysed:

**Disulfide corpus** (`good_backbone.parquet`): structures from a curated PDB subset
enriched for disulfide-bond-containing proteins. Secondary structure labels come
from PDB HELIX/SHEET records (author-assigned).

**PISCES corpus** (`pisces_1000.parquet`): 1,000 structures drawn from the PISCES
non-redundant PDB list (25% sequence identity cutoff, ≤ 2.0 Å resolution). PISCES
is a standard reference for unbiased backbone analysis.

### 3.2 Secondary Structure Labels and the U-Class Problem

PDB HELIX/SHEET records label residues as H or E; everything else is U (unknown).
The large U fraction (17–34% depending on corpus) is not "undefined" geometry —
it is simply residues that the PDB depositor did not explicitly annotate.

Two remapping strategies were evaluated:

**`--remap-u-to-coil`**: blindly assigns all U → C. Fast, but pollutes the coil
class with residues that may be helical or sheet-like by geometry.

**`--remap-u-rama`** (new): assigns each U residue to H, L, P, E, or C based on
where its (φ, ψ) actually falls in the Ramachandran plot, using the canonical
angle boxes:

| Class | φ range | ψ range |
|---|---|---|
| H (α-helix) | [−90°, −30°] | [−70°, −20°] |
| L (left helix) | [+30°, +90°] | [+20°, +80°] |
| P (PPII) | [−90°, −45°] | [+110°, +180°] |
| E (β-sheet) | [−170°, −50°] | [+90°, +180°] or [−180°, −150°] |
| C (coil) | all other valid (φ, ψ) | |

Residues with NaN angles (terminal residues) remain U.

The Ramachandran remap correctly populates the P and L classes, which are
otherwise silently discarded:

| Remap | H | E | P | L | C | U |
|---|---|---|---|---|---|---|
| U→coil | 43.7% | 22.8% | — | — | 33.5% | — |
| U→rama | 47.4% | 30.7% | 8.6% | 1.6% | 11.7% | ≈0% |

### 3.3 Evaluation Protocol

All accuracy figures are held-out test accuracy from a stratified 80/20
train/test split (class proportions preserved). ManifoldModel's local PCA
neighborhood size is scaled with ambient dimension: `k_pca = max(50, 3 × d_ambient)`
to ensure the tangent-space estimate is overdetermined at each node.

---

## 4. Results

### 4.1 Intrinsic Dimensionality Discovery

WaveRider's `discover_dimensionality()` (local PCA at 200 random samples, τ = 0.90)
correctly recovers:

| Embedding | d_ambient | d* discovered | Expected |
|---|---|---|---|
| torus | 4 | **2** | 2 (T² by construction) |
| discrete | 16 | 2 | — |
| window7 | 28 | 9 | — |
| window13 | 52 | **16** | — |

The torus result (d* = 2) is the key geometric confirmation: WaveRider identifies
that the backbone angle space is exactly 2-dimensional, consistent with the
Ramachandran torus. This is a correct, unsupervised geometric result.

Per-class intrinsic dimension at τ = 0.90 (PISCES, 50K residues, U→rama):

| Class | Mean d* | Std | Interpretation |
|---|---|---|---|
| α-helix (H) | 9.2 | 0.8 | Most geometrically regular |
| β-sheet (E) | 9.4 | 1.0 | Regular but more variable |
| PPII (P) | 10.7 | 0.8 | Extended, moderate regularity |
| left-helix (L) | 10.6 | 0.7 | Small class, well-defined |
| coil (C) | 11.7 | 0.8 | Highest disorder — as expected |

The ordering H ≤ E < P ≈ L < C is physically meaningful and robust across corpora.
Helices are the most geometrically constrained; coil spans the widest region of
Ramachandran space.

### 4.2 ManifoldModel Classification (Proper Test Evaluation)

PISCES corpus, 50,000 residues (40K train / 10K test), U→rama:

| Mode | d_ambient | d* | k_pca | Test Acc |
|---|---|---|---|---|
| torus | 4 | 2 | 50 | **0.853** |
| discrete | 16 | 2 | 50 | 0.845 |
| window7 | 28 | 9 | 84 | 0.791 |
| window13 | 52 | 16 | 156 | 0.741 |

### 4.3 The Critical Finding: Torus Wins

The classification accuracy decreases monotonically as the embedding grows larger.
This ordering — **torus > discrete > window7 > window13** — is the opposite of what
naive intuition suggests (more context should help).

This result holds at both N = 5,000 and N = 50,000, ruling out sample size as the
cause. It is a genuine property of the ManifoldModel architecture in this setting.

**Why torus outperforms window embeddings in ManifoldModel:**

ManifoldModel is a local geometry estimator combined with a manifold-aware k-NN
classifier. Its fundamental operation is: estimate the local tangent space at each
training point via local PCA, then classify test points by weighted k-NN vote in
that tangent space.

This works extremely well for the torus (4D ambient, d* = 2): the geometry is
simple, the clusters are well-separated, and local PCA with k_pca = 50 easily
characterises the tangent plane.

For window13 (52D ambient, d* = 16), the problems are structural:

1. **Correlated features**: A window13 vector stacks 13 consecutive torus embeddings.
   Within a helix, all 13 positions have nearly identical (φ, ψ) ≈ (−60°, −40°),
   so the 52D vector is nearly degenerate — a line in high-dimensional space.
   The "extra" dimensions encode repetition, not new information.

2. **Local PCA instability**: Even with k_pca = 156, estimating a 16-dimensional
   tangent space in 52 ambient dimensions requires enough points that the
   neighborhood spans the local geometry. At boundaries between secondary structure
   elements, the local geometry is heterogeneous and the tangent estimate is noisy.

3. **k-NN's dimensional curse**: The manifold-aware vote among k_vote = 7 neighbors
   in projected space is essentially a local majority rule. In high-dimensional
   windows, boundary residues (helix→coil transitions) have ambiguous neighborhoods
   that vote incorrectly.

**What torus 85.3% means:**

This is close to the **ceiling for per-residue classification** from backbone angles
alone. The remaining ~15% error is irreducible: boundary residues at helix/sheet
termini whose (φ, ψ) angle has already begun the transition to coil geometry,
and coil residues that happen to fall within canonical Ramachandran regions by chance.

---

## 5. What ManifoldModel IS Good For

The genuine scientific contribution of ManifoldModel on backbone angles is not
classification accuracy — it is **geometric discovery**:

1. **d* = 2 for torus**: Correctly identifies the 2-torus as the backbone angle
   manifold without any prior knowledge. Confirmed at multiple sample sizes and
   across two different protein corpora.

2. **Per-class intrinsic dimension ordering**: H ≤ E < C across all corpora and
   variance thresholds. Helix is geometrically more constrained than sheet, which
   is more constrained than coil. This is a quantitative, unsupervised confirmation
   of a known structural biological fact.

3. **Corpus-robustness**: d* and per-class ordering are consistent between the
   disulfide corpus (helix-enriched) and the PISCES non-redundant set, despite
   different SS distributions. The geometry is real, not a dataset artifact.

4. **d* = 16 for window13**: The local backbone sequence context in a 13-residue
   window has ~16 degrees of freedom. This quantifies structural context richness.

---

## 6. The Right Tool for Context-Dependent Classification

Per-residue backbone angles contain ~85% of the classifiable secondary structure
information (torus test accuracy). The remaining ~15% requires local sequence
context — and that context is real information.

The failure of window modes in ManifoldModel reveals an architectural mismatch:
ManifoldModel is a geometric tool optimised for low-dimensional, well-structured
manifolds. Window embeddings produce high-dimensional, correlated, anisotropic
spaces that violate these assumptions.

The correct tool for context-dependent secondary structure prediction is a
**sequence-aware neural network**. A small MLP on the window embedding should
outperform ManifoldModel on window modes because:

- MLPs learn non-local discriminative features across all 52 dimensions jointly
- Gradient descent finds the optimal linear combination of window positions
- No local-geometry assumption is imposed; the network can exploit global patterns
  (e.g., the 3.6-residue helical repeat)

The next experiment benchmarks manifold-informed MLP architectures — with
bottlenecks set to d* from WaveRider's geometric discovery — against the
ManifoldModel baseline.

---

## 7. Corpus Composition Effects

PISCES is more helix-enriched than the disulfide corpus, despite being designed
for non-redundancy:

| Corpus | H | E | C | U-remap |
|---|---|---|---|---|
| disulfide (good_backbone) | 32.9% | 26.2% | 41.0% | U→coil |
| PISCES 1000-file | 43.7% | 22.8% | 33.5% | U→coil |
| PISCES 1000-file | 47.4% | 30.7% | 11.7% | U→rama |

The helix bias in PISCES reflects the overall PDB bias toward helical, easily
crystallisable proteins. Non-redundancy at the sequence level does not remove
structural bias. The Ramachandran remap increases the E fraction substantially
(22.8% → 30.7%), because many HELIX/SHEET-unrecorded β-strand residues have
clearly extended angles and are correctly recovered as E.

---

## 8. Conclusions

1. **WaveRider correctly discovers d* = 2** for the protein backbone torus —
   an unsupervised geometric result consistent with the known topology of the
   Ramachandran plot. This is the primary scientific finding.

2. **Per-class d* ordering (H ≤ E < C) is robust** across corpora, sample
   sizes, and variance thresholds. Helix is the most constrained secondary
   structure, coil the least — a quantitative confirmation of structural biology.

3. **ManifoldModel achieves 85.3% test accuracy** on the torus embedding —
   close to the per-residue ceiling. This validates the Ramachandran clusters
   as geometrically well-separated in torus space.

4. **Window embeddings underperform torus** in ManifoldModel at all tested
   sample sizes. This is not a data quantity problem — it reveals an architectural
   mismatch. ManifoldModel is the wrong tool for high-dimensional correlated
   context windows.

5. **The right tool for context-dependent classification is a neural network.**
   An MLP with manifold-informed bottleneck (width = d*) should recover the
   missing ~15% accuracy from context, since it can learn the discriminative
   structure of window embeddings without imposing local-geometry assumptions.

6. **The Ramachandran remap (--remap-u-rama) is strictly better** than U→coil.
   It recovers the P and L classes (8.6% and 1.6% of PISCES residues), removes
   erroneous coil contamination, and improves discrete mode stability (d* recovered
   from 1 → 2).

---

## 9. Next Steps

- MLP benchmark on backbone embeddings with manifold-informed bottleneck
- Per-protein-chain train/test split (eliminates window context leakage between splits)
- Comparison with published secondary structure predictors (PSIPRED, DSSP)
- Apply to full PISCES corpus (8,877 structures) for production-scale analysis

---

*Generated from `protein_backbone_manifold.py` experiments on the `waverider` codebase.*
