[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](https://github.com/Flux-Frontiers/waverider/releases)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

# WaveRider

**Discovering the Intrinsic Dimensionality of Loss Landscapes Through Manifold-Aware Optimization**

*Eric G. Suchanek, PhD — Flux-Frontiers*

[Technical Paper (PDF)](docs/waverider/waverider.md)

---

## Overview

WaveRider is a family of **manifold-aware geometric ML algorithms** that discover the intrinsic dimensionality of data and loss landscapes through local PCA of gradient-diversity samples, then *build models directly from that discovered geometry*.

The central finding: the spaces in which machine learning operates are vastly lower-dimensional than their ambient representations suggest — and knowing this is enough to build models that match or beat systems orders of magnitude larger.

Standard algorithms treat their operating spaces as **isotropic** — every dimension gets equal treatment, even those that are noise. For a space with ambient dimension P and intrinsic dimension d, this means P − d dimensions of wasted computation. Those noise dimensions actively degrade performance by inflating distances, diluting momentum, and introducing off-manifold drift.

WaveRider measures the manifold geometry, designs architectures from it, builds knowledge graphs with it, and navigates through it — from measurement instrument to model builder to interactive explorer.

---

## Key Results

| Dataset | Ambient Dim | Intrinsic Dim | Noise Suppressed | Result |
|---------|-------------|---------------|-----------------|--------|
| **CIFAR-10** | 3,072 | 29 | 99.1% | 3,751-param model beats 820,874-param standard (48.58% vs 48.39%) — **219× reduction** |
| **MNIST** | 784 | 22 | 97.2% | 2,232 params → 95.5% accuracy (standard: 109,386 params → 97.4%) |
| **Digits** | 64 | 11–18 | 71–83% | ManifoldKNN: **97.72%** vs Euclidean KNN 97.33% — zero learned parameters |
| **Iris loss landscape** | 243 (params) | 2–3 | 98.9% | Gradient-diversity PCA exposes that Adam maintains 486 state variables when ~5 do useful work |

---

## Algorithms

| Component | Class | Description |
|-----------|-------|-------------|
| **TurtleND** | `TurtleND` | N-dimensional position + orthonormal frame (navigation primitive) |
| **Manifold Walker** | `ManifoldWalker` | Riemannian gradient descent in discovered tangent space |
| **Manifold Adam** | `ManifoldAdamWalker` | Adam momentum in tangent space — state preserved across re-orientations |
| **Manifold Model** | `ManifoldModel` | Zero-parameter classifier: the manifold *is* the model |
| **Manifold Observer** | `ManifoldObserver` | (N+1)-dimensional extrinsic observer — hovers above the manifold surface |

---

## Method

### Gradient-Diversity PCA

At a point **w** in weight space R^P, mini-batch gradients on S random data subsets are gathered and decomposed:

```
G = [g₁ - ḡ, ..., gₛ - ḡ]ᵀ ∈ R^{S×P}
C = GᵀG / (S-1) = VΛVᵀ
```

The top *d* eigenvectors V_d span the **gradient's active subspace** — the tangent space of the loss manifold. The remaining P − d eigenvectors point into noise.

### Manifold-Projected Step

```
1. Project:   ℓ = Vdᵀ g  (local coords)
              g_proj = Vd ℓ  (back to global, off-manifold zeroed)
2. Adam update on g_proj  (momentum accumulates signal, never noise)
3. Step:      w ← w − η Δw
```

Adam state lives in global R^P — momentum persists across manifold re-orientations without losing memory when the PCA basis rotates.

### Manifold KNN

Rather than measuring Euclidean distance in the full ambient space, ManifoldKNN first projects query and neighbors into the local *d*-dimensional tangent space, then votes in that denoised subspace. The improvement on digits (97.72% vs 97.33%) comes entirely from geometry — no training, no learned weights.

---

## Quick Start

```bash
git clone https://github.com/Flux-Frontiers/waverider.git
cd waverider
poetry install
```

For interactive 3-D voxel visualization (PyVista + SciPy):

```bash
poetry install --with viz
```

For neural network benchmarks (TensorFlow + Metal GPU on Apple Silicon):

```bash
poetry install --with benchmarks
```

Full install (viz + benchmarks):

```bash
poetry install --with viz,benchmarks
```

---

## Installation

**Requirements:** Python 3.10

### From source

```bash
pip install git+https://github.com/Flux-Frontiers/waverider.git
```

### Poetry (recommended)

```bash
poetry add git+https://github.com/Flux-Frontiers/waverider.git
```

---

## Usage

### ManifoldKNN — geometry-aware classification

```python
from waverider import ManifoldModel

model = ManifoldModel(k_pca=20, tau=0.85)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Zero learned parameters — the manifold geometry is the classifier
```

### ManifoldWalker — Riemannian gradient descent

```python
from waverider import ManifoldWalker

walker = ManifoldWalker(k_samples=30, tau=0.90, reorient_every=10)
walker.fit(X, y, epochs=100, lr=0.01)
```

### ManifoldAdamWalker — manifold-projected Adam

```python
from waverider import ManifoldAdamWalker

walker = ManifoldAdamWalker(
    k_samples=30, tau=0.90,
    beta1=0.9, beta2=0.999, lr=0.001,
    reorient_every=10
)
walker.fit(X, y, epochs=200)
```

### TurtleND — N-dimensional frame navigation

```python
from waverider import TurtleND

turtle = TurtleND(dim=10)
turtle.forward(0.1)          # step along heading
turtle.turn(axis=1, angle=0.3)   # rotate frame
print(turtle.position, turtle.frame)
```

### ManifoldObserver — extrinsic manifold sensor

```python
from waverider import ManifoldModel, ManifoldObserver

subject = ManifoldModel(k_graph=10, k_pca=20, k_vote=7, variance_threshold=0.90)
subject.fit(X, y)
observer = ManifoldObserver(subject)
observer.lift_data()
field = observer.observe()   # list of ObservationResult (curvature, height, intrinsic_dim, …)
```

### Manifold Voxel Visualizer — interactive 3-D manifold anatomy

Requires `poetry install --with viz`.

```python
from waverider import fit_and_observe, voxelize, build_grid, render_single

subject, observer, pf, pca_info = fit_and_observe(
    X, y, k_graph=10, k_pca=20, k_vote=7, tau=0.90
)
vox  = voxelize(pf, resolution=32)
grid = build_grid(vox)
render_single(grid, pf, scalar="density", pca_info=pca_info)
```

Or use the installed CLI command:

```bash
# Interactive viewer — synthetic helix (default)
waverider-voxel-viz

# Iris dataset, 2×2 panel of all scalar fields
waverider-voxel-viz --dataset iris --multi-scalar

# Headless PNG export
waverider-voxel-viz --dataset breast_cancer --off-screen --out bc_voxels.png

# CIFAR-10 — subsample 1 000 pts, pre-reduce to 40-D
waverider-voxel-viz --dataset cifar10 --n-points 1000 --pre-pca 40
```

See [docs/waverider/manifold_voxel_viz.md](docs/waverider/manifold_voxel_viz.md) for the full
argument reference and programmatic API.

---

## Benchmarks

```bash
# Run the full benchmark suite
poetry run pytest benchmarks/ -v

# Individual experiments
python benchmarks/digits_manifold.py      # Experiment 1: Digits dataset
python benchmarks/iris_loss_landscape.py  # Experiment 2: Iris MLP
python benchmarks/mnist_architecture.py   # Experiment 3: MNIST manifold-informed MLP
python benchmarks/cifar10_architecture.py # Experiment 4: CIFAR-10 efficiency frontier
python benchmarks/cifar100_architecture.py # Experiment 5: CIFAR-100 efficiency frontier
```

### Reproducing TurtleND paper results

The numerical validation tables in the TurtleND paper (`docs/turtlend/turtlend.tex`)
are produced by two deterministic, seed-locked benchmark scripts in
`benchmarks/canonical_tests/`:

| Script | Manifold | Paper table | Locked numbers |
|---|---|---|---|
| `helix_manifold_observer.py` | Synthetic 1-manifold helix in $\mathbb{R}^5$ | Table 1 | `helix_manifold_observer_results.json` |
| `torus_manifold_observer.py` | Synthetic 2-manifold flat torus in $\mathbb{R}^4$ | Table 2 | `torus_manifold_observer_results.json` |

To reproduce:

```bash
poetry run python benchmarks/canonical_tests/helix_manifold_observer.py
poetry run python benchmarks/canonical_tests/torus_manifold_observer.py
```

Both scripts use seeds 42–51 (10 trials) and write a JSON file alongside
themselves. The committed JSONs are the locked numbers cited in the paper —
running the scripts on a clean checkout should reproduce them bit-for-bit on
the same NumPy / SciPy versions.

---

## Project Structure

```
waverider/
├── pyproject.toml
├── README.md
├── docs/
│   └── waverider/
│       ├── waverider.md              # Technical paper
│       └── manifold_voxel_viz.md     # Voxel visualizer reference
├── src/
│   └── waverider/
│       ├── __init__.py
│       ├── turtleND.py               # N-dim position + orthonormal frame
│       ├── turtle3D.py               # 3D specialization
│       ├── manifold_walker.py        # Riemannian gradient descent
│       ├── manifold_observer.py      # (N+1)-dim extrinsic observer
│       ├── manifold_model.py         # Zero-parameter manifold classifier
│       └── voxel_viz.py              # 3-D voxel visualizer + CLI
├── tests/
├── benchmarks/
```

---

## Theoretical Background

### Why Projection Before Measurement Matters

Operating in the ambient space conflates signal with noise:

- **KNN without projection**: distances inflated by noise dimensions — true neighbors appear farther, non-neighbors appear closer
- **Adam without projection**: momentum accumulates noise, adaptive denominator tracks noise variance, learning rates adapt to the wrong signals

Projecting onto the tangent space is analogous to denoising a signal before feeding it to a filter. The filter then adapts to the real signal.

### Relationship to Natural Gradient

The eigenvalue weighting λᵢ/λ₁ is a form of natural gradient descent using the data covariance as an empirical Fisher information matrix (Amari, 1998). The manifold projection ensures both natural gradient and Adam operate on the right dimensions.

### Noise Suppression as Regularization

By zeroing off-manifold gradient components, WaveRider implicitly regularizes optimization. The model is constrained to move along the data manifold, preventing drift into off-manifold regions that correspond to overfitting — geometrically motivated and data-adaptive, unlike dropout or weight decay.

---

## References

- Bengio, Y. et al. (2013). *Representation Learning: A Review and New Perspectives.* TPAMI.
- Gur-Ari, G. et al. (2018). *Gradient Descent Happens in a Tiny Subspace.* arXiv:1812.04754.
- Ghorbani, B. et al. (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML.
- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning.* Neural Computation.
- Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.

---

## License

[Elastic License 2.0 (ELv2)](https://www.elastic.co/licensing/elastic-license) — see [LICENSE](LICENSE).

Free to use, modify, and distribute. May not be offered as a hosted or managed service to third parties.
