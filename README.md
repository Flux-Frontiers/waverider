[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.7.1-blue.svg)](https://github.com/Flux-Frontiers/waverider/releases)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

# WaveRider

**The geometry of your data tells you the exact size of network you need. Most of what your model is computing is noise.**

*Eric G. Suchanek, PhD — Flux-Frontiers*

[Technical Paper (PDF)](docs/waverider/waverider.md)

---

## The Core Finding

Machine learning spaces are **99% noise** by dimension. CIFAR-10 images live in a 33-dimensional manifold inside a 3,072-dimensional ambient space. Tiny ImageNet: 20 intrinsic dimensions inside 12,288. Standard algorithms treat every dimension equally — spending 99%+ of their compute on dimensions that carry no signal, while momentum, distance metrics, and gradient updates are polluted by that noise.

WaveRider measures the actual geometry, builds models constrained to the signal manifold, and derives a closed-form formula for optimal network width from first principles:

> **w\* = d\* + C − 1**

Measure the intrinsic dimensionality d\*. Count the classes C. That's your optimal bottleneck width. No grid search. No hyperparameter sweep.

---

## Headline Results

→ **[Full documentation and all benchmark reports](docs/INDEX.md)**

### Universal Bottleneck — formula-derived architectures beat ResNet

| Dataset | d\* | C | w\* = d\*+C−1 | ManifoldResNet-UB | Accuracy | vs ResNet-32 | Δ |
|---------|-----|---|--------------|-------------------|----------|-------------|---|
| [**CIFAR-10**](benchmarks/canonical_tests/cifar10_report.md) | 19 | 10 | 28 | 36,942 params | **71.8% ± 0.5%** | 47,978 params → 63.3% ± 2.7% | **+8.5 pp, fewer params** |
| [**Fashion-MNIST**](benchmarks/canonical_tests/mnist_report.md) | 18 | 10 | 27 | 33,868 params | **88.38% ± 0.32%** | 47,338 params → 82.85% ± 2.25% | **+5.5 pp, fewer params** |
| [**MNIST**](benchmarks/canonical_tests/mnist_report.md) | 16 | 10 | 25 | 29,110 params | **98.98% ± 0.18%** | 47,338 params → 99.27% ± 0.12% | within 0.3 pp, fewer params |
| [**CIFAR-100**](benchmarks/canonical_tests/cifar100_report.md) | 19 | 100 | 118 | 644,262 params | **38.3% ± 3.8%** | 50,948 params → 37.6% ± 0.9% | +0.7 pp |

### Zero-parameter classifiers — the manifold is the model

| Dataset | ManifoldModel (0 params) | Best trained | Δ |
|---------|--------------------------|-------------|---|
| [**Heart Disease**](benchmarks/canonical_tests/clinical/heart_report.md) | **83.82% ± 2.47%** | Standard MLP: 80.96% ± 2.91% (7,022 params) | **+2.86 pp with zero parameters** |
| [**Parkinson's**](benchmarks/canonical_tests/clinical/parkinsons_report.md) | **90.77% ± 2.61%** | Standard MLP: 93.33% ± 3.08% (19,802 params) | −2.56 pp vs MLP; beats KNN (89.74%) |
| [**Breast Cancer**](benchmarks/canonical_tests/clinical/breast_cancer_report.md) | 96.31% ± 1.61% | Standard MLP: 97.31% ± 1.41% (36,602 params) | within 1 pp, zero params |
| [**Dermatology**](benchmarks/canonical_tests/clinical/dermatology_report.md) | 95.90% ± 1.51% | Standard MLP: 96.71% ± 2.05% (42,630 params) | within 0.8 pp, zero params |

### Parameter efficiency — noise suppression across datasets

| Dataset | Ambient Dim | Intrinsic d | Noise | Standard baseline | Manifold result | Param reduction |
|---------|-------------|-------------|-------|-------------------|-----------------|-----------------|
| [**Tiny ImageNet**](benchmarks/canonical_tests/tiny_imagenet_report.md) | 12,288 | 20 | 99.9% | 2.66% @ 13.2M params | **3.36% @ 80,400 params** | **164×** |
| [**CIFAR-100**](benchmarks/canonical_tests/cifar100_report.md) | 3,072 | 19 | 99.4% | 5.21% @ 3.7M params | **38.3% @ 644K params** | 5.8× + 7× better acc |
| [**CIFAR-10**](benchmarks/canonical_tests/cifar10_report.md) | 3,072 | 34 | 99.1% | 51.67% @ 3.7M params | 49.12% @ 5,076 params | **724×** at −2.6 pp |
| [**MNIST**](benchmarks/canonical_tests/mnist_report.md) | 784 | 27 | 96.6% | 97.42% @ 109,386 params | 95.11% @ 1,036 params | **105×** at −2.3 pp |

---

## The Dimension Probe

When a network is given a bottleneck of exactly w\* = d\* + C − 1 neurons, it **spontaneously decomposes** that space into exactly d\* geometry dimensions and C−1 class-separation dimensions — with zero instruction.

On CIFAR-10 (d\*=16, C=10, w\*=25): post-bottleneck PCA reveals 7 geometry principal components explaining 90% of variance (Whitney bound), and exactly 9 additional class-separation components — precisely C−1=9. The semantic content is interpretable: PC11 encodes four-legged animals, PC9 encodes flat objects, PC12 encodes wheeled vehicles.

**Gradient descent independently discovers the theorem's decomposition.**

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

**Requirements:** Python 3.12

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

See **[docs/USAGE.md](docs/USAGE.md)** for complete code examples covering all components.

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

### Why Projection Matters

Operating in the ambient space conflates signal with noise:

- **KNN without projection**: distances inflated by noise dimensions — true neighbors appear farther, non-neighbors appear closer
- **Adam without projection**: momentum accumulates noise, adaptive denominator tracks noise variance, learning rates adapt to the wrong signals

### Relationship to Natural Gradient

The eigenvalue weighting λᵢ/λ₁ is a form of natural gradient descent using the data covariance as an empirical Fisher information matrix (Amari, 1998). The manifold projection ensures both natural gradient and Adam operate on the right dimensions.

---

## Benchmarks

All benchmark scripts are run directly with Python — no test runner needed.

```bash
# Standard datasets
python benchmarks/canonical_tests/cifar10_manifold_architecture.py
python benchmarks/canonical_tests/cifar100_manifold_architecture.py
python benchmarks/canonical_tests/mnist_manifold_architecture.py
python benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py
python benchmarks/canonical_tests/digits_manifold_architecture.py
python benchmarks/canonical_tests/iris_manifold_architecture.py

# Clinical datasets
python benchmarks/canonical_tests/clinical/disease_manifold_architecture.py

# Canonical geometry measurements
python benchmarks/canonical_tests/helix_manifold_observer.py
python benchmarks/canonical_tests/torus_manifold_observer.py

# Universal Bottleneck phase boundary
python benchmarks/canonical_tests/mnist_ub_phase_boundary.py
```

Seed-locked results (seeds 42–51, 3–10 trials) are committed as JSON alongside each script. The committed JSONs are the locked numbers cited in the papers. See **[docs/INDEX.md](docs/INDEX.md)** for the full benchmark report index.

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
│   └── canonical_tests/
│       ├── clinical/                 # Heart, breast cancer, Parkinson's, etc.
│       └── *.py / *.json / *.md      # Locked benchmark scripts and results
├── papers/
│   ├── waverider_article/
│   ├── clinical_manifolds/
│   ├── manifold_classification/
│   └── voxel_viz/
```

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
