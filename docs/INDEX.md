# WaveRider — Documentation Index

*Eric G. Suchanek, PhD — Flux-Frontiers*

---

## Getting Started

| Resource | Description |
|----------|-------------|
| [README](../README.md) | Overview, headline results, quick start |
| [USAGE.md](USAGE.md) | Code examples for all components |
| [Quick Start](../README.md#quick-start) | Install in 3 commands |

---

## Technical Documentation

| Document | Description |
|----------|-------------|
| [WaveRider Technical Paper](waverider/waverider.md) | Full algorithmic description, proofs, and results |
| [WaveRider Stack Summary](waverider/waverider_stack_summary.md) | One-page component overview |
| [WaveRider Infographic](waverider/waverider_infographic_summary.md) | Visual summary of the stack |
| [Manifold Voxel Visualizer](waverider/manifold_voxel_viz.md) | Full CLI and API reference for 3-D visualization |
| [ManifoldObserver Spec](manifold_observer/manifold_observer.md) | (N+1)-dimensional extrinsic observer design |
| [ManifoldWalker Spec](manifold_walker_spec/manifold_walker_spec.md) | Riemannian gradient descent specification |

---

## Papers

| Paper | Format | Description |
|-------|--------|-------------|
| [Manifold Classification](../papers/manifold_classification/manifold_classification.pdf) | PDF | ManifoldModel: zero-parameter geometry classifiers across 9 datasets |
| [Clinical Manifolds](../papers/clinical_manifolds/clinical_manifolds.pdf) | PDF | Manifold geometry applied to 5 clinical disease datasets |
| [KAN vs Manifold](../papers/clinical_manifolds/kan_clinical.pdf) | PDF | KAN comparison on clinical datasets |
| [KAN vs Manifold (article)](../papers/clinical_manifolds/kan_manifold_article.md) | Markdown | Narrative article version of the KAN comparison |
| [Voxel Visualizer](../papers/voxel_viz/voxel_viz.pdf) | PDF | 3-D manifold anatomy visualization paper |

---

## Canonical Benchmark Reports

All benchmarks run on Apple M5 Max MacBook Pro, 64 GB RAM, macOS 26.4.
Results are seed-locked (seeds 42–51, 3–10 trials). JSON files are the authoritative locked numbers.

### Standard Datasets

| Dataset | Ambient | Intrinsic d | Noise | Report | PDF |
|---------|---------|-------------|-------|--------|-----|
| **CIFAR-10** | 3,072 | 33 | 99.1% | [cifar10_report.md](../benchmarks/canonical_tests/cifar10_report.md) | [PDF](../benchmarks/canonical_tests/cifar10_report.pdf) |
| **CIFAR-100** | 3,072 | 19 | 99.4% | [cifar100_report.md](../benchmarks/canonical_tests/cifar100_report.md) | [PDF](../benchmarks/canonical_tests/cifar100_report.pdf) |
| **MNIST** | 784 | 27 | 96.6% | [mnist_report.md](../benchmarks/canonical_tests/mnist_report.md) | [PDF](../benchmarks/canonical_tests/mnist_report.pdf) |
| **Tiny ImageNet** | 12,288 | 20 | 99.9% | [tiny_imagenet_report.md](../benchmarks/canonical_tests/tiny_imagenet_report.md) | [PDF](../benchmarks/canonical_tests/tiny_imagenet_report.pdf) |
| **Digits** | 64 | 14 | 78.1% | [digits_report.md](../benchmarks/canonical_tests/digits_report.md) | [PDF](../benchmarks/canonical_tests/digits_report.pdf) |
| **Iris** | 4 | 3 | 25.0% | [iris_report.md](../benchmarks/canonical_tests/iris_report.md) | [PDF](../benchmarks/canonical_tests/iris_report.pdf) |

### Clinical Datasets

| Dataset | Ambient | Intrinsic d | Noise | Report |
|---------|---------|-------------|-------|--------|
| **Heart Disease** | 13 | 9 | 30.8% | [heart_report.md](../benchmarks/canonical_tests/clinical/heart_report.md) |
| **Breast Cancer** | 30 | 9 | 70.0% | [breast_cancer_report.md](../benchmarks/canonical_tests/clinical/breast_cancer_report.md) |
| **Parkinson's** | 22 | 7 | 68.2% | [parkinsons_report.md](../benchmarks/canonical_tests/clinical/parkinsons_report.md) |
| **Dermatology** | 34 | 13 | 61.8% | [dermatology_report.md](../benchmarks/canonical_tests/clinical/dermatology_report.md) |
| **Alzheimer's** | 8 | 6 | 25.0% | [alzheimers_report.md](../benchmarks/canonical_tests/clinical/alzheimers_report.md) |

---

## Reproducing Results

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

# Canonical geometry measurements (helix + torus)
python benchmarks/canonical_tests/helix_manifold_observer.py
python benchmarks/canonical_tests/torus_manifold_observer.py

# Universal Bottleneck phase boundary (MNIST / Fashion-MNIST)
python benchmarks/canonical_tests/mnist_ub_phase_boundary.py
```
