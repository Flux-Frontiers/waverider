# Manifold-Informed Architecture Benchmark — DIGITS

**Generated:** 2026-04-14 20:54:31
**Machine:** Apple M5 Max MacBook Pro, 64 GB RAM, 2TB SSD
**Repository:** waverider @ `4b8002e` (--abbrev-re
4b8002ee9a2e3d56a219d7dab695a80b8efd1e07)
**Commit:** 2026-04-14 20:51:52 -0400 — add: cifar10 results
**Python:** 3.12.13  |  **TensorFlow:** 2.16.2  |  **Device:** CPU
**Host:** Turing  |  **OS:** macOS-26.4-arm64-arm-64bit

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | DIGITS |
| Input dimensionality | 64 |
| Classes | 10 |
| Intrinsic dim (d) | 14 |
| Variance threshold (τ) | 0.9 |
| Epochs | 50 |
| Trials | 3 |
| Batch size | 64 |
| Learning rate | 0.001 |

## Manifold Discovery

Local PCA over the training set, k=30 neighbors.

| τ | Mean d | Std | Min | Max | Noise % |
|---|---|---|---|---|---|
| 0.95 | 14.3 | 1.7 | 8 | 18 | 77.6% |
| 0.90 | 10.9 | 1.5 | 5 | 14 | 83.0% |
| 0.85 | 8.8 | 1.4 | 4 | 11 | 86.3% |
| 0.80 | 7.3 | 1.3 | 3 | 10 | 88.6% |

### Per-Class Intrinsic Dimensionality

| Class | Mean d | Std | Min | Max |
|---|---|---|---|---|
| Digit 0 | 12.6 | 0.6 | 11 | 14 |
| Digit 8 | 12.5 | 1.0 | 7 | 13 |
| Digit 9 | 11.8 | 0.7 | 10 | 13 |
| Digit 5 | 11.6 | 0.9 | 10 | 13 |
| Digit 3 | 11.5 | 0.7 | 10 | 13 |
| Digit 6 | 10.7 | 1.3 | 4 | 12 |
| Digit 7 | 10.6 | 1.1 | 6 | 12 |
| Digit 2 | 10.2 | 0.9 | 8 | 11 |
| Digit 4 | 9.4 | 1.2 | 5 | 11 |
| Digit 1 | 8.8 | 1.7 | 5 | 11 |

## Architecture Comparison

| Architecture | Params | Test Acc (mean ± std) | Test Loss | Acc/Kparam |
|---|---|---|---|---|
| Euclidean KNN (k=7) | 0 | 0.9733 ± 0.0054 | N/A | N/A |
| ManifoldModel (τ=0.9) | 0 | 0.9727 ± 0.0054 | N/A | N/A |
| Standard (128→64) | 17,226 | 0.9787 ± 0.0051 | 0.0825 | 0.0568 |
| Wide Manifold (d→d+1→d, d=14) | 1,509 | 0.9521 ± 0.0061 | 0.1780 | 0.6310 |
| Manifold (2d→d, d=14) | 2,376 | 0.9668 ± 0.0079 | 0.1227 | 0.4069 |
| PCA→14D + MLP (2d→d) | 976 | 0.9375 ± 0.0116 | 0.2006 | 0.9605 |
| Intrinsic Dim (PCA→14D→output) | 360 | 0.9178 ± 0.0108 | 0.2837 | 2.5495 |

## Key Findings

- **Best architecture:** Standard (128→64)
  — test accuracy 0.9787 ± 0.0051
- **Manifold compression:** 64D → 14D (78.1% of ambient dimensions are noise)

## Result Figure

![DIGITS Results](digits_manifold_architecture_results.png)
