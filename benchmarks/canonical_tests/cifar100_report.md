# Manifold-Informed Architecture Benchmark — CIFAR100

**Generated:** 2026-04-14 20:54:33
**Machine:** Apple M5 Max MacBook Pro, 64 GB RAM, 2TB SSD
**Repository:** waverider @ `4b8002e` (--abbrev-re
4b8002ee9a2e3d56a219d7dab695a80b8efd1e07)
**Commit:** 2026-04-14 20:51:52 -0400 — add: cifar10 results
**Python:** 3.12.13  |  **TensorFlow:** 2.16.2  |  **Device:** Metal GPU (/physical_device:GPU:0)
**Host:** Turing  |  **OS:** macOS-26.4-arm64-arm-64bit

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR100 |
| Input dimensionality | 3,072 |
| Classes | 100 |
| Intrinsic dim (d) | 19 |
| Variance threshold (τ) | 0.9 |
| Epochs | 30 |
| Trials | 3 |

## Manifold Discovery

Local PCA over the training set, k=not recorded neighbors.

| τ | Mean d | Std | Min | Max | Noise % |
|---|---|---|---|---|---|
| 0.95 | 18.9 | 0.9 | 15 | 21 | 99.4% |
| 0.90 | 15.7 | 1.1 | 11 | 18 | 99.5% |
| 0.85 | 13.3 | 1.1 | 9 | 16 | 99.6% |
| 0.80 | 11.5 | 1.1 | 8 | 14 | 99.6% |

### Per-Class Intrinsic Dimensionality

*Showing 10 hardest + 10 easiest classes (sorted by mean d)*

| Class | Mean d | Std | Min | Max |
|---|---|---|---|---|
| 51 | 18.0 | 0.0 | 18 | 18 |
| 81 | 18.0 | 0.6 | 17 | 19 |
| 48 | 17.8 | 0.4 | 17 | 18 |
| 13 | 17.6 | 0.5 | 17 | 18 |
| 14 | 17.6 | 0.5 | 17 | 18 |
| 66 | 17.6 | 1.0 | 16 | 19 |
| 6 | 17.4 | 0.5 | 17 | 18 |
| 37 | 17.4 | 0.5 | 17 | 18 |
| 58 | 17.4 | 0.5 | 17 | 18 |
| 43 | 17.2 | 0.4 | 17 | 18 |
| … | … | … | … | … |
| 94 | 14.2 | 1.0 | 13 | 15 |
| 61 | 14.0 | 0.9 | 13 | 15 |
| 9 | 13.8 | 0.4 | 13 | 14 |
| 24 | 13.8 | 0.7 | 13 | 15 |
| 73 | 13.4 | 0.8 | 13 | 15 |
| 67 | 13.0 | 0.9 | 12 | 14 |
| 69 | 13.0 | 0.9 | 12 | 14 |
| 23 | 12.4 | 0.5 | 12 | 13 |
| 60 | 12.4 | 0.5 | 12 | 13 |
| 71 | 11.0 | 0.9 | 10 | 12 |

## Architecture Comparison

| Architecture | Params | Test Acc (mean ± std) | Test Loss | Acc/Kparam |
|---|---|---|---|---|
| Standard (1024→512) | 3,722,852 | 0.0521 ± 0.0048 | 33499.9707 | 0.0000 |
| Manifold (2d→d, d=19) | 317,400 | 0.0833 ± 0.0043 | 12.7129 | 0.0003 |
| Manifold + ManifoldAdam (d=19) | 317,400 | 0.0614 ± 0.0039 | 11.9235 | 0.0002 |
| ResNet (Adam) | 50,948 | 0.3755 ± 0.0089 | 2.4784 | 0.0074 |
| ManifoldResNet-d (d=19) | 19,176 | 0.3116 ± 0.0068 | 2.6885 | 0.0162 |
| ManifoldResNet-d+C (d=19) | 29,276 | 0.3051 ± 0.0058 | 2.8108 | 0.0104 |
| PCA→d*→C→C (d=19) | 12,100 | 0.0966 ± 0.0048 | 4.3031 | 0.0080 |
| ManifoldResNet-2d (2d=38) | 70,742 | 0.3719 ± 0.0195 | 2.5369 | 0.0053 |
| PCA(100) Whitney(2d=38)→100 | 7,738 | 0.1502 ± 0.0019 | 3.8427 | 0.0194 |
| Intrinsic Dim (PCA→19D→output) | 2,380 | 0.1212 ± 0.0052 | 3.9027 | 0.0509 |
| ManifoldResNet-UB (w*=118) ✦ | 644,262 | 0.3829 ± 0.0377 | 4.1482 | 0.0006 |
| UB-PCA-MLP (→119→PCA→119→100) | 391,967 | 0.1309 ± 0.0021 | 4.0014 | 0.0003 |

## Key Findings

- **Best architecture:** ManifoldResNet-UB (w*=118)
  — test accuracy 0.3829 ± 0.0377
- **vs Standard:** +0.3308 (33.08 pp) accuracy gain
- **Parameter reduction:** 5.8× fewer parameters (644,262 vs 3,722,852)
- **Parameter efficiency:** 0.0006 acc/Kparam vs 0.0000 for Standard (42.5× improvement)
- **Manifold compression:** 3,072D → 19D (99.4% of ambient dimensions are noise)

## Result Figure

![CIFAR100 Results](cifar100_architecture_results.png)
