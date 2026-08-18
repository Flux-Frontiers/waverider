# WaveRider — Benchmark Results

*Eric G. Suchanek, PhD — Flux-Frontiers*

The complete headline results tables, with the provenance notes that qualify
them. Every figure is traceable to a results JSON committed beside its
benchmark script; each dataset links its rendered report. See
[INDEX.md](INDEX.md) for the full report index and
[the README](../README.md) for the summary view.

---

## Universal Bottleneck — formula-derived architectures beat ResNet

| Dataset | d\* | C | w\* = d\*+C−1 | ManifoldResNet-UB+Drop | Accuracy | vs ResNet-32 | Δ |
|---------|-----|---|--------------|------------------------|----------|-------------|---|
| [**CIFAR-10**](../benchmarks/canonical_tests/cifar10_report.md) | 19 | 10 | 28 | 36,942 params | **71.83% ± 0.60%** | 47,978 params → 63.26% ± 3.09% | **+8.57 pp, 23% fewer params** |
| [**Fashion-MNIST**](../benchmarks/canonical_tests/mnist_report.md) | 18 | 10 | 27 | 33,868 params | **88.38% ± 0.37%** | 47,338 params → 82.85% ± 2.60% | **+5.53 pp, 28% fewer params** |
| [**MNIST**](../benchmarks/canonical_tests/mnist_report.md) | 16 | 10 | 25 | 29,110 params | **98.98% ± 0.21%** | 47,338 params → 99.27% ± 0.13% | within 0.3 pp, 38% fewer params |
| [**CIFAR-100**](../benchmarks/canonical_tests/cifar100_report.md) | 19 | 100 | 118 | 644,262 params | **38.3% ± 3.8%** | 50,948 params → 37.6% ± 0.9% | +0.7 pp |

*UB+Drop = w\* filters with dropout=0.3; bare UB (no dropout) underperforms — dropout is the regularizer that lets the formula-derived width generalize. See [resnet_manifold_architecture_results.json](../benchmarks/canonical_tests/resnet_manifold_architecture_results.json) (CIFAR-10) and [mnist_ub_phase_boundary_*_results.json](../benchmarks/canonical_tests/) (MNIST/Fashion-MNIST) for raw trial data.*

---

## Zero-parameter classifiers — the manifold is the model

| Dataset | ManifoldModel (0 params) | Best trained | Δ |
|---------|--------------------------|-------------|---|
| [**Heart Disease**](../benchmarks/canonical_tests/clinical/heart_report.md) | **83.82% ± 2.47%** | Standard MLP: 80.96% ± 2.91% (7,022 params) | **+2.86 pp with zero parameters** |
| [**Parkinson's**](../benchmarks/canonical_tests/clinical/parkinsons_report.md) | **90.77% ± 2.61%** | Standard MLP: 93.33% ± 3.08% (19,802 params) | −2.56 pp vs MLP; beats KNN (89.74%) |
| [**Breast Cancer**](../benchmarks/canonical_tests/clinical/breast_cancer_report.md) | 96.31% ± 1.61% | Standard MLP: 97.31% ± 1.41% (36,602 params) | within 1 pp, zero params |
| [**Dermatology**](../benchmarks/canonical_tests/clinical/dermatology_report.md) | 95.90% ± 1.51% | Standard MLP: 96.71% ± 2.05% (42,630 params) | within 0.8 pp, zero params |

---

## Parameter efficiency — noise suppression across datasets

| Dataset | Ambient Dim | Intrinsic d | Noise | Standard baseline | Manifold result | Param reduction |
|---------|-------------|-------------|-------|-------------------|-----------------|-----------------|
| [**Tiny ImageNet**](../benchmarks/canonical_tests/tiny_imagenet_report.md) | 12,288 | 20 | 99.8% | 2.66% @ 13.2M params | **3.36% @ 80,400 params** | **164×** |
| [**CIFAR-100**](../benchmarks/canonical_tests/cifar100_report.md) | 3,072 | 19 | 99.4% | 21.31% @ 3.7M params | **38.3% @ 644K params** | 5.8× + 1.8× better acc |
| [**CIFAR-10**](../benchmarks/canonical_tests/cifar10_report.md) | 3,072 | 34 | 98.9% | 51.67% @ 3.7M params | 49.12% @ 5,076 params | **724×** at −2.6 pp |
| [**MNIST**](../benchmarks/canonical_tests/mnist_report.md) | 784 | 27 | 96.6% | 97.42% @ 109,386 params | 95.11% @ 1,036 params | **105×** at −2.3 pp |

*Mixed estimator settings, retained as-run.* The CIFAR-10 and MNIST rows are
k=50 readings (34 and 27); CIFAR-100 is k=25 (19) and Tiny ImageNet does not
record its k. Only the CIFAR-100 row is at the canonical convention. The
parameter counts and accuracies are unaffected — d enters this table as the
noise fraction, not as a trained width — but the "Intrinsic d" column is not
comparable row-to-row and is not comparable with the Universal Bottleneck table
above. See the provenance note below.

---

## Provenance notes

> **Why d differs between tables — and the canonical convention.** d\* is not a
> property of a dataset. It is the output of local PCA under three choices —
> the neighbourhood size **k**, the cumulative-variance threshold **τ**, and the
> **aggregation** — and it moves with all three. The canonical convention for
> this project is **k=25, τ=0.90, per-class maximum**; that is the default in
> every benchmark script as of the estimator-provenance change, and it is what
> the Universal Bottleneck table above reports. Everything else on this page is
> a sensitivity reading and is labelled as one.
>
> | Dataset | Run | k | τ | Aggregation | d |
> |---|---|---|---|---|---|
> | CIFAR-10 | [`resnet_manifold_architecture_results.json`](../benchmarks/canonical_tests/resnet_manifold_architecture_results.json) — **canonical, UB table** | 25 | 0.90 | per-class max | **19** |
> | CIFAR-10 | same run, other statistic | 25 | 0.90 | global mean | 16 |
> | CIFAR-10 | [`cifar10_architecture_results.json`](../benchmarks/canonical_tests/cifar10_architecture_results.json) — efficiency table | 50 | 0.90 | per-class max | 34 |
> | CIFAR-10 | same run, other statistic | 50 | 0.90 | global mean | 29 |
> | CIFAR-10 | [`manifold_dim_probe_results.json`](../benchmarks/canonical_tests/manifold_dim_probe_results.json) — probe | 25 | 0.90 | global mean | 16 |
> | MNIST | [`mnist_ub_phase_boundary_mnist_results.json`](../benchmarks/canonical_tests/mnist_ub_phase_boundary_mnist_results.json) — canonical, UB table | 25 | 0.90 | per-class max | **16** |
> | MNIST | [`mnist_architecture_results.json`](../benchmarks/canonical_tests/mnist_architecture_results.json) — efficiency table | 50 | 0.90 | per-class max | 27 |
>
> **The 34-vs-19 gap is `k`, not preprocessing.** An earlier version of this note
> attributed it to differing preprocessing. That was wrong: both scripts apply
> the same `StandardScaler` before discovery
> (`cifar_architecture_sweep.py` and `resnet_manifold_architecture.py`, each
> immediately before its discovery call). The difference was the `--k-pca`
> default — 50 in the sweep, 25 in the ResNet script — and a wider neighbourhood
> probes a larger radius, at which local PCA stops measuring a tangent space and
> starts mixing in curvature. On synthetic data with three regions of known
> local dimension 8/16/24, moving k from 10 to 100 moved the reported per-class
> maximum from 8 to 25 with the data held fixed. The defaults are now unified on
> `DEFAULT_K_PCA = 25` and every run records k, τ, the aggregation, the probe
> budget and the seed in an `"estimator"` block in its own JSON. Coverage in the
> older artifacts is uneven — `cifar10_architecture_results.json` and
> `mnist_architecture_results.json` do record `k_pca`, while
> `resnet_manifold_architecture_results.json` records τ but not k, so the
> headline `d*=19 → w*=28` was not reproducible from its own artifact. Where the
> JSON is silent, the k column above is the script default that produced it,
> read from the script.
>
> **Two caveats on the numbers themselves.** The per-class maximum is an order
> statistic over the probe budget, so it drifts upward as that budget grows and
> is not comparable across runs with different `samples_per_class`. And
> discovery was unseeded until the same change, so no d\* on this page is
> reproducible by re-running the script that produced it — repeats at the
> shipped defaults moved the mean statistic by roughly a dimension. Discovery is
> seeded by default now (`--discovery-seed`, default 0).

> **Caveat: the CIFAR-100 baseline and manifold figures come from different
> runs.** The 21.31% baseline is the `Standard (1024→512)` arm of
> [`cifar100_architecture_results.json`](../benchmarks/canonical_tests/cifar100_architecture_results.json)
> (100 epochs, 3 trials, test loss 3.8); the 38.3% manifold result is
> `ManifoldResNet-UB` from
> [`cifar100_resnet_manifold_architecture_results.json`](../benchmarks/canonical_tests/cifar100_resnet_manifold_architecture_results.json)
> (30 epochs). The same-configuration baseline inside that second run reaches
> only 5.21%, because all three of its trials diverged — test losses of 23,371 /
> 30,574 / 46,555, against ~2.5 for the manifold arms. Quoting 5.21% would
> make the comparison same-run but would measure the manifold against a network
> that failed to train, so the row cites the converged 21.31% instead and the
> accuracy ratio is 1.8×, not 7×. The divergence is itself worth knowing: the
> 3.7M-parameter dense net is unstable at the 30-epoch setting.

---

## Regenerating reports

Each benchmark ships a rendered report in three forms — `*_report.md`,
`*_report.tex`, and a typeset `*_report.pdf` — all generated from the committed
results JSON by `report_generator.py`:

```bash
python benchmarks/canonical_tests/report_generator.py <dataset>_architecture_results.json
```
