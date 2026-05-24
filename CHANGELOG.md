# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`--plot-only` flag** — all canonical benchmark scripts (CIFAR-10, CIFAR-100, CIFAR-100+ResNet, Digits, MNIST, Iris, Iris Adam-vs-Manifold, Iris ManifoldAdamWalker, Clinical/Disease, Tiny ImageNet, ResNet) now accept `--plot-only` to regenerate the results figure from an existing JSON file without re-running training. Enables fast figure-tweak cycles after long runs.

### Changed
- **`mnist_ub_phase_boundary`** — replaced `verbose=1` with a `_ThrottledProgbar`
  callback that redraws the Keras progress bar every 5% of steps per epoch,
  reducing terminal I/O overhead on long runs without losing live feedback.
- **README** — UB table updated with 60-epoch / 4-trial results: Fashion-MNIST
  UB+Drop 88.38% ± 0.32% (+5.5 pp over ResNet, fewer params); MNIST UB+Drop
  98.98% ± 0.18% (within 0.3 pp of ResNet, 38% fewer params). Version badge
  updated to 0.7.1. CIFAR-10 parameter-efficiency row updated (d\*=34, 724×,
  49.12% @ 5,076 params).
- **`cifar10_report.md` → `cifar10_analysis.md`** — renamed; refreshed with
  60-epoch / 5-trial / batch-512 results: d\*=34 (truck class drives the
  ceiling), 724× parameter reduction, PCA+MLP achieves 95.1% of standard
  accuracy (49.12% vs 51.67%). Added analysis: 43pp train–test gap in Standard
  MLP reveals memorization; ManifoldAdam on overparameterized arch regresses
  5.4pp vs vanilla Adam; cold-init projection layer identified as root cause for
  Manifold/Wide-Manifold underperformance vs PCA+MLP.
- **`docs/waverider/waverider.md`** — Finding 8 table refreshed with 60-epoch /
  5-trial numbers (d\*=34); analysis section expanded with overparameterization
  diagnosis and PCA warm-start recommendation.
- **`papers/manifold_classification/DATA.md`** — CIFAR-10 section updated with
  60-epoch / batch-512 results, prior-run row archived for comparison, analysis
  notes added for ManifoldAdam gradient-projection pathology and cold-init fix.
- **Plot layout** — all canonical benchmark scripts: figure height increased,
  bottom-row grid-spec ratio raised to 1.2, `hspace` increased to 0.55; x-axis
  tick labels rotated from 30° to 45° with `ha="right"` alignment to prevent
  label overlap in accuracy and parameter-count bar charts.

### Benchmarks
- **UB phase boundary — MNIST** (60 epochs, 4 trials): ResNet 99.27% ± 0.12%
  (47,338 params); UB+Drop 98.98% ± 0.18% (29,110 params) — matched at 38%
  fewer params. Whitney-dominated regime (C=10 ≤ d\*=16) confirmed.
- **UB phase boundary — Fashion-MNIST** (60 epochs, 4 trials): UB+Drop wins
  at 88.38% ± 0.32% vs ResNet 82.85% ± 2.25% — **+5.5 pp with 38% fewer
  params**. UB theorem prediction confirmed on the harder dataset.

## [0.7.1] - 2026-05-23

### Added
- **`benchmarks/tf_setup.py`** — shared TensorFlow bootstrap module replacing
  ~25 lines of duplicated setup boilerplate across all benchmark scripts.
  Accepts `gpu_flag` and `argv` parameters so Metal GPU benchmarks can be
  enabled with `--metal` without touching individual scripts.
- **`__init__.py`** files in `benchmarks/`, `benchmarks/canonical_tests/`,
  `benchmarks/canonical_tests/clinical/`, and `benchmarks/manifold_model/`
  to enable package-style imports (`from benchmarks.tf_setup import …`).

### Fixed
- **macOS import-order deadlock** — all benchmark scripts now import `tensorflow`
  before `numpy` / `sklearn` to prevent Arrow's `libarrow.dylib` from binding
  TensorFlow's `_AbslInternalPerThreadSemWait_*` symbols first, which caused
  non-deterministic deadlocks on the first `model.fit()` call on Apple Silicon.
- **`manifold_optimizer`** — `import keras` reordered to precede numpy and
  sklearn imports, consistent with the package-wide TF-first policy.
- **`mnist_manifold_model`** — `tensorflow` promoted from deferred local import
  inside `main()` to module-level, consistent with all other benchmark scripts.
- **`mnist_ub_phase_boundary`** — `verbose=0` changed to `verbose=1` in
  `model.fit()` so training progress is visible during long runs.

### Changed
- Reformatted long function-call argument lists across all canonical benchmark
  scripts to Black's multi-line style (one argument per line, trailing comma).
- Minor style fixes across source modules: comment alignment, forward-reference
  type hints replaced with bare names, removed stray blank lines in tests.

## [0.7.0] - 2026-04-15

Initial public release of the WaveRider geometric ML stack.

### Core Components
- **`ManifoldModel`** — zero-parameter classifier operating entirely in tangent space; local PCA geometry drives both fit and predict; vectorized fit and predict via chunked BLAS GEMM for production-scale throughput
- **`TurtleND`** — N-dimensional navigational primitive with Givens-rotation orthonormal frame update; formal specification in `docs/turtlend/`
- **`ManifoldWalker`** / **`ManifoldAdamWalker`** — Riemannian gradient descent and Adam-momentum descent in tangent space
- **`ManifoldObserver`** — (N+1)-dimensional extrinsic observer hovering above the manifold surface
- **`GeodesicEncoder`** — ambient → tangent-projected geodesic distance coordinates with k-means++ anchor selection
- **`voxel_viz`** — interactive 3-D manifold anatomy tool: projects observer field onto a 3D PCA subspace, rasterizes into an N³ voxel grid (density / curvature / height / intrinsic_dim / class_vote), renders with PyVista orthogonal slice planes; headless PNG export; `waverider-voxel-viz` CLI entry point

### Benchmarks
- **Clinical Manifolds** — ManifoldModel benchmarked on six disease datasets (Alzheimer's, Breast Cancer, Dermatology, Diabetes, Heart Disease, Parkinson's); wins on Heart Disease with **zero trainable parameters** and an average **80× parameter reduction** vs. standard MLPs
- **CIFAR-10** — best manifold-informed model (4,795 parameters, PCA+MLP) achieves 48.70% vs 52.04% for a 3,676,682-parameter standard architecture — **766× parameter reduction** at 93.6% of standard performance; intrinsic d*=33
- **CIFAR-100** — Intrinsic Dim head achieves 25.29% vs 20.44% for Standard MLP (+23.8% relative, 184× parameter reduction)
- **MNIST, Iris, Digits, Tiny ImageNet** — full architecture sweep results locked in `benchmarks/canonical_tests/`
- **Protein Backbone** — Ramachandran plot rediscovery from raw (φ, ψ) angles via proteusPy; window13 achieves 0.906 accuracy, intrinsic d*=16; per-class ordering H (9.6) < E (10.0) < C (10.7) — biologically correct

### Papers Included
- `papers/waverider_article/waverider_jmlr.tex`: WaveRider full-stack JMLR draft
- `papers/manifold_classification/manifold_classification.tex`: manifold-informed architecture paper
- `papers/clinical_manifolds/clinical_manifolds.tex`: clinical disease manifolds paper
- `papers/voxel_viz/voxel_viz.tex`: Manifold Voxel Visualizer paper
- `docs/turtlend/turtlend.tex`: TurtleND formal paper (submitted TMLR 2026-04-14)
- `docs/ub_theorem/ub_theorem.tex`: Universal Bottleneck Theorem paper
