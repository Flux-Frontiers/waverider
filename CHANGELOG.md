# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
