# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`UniversalEmbedder` (`src/waverider/universal_embedder.py`)** — geometry-grounded, modality-agnostic dimensionality reducer with the same `fit` / `transform` / `fit_transform` surface as `sklearn.decomposition.PCA`. Discovers d\* from local manifold geometry (`ManifoldModel` under the hood) and then auto-selects a coordinate strategy via the **Manifold Linearity Index** `MLI = global_d_at_τ / d*`:
  - `MLI ≤ mli_threshold` (default 3.0) → **`"pca"`** strategy: global linear projection sized to d\* (lossless rotation, optimal for near-linear data — most tabular/image datasets).
  - `MLI > mli_threshold` → **`"turtle"`** strategy: BFS Procrustes-transported TurtleND frames + k-means++ anchors, with class-balanced anchor selection when `y` is supplied (optimal for genuinely curved manifolds — proteins, molecular conformations).
  - Explicit modes `"pca"` / `"turtle"` / `"tangent"` (raw sign-corrected PCA frames, no Procrustes transport) bypass auto-selection. All strategies output `(n, d*)` or `(n, n_components)` float32 arrays. Exposes `d_star`, `strategy`, `mli`, and a `manifold_summary` dict for downstream logging.
- **`tests/test_universal_embedder.py`** — 362 lines covering construction/repr, mode validation, shape & dtype invariants across all four modes, MLI dispatch behaviour on swiss-roll vs linear vs tabular fixtures, and the sklearn-PCA drop-in contract.
- **`CIFAR10_CLAIM_VERIFICATION.md`** — audit report verifying the README's CIFAR-10 "+8.5 pp over ResNet" claim against the raw JSON. Confirms the number is reproducible from [`resnet_manifold_architecture_results.json`](benchmarks/canonical_tests/resnet_manifold_architecture_results.json) (4 trials × 60 epochs): ManifoldResNet-UB+Drop reaches 71.83% ± 0.60% at 36,942 params vs ResNet baseline 63.26% ± 3.09% at 47,978 params (+8.57 pp at 23% fewer parameters). Includes a "What 'Matched' Means" section spelling out that the two architectures share topology, residual primitive, optimizer, and training schedule — differing only in filter width (32 → w\*=28) and added dropout=0.3.
- **`AGENT_BRIEF_CIFAR10_CLAIM.md`** — the verification request that triggered the audit; kept in-tree as a traceable provenance record alongside the report it produced.
- **`README.md` — Manifold Voxel Visualizer section** — new dedicated section under the Algorithms table with the visualizer's hero figure, CLI examples (helix / iris / cifar10 / breast_cancer / off-screen), per-voxel scalar-field inventory (`density`, `curvature`, `height`, `intrinsic_dim`, `class_vote`), built-in dataset catalogue, and links to the CLI+API reference, USAGE examples, and method paper. Algorithms table gets a new `Voxel Visualizer` row pointing at `waverider.voxel_viz` / `waverider-voxel-viz`.

### Changed
- **`README.md` — UB headline table**:
  - Column header renamed `ManifoldResNet-UB` → `ManifoldResNet-UB+Drop` to honestly attribute the headline numbers to the dropout variant (bare UB without dropout underperforms — verified per-dataset against the JSONs).
  - Per-row stats re-aligned to JSON-computed values (sample std over 4 trials), e.g. CIFAR-10 `71.8% ± 0.5%` → `71.83% ± 0.60%`; ResNet baseline `63.3% ± 2.7%` → `63.26% ± 3.09%`; Fashion-MNIST and MNIST rows likewise re-synced. Δ-column now also states the parameter savings as a percentage (23% / 28% / 38% fewer than the matched ResNet baseline).
  - New footnote under the table points readers to the raw `resnet_manifold_architecture_results.json` (CIFAR-10) and `mnist_ub_phase_boundary_*_results.json` (MNIST/Fashion-MNIST) so a cloning reviewer can recompute the means and stds directly.
  - Dead links to `cifar10_report.md` / `cifar100_report.md` (which never existed in markdown form) redirected to the existing `cifar10_report.pdf` / `cifar100_report.pdf`.
- **`README.md` — file tree** refreshed to list `docs/INDEX.md`, `docs/USAGE.md`, and the `waverider-voxel-viz` CLI alongside `voxel_viz.py`.
- **`pyproject.toml`** — section-header comment retitled `CodeKG / DocKG index configuration` → `PyCodeKG / DocKG index configuration` to match the renamed package.
- **Version bump 0.8.1 → 0.9.0** (`pyproject.toml`, `src/waverider/__init__.py`) — minor bump for the new `UniversalEmbedder` public API surface. `__init__.py` now imports and re-exports `UniversalEmbedder`; module-header docstring lists it alongside the other core components and gains an explicit `License: Elastic 2.0` line.

### Removed

### Fixed

## [0.8.1] - 2026-05-25

### Changed
- **`CITATION.cff`** — `doi` field activated with the minted Zenodo identifier `10.5281/zenodo.20383651`.
- **`README.md`** — DOI badge (header + Citation section) wired to the live Zenodo concept-DOI badge (`zenodo.org/badge/1234120398.svg`); prose citation now resolves to `https://doi.org/10.5281/zenodo.20383651`; BibTeX entry replaces the placeholder `note` with a proper `doi` field.

## [0.8.0] - 2026-05-25

### Added
- **`CITATION.cff`** — machine-readable citation metadata (ORCID, affiliation, version, license, keywords) for Zenodo attribution and citation-aware tooling.
- **`cifar100_analysis.md`** — new persistent analysis file (replaces auto-generated `cifar100_report.md`, which was clobbered on every run). Contains manifold discovery results, full architecture comparison table, Key Findings, τ/d sweep table with three structural findings, and design rule.
- **`cifar100_tau_sweep_results.json`** — serialised results from the τ/d sweep (9 d-values, PCA+MLP + IntDim, patience=10 early stopping). Best: Intrinsic Dim d=75 → 25.70% ± 0.39% at 13,300 params, +4.37 pp vs Standard at 279× fewer parameters.
- **Early-stopping support in `cifar100_manifold_architecture.py`** — `--patience` flag (default 10) adds `EarlyStopping(monitor="val_accuracy", restore_best_weights=True)` uniformly to all architectures. `EpochHeartbeat` callback prints periodic epoch/acc/val_acc progress. `run_trial()` now returns `best_val_acc`, `best_val_epoch`, and `stopped_epoch` metadata. `patience` is now persisted in the results JSON.
- **`--tau-sweep` mode** — Phase 4 sweeps d across τ-derived values `{d*(τ) for τ ∈ {0.80,0.85,0.90,0.95}}` plus fixed grid `{50,75,100,150,200}`, training PCA+MLP and Intrinsic Dim only, saving `cifar100_tau_sweep_results.json`.
- **`--no-plot` flag** in `cifar100_manifold_architecture.py` and `cifar10_manifold_architecture.py` via `argparse.BooleanOptionalAction`.
- **`--plot-only` flag** — all canonical benchmark scripts (CIFAR-10, CIFAR-100, CIFAR-100+ResNet, Digits, MNIST, Iris, Iris Adam-vs-Manifold, Iris ManifoldAdamWalker, Clinical/Disease, Tiny ImageNet, ResNet) now accept `--plot-only` to regenerate the results figure from an existing JSON file without re-running training. Enables fast figure-tweak cycles after long runs.

### Changed
- **`README.md`** — added DOI badge placeholder and new **Citation** section (prose citation + BibTeX); both wired with `TODO` stubs for the Zenodo DOI/badge ID to fill in after repository activation.
- **`cifar100_report.md` → `cifar100_analysis.md`** — renamed to protect from overwrite by `report_generator.py`. Analysis section updated with re-run numbers (patience=10, uniform early stopping on all architectures):
  - Standard (with early stopping): 21.31% ± 0.28% (was 20.44% without early stopping; stopped at ep18)
  - Intrinsic Dim PCA→100D: 25.60% ± 0.12% (+4.29 pp, 184× fewer params)
  - Manifold + ManifoldAdam (d=100): 24.02% ± 0.64% (new; early stopping helps gradient-projection)
  - Manifold (d=100): 20.89% ± 0.19% (was 19.12% without early stopping)
- **`benchmarks/tf_setup.py`** — CPU is now the default device (not forced); universal `--gpu`/`--metal` argv detection added regardless of `gpu_flag` param. Device label changed from `"CPU (forced)"` to `"CPU"`.
- **`docs/waverider/waverider.md`** — comprehensive update to CIFAR-100 sections:
  - Abstract/intro: Standard accuracy updated to 21.31% (early-stopping baseline)
  - §4.5 evaluation description updated to "100 epochs max, patience=10 early stopping (all architectures)"
  - Finding 11 table: Standard updated 20.44% → 21.31%, IntDim d=100 updated 25.56% → 25.60%, Manifold updated 19.12% → 20.89%, new Manifold+ManifoldAdam row
  - Finding 9 efficiency frontier: Standard updated 20.44% → 21.31%, gap updated to +4.39 pp
  - Scorecard §5.6: Standard updated to 21.31%
  - τ/d sweep design rule extended to note optimal d ≈ 0.75×n\_classes
  - Finding 8 table refreshed with 60-epoch / 5-trial CIFAR-10 numbers (d\*=34); analysis section expanded with overparameterization diagnosis and PCA warm-start recommendation.
- **`papers/manifold_classification/DATA.md`** — CIFAR-100 Run A updated to re-run results (uniform early stopping, 100 epochs max, patience=10); Run B authoritative Standard reference updated 20.44% → 21.31%; Run C crossover note updated. CIFAR-10 section updated with 60-epoch / batch-512 results, prior-run row archived for comparison, analysis notes added for ManifoldAdam gradient-projection pathology and cold-init fix.
- **`mnist_ub_phase_boundary`** — replaced `verbose=1` with a `_ThrottledProgbar`
  callback that redraws the Keras progress bar every 5% of steps per epoch,
  reducing terminal I/O overhead on long runs without losing live feedback.
- **README UB table** — updated with 60-epoch / 4-trial results: Fashion-MNIST
  UB+Drop 88.38% ± 0.32% (+5.5 pp over ResNet, fewer params); MNIST UB+Drop
  98.98% ± 0.18% (within 0.3 pp of ResNet, 38% fewer params). CIFAR-10
  parameter-efficiency row updated (d\*=34, 724×, 49.12% @ 5,076 params).
- **`cifar10_report.md` → `cifar10_analysis.md`** — renamed; refreshed with
  60-epoch / 5-trial / batch-512 results: d\*=34 (truck class drives the
  ceiling), 724× parameter reduction, PCA+MLP achieves 95.1% of standard
  accuracy (49.12% vs 51.67%). Added analysis: 43pp train–test gap in Standard
  MLP reveals memorization; ManifoldAdam on overparameterized arch regresses
  5.4pp vs vanilla Adam; cold-init projection layer identified as root cause for
  Manifold/Wide-Manifold underperformance vs PCA+MLP.
- **Plot layout** — all canonical benchmark scripts: figure height increased,
  bottom-row grid-spec ratio raised to 1.2, `hspace` increased to 0.55; x-axis
  tick labels rotated from 30° to 45° with `ha="right"` alignment to prevent
  label overlap in accuracy and parameter-count bar charts.

### Benchmarks
- **CIFAR-100 flat-MLP re-run (3 trials, patience=10, batch=256)** — controlled comparison with uniform early stopping across all seven architectures. Standard converges fast (stopped ep18) to 21.31%; Intrinsic Dim d=100 converges slowly (stopped ep65) to 25.60%. Even at Standard's val-peak, manifold-informed model wins by +4.29 pp at 184× fewer parameters.
- **CIFAR-100 τ/d sweep (3 trials, patience=10, 100 epochs max)** — optimal PCA compression d=75 (not n\_classes=100); crossover from PCA+MLP to IntDim winning at d≈50; geometric τ-values (d=13–21) insufficient for 100-class discrimination; design rule established: optimal d ∈ (d\*, n\_classes), empirically ≈0.75×n\_classes.
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
