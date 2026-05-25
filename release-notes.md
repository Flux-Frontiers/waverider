# Release Notes — v0.8.0

> Released: 2026-05-25

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

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
