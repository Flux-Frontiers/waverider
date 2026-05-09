# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `benchmarks/canonical_tests/protein_backbone_manifold.py`: **Protein Backbone Latent-Space Discovery** — full pipeline benchmarking whether WaveRider can rediscover the Ramachandran plot from raw (φ, ψ) angles; integrates proteusPy `BackboneLoader` for PDB corpus ingestion; four embedding modes (torus/discrete/window7/window13); headline result: window13 achieves 0.906 accuracy, intrinsic d*=16 with DSSP labels, per-class d* confirming H (9.6) < E (10.0) < C (10.7) — biologically correct ordering
- New CLI flags: `--cache-file` / `--rebuild-cache` / `--cache-only` / `--check-cache` / `--batch-files` for memory-safe batched Parquet cache build over the full PDB corpus (~33M residues, 760 MB zstd); `--dssp` for pydssp re-annotation; `--remap-u-to-coil` to fold unknown SS into coil for clean 3-class experiments; `--sample-n` for stratified proportional subsampling; `--report` for Markdown reports with full git provenance and timing
- `papers/backbone_manifold/backbone_report.md`: initial backbone experiment report (synthetic + PDB, torus/window modes)
- `papers/backbone_manifold/backbone_dssp_report.md`: DSSP-annotated 100k-residue report — embedding mode comparison table and per-class intrinsic dimensionality landscape (τ=0.85/0.90/0.95)

### Changed
- `pyproject.toml`: added `proteuspy = { version = ">=0.100.0", optional = true }` to `[tool.poetry.dependencies]`; added to `kg` extras — backbone pipeline depends on proteusPy BackboneLoader
- `poetry.lock`: regenerated for proteuspy addition
- `src/waverider/backbone_manifold.py`: added flushed progress prints after `ManifoldModel.fit()` and `GeodesicEncoder.fit()` / `transform()` to surface the hang point during long window13 runs; fixed `ManifoldModel(k=…)` → `ManifoldModel(k_graph=…)` kwarg (wrong keyword crashed all PDB-corpus runs)
- `src/waverider/manifold_model.py`: added chunk-level progress counter in `_predict_vectorized` (10 % increments, flushed) so long predict() calls are observable rather than silent

### Fixed
- `src/waverider/dimensionality_discovery.py`: guard against zero-size array crash in `discover_dimensionality` when a variance-threshold bucket accumulates no samples (degenerate neighborhood); now returns a zero-stats dict instead of raising `ValueError`

### Changed
- `pyproject.toml`: aligned with sister repos (`code_kg`, `kgrag`) — `[build-system]` moved first with `poetry-core>=2.0.0`; `target-version` and `mypy python_version` corrected to `py312`/`3.12` (matches `python = ">=3.12"` constraint); added `[tool.ruff.lint]` with `select`/`ignore`; dev deps switched from `^` to `>=`; `doc-kg`, `pycode-kg`, `ftree-kg` switched from git URLs to PyPI version constraints (`>=0.14.0`, `>=0.19.0`, `>=0.8.0`); `agent-kg` moved to optional `[tool.poetry.group.kgdeps]` group (not yet on PyPI); `kg` extra updated to `["pycode-kg", "ftree-kg", "doc-kg"]`; added `[tool.poe.tasks.docs]`; added header comment with install/test/build instructions; build commands corrected (`pycodekg build`, `dockg build`)
- `poetry.lock`: regenerated to reflect dependency changes above

### Added
- `src/waverider/geodesic_coords.py`: `GeodesicEncoder` — ambient → tangent-projected geodesic distance coordinates; encodes each point as distances to $d^*$ manifold anchors selected via k-means++ initialization; supports signed coordinate pairs (sign + magnitude) and class-aware anchor selection; exported from `waverider` package
- `papers/clinical_manifolds/kan_clinical.tex`: new LaTeX article "Reading the Rules Written in Disease: Kolmogorov-Arnold Networks on Clinical Manifolds" — 5-dataset benchmark (heart, breast, diabetes, Parkinson's, dermatology); three-way comparison (ManifoldModel / KAN-raw / KAN-pca); symbolic regression recovering `2.557·PC1 − 0.035` for heart disease; activation plots for all datasets; reports Riemannian coordinate (GeodesicEncoder) experiment as a completed negative result
- `papers/clinical_manifolds/clinical_manifolds.bib`: added `liu2024kan` (KAN paper), `kolmogorov1957representation`, and `detrano1989international` citations

### Changed
- `benchmarks/canonical_tests/clinical/kan_clinical.py`: removed KAN-geo arm (GeodesicEncoder-based coordinates consistently underperformed KAN-pca across all 5 datasets); replaced `auto_symbolic` with per-edge `suggest_symbolic` loop (R²≥0.60 threshold) to fix zero-active-edge collapse; switched to 1-layer symbolic KAN architecture `[d*, n_out]` for clean additive formulas; fixed ManifoldModel multiclass AUC (one-hot encode hard predictions for `roc_auc_score`); fixed activation plot output labels to use dataset-specific class names; renamed ambiguous variable `l` → `li` (E741); `class_names[1]` for heart disease corrected from `"Disease"` to `"Heart disease"`; `varscale` now computed as `min(0.8, 4.0 / max(d_star, n_classes))` so label size scales down for wide/multiclass networks (fixes illegible dermatology plot); `dpi` raised to 150
- `papers/clinical_manifolds/kan_{heart,breast,diabetes,parkinsons,dermatology}/activations.png`: all five plots regenerated — corrected heart disease output label, proportionally scaled text labels across all datasets
- `benchmarks/canonical_tests/clinical/kan_{breast,dermatology,diabetes,heart,parkinsons}_results.json`: updated with current benchmark numbers including `simplified_formula`, `n_active_edges`, `active_edges`, and `pca_loadings` fields

### Added
- `docs/GraphReasoner_Formal_Specification.md` + `docs/graph_reasoner_spec/graph_reasoner_spec.md`: formal specification for GraphReasoner — discrete knowledge-graph traversal engine with TurtleND heading, three steering strategies (TargetSteering / GradientSteering / ExplorationSteering), greedy/targeted/beam-search reasoning modes, backtracking, and correctness properties P1–P5; includes complexity table and KGRAG integration notes
- `docs/graph_reasoner_spec/summary.md`: "The graph reasons. The LLM synthesizes." — architecture overview, three-navigator comparison (TurtleND / ManifoldWalker / GraphReasoner), DisulfideTree KGRAG integration, and implementation table
- `docs/manifold_walker_spec/manifold_walker_spec.pdf`: compiled PDF formal specification for ManifoldWalker

### Changed
- `src/waverider/voxel_viz.py`: `render_multi` now accepts an optional `scalars` parameter (list of `(field, cmap, title)` tuples); defaults to the existing intrinsic_dim/curvature/height/class_vote quartet — callers can now override panels per dataset without patching the function
- `benchmarks/canonical_tests/clinical/gen_voxel_viz.py`: added `_BREAST_CANCER_SCALARS` constant (density/curvature/height/class_vote) and wired it into `run_one()`; breast cancer voxel now renders density upper-left instead of intrinsic_dim, matching the actual figure
- `papers/clinical_manifolds/breast_cancer_manifold_voxel.png`: re-rendered with corrected scalars (density upper-left)
- `papers/clinical_manifolds/clinical_manifolds.tex`: breast cancer paragraph rewritten — corrected from "two-lobe density anatomy" to spatial segregation (benign majority dense central volume; malignant cluster in upper-left PCA quadrant); figure caption updated to reflect density upper-left panel; curvature described as uniformly low (flat embedding, gap-separated classes)

### Added
- `waverider_missions/journal_2026.107.md`: Admiral's Journal stardate 2026.107 — two entries covering the bioRxiv rejection cascade (arXiv → bioRxiv → Hopkins Tech Ventures, all blocked by lack of affiliation), the Springer in-press status of the disulfide paper, and the zero-AM topology of institutional gatekeeping
- `waverider_missions/bioarxiv_letter_2026.107.md`: formal appeal letter to Cold Spring Harbor Laboratory/bioRxiv requesting reconsideration of manuscript BIORXIV/2026/717769 and policy review; invokes Einstein, Newton, Faraday, Tesla, and Darwin as precedent; argues Springer acceptance constitutes the peer review bioRxiv explicitly disclaims providing
- `waverider_missions/bioarxiv_retraction_2026.107.md`: satirical notice of retraction — not of the science (in press, correct, reproducible) but of the assumption that a preprint server would make science openly accessible to scientists lacking institutional affiliation
- `waverider_missions/arxiv_endorsement_request.md`: arXiv cs.LG endorsement request template for TurtleND preprint (endorsement code X3GMXJ); briefly describes the navigational primitive, Givens rotation frame update, and validation datasets

- `benchmarks/arc_agi/arc_onnx.py`: two new zero-parameter ONNX template families — `endpoint_row_fill` (task 22eb0ac0: fill row when left==right endpoint, non-background) and `row_uniform_fill` (task 25d8a9c8: uniform rows → fixed fill color, non-uniform rows → zero); each has a `build_*` graph constructor and a `fit_*` fitter wired into `build_onnx_network` / `fit_task`
- `benchmarks/arc_agi/eval_results/onnx/22eb0ac0.onnx` + `25d8a9c8.onnx`: compiled ONNX graphs for both new families
- `docs/waverider/ship_dimensions_scales.md`: canonical WaveRider ship class proportions and real dimensions for Probe (42 m), Cruiser (300 m), and Galaxy (640 m) classes; print-scale table (1:50 – 1:5000) and Unity / Blender export workflow
- `docs/waverider/ship_ortho_capital_prompt.md`: image-gen orthographic reference prompt for the Capital-class WaveRider (Caldari heavy explorer, vast hexagonal carapace)
- `docs/waverider/ship_ortho_gallente_prompt.md`: image-gen prompt for the organic Gallente WaveRider variant (smooth lenticular hull, dolphin-nose bow, iridescent bronze-green)
- `docs/waverider/ship_ortho_turtle_ufo_prompt.md`: image-gen prompt for the Turtle UFO class (oblate disc hull, full-dorsal carapace, Meshy <800-char variant)
- `docs/waverider/ship_ortho_waverider_cruiser_prompt.md`: detailed cruiser-class prompt with Meshy short form, design intent narrative, element key, style notes, and full PaperBanana long-form prompt
- `docs/waverider/steampunk_armillary_manifold_prompt.md`: image-gen prompt for a steampunk armillary sphere with holographic voxel manifold interior — visualization centrepiece for the clinical manifolds paper
- `waverider_missions/journal_2026.106.md`: Admiral's Journal stardate 2026.106 — two entries + addendum covering NeuroGolf Phase 2 milestone (27 tasks, 720-byte cost floor), bureaucratic obstacles (Absil review, Hopkins portal), clinical manifolds results (ManifoldModel beats trained MLP on heart disease at zero parameters), and the voxel anatomy of cancer

### Changed
- `benchmarks/arc_agi/NEUROGOLF_STRATEGY.md`: updated submission stats to 28 tasks / 484.7 pts / 10,525 B; added `endpoint_row_fill` and `row_uniform_fill` to the solved-task family table; updated tractability table (Tier-3 composite 51→62, variable-input 14→2); revised ceiling paragraph to reflect new families and remaining failure modes
- `benchmarks/arc_agi/arc_onnx.py`: extracted shared `_NO_BG_MASK` constant (replaces inline arrays in `build_diag_fill` and new builders)
- `benchmarks/arc_agi/arc_fit.py`: added `endpoint_row_fill` and `row_uniform_fill` to `_FALLBACK_FAMILIES`; fixed `import onnx` noqa placement
- `docs/waverider/ship_ortho_textured_prompt.md`: upgraded from 2×2 to 2×3 panel layout; added ventral dome (clear glass), dorsal dome (dark glass), rabbit-foot landing pads; removed panel text labels (pure orthographic reference)
- `docs/waverider/logo_prompt.md`: switched concept art faction from Caldari to Gallente (smooth organic hull replacing angular modular hull)

### Added
- `benchmarks/canonical_tests/clinical/gen_voxel_viz.py`: batch renderer — runs full voxel pipeline for all 6 clinical datasets and saves 2×2 multi-scalar PNG panels to `papers/clinical_manifolds/`
- `benchmarks/canonical_tests/gen_gallery_voxels.py`: gallery renderer — generates voxel PNGs for synthetic (helix, swiss_roll, torus), tabular (iris, digits), and high-dim (mnist, cifar10) datasets; outputs to `papers/voxel_viz/figures/`
- `papers/clinical_manifolds/{dataset}_manifold_voxel.png` (6 files): voxel anatomy panels for all clinical datasets, re-rendered with corrected panel order (intrinsic_dim upper-left)
- `papers/clinical_manifolds/{dataset}_disease_architecture_results.png` (6 files): benchmark bar-chart figures for all clinical datasets
- `papers/clinical_manifolds/manifold_viz.png`: updated pipeline overview figure (three-stage annotated diagram)
- `papers/voxel_viz/figures/{helix,swiss_roll,torus,iris,digits,mnist,cifar10}_voxel.png` (7 files): voxel gallery figures for the voxel_viz paper appendix
- `docs/waverider/ship_ortho_prompt.md`: orthographic reference sheet prompt for WaveRider ship concept art

### Changed
- `src/waverider/voxel_viz.py` + `benchmarks/canonical_tests/manifold_voxel_viz.py`: `render_multi` 2×2 panel upper-left field corrected from `density` ("Point density") to `intrinsic_dim` ("Intrinsic dim (d*)") — aligns rendered figures with clinical paper text which references the d* field
- `papers/clinical_manifolds/clinical_manifolds.tex`: expanded to 6 datasets (added Pima Indians diabetes); pipeline figure updated to `manifold_viz.png` with three-stage caption; all 6 voxel figures converted to non-floats (`\captionof`) for inline placement; zero-parameter subsection relocated adjacent to heart disease results; panel description updated to reflect intrinsic_dim upper-left
- `papers/voxel_viz/voxel_viz.tex`: matched style to `clinical_manifolds.tex` (added `float`/`placeins`/`caption` packages, `[Suchanek]` author arg, `, PhD`); pipeline figure replaced placeholder with real `manifold_viz.png`; gallery section moved to appendix with subfigure panels labelled (a)/(b)/(c) using minipage + `\captionof`; figure labels placed above panels
- `.gitignore`: excluded `3D_Objs/` (WaveRider 3D ship mesh assets)
- `pyproject.toml` + `src/waverider/__init__.py`: version bumped to 0.7.0
- `docs/waverider/logo_prompt.md`: updated WaveRider logo generation prompt

### Previous Added
- `src/waverider/voxel_viz.py`: **Manifold Voxel Visualizer** — interactive 3-D anatomy tool; projects observer field onto a 3D PCA subspace, rasterizes into an N³ voxel grid (density / curvature / height / intrinsic_dim / class_vote), and renders with PyVista orthogonal slice planes; single-scalar and 2×2 multi-scalar panel modes; headless PNG export; installed as `waverider-voxel-viz` CLI entry point
- `papers/voxel_viz/voxel_viz.tex` + `.pdf` + `.bib`: standalone JMLR-format paper on the Manifold Voxel Visualizer
- `papers/clinical_manifolds/clinical_manifolds.tex` + `.pdf` + `.bib` + `medium_article.md`: **Clinical Manifolds paper** — ManifoldModel benchmarked on six disease datasets (Alzheimer's, Breast Cancer, Dermatology, Diabetes, Heart Disease, Parkinson's); ManifoldModel wins on Heart Disease with **zero trainable parameters** and an average **80× parameter reduction** vs. standard MLPs; targets MLHC/ML4H
- `benchmarks/canonical_tests/disease_manifold_architecture.py`: disease manifold benchmark runner — six UCI/sklearn disease datasets with the same 7-architecture sweep used for CIFAR/MNIST; produces `*_disease_architecture_results.{json,png}` and LaTeX/Markdown reports
- `benchmarks/canonical_tests/{alzheimers,breast_cancer,dermatology,diabetes,heart,parkinsons}_disease_architecture_results.{json,png}`: locked disease benchmark results for all six datasets
- `benchmarks/canonical_tests/{alzheimers,breast_cancer,dermatology,heart,parkinsons}_report.{md,tex}`: auto-generated Markdown and LaTeX reports for five disease benchmarks
- `benchmarks/canonical_tests/figures/iris_density.png`: density voxel slice figure for Iris dataset
- `docs/waverider/manifold_voxel_viz.png` + `manifold_voxel_viz_light.png`: voxel visualizer screenshot figures (dark and light themes)
- `docs/waverider/logo1.png` + `logo2.png`: WaveRider project logos
- `.mcp.json`: MCP server configuration for `pycodekg` and `dockg` in-project servers
- `analysis/waverider_analysis_20260415.md`: CodeKG architectural analysis snapshot (2026-04-15) — 11,857 nodes, 11,704 edges, grade C/70; documents fan-in ranking, module coupling, and public API surface

### Changed
- `README.md`: updated install instructions for new `viz` optional group (`poetry install --with viz`); added Manifold Voxel Visualizer API and CLI examples; updated project tree with `voxel_viz.py`
- `pyproject.toml` + `src/waverider/__init__.py`: version bumped to 0.6.0; added `viz` optional dependency group (`pyvista`, `scipy`)
- `docs/waverider/manifold_voxel_viz.md`: updated voxel visualizer reference documentation
- `.gitignore`: added `benchmarks/data/` exclusion for OASIS restricted-access data

### Added
- `PUBLICATION_PLAN.md`: Hopkins Partnership section — MetaboKG / KGRAG wedge strategy targeting Johns Hopkins via Dr. Wolberg (Biophysics chair, alumni connection); documents proposed deal structure (academic license + sponsored research + royalty stream), valuation anchor vs. BioCyc/Pathway Tools, JHTV engagement path, and next-step checklist
- `benchmarks/canonical_tests/report_generator.py`: LaTeX report generation — `_tex_escape()` and `_write_tex()` produce self-contained `.tex` files compilable with `latexmk -pdf` or `pdflatex`; handles Unicode→LaTeX math mapping, booktabs tables, and figure inclusion; falls back to matplotlib PDF if no LaTeX compiler is present
- `benchmarks/canonical_tests/{cifar100,cifar10,digits,iris,mnist,tiny_imagenet}_report.tex`: LaTeX benchmark reports generated for all six canonical datasets
- `pyproject.toml`: documented system LaTeX dependency for `.tex`→PDF compilation in the benchmarks group (macOS: `brew install --cask mactex`; Ubuntu: `apt install texlive-latex-extra latexmk`)

### Changed
- **CIFAR-10 results updated** (`docs/waverider/waverider.md`, `papers/waverider_article/waverider_jmlr.tex`, benchmark reports): intrinsic dimensionality revised to 33 (from 29); best manifold-informed model is the 4,795-parameter PCA+MLP at 48.70% vs 52.04% for a 3,676,682-parameter standard architecture — a **766x parameter reduction** at 93.6% of standard performance; all benchmark `.md` reports and CIFAR-10 results JSON/PNG regenerated to match

### Fixed
- `benchmarks/canonical_tests/report_generator.py`: expanded `KNOWN_RESULTS` to include all nine result files that exist on disk (`iris_architecture_results.json`, both ResNet-manifold variants, two Tiny ImageNet results); previously `--all` silently skipped six of them; updated module docstring to match the full dataset coverage

## [0.5.0] - 2026-04-14

### Added
- `docs/turtlend_tmlr/turtlend_tmlr.tex` + `.pdf`: **TurtleND paper submitted to TMLR (OpenReview, 2026-04-14).** Double-blind 8-page submission covering the TurtleND primitive, Normal Extension (Proposition 2), and numerical validation on helix and flat-torus synthetic manifolds. Recovers $d^*$ to within 0.02 and lifts $(N{+}1)$-frames to machine-precision orthonormality across 10 seeds.

### Fixed
- `docs/turtlend_tmlr/turtlend_tmlr.tex`: corrected "thin QR" → "full QR" in §4 Normal Extension constructive recovery. Thin QR of an $(N{+}1)\times N$ matrix yields $Q\in\mathbb{R}^{(N+1)\times N}$; the full/complete QR is required to produce the $(N{+}1)\times(N{+}1)$ orthogonal matrix and extract the manifold normal vector. Caught during final pre-submission review.

### Changed
- `benchmarks/canonical_tests/tiny_imagenet_resnet_manifold_architecture.py`: added `PCA→d*D + MLP (2d→d)` and `PCA→d*D + MLP-wide (4d→2d)` architectures to the Tiny ImageNet sweep, matching the CIFAR architecture suite for cross-dataset comparison; reorganized architecture docstring into MLP and ResNet sections; fixed UB-PCA key to use symbolic name `UB-PCA (PCA→d*→w*→C)` (was parameterized string, causing lookup failures); added `build_pca_mlp_wide` and `build_pca_model` imports from `model_builder`
- `papers/manifold_classification/manifold_classification.tex`: removed `height=0.70\textheight,keepaspectratio` constraint from CIFAR-10 ResNet result figure — `\includegraphics[width=\textwidth]` alone gives cleaner full-width rendering without distortion
- `.gitignore`: added `.agentkg/` exclusion — AgentKG conversational memory graph is local/transient and not versioned
- `pyproject.toml`, `src/waverider/__init__.py`: version bumped to 0.5.0
- `poetry.lock`: refreshed to Poetry 2.3.2; `agent-kg 0.5.1` resolved as optional KG dependency

## [0.4.0] - 2026-04-12

### Added
- `papers/manifold_classification/cifar100_resnet_manifold_architecture_results.png`, `resnet_manifold_architecture_results.png`: new result figures for the manifold classification paper covering ResNet comparison benchmarks
- `UB-PCA` architecture (`build_pca_mlp_wide` in `model_builder.py`): Universal Bottleneck PCA — PCA→d\* then a single hidden layer of prescribed width w\*=d\*+C−1 before classification; 1,945 parameters on CIFAR-10, 40,099 on CIFAR-100
- Apple Accelerate BLAS FPE guard in `cifar_architecture_sweep.py`: `_fpe_ctx()` suppresses spurious divide/overflow/invalid warnings fired by Accelerate on large float64 matmuls; PCA outputs sanitized with `nan_to_num`

### Changed
- **Keras import** — all benchmark scripts and `src/model_builder.py`, `src/waverider/manifold_optimizer.py` changed from `from tensorflow import keras` to `import keras` (standalone Keras 3 path)
- **CIFAR-10 results corrected** (`DATA.md`, `manifold_classification.tex`): Standard MLP baseline now correctly achieves 50.99%±0.41% (was 20.80%); all PCA-preprocessed architectures cluster within 3.81 pp of the baseline while using 249×–2,387× fewer parameters; UB-PCA achieves best-below-2K-params at 48.02%±0.19%
- **CIFAR-100 results corrected** (`DATA.md`, `manifold_classification.tex`): d set to max(intrinsic_dim=35, n_classes=100)=100; Intrinsic Dim head now achieves 25.29%±0.32% vs 20.44% for Standard MLP (+23.8% relative, 184× parameter reduction); UB-PCA matches at 25.15%±0.13%
- `cifar_architecture_sweep.py`: removed LDA+PCA Augmented, PCA-MLP-deep, and UB-PCA-deep architectures; tightened architecture set to Standard MLP, PCA+MLP, PCA+MLP-wide, UB-PCA, and Intrinsic Dim; DPI raised 150→200
- `cifar100_resnet_manifold_architecture.py`, `resnet_manifold_architecture.py`: figure size 16×20→16×16; DPI raised 150→200 for publication-quality output
- `papers/manifold_classification/manifold_classification.tex`: added UB-PCA architecture description (§Architecture Families), updated results narrative and table to reflect corrected CIFAR-10/100 numbers, improved table font (footnotesize), fixed figure placement (`[tp]`, `[p]`)
- `pyproject.toml`, `src/waverider/__init__.py`: version bumped to 0.4.0

### Added
- `benchmarks/arc_agi/`: ARC-AGI scout — Paper #4 foundation. Hand-crafted 142-dim pair embedding (color histogram + shape meta + D4 self-symmetry + per-color connected-component stats + edge co-occurrence + 8-dim delta), D4 × color-permutation augmentation (128 copies per pair), `ManifoldModel` adapter with `last_train` and `official_test` holdout protocols, evaluation harness with scaling sweeps and confusion analysis, and a foreign-task OOD probe via chunked BLAS-matmul kNN with Mann-Whitney AUC. Scout reframing documented: the manifold hypothesis is in embedding space (augmented pairs populate a low-intrinsic-dim submanifold of R^142), not in raw grids — D4 is a finite group, not a Riemannian structure
- `benchmarks/arc_agi/eval_results/locked_scale_sweep_v1.json`: locked `last_train` scaling sweep over n_tasks ∈ {10, 25, 50, 100, 200} on ARC-AGI-1. Intrinsic dimensionality stays flat at d ≈ 6 (min 5.34, max 6.34) across a 20× task-count scale-up; accuracy 0.978 → 0.574; lift over chance 9.8× → 114.8×. Reproducible via `arc_eval.py --holdout last_train --lock`
- `benchmarks/arc_agi/eval_results/locked_scale_sweep_v2.json`: locked `official_test` scaling sweep over n_tasks ∈ {10, 25, 50, 100, 200} — trains on all training pairs, probes on augmented official ARC test pairs. Every scale beats the matching `last_train` row (0.994/0.948/0.830/0.729/0.673 vs 0.978/0.894/0.763/0.633/0.573), retiring the "within-task consistency" critique: scout generalizes cleanly to held-out ARC test pairs, not just to augmentations of its own training data. 200-row lift 134.6× over chance, d = 6.35
- `benchmarks/arc_agi/eval_results/locked_probe_v1.json`: locked foreign-task OOD probe, 50 TRAIN / 50 FOREIGN tasks, 6 400 familiar + 6 400 foreign probes. Separation ratio 1.40×, ROC-AUC 0.710 — moderate (not dramatic) signal, consistent with shared-substrate reading: foreign ARC rules sit in previously-unpopulated regions of the same ~6D manifold, not off-manifold
- `benchmarks/arc_agi/eval_results/locked_probe_v2.json`: locked foreign probe at 100/100 split (12 800 familiar + 12 800 foreign). Separation ratio rises to 1.49× (from 1.40×) while AUC holds essentially flat at 0.707. Scaling invariance confirms the gap is a manifold property, not a sampling-density artifact
- `benchmarks/canonical_tests/*_results.json`, `*_results.png`, `manifold_dim_probe{,_cifar10}.png`: unlocked 14 canonical benchmark result JSONs (MNIST, CIFAR-10/100, digits, iris, resnet manifold architecture, UB phase boundary) and 14 figure PNGs so papers can cite and re-plot without re-running the full benchmark suite. The 3 `manifold_dim_probe*_arrays.npz` raw-array dumps (~17 MB) and `manifold_dim_probe_cifar100.png` (1.3 MB) exceed the repo's 1 MB pre-commit size guard and are regenerated on demand rather than versioned
- `docs/ARC_PAPER_PLAN.md`: preliminary scout results section (2026-04-11) — both protocol tables through 200 tasks (10 rows total), foreign-probe tables at 50/50 and 100/100, decision-gate revision noting the geometric precondition (d flat, lift grows) is already cleared and Stage 2 investment is defensible on those grounds
- `benchmarks/arc_agi/README.md`: full scout documentation with framing, architecture diagram, preliminary result tables, reproduction commands, and status table
- `benchmarks/canonical_tests/torus_manifold_observer.py`: synthetic 2-manifold flat-torus benchmark for the ManifoldObserver Normal Extension (Proposition 1 of TurtleND) — flat torus in R^4 embedded in R^6 with a 4-class topologically non-convex quadrant checkerboard; recovers d*=2 to 2.019±0.009, subject 93.94%, observer 94.69%, agreement 93.75% over 10 seeds; produces `torus_manifold_observer.png` diagnostic (training points / d* histogram / per-trial accuracy)
- `docs/turtlend/turtlend.tex`: new §7.2 "Synthetic 2-Manifold Flat Torus" with setup, results table, figure, and findings; abstract and contributions updated to report both helix and torus validations; disulfide chapter (`suchanek26chapter`, Springer in press) now cited in the origin section and added as a full bibentry (replacing the TODO placeholder); unused `krizhevsky09` bibentry removed
- `docs/turtlend_arxiv/`: non-anonymized arXiv build of TurtleND (`turtlend_arxiv.tex`, `.bib`, figures, compiled `.pdf`) — Eric G. Suchanek + ORCID, ready for arXiv upload
- `docs/turtlend_tmlr/`: double-blind TMLR build of TurtleND (`turtlend_tmlr.tex`, `.bib`, `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, figures, compiled `.pdf`) — anonymous authors, TMLR-styled
- `docs/TURTLEND_SUBMISSION_PLAN.md`: dual-track submission plan (arXiv first, TMLR second) with OpenReview profile checklist, arXiv account setup, build/upload steps, and scope-protection notes for the future WaveRider JMLR flagship
- `src/waverider/arc/`: WaveRider-ARC Phase 0 package for the ARC Prize 2025 self-contained track. Thesis: a Riemannian refinement loop over a learned grid-transformation manifold replaces neural test-time training as the per-task ARC learner — zero LLM inference at test time, all learning happens offline on the 1000 public training tasks. Six modules: `task.py` (`Grid`/`Pair`/`Task` dataclasses + `load_task`/`load_task_dir` JSON loader, int8 grids, `None` outputs for private test pairs, `grids_equal` helper); `features.py` (deterministic O(H·W) `GridFeatures` — shape, palette, 10-bin histogram, background, non-background count, horizontal/vertical/180°/diagonal symmetry flags; no connected components yet, by design); `baselines.py` (three floor solvers — `IdentitySolver`, `ZerosLikeInputSolver`, `TrainOutputModeSolver`); `harness.py` (`Solver` Protocol, `evaluate()`, `score_task()`, `TaskResult`/`EvalReport` — ARC-AGI-2 rule: `max_attempts=2`, task correct iff every test input correct under any attempt, solver strictly blind to test outputs); `render.py` (24-bit ANSI terminal renderer for grids and tasks using the arcprize.org palette); `__init__.py` (public surface). Solvers come later; Phase 0 is the fixture for everything downstream
- `docs/ARC_TAXONOMY_V1.md`: hand-analyzed sample of 7 / 20 ARC-AGI-2 public eval tasks (seed 7699, random sample). Identifies 34 cognitive primitives and tag-frequency table; surfaces where the geometric stack should outperform current self-contained SOTA (CompressARC 4 %, TRM 8 %). 5 / 7 tasks hit CompressARC disabilities (variable-output-size, legend-driven recolor, topology-based rules). Drives the featurizer v2 requirements
- `tests/test_arc_harness.py`: pytest suite for the Phase 0 ARC package — loader round-trip, featurizer determinism and symmetry flags, `score_task` correctness under multi-attempt and multi-test-input cases, baseline solver behavior, `evaluate()` aggregation

### Changed
- `src/waverider/manifold_model.py`: **vectorized `fit()`** — replaced the per-node `np.linalg.norm(X - point, axis=1)` loop (O(N) Python calls with large per-call overhead) with chunked BLAS GEMM that expands `‖a−b‖² = ‖a‖² + ‖b‖² − 2·a·b` into one `(chunk × N)` matrix multiply per block. New `_fit_vectorized()` performs both phases (local-PCA geometry and manifold-aware edge construction) as two chunked passes; chunk size auto-tunes to keep a single distance block under ~256 MB. Edge construction also vectorizes the tangent projection as one `diffs @ basis[:d].T` matmul. Per-edge `Graph.add_edge` calls remain (graph API is the sequential bottleneck now, not the distance math)
- `src/waverider/manifold_model.py`: **vectorized `predict()`** — new `_predict_vectorized()` replaces the per-query loop that dominated scout wall-clock (profile showed 44% of total in predict, ~99% of which was serial `np.linalg.norm` / `argpartition` / `np.linalg.svd` calls at ~2.5 ms each of mostly Python→LAPACK dispatch overhead). Queries now flow through chunked GEMM distance blocks, batched `argpartition` for candidate pools, a single batched SVD for per-query local PCA, and a batched `einsum` for tangent projection. Per-query variance-threshold truncation is handled via a per-query active-component mask over the full projection tensor. Only graph-walk + majority vote stay per-query, and at that point each query is cheap dict-level work. ARC 100-task scout wall-clock on dev box (M5 Max) drops from 71.7 s → 41.7 s (42% faster); full v1 sweep (10/25/50/100/200) from ~15 min → **2m57s** (~5× faster); v2 `official_test` 200-task (87k train × 26k probes) from tesla's 366 s → **162.6 s** (2.25× faster). Accuracy drifts by ≤ 0.0004 at 100/200 tasks on `last_train` (2–10 predictions out of 12 800/25 600) from FP op-order changes in `argpartition` tie-breaks and batched-vs-sequential SVD sign conventions; `official_test` sweep is byte-identical at all five scales. Locked `locked_scale_sweep_v{1,2}.json` refreshed; v1 200-task accuracy 0.573 → 0.574 (favorable drift), v2 unchanged
- `.gitignore`: replaced narrow `helix` / `torus` whitelists under `benchmarks/canonical_tests/` with broad exceptions for `*_results.json`, `*_results.png`, `manifold_dim_probe*.png`, and `manifold_dim_probe*_arrays.npz`; added `benchmarks/arc_agi/data/` (upstream ARC corpora) and a `locked_*.json` whitelist under `benchmarks/arc_agi/eval_results/` for paper-grade scout numbers
- `benchmarks/arc_agi/arc_augment.py`: ruff cleanup — removed one extraneous `f` prefix on a placeholder-less f-string, reformatted
- `docs/ub_theorem/ub_theorem.tex` + `.pdf`: clarified the two coexisting d* estimators (per-class-max vs. global local-PCA) via a new footnote in §Background; dimension-probe section now documents subsample-size sensitivity (probe uses d*=16, main CIFAR-10 experiment uses d*=19) in a dedicated footnote; §Negative-Result intro rewritten to note its linear baselines use the global estimate; Table caption expanded to spell out which d* values drive which rows; abstract/contributions updated to list four datasets (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST)
- `docs/turtlend/disulfide_flight_results.png`, `benchmarks/disulfide_flight_results.png`: refreshed disulfide flight diagnostic
- `poetry.lock`: refreshed for Python 3.13 + latest TensorFlow 2.19 (neural group) to match tesla; no `pyproject.toml` version constraints changed

## [0.2.0] - 2026-04-10

### Added
- `benchmarks/canonical_tests/manifold_voxel_app.py`: Streamlit web UI for the manifold voxel pipeline — three view modes (Interactive 3D via Plotly WebGL with X/Y/Z slice planes, Static single-field PNG, Static all-fields 2×2 PNG grid); full sidebar for dataset, ManifoldModel params, PCA axis selection, voxel resolution, and display options; model fitted once and `@st.cache_data`-cached so visual parameter changes re-render without refitting
- `_add_nav_help()` in `manifold_voxel_viz.py`: orientation-cube widget (`add_camera_orientation_widget`) and keyboard-shortcut overlay (rotate/zoom/pan/slice/reset/screenshot) added to interactive PyVista renders

### Changed
- `benchmarks/canonical_tests/manifold_voxel_viz.py`: PCA arrow length now scaled by `sqrt(variance_ratio / variance_ratio[0])` with a floor of 0.30 — compresses dynamic range while preserving axis ordering and preventing low-variance axes from disappearing; arrow tip/shaft resolution raised to 32 for smoother geometry; point size and opacity increased in `render_single` (3 px / 0.4 → 8 px / 0.7) and `render_multi` (2 px / 0.3 → 6 px / 0.6) for better visibility; nav help injected into all interactive renders
- `docs/turtlend/turtlend.tex` + `turtlend.pdf`: made standalone for independent submission — softened forward references to WaveRider and UB Theorem; WaveRider and UB Theorem now framed as downstream applications of TurtleND rather than dependencies
- `docs/ub_theorem/ub_theorem.tex` + `ub_theorem.pdf`: added citations to TurtleND (for d* measurement primitive) and WaveRider (for broader manifold stack) in related work and bibliography; establishes the citation chain TurtleND → UB Theorem → WaveRider
- `papers/waverider_article/waverider_jmlr.tex`: TurtleND section (§3.1) compressed from full derivation to summary with `\cite{turtlend}`; UB Theorem section (§5.7) compressed from full re-derivation to summary with `\cite{ub_theorem}`; `ub_theorem` bibentry added — WaveRider now cites both upstream papers rather than re-deriving them
- `papers/manifold_classification/manifold_classification.bib`: added latest bibentry for the Springer paper
- `pyproject.toml`, `src/waverider/__init__.py`: version bumped to 0.2.0

### Added
- `PUBLICATION_PLAN.md`: full four-paper publication roadmap (TurtleND → Dimension Probe → UB Theorem → WaveRider Full Stack) with priority queue, submission targets, citation chain, and license policy (arXiv non-exclusive default)
- `docs/turtlend/turtlend.tex` + `turtlend.pdf`: TurtleND paper draft — N-dimensional generalisation of the Turtle3D primitive via Givens rotations; establishes the foundational primitive that all other WaveRider papers cite
- `docs/turtlend/TURTLEND_PLAN.md`: TurtleND paper plan with key content, target venue (TMLR), and arXiv category
- `docs/turtlend/waverider_approach.png`, `waverider_arch.png`: architecture figures for TurtleND paper
- `docs/ub_theorem/ub_theorem_arxiv.tar.gz`: arXiv submission package for UB Theorem paper (ready to upload, pending lede fix)
- `benchmarks/canonical_tests/cifar_architecture_sweep.py`: unified CIFAR-10/100 architecture sweep consolidating per-dataset scripts; adds **Class-Augmented PCA** architecture (PCA(d\*+C) → C, no hidden bottleneck) alongside the existing MLP suite; `--dataset`, `--only`, and `--plot` flags
- `benchmarks/canonical_tests/cfar-100.csv`, `cifar_10_probe.csv`: locked canonical benchmark result CSVs
- `.codekg/graph.sqlite`, `.codekg/lancedb/`: CodeKG semantic index for the waverider codebase

### Changed
- `UB_PAPER_PLAN.md`: streamlined to reflect current state — priority set to #3 (after TurtleND and Dimension Probe), lede problem documented, Fashion-MNIST results added, redundant checklists removed
- `src/model_builder.py`: added `build_class_augmented_pca()` (PCA(d\*+C) → Dense(C)) and `build_class_augmented_mlp()` — Class-Augmented PCA classifiers that explicitly allocate d\* geometry dims + C class dims with no Shannon bottleneck
- `benchmarks/canonical_tests/manifold_dim_probe.py`: added `--dataset cifar10|cifar100` flag; full CIFAR-100 class list (100 entries); per-dataset output filenames (`manifold_dim_probe_{cifar10|cifar100}_{results,arrays,png}`); machine/device/invocation metadata in JSON and plot titles; elapsed-time reporting; wider `tab20` colormap for 100-class plots
- `docs/ub_theorem/ub_theorem.tex` + `ub_theorem.pdf`: paper updated with Fashion-MNIST results and revised key-numbers table

### Added
- `CONSTITUTION.md`: project constitution — documents the Universal Bottleneck Theorem (w* = d* + C − 1), key files, benchmark rules (CPU vs Metal), and Spock persona; serves as the canonical orientation doc for new sessions
- `HANDOFF.md`: session handoff from Stardate 2026.093 — captures breakthrough results, key numbers, completed actions, and priority next steps for continuity across sessions
- `UB_PAPER_PLAN.md`: full submission plan for the standalone UB theorem paper targeting arXiv (≤ 2026-04-07), TMLR (rolling), and NeurIPS 2026; includes paper structure, abstract draft, figure checklist, and locked key numbers
- `benchmarks/canonical_tests/mnist_ub_phase_boundary.py`: MNIST phase boundary benchmark confirming the Whitney-dominated regime (C ≤ d*); ManifoldResNet-UB+Drop achieves 99.03% ± 0.03% — winner on accuracy, variance, and parameter efficiency simultaneously
- `waverider_missions/notes/stardate_2026_093_cifar100_ub_probe.md`: mission log — two-regime analysis (Whitney-dominated vs floor-dominated)
- `waverider_missions/notes/stardate_2026_093b_ub_pca_mlp.md`: mission log — UB-PCA-MLP negative control confirming convolutional inductive bias is load-bearing
- `waverider_missions/notes/stardate_2026_093c_ub_dropout_confirmation.md`: mission log — ManifoldResNet-UB+Dropout(0.3) achieves 72.1% ± 0.008 on CIFAR-10 (+7.9 pp over ResNet-32, 28% fewer params)
- `waverider_missions/notes/stardate_2026_094_mnist_phase_boundary.md`: mission log — MNIST Whitney-dominated regime confirmation

### Changed
- `src/model_builder.py`: added `build_ub_pca_mlp` and `build_universal_bottleneck_mlp`; added `dropout=0.3` to `build_manifold_resnet` — dropout is now part of the UB prescription (without it, UB overfits at 60 epochs)
- `benchmarks/canonical_tests/resnet_manifold_architecture.py`: added ManifoldResNet-UB and ManifoldResNet-UB+Drop architectures; summary block made defensive against cached-d vs current-d mismatch
- `benchmarks/canonical_tests/cifar100_resnet_manifold_architecture.py`: full rewrite for UB theorem validation in the high-C (floor-dominated) regime; added ManifoldResNet-UB (w*=118); added `--only` and `--plot-only` incremental-run flags
- `docs/waverider/article/waverider_jmlr.tex`: added Universal Bottleneck Theorem section, dimension probe mechanistic proof, MNIST phase boundary results, and two-regime analysis
- `docs/manifold_walker_spec/manifold_walker_spec.tex`: updated with UB theorem and proof sketch
- `waverider_missions/MISSIONS.md`: logged Stardates 2026.093 and 2026.094 experiments
- `waverider_missions/notes/stardate_2026_092_universal_bottleneck.md`: updated with final theorem statement
- `.gitignore`: added LaTeX intermediate file extensions (*.aux, *.bbl, *.blg, *.fdb_latexmk, *.fls, *.out, *.synctex.gz, *.toc, *.nav, *.snm, *.vrb)
- `.claude/settings.json`: added ls allowances for benchmark PNG and JSON outputs

- `src/waverider/manifold_optimizer.py`: `ManifoldAdam` — Keras Adam subclass that projects gradients onto the top-d PCA principal axes before each update, zeroing noise-dimension components; `make_basis()` helper extracts the projection matrix from a fitted sklearn PCA
- `benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py`: Tiny ImageNet benchmark (64×64×3, 12,288D, 200 classes) using the same three-phase manifold discovery → architecture comparison → evaluation pipeline as CIFAR-10/100; auto-downloads and caches dataset via `tensorflow_datasets`
- `docs/waverider/m5_max_performance_notes.md`: documented M5 Max performance findings — Metal TF plugin per-op sync overhead, CPU/Accelerate/AMX path advantages for small MLPs, and the eigvalsh→thin SVD speedup (~3700× fewer flops for local PCA)
- `src/waverider/graph_reasoner.py`: relocated from project root into the `waverider` package

### Changed
- `src/waverider/dimensionality_discovery.py`: replaced `np.linalg.eigvalsh` on (n_dims×n_dims) covariance matrix with `scipy.linalg.svd` thin decomposition on the (k×n_dims) neighbor matrix — O(k²n) vs O(n³), ~3700× fewer flops for CIFAR; introduced `_local_eigenvalues()` helper shared by both discovery functions
- `src/model_builder.py`: `build_standard_model` widened to canonical 1024→512; added optional `optimizer` parameter to `build_standard_model` and `build_manifold_model` to support `ManifoldAdam` injection; `build_wide_manifold_model` input shape fixed
- `benchmarks/canonical_tests/cifar100_manifold_architecture.py`: forced CPU (`CUDA_VISIBLE_DEVICES=""`); added `ManifoldAdam` and `Manifold + ManifoldAdam` architecture entries; upgraded plot to 5-panel layout with architecture schematics, timestamp, runtime, and right-side legend; saved `elapsed_s` and `d` in results JSON; fixed hardcoded `ylim` to auto-scale
- `benchmarks/canonical_tests/cifar10_manifold_architecture.py`: forced CPU; added `--samples-per-class` CLI arg (default 10); legend moved to Training Loss panel only
- `benchmarks/canonical_tests/iris_adam_vs_manifold.py`, `iris_manifold_adam_walker.py`, `iris_manifold_architecture.py`, `mnist_manifold_architecture.py`: forced CPU; removed `metal` detection; simplified `DEVICE_INFO`
- `pyproject.toml`: removed `tensorflow-metal` dependency (Metal per-op sync overhead hurts small MLPs; Accelerate/AMX via CPU path is faster on M-series)

## [0.1.0] - 2026-03-30

### Added
- `src/waverider/vector3D.py`: `Vector3D` class ported from proteusPy for use by `Turtle3D`
- `graph_reasoner.py`: `GraphReasoner` — heading-aware semantic reasoning engine that traverses knowledge graphs using a `TurtleND` navigator; supports beam search, lazy edge discovery, and cross-corpus bridging
- `README.md`: comprehensive project README covering algorithms, key benchmark results, quick-start, and theoretical background
- `LICENSE`: CC BY 4.0 license
- `.pre-commit-config.yaml`: full pre-commit pipeline — `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-merge-conflict`, `check-added-large-files`, `debug-statements`, `pylint`, `poetry-check`, `mypy`, `pytest`, `detect-secrets`, `ruff`, and `ruff-format`
- `.secrets.baseline`: initial `detect-secrets` baseline (no secrets found)
- `.vscode/settings.json`: VSCode pytest integration
- `poetry.lock` / `poetry.toml`: locked dependency manifest and in-project venv configuration
- `docs/graph_reasoner_spec/`: GraphReasoner design specification and summary
- `docs/manifold_observer/manifold_observer.md`: ManifoldObserver design document
- `docs/manifold_walker_spec/`: ManifoldWalker specification (Markdown + LaTeX)
- `docs/waverider/article/`: arXiv paper source, PDF, and figures
- `docs/waverider/*.md`: WaveRider overview, mission, stack summary, and infographic

### Changed
- `src/waverider/turtle3D.py`: restored original proteusPy provenance in module docstring; moved imports above `numpy.set_printoptions()` to fix E402; updated module header to CC 4.0 license
- `src/waverider/turtleND.py`: added `list[np.ndarray]` type annotation to `_tape` to satisfy mypy
- `src/waverider/manifold_walker.py`: added `list[tuple[np.ndarray, float]]` type annotation to `_history` to satisfy mypy
- `graph_reasoner.py`: moved module docstring before `from __future__ import annotations` and imports to fix E402; moved `__pdoc__` after imports
- `pyproject.toml`: commented out unresolvable `[tool.poetry.extras]` `kg` group; added mypy `[[overrides]]` to suppress type errors in legacy `turtle3D` and pre-fit Optional access in `manifold_model` / `manifold_observer`; ran `poetry lock` to regenerate lock file
- `benchmarks/canonical_tests/*.py`: ruff-format reformatting (trailing commas, one-item-per-line lists, multi-arg calls); updated module docstrings from proteusPy to WaveRider provenance; removed unused `sys` and `pathlib.Path` imports
- `waverider_missions/`: updated MISSIONS.md, CLAUDE_NOTES.md, GLOSSARY.md, chapter 3 narrative, and temporal momentum hypothesis notes

### Fixed
- All 14 pre-commit hooks now pass cleanly: ruff (6 unused imports auto-fixed), ruff-format (19 files reformatted), mypy (68 errors resolved via annotations and targeted overrides), poetry-check (stale extras removed, lock regenerated), detect-secrets (baseline created)
