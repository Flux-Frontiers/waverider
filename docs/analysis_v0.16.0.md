> **Analysis Report Metadata**  
> - **Generated:** 2026-09-21T23:11:56Z  
> - **Version:** pycode-kg 0.28.0  
> - **Commit:** 156b286 (main)  
> - **Index freshness:** [WARN] 11 uncommitted change(s) — the index may not reflect current file contents; line numbers and edge counts can drift. Re-run `pycodekg build` before trusting them.  
> - **Platform:** macOS 27.0 | arm64 (arm) | turing | Python 3.12.8  
> - **Graph:** 16520 nodes · 16103 edges (753 meaningful)  
> - **Included directories:** benchmarks, src  
> - **Excluded directories:** none  
> - **Elapsed time:** 9s  

# waverider Analysis

**Generated:** 2026-09-21 23:11:56 UTC

---

## Executive Summary

This report provides a comprehensive architectural analysis of the **waverider** repository using PyCodeKG's knowledge graph. The analysis covers complexity hotspots, module coupling, key call chains, and code quality signals to guide refactoring and architecture decisions.

| Overall Quality | Grade | Score |
| :--- | :--- | :--- |
| [D] **Needs Work** | **D** | 55.3 / 100 |

Score components:

| Component | Points | Max | Basis |
| :--- | ---: | ---: | :--- |
| Docstring coverage | 30.0 | 40 | 67.5% documented (full marks at 90%) |
| Dead code | 6.3 | 25 | 26 candidates / 695 definitions scanned (3.7%; zero points at 5%) |
| High fan-out | 4.0 | 20 | 4 orchestrator(s); −4 pts each |
| Circular dependencies | 15.0 | 15 | 0 cycle(s); −5 pts each |

---

## Baseline Metrics

| Metric | Value |
| :--- | :--- |
| **Total Nodes** | 16520 |
| **Total Edges** | 16103 |
| **Modules** | 58 (of 58 total) |
| **Functions** | 416 |
| **Classes** | 44 |
| **Methods** | 235 |

### Edge Distribution

| Relationship Type | Count |
| :--- | ---: |
| CALLS | 6394 |
| CONTAINS | 695 |
| IMPORTS | 676 |
| ATTR_ACCESS | 5516 |
| INHERITS | 19 |

_Excludes 2,803 `RESOLVES_TO` edges: internal symbol-stub resolutions, not relationships between two pieces of code. This table therefore does not sum to Total Edges._

---

## Fan-In Ranking

Most-called functions and methods — potential bottlenecks or core functionality.  Classes are omitted: instantiation counts are not architectural fan-in.

| # | Kind | Function | Module | Callers |
| ---: | :--- | :--- | ---: | :--- |
| 1 | method | `fit()` | src/waverider/backbone_embedder.py | **41** |
| 2 | method | `fit()` | src/waverider/geodesic_coords.py | **41** |
| 3 | method | `fit()` | src/waverider/universal_embedder.py | **41** |
| 4 | function | `_require_viz()` | src/waverider/voxel_viz.py | **14** |
| 5 | method | `predict()` | src/waverider/manifold_model.py | **13** |
| 6 | method | `observe()` | src/waverider/manifold_observer.py | **6** |
| 7 | function | `_probe_indices()` | src/waverider/dimensionality_profile.py | **5** |
| 8 | function | `_distance_rows()` | src/waverider/dimensionality_profile.py | **5** |
| 9 | method | `lift_data()` | src/waverider/manifold_observer.py | **5** |
| 10 | method | `node_ids()` | src/waverider/graph_reasoner.py | **4** |
| 11 | function | `_subsample()` | src/waverider/voxel_viz.py | **4** |
| 12 | function | `_compose_ct_scene()` | src/waverider/voxel_viz.py | **4** |
| 13 | function | `_compose_tvb_scene()` | src/waverider/voxel_viz.py | **4** |

**Insight:** Functions with high fan-in are either core APIs or bottlenecks. Review these for:

- Thread safety and performance
- Clear documentation and contracts
- Potential for breaking changes

---

## High Fan-Out Functions (Orchestrators)

Functions that call many others may indicate complex orchestration logic or poor separation of concerns.  Only repo-internal callees are counted — stdlib and third-party calls are not orchestration.

| # | Function | Module | Internal Calls | Type |
| ---: | :--- | :--- | ---: | :--- |
| 1 | `__init__()` | benchmarks/canonical_tests/cifar100_manifold_architecture.py | **28** | Coordinator |
| 2 | `__init__()` | benchmarks/canonical_tests/cifar10_manifold_architecture.py | **28** | Coordinator |
| 3 | `__init__()` | src/waverider/manifold_optimizer.py | **28** | Coordinator |
| 4 | `__init__()` | src/waverider/manifold_walker.py | **28** | Coordinator |

---

## Module Architecture

Top modules by dependency coupling and cohesion (showing 10 of 58 with activity).
Cohesion = incoming / (incoming + outgoing + 1); higher = more internally focused.  Modules with no in-repo callers are externally driven (MCP router, CLI, GUI event loop) — their 0.00 cohesion is expected, not a coupling problem.

| Module | Functions | Classes | Incoming | Outgoing | Cohesion | Note |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `src/waverider/graph_reasoner.py` | 3 | 12 | 1 | 0 | 0.50 |  |
| `src/waverider/voxel_viz.py` | 38 | 2 | 3 | 8 | 0.25 |  |
| `src/waverider/manifold_model.py` | 0 | 4 | 20 | 0 | 0.95 |  |
| `benchmarks/canonical_tests/iris_adam_vs_manifold.py` | 14 | 4 | 4 | 2 | 0.57 |  |
| `src/waverider/backbone_angles.py` | 3 | 2 | 6 | 1 | 0.75 |  |
| `benchmarks/canonical_tests/iris_manifold_adam_walker.py` | 13 | 3 | 4 | 2 | 0.57 |  |
| `src/waverider/manifold_observer.py` | 0 | 2 | 6 | 1 | 0.75 |  |
| `benchmarks/canonical_tests/manifold_voxel_viz.py` | 22 | 2 | 2 | 8 | 0.18 |  |
| `src/model_builder.py` | 23 | 0 | 11 | 0 | 0.92 |  |
| `src/waverider/manifold_walker.py` | 0 | 2 | 3 | 0 | 0.75 |  |

---

## Key Call Chains

Deepest call chains in the codebase.

**Chain 1** (depth: 6)

```
backbone_embedder.py:fit_transform → backbone_embedder.py:fit → backbone_angles.py:to_combined_codes → backbone_angles.py:combined_code → backbone_angles.py:phi_bin → backbone_angles.py:quantize_angle
```

**Chain 2** (depth: 4)

```
universal_embedder.py:fit_transform → universal_embedder.py:fit → universal_embedder.py:_build_tangent_frames → universal_embedder.py:_padded_basis
```

**Chain 3** (depth: 4)

```
manifold_model.py:score → manifold_model.py:predict → manifold_model.py:_predict_vectorized → manifold_model.py:_gather_graph_neighbors
```

---

## Public API Surface

Definitions re-exported from an `__init__.py` or otherwise reachable as public entry points, ranked by fan-in.  Top 10 of 56 shown.

| Name | Module | Fan-In | Kind |
| :--- | :--- | ---: | :--- |
| `ManifoldModel` | src/waverider/manifold_model.py | 18 | class |
| `BackboneAngleList` | src/waverider/backbone_angles.py | 7 | class |
| `ManifoldObserver` | src/waverider/manifold_observer.py | 5 | class |
| `voxelize()` | src/waverider/voxel_viz.py | 5 | function |
| `BackboneEmbedder` | src/waverider/backbone_embedder.py | 4 | class |
| `BackboneResidue` | src/waverider/backbone_angles.py | 4 | class |
| `ReasoningPath` | src/waverider/graph_reasoner.py | 4 | class |
| `build_grid()` | src/waverider/voxel_viz.py | 4 | function |
| `fit_and_observe()` | src/waverider/voxel_viz.py | 4 | function |
| `render_multi()` | src/waverider/voxel_viz.py | 4 | function |

---

## Docstring Coverage

Docstring coverage directly determines semantic retrieval quality. Nodes without docstrings embed only structured identifiers (`KIND/NAME/QUALNAME/MODULE`), where keyword search is as effective as vector embeddings. The semantic model earns its value only when a docstring is present.

| Kind | Documented | Total | Coverage |
| :--- | ---: | ---: | :--- |
| `function` | 254 | 416 | [WARN] 61.1% |
| `method` | 156 | 235 | [WARN] 66.4% |
| `class` | 41 | 44 | [OK] 93.2% |
| `module` | 57 | 58 | [OK] 98.3% |
| **total** | **508** | **753** | **[WARN] 67.5%** |

> **Recommendation:** 245 nodes lack docstrings. Prioritize documenting high-fan-in functions and public API surface first — these have the highest impact on query accuracy.

---

## Structural Importance Ranking (SIR)

Weighted PageRank aggregated by module — reveals architectural spine. Cross-module edges boosted 1.5×; private symbols penalized 0.85×. Node-level detail: `pycodekg centrality --top 25`

| Rank | Score | Members | Module |
| ---: | ---: | ---: | :--- |
| 1 | 0.093012 | 34 | `src/waverider/backbone_angles.py` |
| 2 | 0.090163 | 39 | `src/waverider/manifold_model.py` |
| 3 | 0.075562 | 59 | `src/waverider/graph_reasoner.py` |
| 4 | 0.068464 | 41 | `src/waverider/voxel_viz.py` |
| 5 | 0.041624 | 19 | `src/waverider/universal_embedder.py` |
| 6 | 0.032117 | 26 | `src/waverider/manifold_observer.py` |
| 7 | 0.031000 | 12 | `src/waverider/backbone_embedder.py` |
| 8 | 0.029974 | 23 | `src/waverider/manifold_walker.py` |
| 9 | 0.028916 | 10 | `src/waverider/geodesic_coords.py` |
| 10 | 0.021575 | 15 | `src/waverider/dimensionality_profile.py` |
| 11 | 0.015657 | 6 | `src/waverider/dimensionality_discovery.py` |
| 12 | 0.015592 | 7 | `src/waverider/manifold_optimizer.py` |
| 13 | 0.014886 | 11 | `src/waverider/manifold_reach.py` |
| 14 | 0.005034 | 5 | `src/waverider/backbone_manifold.py` |
| 15 | 0.000793 | 1 | `src/waverider/__init__.py` |

---

## Code Quality Issues

- [WARN] Moderate docstring coverage (67.5%) — semantic retrieval quality is degraded for undocumented nodes; BM25 is as effective as embeddings without docstrings
- [WARN] 26 dead-code candidates found (`manifold_observer.py:observe_path`, `manifold_model.py:_predict_single`, `model_builder.py:build_manifold_resnet_nc`, `model_builder.py:build_class_augmented_resnet`, `tf_setup.py:setup_tensorflow`, `manifold_model.py:_compute_local_geometry`, `manifold_model.py:_build_manifold_edges`, `backbone_angles.py:ramachandran_plot` and 18 more) -- no callers in code or tests; verify against downstream consumers, then remove or archive
- [INFO] 13 definitions are unused in production code but exercised by tests -- likely public API for downstream packages; not counted against the quality grade
- [WARN] 4 functions with high fan-out -- potential orchestrators or god objects
- [WARN] `graph_reasoner.py` has 58 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `voxel_viz.py` has 40 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `manifold_model.py` has 38 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `iris_adam_vs_manifold.py` has 35 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `backbone_angles.py` has 33 functions/methods/classes -- consider splitting into focused submodules

---

## Architectural Strengths

- Well-structured with 15 core functions identified

---

## Recommendations

### Immediate Actions
1. **Improve docstring coverage** — 245 nodes lack docstrings; prioritize high-fan-in functions and public APIs first for maximum semantic retrieval gain
2. **Triage dead-code candidates** — `observe_path`, `_predict_single`, `build_manifold_resnet_nc`, `build_class_augmented_resnet`, `setup_tensorflow` (and 21 more) have zero callers in code and tests; confirm no downstream package consumes them, then remove
3. **Refactor high fan-out orchestrators** — `__init__` calls 28 functions; consider splitting into smaller, focused coordinators

### Medium-term Refactoring
1. **Harden high fan-in functions** — `fit`, `fit`, `fit` are widely depended upon; review for thread safety, clear contracts, and stable interfaces
2. **Reduce module coupling** — consider splitting tightly coupled modules or introducing interface boundaries
3. **Add tests for key call chains** — the identified call chains represent well-traveled execution paths that benefit most from regression coverage

### Long-term Architecture
1. **Version and stabilize the public API** — document breaking-change policies for `ManifoldModel`, `BackboneAngleList`, `ManifoldObserver`
2. **Enforce layer boundaries** — add linting or CI checks to prevent unexpected cross-module dependencies as the codebase grows
3. **Monitor hot paths** — instrument the high fan-in functions identified here to catch performance regressions early

---

## Inheritance Hierarchy

**19** INHERITS edges across **21** classes. Max depth: **1**.

| Class | Module | Depth | Parents | Children |
| :--- | :--- | ---: | ---: | ---: |
| `EigenWeightedManifoldKNN` | benchmarks/canonical_tests/digits_manifold_knn.py | 1 | 1 | 0 |
| `DirectedDiscoverer` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `ExplorationSteering` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `GradientSteering` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `KNNDiscoverer` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `RadiusDiscoverer` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `TargetSteering` | src/waverider/graph_reasoner.py | 1 | 1 | 0 |
| `ManifoldAdamWalker` | src/waverider/manifold_walker.py | 1 | 1 | 0 |
| `EpochHeartbeat` | benchmarks/canonical_tests/cifar100_manifold_architecture.py | 0 | 1 | 0 |
| `EpochHeartbeat` | benchmarks/canonical_tests/cifar10_manifold_architecture.py | 0 | 1 | 0 |
| `ManifoldKNN` | benchmarks/canonical_tests/digits_manifold_knn.py | 0 | 0 | 1 |
| `PCAInfo` | benchmarks/canonical_tests/manifold_voxel_viz.py | 0 | 1 | 0 |
| `PointField` | benchmarks/canonical_tests/manifold_voxel_viz.py | 0 | 1 | 0 |
| `_ThrottledProgbar` | benchmarks/canonical_tests/mnist_ub_phase_boundary.py | 0 | 1 | 0 |
| `PeakClampingCallback` | benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py | 0 | 1 | 0 |
| `EdgeDiscoverer` | src/waverider/graph_reasoner.py | 0 | 1 | 3 |
| `SteeringStrategy` | src/waverider/graph_reasoner.py | 0 | 1 | 3 |
| `ManifoldAdam` | src/waverider/manifold_optimizer.py | 0 | 1 | 0 |
| `ManifoldWalker` | src/waverider/manifold_walker.py | 0 | 0 | 1 |
| `PCAInfo` | src/waverider/voxel_viz.py | 0 | 1 | 0 |

---

## Snapshot History

No snapshots found. Run `pycodekg snapshot save <version>` to capture one.

---

## Orphaned Code

26 definitions have no callers in code or tests (dead-code candidates).  Framework-dispatched entry points — dunder/protocol methods, properties, Click commands, MCP tools, `ast.NodeVisitor` dispatch, SDK protocol overrides, console scripts, `__main__` guards — are already excluded.  Top 15 of 26 by size shown.

| Name | Kind | Module | Lines |
| :--- | :--- | :--- | ---: |
| `observe_path()` | method | src/waverider/manifold_observer.py | 141 |
| `_predict_single()` | method | src/waverider/manifold_model.py | 71 |
| `build_manifold_resnet_nc()` | function | src/model_builder.py | 63 |
| `build_class_augmented_resnet()` | function | src/model_builder.py | 62 |
| `setup_tensorflow()` | function | benchmarks/tf_setup.py | 51 |
| `_compute_local_geometry()` | method | src/waverider/manifold_model.py | 50 |
| `_build_manifold_edges()` | method | src/waverider/manifold_model.py | 49 |
| `ramachandran_plot()` | method | src/waverider/backbone_angles.py | 48 |
| `build_class_augmented_mlp()` | function | src/model_builder.py | 43 |
| `build_lda_pca_augmented()` | function | src/model_builder.py | 41 |
| `sync()` | method | src/waverider/manifold_observer.py | 36 |
| `build_pca_c_expand_c()` | function | src/model_builder.py | 34 |
| `topology_summary()` | method | src/waverider/manifold_observer.py | 34 |
| `build_pca_mlp_deep()` | function | src/model_builder.py | 30 |
| `build_pca_ub_deep()` | function | src/model_builder.py | 30 |

13 further definitions are unused in production code but exercised by tests — likely public API consumed by downstream packages.  Not counted against the quality grade; review for intentional export.

| Name | Kind | Module | Lines |
| :--- | :--- | :--- | ---: |
| `beam_reason()` | method | src/waverider/graph_reasoner.py | 90 |
| `walk()` | method | src/waverider/manifold_walker.py | 45 |
| `reason_toward()` | method | src/waverider/graph_reasoner.py | 42 |
| `reason()` | method | src/waverider/graph_reasoner.py | 39 |
| `probe()` | method | src/waverider/manifold_walker.py | 25 |
| `discover_neighbors()` | method | src/waverider/graph_reasoner.py | 16 |
| `fly_to_nearest()` | method | src/waverider/manifold_model.py | 16 |
| `get_neighbors()` | method | src/waverider/manifold_model.py | 15 |
| `add_edge()` | method | src/waverider/manifold_model.py | 8 |
| `copy()` | method | src/waverider/graph_reasoner.py | 7 |
| `reset_flight()` | method | src/waverider/manifold_model.py | 3 |
| `add_edge()` | method | src/waverider/graph_reasoner.py | 2 |
| `get_node()` | method | src/waverider/graph_reasoner.py | 2 |

13 further methods override a base class outside the indexed graph — the caller may live in that dependency, so these cannot be judged dead or alive.  Not counted against the quality grade.

| Name | Kind | Module | Lines |
| :--- | :--- | :--- | ---: |
| `on_epoch_end()` | method | benchmarks/canonical_tests/cifar10_manifold_architecture.py | 22 |
| `score_candidate()` | method | src/waverider/graph_reasoner.py | 13 |
| `on_epoch_end()` | method | benchmarks/canonical_tests/cifar100_manifold_architecture.py | 12 |
| `score_candidate()` | method | src/waverider/graph_reasoner.py | 12 |
| `on_epoch_end()` | method | benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py | 11 |
| `score_candidate()` | method | src/waverider/graph_reasoner.py | 11 |
| `score_candidate()` | method | src/waverider/graph_reasoner.py | 9 |
| `on_train_batch_end()` | method | benchmarks/canonical_tests/mnist_ub_phase_boundary.py | 7 |
| `on_train_end()` | method | benchmarks/canonical_tests/tiny_imagenet_manifold_architecture.py | 5 |
| `on_train_begin()` | method | benchmarks/canonical_tests/cifar10_manifold_architecture.py | 3 |
| `on_epoch_begin()` | method | benchmarks/canonical_tests/mnist_ub_phase_boundary.py | 3 |
| `on_train_begin()` | method | benchmarks/canonical_tests/cifar100_manifold_architecture.py | 2 |
| `on_epoch_end()` | method | benchmarks/canonical_tests/mnist_ub_phase_boundary.py | 2 |

---

## CodeRank — Global Structural Importance

Weighted PageRank over CALLS + IMPORTS + INHERITS edges (test paths excluded). Scores are normalized to sum to 1.0. This ranking seeds fan-in discovery and the concern queries below.  Top 20 of 25 shown.

| Rank | Score | Kind | Name | Module |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.000200 | function | `quantize_angle()` | src/waverider/backbone_angles.py |
| 2 | 0.000168 | function | `_require_viz()` | src/waverider/voxel_viz.py |
| 3 | 0.000142 | class | `BackboneAngleList` | src/waverider/backbone_angles.py |
| 4 | 0.000140 | method | `KnowledgeGraph.node_ids()` | src/waverider/graph_reasoner.py |
| 5 | 0.000128 | method | `ManifoldObserver._compute_curvature()` | src/waverider/manifold_observer.py |
| 6 | 0.000125 | class | `ReasoningPath` | src/waverider/graph_reasoner.py |
| 7 | 0.000108 | method | `ManifoldModel.n_nodes()` | src/waverider/manifold_model.py |
| 8 | 0.000108 | method | `GraphReasoner.current_node()` | src/waverider/graph_reasoner.py |
| 9 | 0.000108 | function | `_subsample()` | src/waverider/voxel_viz.py |
| 10 | 0.000106 | method | `ManifoldObserver.locate()` | src/waverider/manifold_observer.py |
| 11 | 0.000105 | method | `ManifoldWalker._local_pca()` | src/waverider/manifold_walker.py |
| 12 | 0.000104 | method | `ManifoldObserver.observe()` | src/waverider/manifold_observer.py |
| 13 | 0.000103 | method | `BackboneEmbedder._aa_features()` | src/waverider/backbone_embedder.py |
| 14 | 0.000101 | method | `BackboneEmbedder._aa_dim()` | src/waverider/backbone_embedder.py |
| 15 | 0.000101 | function | `_rng()` | src/waverider/manifold_reach.py |
| 16 | 0.000101 | method | `ReasoningPath.total_score()` | src/waverider/graph_reasoner.py |
| 17 | 0.000099 | function | `_rng()` | src/waverider/dimensionality_profile.py |
| 18 | 0.000097 | method | `UniversalEmbedder._padded_basis()` | src/waverider/universal_embedder.py |
| 19 | 0.000097 | method | `UniversalEmbedder._sign_correct()` | src/waverider/universal_embedder.py |
| 20 | 0.000097 | method | `ManifoldModel._fit_vectorized_inner()` | src/waverider/manifold_model.py |

---

## Concern-Based Hybrid Ranking

Top structurally-dominant nodes per architectural concern (0.60 × semantic + 0.25 × CodeRank + 0.15 × graph proximity).

### Configuration Loading Initialization Setup

| Rank | Score | Kind | Name | Module |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.7259 | method | `EpochHeartbeat.__init__()` | benchmarks/canonical_tests/cifar100_manifold_architecture.py |
| 2 | 0.7246 | method | `EpochHeartbeat.__init__()` | benchmarks/canonical_tests/cifar10_manifold_architecture.py |
| 3 | 0.7244 | function | `_setup_tf()` | benchmarks/canonical_tests/estimator_calibration.py |
| 4 | 0.7234 | function | `setup_tensorflow()` | benchmarks/tf_setup.py |
| 5 | 0.7213 | method | `UniversalEmbedder.__init__()` | src/waverider/universal_embedder.py |

### Data Persistence Storage Database

| Rank | Score | Kind | Name | Module |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.7687 | function | `_load_ucimlrepo()` | benchmarks/canonical_tests/clinical/disease_manifold_architecture.py |
| 2 | 0.7509 | function | `run_trial()` | benchmarks/canonical_tests/torus_manifold_observer.py |
| 3 | 0.7447 | function | `run_trial()` | benchmarks/canonical_tests/helix_manifold_observer.py |
| 4 | 0.7413 | function | `plot_results()` | benchmarks/canonical_tests/digits_manifold_architecture.py |
| 5 | 0.7409 | method | `StandaloneManifoldAdam._save_adam_state()` | benchmarks/canonical_tests/iris_manifold_adam_walker.py |

### Query Search Retrieval Semantic

| Rank | Score | Kind | Name | Module |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.747 | function | `_usable_gap()` | benchmarks/canonical_tests/manifold_reach_embeddings.py |
| 2 | 0.7469 | function | `main()` | benchmarks/canonical_tests/manifold_reach_embeddings.py |
| 3 | 0.7463 | method | `ManifoldModel._predict_single()` | src/waverider/manifold_model.py |
| 4 | 0.7453 | method | `KnowledgeGraph.__contains__()` | src/waverider/graph_reasoner.py |
| 5 | 0.7452 | method | `ManifoldKNN.fit()` | benchmarks/canonical_tests/digits_manifold_knn.py |

### Graph Traversal Node Edge

| Rank | Score | Kind | Name | Module |
| ---: | ---: | :--- | :--- | :--- |
| 1 | 0.7516 | method | `ManifoldModel._gather_graph_neighbors()` | src/waverider/manifold_model.py |
| 2 | 0.7497 | method | `GraphReasoner.step()` | src/waverider/graph_reasoner.py |
| 3 | 0.7428 | method | `KnowledgeGraph.discover_neighbors()` | src/waverider/graph_reasoner.py |
| 4 | 0.7343 | method | `ManifoldModel.get_neighbors()` | src/waverider/manifold_model.py |
| 5 | 0.7338 | method | `KnowledgeGraph.add_discoverer()` | src/waverider/graph_reasoner.py |

---

*Report generated by PyCodeKG Thorough Analysis Tool — analysis completed in 9.9s*
