> **Analysis Report Metadata**  
> - **Generated:** 2026-08-04T15:33:55Z  
> - **Version:** pycode-kg 0.19.2  
> - **Commit:** 2de0d32 (main)  
> - **Platform:** macOS 27.0 | arm64 (arm) | turing | Python 3.12.13  
> - **Graph:** 16275 nodes · 16338 edges (798 meaningful)  
> - **Included directories:** benchmarks, src  
> - **Excluded directories:** none  
> - **Elapsed time:** 8s  

# waverider Analysis

**Generated:** 2026-08-04 15:33:55 UTC

---

## Executive Summary

This report provides a comprehensive architectural analysis of the **waverider** repository using PyCodeKG's knowledge graph. The analysis covers complexity hotspots, module coupling, key call chains, and code quality signals to guide refactoring and architecture decisions.

| Overall Quality | Grade | Score |
|----------------|-------|-------|
| [C] **Fair** | **C** | 70 / 100 |

---

## Baseline Metrics

| Metric | Value |
|--------|-------|
| **Total Nodes** | 16275 |
| **Total Edges** | 16338 |
| **Modules** | 55 (of 55 total) |
| **Functions** | 375 |
| **Classes** | 48 |
| **Methods** | 320 |

### Edge Distribution

| Relationship Type | Count |
|-------------------|-------|
| CALLS | 6215 |
| CONTAINS | 743 |
| IMPORTS | 597 |
| ATTR_ACCESS | 5618 |
| INHERITS | 19 |

---

## Fan-In Ranking

Most-called functions are potential bottlenecks or core functionality. These functions are heavily depended upon across the codebase.

| # | Function | Module | Callers |
|---|----------|--------|---------|
| 1 | `copy()` | src/waverider/vector3D.py | **62** |
| 2 | `fit()` | src/waverider/backbone_embedder.py | **41** |
| 3 | `fit()` | src/waverider/geodesic_coords.py | **41** |
| 4 | `fit()` | src/waverider/universal_embedder.py | **41** |
| 5 | `Vector3D()` | src/waverider/vector3D.py | **24** |
| 6 | `predict()` | benchmarks/canonical_tests/digits_manifold_knn.py | **12** |
| 7 | `predict()` | src/waverider/manifold_model.py | **12** |
| 8 | `_require_viz()` | src/waverider/voxel_viz.py | **10** |
| 9 | `BackboneAngleList()` | src/waverider/backbone_angles.py | **7** |
| 10 | `_hr()` | benchmarks/canonical_tests/protein_backbone_manifold.py | **7** |
| 11 | `build_manifold_resnet()` | src/model_builder.py | **7** |
| 12 | `_rotate()` | src/waverider/turtleND.py | **6** |
| 13 | `_compile()` | benchmarks/canonical_tests/mnist_manifold_architecture.py | **6** |
| 14 | `_scale()` | benchmarks/canonical_tests/clinical/gen_voxel_viz.py | **6** |
| 15 | `__init__()` | src/waverider/turtle3D.py | **6** |


**Insight:** Functions with high fan-in are either core APIs or bottlenecks. Review these for:
- Thread safety and performance
- Clear documentation and contracts
- Potential for breaking changes

---

## High Fan-Out Functions (Orchestrators)

Functions that call many others may indicate complex orchestration logic or poor separation of concerns.

No extreme high fan-out functions detected. Well-balanced architecture.

---

## Module Architecture

Top modules by dependency coupling and cohesion (showing up to 10 with activity).
Cohesion = incoming / (incoming + outgoing + 1); higher = more internally focused.

| Module | Functions | Classes | Incoming | Outgoing | Cohesion |
|--------|-----------|---------|----------|----------|----------|
| `src/waverider/graph_reasoner.py` | 3 | 12 | 1 | 1 | 0.33 |
| `src/waverider/manifold_model.py` | 0 | 4 | 20 | 1 | 0.91 |
| `benchmarks/canonical_tests/iris_adam_vs_manifold.py` | 14 | 4 | 4 | 2 | 0.57 |
| `src/waverider/voxel_viz.py` | 32 | 2 | 3 | 9 | 0.23 |
| `src/waverider/backbone_angles.py` | 3 | 2 | 6 | 2 | 0.67 |
| `src/waverider/turtleND.py` | 0 | 1 | 5 | 0 | 0.83 |
| `src/waverider/turtle3D.py` | 0 | 1 | 0 | 1 | 0.00 |
| `benchmarks/canonical_tests/iris_manifold_adam_walker.py` | 13 | 3 | 4 | 2 | 0.57 |
| `src/waverider/vector3D.py` | 5 | 1 | 2 | 0 | 0.67 |
| `src/waverider/manifold_observer.py` | 0 | 2 | 6 | 2 | 0.67 |

---

## Key Call Chains

Deepest call chains in the codebase.

**Chain 1** (depth: 6)

```
fit_transform → fit → to_combined_codes → combined_code → phi_bin → quantize_angle
```

**Chain 2** (depth: 5)

```
fit_transform → fit → _select_anchors → copy → ReasoningPath
```

**Chain 3** (depth: 6)

```
fit_transform → fit → _build_tangent_frames → _padded_basis → copy → ReasoningPath
```

---

## Public API Surface

Identified public APIs (module-level functions with high usage).

| Function | Module | Fan-In | Type |
|----------|--------|--------|------|
| `Vector3D()` | src/waverider/vector3D.py | 24 | class |
| `ManifoldModel()` | src/waverider/manifold_model.py | 18 | class |
| `BackboneAngleList()` | src/waverider/backbone_angles.py | 7 | class |
| `build_manifold_resnet()` | src/model_builder.py | 7 | function |
| `ManifoldObserver()` | src/waverider/manifold_observer.py | 5 | class |
| `voxelize()` | src/waverider/voxel_viz.py | 5 | function |
| `TurtleND()` | src/waverider/turtleND.py | 5 | class |
| `ReasoningPath()` | src/waverider/graph_reasoner.py | 4 | class |
| `BackboneResidue()` | src/waverider/backbone_angles.py | 4 | class |
| `load_iris()` | benchmarks/canonical_tests/iris_manifold_adam_walker.py | 4 | function |
---

## Docstring Coverage

Docstring coverage directly determines semantic retrieval quality. Nodes without
docstrings embed only structured identifiers (`KIND/NAME/QUALNAME/MODULE`), where
keyword search is as effective as vector embeddings. The semantic model earns its
value only when a docstring is present.

| Kind | Documented | Total | Coverage |
|------|-----------|-------|----------|
| `function` | 233 | 375 | [WARN] 62.1% |
| `method` | 230 | 320 | [WARN] 71.9% |
| `class` | 45 | 48 | [OK] 93.8% |
| `module` | 54 | 55 | [OK] 98.2% |
| **total** | **562** | **798** | **[WARN] 70.4%** |

> **Recommendation:** 236 nodes lack docstrings. Prioritize documenting high-fan-in functions and public API surface first — these have the highest impact on query accuracy.

---

## Structural Importance Ranking (SIR)

Weighted PageRank aggregated by module — reveals architectural spine. Cross-module edges boosted 1.5×; private symbols penalized 0.85×. Node-level detail: `pycodekg centrality --top 25`

| Rank | Score | Members | Module |
|------|-------|---------|--------|
| 1 | 0.152628 | 27 | `src/waverider/vector3D.py` |
| 2 | 0.101946 | 59 | `src/waverider/graph_reasoner.py` |
| 3 | 0.071880 | 34 | `src/waverider/backbone_angles.py` |
| 4 | 0.062512 | 39 | `src/waverider/manifold_model.py` |
| 5 | 0.045885 | 35 | `src/waverider/voxel_viz.py` |
| 6 | 0.036764 | 32 | `src/waverider/turtleND.py` |
| 7 | 0.035140 | 25 | `benchmarks/canonical_tests/manifold_voxel_viz.py` |
| 8 | 0.031392 | 19 | `src/waverider/universal_embedder.py` |
| 9 | 0.030086 | 36 | `benchmarks/canonical_tests/iris_adam_vs_manifold.py` |
| 10 | 0.028201 | 12 | `src/waverider/backbone_embedder.py` |
| 11 | 0.024466 | 31 | `src/waverider/turtle3D.py` |
| 12 | 0.024212 | 24 | `src/model_builder.py` |
| 13 | 0.023693 | 28 | `benchmarks/canonical_tests/iris_manifold_adam_walker.py` |
| 14 | 0.023644 | 26 | `src/waverider/manifold_observer.py` |
| 15 | 0.021130 | 10 | `src/waverider/geodesic_coords.py` |



---

## Code Quality Issues

- [WARN] Moderate docstring coverage (70.4%) — semantic retrieval quality is degraded for undocumented nodes; BM25 is as effective as embeddings without docstrings
- [WARN] 1 orphaned functions found (`main`) -- consider archiving or documenting
- [WARN] `graph_reasoner.py` has 58 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `manifold_model.py` has 38 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `iris_adam_vs_manifold.py` has 35 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `voxel_viz.py` has 34 functions/methods/classes -- consider splitting into focused submodules
- [WARN] `backbone_angles.py` has 33 functions/methods/classes -- consider splitting into focused submodules

---

## Architectural Strengths

- Well-structured with 15 core functions identified
- No god objects or god functions detected

---

## Recommendations

### Immediate Actions
1. **Improve docstring coverage** — 236 nodes lack docstrings; prioritize high-fan-in functions and public APIs first for maximum semantic retrieval gain
2. **Remove or archive orphaned functions** — `main` have zero callers and add maintenance burden

### Medium-term Refactoring
1. **Harden high fan-in functions** — `copy`, `fit`, `fit` are widely depended upon; review for thread safety, clear contracts, and stable interfaces
2. **Reduce module coupling** — consider splitting tightly coupled modules or introducing interface boundaries
3. **Add tests for key call chains** — the identified call chains represent well-traveled execution paths that benefit most from regression coverage

### Long-term Architecture
1. **Version and stabilize the public API** — document breaking-change policies for `Vector3D`, `ManifoldModel`, `BackboneAngleList`
2. **Enforce layer boundaries** — add linting or CI checks to prevent unexpected cross-module dependencies as the codebase grows
3. **Monitor hot paths** — instrument the high fan-in functions identified here to catch performance regressions early

---

## Inheritance Hierarchy

**19** INHERITS edges across **21** classes. Max depth: **1**.

| Class | Module | Depth | Parents | Children |
|-------|--------|-------|---------|----------|
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

## Appendix: Orphaned Code

Functions with zero callers (potential dead code):

| Function | Module | Lines |
|----------|--------|-------|
| `main()` | benchmarks/canonical_tests/cifar_architecture_sweep.py | 458 |
---

## CodeRank -- Global Structural Importance

Weighted PageRank over CALLS + IMPORTS + INHERITS edges (test paths excluded). Scores are normalized to sum to 1.0. This ranking seeds Phase 2 fan-in discovery and Phase 15 concern queries.

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.000376 | class | `Vector3D` | src/waverider/vector3D.py |
| 2 | 0.000210 | function | `_load_ucimlrepo` | benchmarks/canonical_tests/clinical/disease_manifold_architecture.py |
| 3 | 0.000209 | method | `TurtleND._rotate` | src/waverider/turtleND.py |
| 4 | 0.000202 | function | `quantize_angle` | src/waverider/backbone_angles.py |
| 5 | 0.000174 | method | `Turtle3D.unit` | src/waverider/turtle3D.py |
| 6 | 0.000172 | function | `_compile` | benchmarks/canonical_tests/mnist_manifold_architecture.py |
| 7 | 0.000143 | class | `BackboneAngleList` | src/waverider/backbone_angles.py |
| 8 | 0.000142 | method | `KnowledgeGraph.node_ids` | src/waverider/graph_reasoner.py |
| 9 | 0.000141 | function | `_require_viz` | src/waverider/voxel_viz.py |
| 10 | 0.000133 | function | `_compile` | benchmarks/canonical_tests/backbone_mlp_benchmark.py |
| 11 | 0.000131 | function | `_prep` | benchmarks/canonical_tests/clinical/kan_clinical.py |
| 12 | 0.000130 | method | `ManifoldObserver._compute_curvature` | src/waverider/manifold_observer.py |
| 13 | 0.000127 | class | `ReasoningPath` | src/waverider/graph_reasoner.py |
| 14 | 0.000121 | function | `_hr` | benchmarks/canonical_tests/protein_backbone_manifold.py |
| 15 | 0.000119 | function | `_uci` | benchmarks/canonical_tests/clinical/kan_clinical.py |
| 16 | 0.000116 | method | `ManifoldAdamOptimizer._set_flat_weights` | benchmarks/canonical_tests/iris_adam_vs_manifold.py |
| 17 | 0.000115 | function | `_scale` | benchmarks/canonical_tests/clinical/gen_voxel_viz.py |
| 18 | 0.000111 | method | `StandaloneManifoldAdam._set_flat_weights` | benchmarks/canonical_tests/iris_manifold_adam_walker.py |
| 19 | 0.000109 | function | `build_manifold_resnet` | src/model_builder.py |
| 20 | 0.000109 | method | `ManifoldModel.n_nodes` | src/waverider/manifold_model.py |

---

## Concern-Based Hybrid Ranking

Top structurally-dominant nodes per architectural concern (0.60 × semantic + 0.25 × CodeRank + 0.15 × graph proximity).

### Configuration Loading Initialization Setup

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7104 | method | `EpochHeartbeat.__init__` | benchmarks/canonical_tests/cifar10_manifold_architecture.py |
| 2 | 0.7101 | method | `EpochHeartbeat.__init__` | benchmarks/canonical_tests/cifar100_manifold_architecture.py |
| 3 | 0.7082 | function | `setup_tensorflow` | benchmarks/tf_setup.py |
| 4 | 0.7055 | method | `UniversalEmbedder.__init__` | src/waverider/universal_embedder.py |
| 5 | 0.7024 | method | `_ThrottledProgbar.on_epoch_begin` | benchmarks/canonical_tests/mnist_ub_phase_boundary.py |

### Data Persistence Storage Database

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.765 | function | `_load_ucimlrepo` | benchmarks/canonical_tests/clinical/disease_manifold_architecture.py |
| 2 | 0.7509 | function | `run_trial` | benchmarks/canonical_tests/torus_manifold_observer.py |
| 3 | 0.7412 | function | `run_trial` | benchmarks/canonical_tests/helix_manifold_observer.py |
| 4 | 0.7363 | function | `plot_results` | benchmarks/canonical_tests/digits_manifold_architecture.py |
| 5 | 0.7352 | method | `StandaloneManifoldAdam._save_adam_state` | benchmarks/canonical_tests/iris_manifold_adam_walker.py |

### Query Search Retrieval Semantic

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.743 | method | `ManifoldModel._predict_single` | src/waverider/manifold_model.py |
| 2 | 0.7426 | method | `KnowledgeGraph.__contains__` | src/waverider/graph_reasoner.py |
| 3 | 0.7425 | method | `ManifoldKNN.fit` | benchmarks/canonical_tests/digits_manifold_knn.py |
| 4 | 0.7415 | function | `_results_to_dicts` | benchmarks/canonical_tests/iris_adam_vs_manifold.py |
| 5 | 0.6924 | class | `KnowledgeGraph` | src/waverider/graph_reasoner.py |

### Graph Traversal Node Edge

| Rank | Score | Kind | Name | Module |
|------|-------|------|------|--------|
| 1 | 0.7517 | method | `ManifoldModel._gather_graph_neighbors` | src/waverider/manifold_model.py |
| 2 | 0.7475 | method | `GraphReasoner.step` | src/waverider/graph_reasoner.py |
| 3 | 0.7386 | method | `KnowledgeGraph.discover_neighbors` | src/waverider/graph_reasoner.py |
| 4 | 0.7243 | method | `ManifoldModel.get_neighbors` | src/waverider/manifold_model.py |
| 5 | 0.7242 | method | `KnowledgeGraph.add_discoverer` | src/waverider/graph_reasoner.py |



---

*Report generated by PyCodeKG Thorough Analysis Tool — analysis completed in 9.0s*
