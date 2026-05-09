# WaveRider: Manifold-Aware Machine Learning Framework

**Core Insight**: 71–99% of dimensions in real data spaces are noise. Discover the true manifold, operate within it.

**Motto**: *"The manifold IS the model."*

---

## Layer 1: TurtleND — Navigation Primitive

*`src/waverider/turtleND.py`*

**What it is**: N-dimensional generalization of turtle graphics (Logo 1982 → Turtle3D 1986 → TurtleND 2024)

### State

- Position **p** ∈ ℝ^N
- Orthonormal frame **F** (N×N matrix, rows = basis vectors)
  - `frame[0]` = heading, `frame[1]` = left, `frame[2]` = up, `frame[3..N-1]` = higher dims

### Primitives

- `move(distance)` — translate along heading
- `rotate(angle, i, j)` — Givens rotation in basis[i]–basis[j] plane
- `turn(angle)` — heading↔left (basis 0–1)
- `pitch(angle)` — heading↔up (basis 0–2)
- `roll(angle)` — left↔up (basis 1–2)
- `yaw(angle)` — heading↔left with 180° convention (molecular building)
- `to_local(vec)` / `to_global(vec)` — coordinate transforms
- `orient(pos, heading, left)` — set position + frame, Gram-Schmidt completion
- `orthonormalize()` — QR re-normalization to prevent drift

**Key Property**: All rotations preserve orthonormality (Frame Invariant Proposition)

---

## Layer 2: ManifoldWalker — Continuous Optimization Engine

*`src/waverider/manifold_walker.py`*

**What it is**: Riemannian-approximate gradient descent using local PCA to discover and follow the data manifold

### Algorithm (each step)

1. **KNN** — find k nearest neighbors in embedding space
2. **Local PCA** — eigendecompose neighborhood covariance → eigenvalues λ, eigenvectors V
3. **Intrinsic dim** — d = min dimensions capturing τ% cumulative variance
4. **Orient turtle** — align frame to eigenvectors (descending λ order)
5. **Project gradient** — transform to local frame, zero components d+1..N
6. **Eigenvalue weight** — scale each component by λᵢ/λ₁ (natural gradient)
7. **Step** — convert back to global, descend: **p** ← **p** − η · **s**

**Parameters**: k=50, τ=0.95, η=0.01, ε=1e-5

### Additional Methods

- `walk(n_steps, tol)` — multi-step optimization with convergence check
- `probe(direction_index, distances)` — evaluate objective along a basis direction without moving

### Layer 2a: ManifoldAdamWalker — Adam + Manifold Projection

*Subclass of ManifoldWalker*

**Key difference**: Replaces eigenvalue weighting with Adam's momentum + adaptive LR

#### Algorithm modification at step 5–7

1. Project gradient onto manifold, zero off-manifold (same as base)
2. **No** eigenvalue weighting — Adam handles step-size adaptation
3. Map projected gradient back to global coordinates
4. Adam update: m ← β₁m + (1−β₁)g_proj, v ← β₂v + (1−β₂)g_proj²
5. Step: **p** ← **p** − η · m̂/(√v̂ + ε)

**Critical design**: Adam state (m, v) lives in **global** coordinates so momentum persists across frame re-orientations

**Parameters**: β₁=0.9, β₂=0.999, adam_ε=1e-7

---

## Layer 3: ManifoldModel — Zero-Parameter Geometric Classifier

*`src/waverider/manifold_model.py`*

**What it is**: The manifold itself becomes the trained model. No learned weights. A map you can walk through.

**The "trained model" consists of**:

1. The graph (connectivity/topology)
2. Local basis vectors at each node (geometry)
3. The eigenvalue field (how geometry varies)

### Phase 1 — Explore (`fit`)

1. Add all training points as graph nodes
2. At each node: local PCA via SVD (O(k²·ndim), efficient for high-D)
   - Store truncated basis (d × ndim), eigenvalues, intrinsic_dim, centroid, label
3. Build manifold-aware edges:
   - For each node, find k_graph Euclidean nearest neighbors
   - Recompute distance by projecting into source tangent space
   - `blended_dist = 0.8 × manifold_dist + 0.2 × euclidean_dist`
   - `edge.weight = 1 / (1 + blended_dist)`
4. Initialize TurtleND for navigation

**Parameters**: k_graph=15, k_pca=50, k_vote=7, τ=0.95, manifold_weight=0.8

### Phase 2 — Navigate (`predict`)

1. Find k_pca Euclidean nearest neighbors of query point
2. Local PCA **at the query point** via SVD → discover its tangent space
3. Project query + candidates into tangent space
4. Walk graph from nearest node (2 hops) to gather additional candidates
5. Merge PCA-local + graph-connected candidates
6. Distance-weighted majority vote among k_vote nearest in projected space

### Phase 3 — Fly (Interactive Exploration)

| Method | What it does |
|--------|-------------|
| `fly_to(node_id)` | Teleport to node, orient frame to its PCA basis (QR-padded to full rank) |
| `fly_to_nearest(point)` | Enter manifold from any ambient point → nearest node → `fly_to` |
| `fly_step(direction?)` | One graph hop; score = **0.6 × alignment + 0.4 × edge_weight**; re-orient at destination |
| `fly_toward(target, max_steps=20)` | Repeated `fly_step` steering toward target; stops on dead end or no progress |
| `reset_flight()` | Clear current node + flight path |

### Supporting Queries

- `get_geometry(node_id)` → NodeGeometry (basis, eigenvalues, intrinsic_dim, centroid, label)
- `get_neighbors(node_id)` → sorted (node_id, weight) pairs
- `geometry_summary()` → mean/std/min/max intrinsic_dim, node count, edge count

---

## Layer 3a: ManifoldObserver — Extrinsic Observer

*`src/waverider/manifold_observer.py`*

**What it is**: An (N+1)-dimensional turtle that sees into the N-dimensional manifold from above. One extra orthonormal dimension = the manifold normal = sight.

**Core insight**: The ManifoldModel walks *on* the surface — it can only discover structure by searching. The ManifoldObserver hovers *above* the surface — it can *see* curvature, topology, and class boundaries directly.

### Construction

1. Take subject's N×N frame **F** at any node
2. Pad to (N+1)×(N+1): append zeros + normal vector [0,…,0,1]
3. Orthonormalize via QR → valid (N+1)-dim frame
4. Position: [p₀,…,p_{N-1}, altitude] in R^{N+1}

### What It Sees (that the subject can't)

| Observable | Method | Subject alternative |
|-----------|--------|-------------------|
| **Curvature** | Principal angles between neighboring tangent subspaces | Multi-hop walk |
| **Height** | Reconstruction error from local tangent plane | Lives on surface — can't measure |
| **Topology** | Full observation field in one pass | Explore node by node |
| **Classification** | Project down, read label — O(n) | Graph walk + vote — O(hops × k) |

### Key Methods

| Method | What it does |
|--------|-------------|
| `observe()` | See entire manifold in one pass — curvature, height, intrinsic dim at every node |
| `locate(query)` | Project straight down to surface — no graph walking |
| `predict(X)` | Classify by observation, not by search |
| `curvature_at(node)` | Local curvature from tangent-basis rotation rate |
| `curvature_field()` | Curvature at all nodes |
| `topology_summary()` | Manifold shape statistics from above |
| `sync(node)` | Extend subject's N-dim frame to (N+1)-dim observer frame |
| `lift_off(h)` / `look_down()` | Rise above / project back to surface |
| `pan(dir, dist)` / `orbit(angle)` | Navigate (N+1)-space around the surface |

### Data Structures

**ObservedGeometry** (per node):
- `position_lifted` — (N+1)-dim coordinates
- `normal` — unit normal to manifold
- `curvature` — rate of tangent-plane rotation (mean principal angle, radians)
- `tangent_spread` — curvature variation among neighbors
- `height` — distance from ideal tangent plane
- `intrinsic_dim` — local dimensionality

**For disulfides**: Subject is TurtleND(5) in torsion-angle space. Observer is TurtleND(6) — the 6th dimension is the normal to the 5D structural manifold.

---

## Layer 4: GraphReasoner — Semantic Reasoning Engine

*`src/waverider/graph_reasoner.py`*

**What it is**: Traverses discrete knowledge graphs using TurtleND for directional coherence. The heading encodes the current "line of reasoning."

### Design Principles

- Lazy edge discovery — neighbors found on arrival, not precomputed
- Heading-aware steering — candidates scored for alignment with current reasoning direction
- Backtracking — visited set prevents revisits, forces alternative exploration

### Reasoning Modes

| Mode | Method | Behavior |
|------|--------|----------|
| **Greedy** | `reason(start, max_hops)` | Best-first at each step, stop on dead end or min_score |
| **Targeted** | `reason_toward(start, target)` | Installs TargetSteering, orients heading toward target embedding |
| **Beam Search** | `beam_reason(start, beam_width, max_hops)` | Maintains beam_width parallel hypothesis paths, prunes at each hop |
| **Single Step** | `step(edge_type?)` | One hop, returns node_id or None |
| **Backtrack** | `backtrack(n_steps)` | Rewind path without clearing visited set |

### Steering Strategies (Pluggable)

| Strategy | Formula | Use case |
|----------|---------|----------|
| **TargetSteering** | 0.5 × alignment + 0.5 × progress + 0.1 × edge_weight | Navigate toward a known destination |
| **GradientSteering** | 0.5 × field_improvement + 0.3 × alignment + 0.2 × edge_weight | Follow a scalar field (loss landscape) |
| **ExplorationSteering** | 0.4 × novelty + 0.3 × alignment + 0.3 × edge_weight | Maximize coverage, avoid redundancy |

---

## Layer 5: Knowledge Graph Infrastructure

*`src/waverider/graph_reasoner.py`*

### KnowledgeGraph

- **Nodes**: string ID + embedding vector (ℝ^N) + optional payload (e.g. Disulfide object)
- **Edges**: `SemanticEdge(source, target, weight ∈ [0,1], type, metadata)`
- Supports both **pre-computed** and **lazily-discovered** edges
- Vectorized batch operations via cached embedding matrix

### Edge Discoverers (Pluggable, Lazy)

| Discoverer | Behavior |
|------------|----------|
| **RadiusDiscoverer** | All nodes within distance threshold; weight = 1 − dist/threshold |
| **KNNDiscoverer** | k nearest neighbors; weight = 1 − dist/max_dist |
| **DirectedDiscoverer** | Wraps any discoverer with a forward-cone filter (cos angle ≥ threshold) |

### Distance Functions

- `euclidean_distance(a, b)` — standard L2
- `angular_distance(a, b)` — wraps at ±180° for dihedral angles

### Domain Factory

- `graph_from_disulfides(sslist)` — builds 5D torsion-angle KnowledgeGraph from disulfide bonds with angular RadiusDiscoverer

---

## Data Structures

### NodeGeometry (stored per graph node in ManifoldModel)

- `basis` — truncated PCA basis, shape (d, ndim)
- `eigenvalues` — descending, length ndim
- `intrinsic_dim` — dimensions capturing τ% variance
- `centroid` — neighborhood mean
- `label` — class label (optional)
- `index` — index into training array

### ReasoningPath (output of GraphReasoner)

- `node_ids` — ordered list of visited nodes
- `edges` — SemanticEdge at each hop
- `embeddings` — position at each node
- `scores` — steering score at each step
- Properties: `length`, `total_score`, `mean_score`

---

## Experimental Results

| Dataset | Ambient D | Intrinsic d | Noise % | WaveRider Params | Standard Params | Reduction | Winner |
|---------|-----------|-------------|---------|-----------------|-----------------|-----------|--------|
| Digits | 64 | 11–18 | 71–83% | **0** | 0 | — | WaveRider (97.72% vs 97.33%) |
| Iris | 4 | 2–3 | 98.9% | ~5 eff. | 243 | — | Reveals noise; Adam faster |
| MNIST | 784 | 22 | 97.2% | **2,232** | 109,386 | **49×** | Standard by 1.9% |
| CIFAR-10 | 3,072 | 29 | 99.1% | **3,751** | 820,874 | **219×** | **WaveRider** (48.58% vs 48.39%) |

**Pattern**: As problem complexity and ambient dimensionality grow, WaveRider's advantage grows. At CIFAR-10 scale it wins outright.
