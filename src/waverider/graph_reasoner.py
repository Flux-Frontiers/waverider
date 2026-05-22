"""
GraphReasoner: A semantic reasoning engine for knowledge graphs.

Navigates knowledge graphs along semantically weighted edges using an
N-dimensional turtle navigator to maintain directional coherence. Unlike
the ManifoldWalker (which optimizes scalar objectives on continuous manifolds),
the GraphReasoner traverses discrete graph structures by evaluating and
following chains of semantically coherent edges.

The engine discovers reasoning paths through knowledge graphs by:
  1. Lazy edge discovery — neighbors found on arrival, not precomputed
  2. Heading-aware steering — the turtle's orientation encodes the current
     "line of reasoning" and candidates are scored for alignment
  3. Multi-hop chaining — paths are extended step-by-step with backtracking
  4. Beam search — multiple hypothesis paths explored in parallel
  5. Cross-corpus bridging — edges can span knowledge bases

Part of the program proteusPy, https://github.com/suchanek/proteusPy,
a Python package for the manipulation and analysis of macromolecules.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from proteusPy.turtleND import TurtleND

__pdoc__ = {"__all__": True}

# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


def angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in angle space with wrapping at +/-180 degrees.

    Handles the circular topology of dihedral angles: the distance between
    -170 and +170 degrees is 20, not 340.

    Parameters
    ----------
    a, b : np.ndarray
        Angle vectors of equal length (in degrees).

    Returns
    -------
    float
        Euclidean distance after wrapping each component to [0, 180].
    """
    diff = np.abs(np.asarray(a, dtype="d") - np.asarray(b, dtype="d"))
    diff = np.where(diff > 180, 360 - diff, diff)
    return float(np.linalg.norm(diff))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SemanticEdge:
    """A weighted, typed connection between two knowledge-graph nodes.

    Attributes
    ----------
    source_id : str
        Origin node identifier.
    target_id : str
        Destination node identifier.
    weight : float
        Edge strength in [0, 1]. Higher means semantically closer.
    edge_type : str
        Category label (e.g. ``"torsion"``, ``"spatial"``, ``"cross-corpus"``).
    metadata : dict
        Arbitrary key-value pairs for provenance, distance, etc.
    """

    source_id: str
    target_id: str
    weight: float
    edge_type: str = "semantic"
    metadata: dict = field(default_factory=dict)


@dataclass
class ReasoningPath:
    """An ordered chain of nodes and edges representing an inference trace.

    The path records every node visited, every edge followed, the embedding
    at each node, and the steering score assigned to each step.
    """

    node_ids: list[str] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Number of nodes in the path."""
        return len(self.node_ids)

    @property
    def total_score(self) -> float:
        """Sum of all step scores."""
        return sum(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        """Average step score (0.0 for empty paths)."""
        return self.total_score / len(self.scores) if self.scores else 0.0

    def copy(self) -> ReasoningPath:
        """Return a deep copy of this path."""
        return ReasoningPath(
            node_ids=list(self.node_ids),
            edges=list(self.edges),
            embeddings=[e.copy() for e in self.embeddings],
            scores=list(self.scores),
        )

    def __repr__(self) -> str:
        return (
            f"ReasoningPath(length={self.length}, "
            f"mean_score={self.mean_score:.4f}, "
            f"nodes={self.node_ids})"
        )


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """A semantic knowledge graph with lazy edge discovery.

    Nodes are identified by string IDs and carry an embedding vector plus
    an optional arbitrary payload.  Edges may be pre-computed *or* discovered
    lazily by registered :class:`EdgeDiscoverer` instances when the reasoner
    arrives at a node.

    Parameters
    ----------
    ndim : int
        Dimensionality of the embedding space.
    name : str
        Human-readable graph name.
    """

    def __init__(self, ndim: int, name: str = ""):
        self.ndim = ndim
        self.name = name
        self._nodes: dict[str, Any] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._edges: dict[str, list[SemanticEdge]] = {}
        self._discoverers: list[EdgeDiscoverer] = []
        # Cached matrix for vectorized operations
        self._id_list: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._dirty = True

    # ----- Node operations -----

    def add_node(self, node_id: str, embedding: np.ndarray, node: Any = None) -> None:
        """Add a node with its embedding vector and optional payload.

        Parameters
        ----------
        node_id : str
            Unique identifier for this node.
        embedding : np.ndarray
            Coordinate in the graph's embedding space.  Must have length ``ndim``.
        node : Any, optional
            Arbitrary payload (e.g. a ``Disulfide`` object).
        """
        emb = np.asarray(embedding, dtype="d")
        if emb.shape != (self.ndim,):
            raise ValueError(f"Embedding must have shape ({self.ndim},), got {emb.shape}")
        self._nodes[node_id] = node
        self._embeddings[node_id] = emb.copy()
        self._dirty = True

    def get_embedding(self, node_id: str) -> np.ndarray:
        """Return the embedding vector for *node_id*."""
        return self._embeddings[node_id]

    def get_node(self, node_id: str) -> Any:
        """Return the payload object for *node_id*, or ``None``."""
        return self._nodes.get(node_id)

    @property
    def node_ids(self) -> list[str]:
        """All node IDs in insertion order."""
        return list(self._nodes.keys())

    # ----- Edge operations -----

    def add_edge(self, edge: SemanticEdge) -> None:
        """Register a pre-computed edge."""
        self._edges.setdefault(edge.source_id, []).append(edge)

    def add_discoverer(self, discoverer: EdgeDiscoverer) -> None:
        """Register an :class:`EdgeDiscoverer` for lazy neighbor lookup."""
        self._discoverers.append(discoverer)

    def discover_neighbors(
        self, node_id: str, heading: np.ndarray | None = None
    ) -> list[SemanticEdge]:
        """Return all edges from *node_id*: pre-computed + dynamically discovered.

        Parameters
        ----------
        node_id : str
            The source node.
        heading : np.ndarray, optional
            Current turtle heading; passed to discoverers that support
            directional filtering.
        """
        edges = list(self._edges.get(node_id, []))
        for disc in self._discoverers:
            edges.extend(disc.discover(node_id, self, heading))
        return edges

    # ----- Batch-matrix helpers -----

    def _rebuild_matrix(self) -> None:
        """Rebuild the NxD embedding matrix from the node dict (if dirty)."""
        if not self._dirty:
            return
        self._id_list = list(self._embeddings.keys())
        self._id_to_idx = {nid: i for i, nid in enumerate(self._id_list)}
        self._embedding_matrix = np.array(
            [self._embeddings[nid] for nid in self._id_list], dtype="d"
        )
        self._dirty = False

    # ----- Dunder -----

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(name={self.name!r}, ndim={self.ndim}, "
            f"nodes={len(self._nodes)}, "
            f"discoverers={len(self._discoverers)})"
        )


# ---------------------------------------------------------------------------
# Edge discoverers
# ---------------------------------------------------------------------------


class EdgeDiscoverer(ABC):
    """Abstract base class for lazy edge discovery strategies.

    A discoverer is called when the reasoner arrives at a node and needs
    to know which neighbors are reachable.  Implementations should return
    a list of :class:`SemanticEdge` objects.
    """

    @abstractmethod
    def discover(
        self,
        node_id: str,
        graph: KnowledgeGraph,
        heading: np.ndarray | None = None,
    ) -> list[SemanticEdge]:
        """Discover edges from *node_id*.

        Parameters
        ----------
        node_id : str
            Source node.
        graph : KnowledgeGraph
            The graph to search.
        heading : np.ndarray, optional
            Current heading vector for directional filtering.
        """
        ...


class RadiusDiscoverer(EdgeDiscoverer):
    """Discover edges to all nodes within a distance threshold.

    Uses vectorized numpy operations over the full embedding matrix
    for performance on large graphs.

    Parameters
    ----------
    threshold : float
        Maximum distance to create an edge.
    edge_type : str
        Label for discovered edges.
    angular : bool
        If True, use angular wrapping at +/-180 degrees.
    """

    def __init__(self, threshold: float, edge_type: str = "semantic", angular: bool = False):
        self.threshold = threshold
        self.edge_type = edge_type
        self.angular = angular

    def discover(self, node_id, graph, heading=None):
        source_emb = graph.get_embedding(node_id)
        graph._rebuild_matrix()

        diffs = np.abs(graph._embedding_matrix - source_emb)
        if self.angular:
            diffs = np.where(diffs > 180, 360 - diffs, diffs)
        distances = np.linalg.norm(diffs, axis=1)

        # Exclude self
        self_idx = graph._id_to_idx.get(node_id)
        if self_idx is not None:
            distances[self_idx] = np.inf

        mask = distances <= self.threshold
        indices = np.where(mask)[0]

        edges = []
        for idx in indices:
            target_id = graph._id_list[idx]
            weight = 1.0 - (distances[idx] / self.threshold)
            edges.append(
                SemanticEdge(
                    node_id,
                    target_id,
                    float(weight),
                    self.edge_type,
                    {"distance": float(distances[idx])},
                )
            )
        return edges


class KNNDiscoverer(EdgeDiscoverer):
    """Discover edges to the *k* nearest neighbors.

    Parameters
    ----------
    k : int
        Number of neighbors.
    edge_type : str
        Label for discovered edges.
    angular : bool
        If True, use angular wrapping at +/-180 degrees.
    """

    def __init__(self, k: int, edge_type: str = "semantic", angular: bool = False):
        self.k = k
        self.edge_type = edge_type
        self.angular = angular

    def discover(self, node_id, graph, heading=None):
        source_emb = graph.get_embedding(node_id)
        graph._rebuild_matrix()

        diffs = np.abs(graph._embedding_matrix - source_emb)
        if self.angular:
            diffs = np.where(diffs > 180, 360 - diffs, diffs)
        distances = np.linalg.norm(diffs, axis=1)

        self_idx = graph._id_to_idx.get(node_id)
        if self_idx is not None:
            distances[self_idx] = np.inf

        k = min(self.k, len(distances) - 1)
        if k <= 0:
            return []

        indices = np.argpartition(distances, k)[:k]
        indices = indices[np.argsort(distances[indices])]

        max_dist = distances[indices[-1]]

        edges = []
        for idx in indices:
            target_id = graph._id_list[idx]
            w = 1.0 - (distances[idx] / max(max_dist, 1e-10))
            edges.append(
                SemanticEdge(
                    node_id,
                    target_id,
                    float(w),
                    self.edge_type,
                    {"distance": float(distances[idx])},
                )
            )
        return edges


class DirectedDiscoverer(EdgeDiscoverer):
    """Decorator that filters an inner discoverer to a forward cone.

    Only edges whose direction from the current node has a positive
    projection onto the turtle's heading (within *cone_angle* degrees)
    are kept.

    Parameters
    ----------
    inner : EdgeDiscoverer
        The discoverer to wrap.
    cone_angle : float
        Half-angle of the forward cone in degrees.  90 = hemisphere.
    """

    def __init__(self, inner: EdgeDiscoverer, cone_angle: float = 90.0):
        self.inner = inner
        self.cone_cos = math.cos(math.radians(cone_angle))

    def discover(self, node_id, graph, heading=None):
        edges = self.inner.discover(node_id, graph, heading)
        if heading is None:
            return edges

        source_emb = graph.get_embedding(node_id)
        result = []
        for edge in edges:
            target_emb = graph.get_embedding(edge.target_id)
            direction = target_emb - source_emb
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            cos_angle = np.dot(direction / norm, heading)
            if cos_angle >= self.cone_cos:
                result.append(edge)
        return result


# ---------------------------------------------------------------------------
# Steering strategies
# ---------------------------------------------------------------------------


class SteeringStrategy(ABC):
    """Abstract base for reasoning-step scoring.

    A strategy scores a candidate next-node based on the current embedding,
    the candidate embedding, the turtle's heading, the edge, and the path
    so far.  Higher scores are preferred.
    """

    @abstractmethod
    def score_candidate(
        self,
        current_embedding: np.ndarray,
        candidate_embedding: np.ndarray,
        heading: np.ndarray,
        edge: SemanticEdge,
        path: ReasoningPath,
    ) -> float:
        """Return a scalar score for following *edge* to *candidate_embedding*."""
        ...


class TargetSteering(SteeringStrategy):
    """Steer toward a specific target embedding.

    Combines three signals:
      * **alignment** — cosine of angle between step direction and heading
      * **progress** — fraction of remaining distance closed by this step
      * **edge weight** — intrinsic edge quality

    Parameters
    ----------
    target : np.ndarray
        Destination embedding.
    alignment_weight : float
        Balance between alignment (higher) and progress (lower).
    """

    def __init__(self, target: np.ndarray, alignment_weight: float = 0.5):
        self.target = np.asarray(target, dtype="d")
        self.alignment_weight = alignment_weight

    def score_candidate(self, current, candidate, heading, edge, path):
        direction = candidate - current
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return 0.0

        alignment = float(np.dot(direction / norm, heading))
        current_dist = float(np.linalg.norm(self.target - current))
        candidate_dist = float(np.linalg.norm(self.target - candidate))
        progress = (current_dist - candidate_dist) / max(current_dist, 1e-10)

        w = self.alignment_weight
        return w * alignment + (1 - w) * progress + 0.1 * edge.weight


class GradientSteering(SteeringStrategy):
    """Steer along the gradient of a scalar field.

    Parameters
    ----------
    field_fn : callable
        Maps an embedding vector to a scalar.
    minimize : bool
        If True, prefer decreasing field values.
    """

    def __init__(self, field_fn: Callable[[np.ndarray], float], minimize: bool = True):
        self.field_fn = field_fn
        self.minimize = minimize

    def score_candidate(self, current, candidate, heading, edge, path):
        current_val = self.field_fn(current)
        candidate_val = self.field_fn(candidate)
        improvement = current_val - candidate_val if self.minimize else candidate_val - current_val

        direction = candidate - current
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return 0.0
        alignment = float(np.dot(direction / norm, heading))

        return 0.5 * improvement + 0.3 * alignment + 0.2 * edge.weight


class ExplorationSteering(SteeringStrategy):
    """Steer to maximize coverage and information gain.

    Favors candidates that are distant from all previously visited nodes.
    """

    def score_candidate(self, current, candidate, heading, edge, path):
        direction = candidate - current
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return 0.0

        alignment = float(np.dot(direction / norm, heading))

        if path.embeddings:
            min_dist = min(float(np.linalg.norm(candidate - e)) for e in path.embeddings)
        else:
            min_dist = norm

        return 0.4 * min_dist + 0.3 * alignment + 0.3 * edge.weight


# ---------------------------------------------------------------------------
# The reasoning engine
# ---------------------------------------------------------------------------


class GraphReasoner:
    """Semantic reasoning engine for knowledge graphs.

    Steers through a :class:`KnowledgeGraph` along weighted edges using a
    :class:`TurtleND` to maintain directional coherence.  The turtle's
    heading encodes the current "line of reasoning"; candidates are scored
    by a :class:`SteeringStrategy` that evaluates alignment, edge quality,
    and task-specific criteria.

    Parameters
    ----------
    graph : KnowledgeGraph
        The knowledge graph to reason over.
    strategy : SteeringStrategy
        Scoring function for candidate edges.

    Examples
    --------
    >>> g = KnowledgeGraph(ndim=3)
    >>> for i in range(5):
    ...     g.add_node(f"n{i}", np.array([float(i), 0.0, 0.0]))
    >>> g.add_discoverer(RadiusDiscoverer(threshold=1.5))
    >>> r = GraphReasoner(g, ExplorationSteering())
    >>> path = r.reason("n0", max_hops=4)
    >>> path.length >= 2
    True
    """

    def __init__(self, graph: KnowledgeGraph, strategy: SteeringStrategy):
        self.graph = graph
        self.strategy = strategy
        self.turtle = TurtleND(graph.ndim)
        self._path = ReasoningPath()
        self._visited: set[str] = set()

    # ----- Properties -----

    @property
    def path(self) -> ReasoningPath:
        """The current reasoning path."""
        return self._path

    @property
    def position(self) -> np.ndarray:
        """Current embedding position."""
        return self.turtle.position

    @property
    def heading(self) -> np.ndarray:
        """Current heading direction."""
        return self.turtle.heading

    @property
    def current_node(self) -> str | None:
        """ID of the node the reasoner is currently at."""
        return self._path.node_ids[-1] if self._path.node_ids else None

    # ----- Core reasoning -----

    def start(self, node_id: str) -> None:
        """Begin reasoning from *node_id*.

        Resets the path, positions the turtle at the node's embedding,
        and marks the node as visited.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node {node_id!r} not in graph")

        emb = self.graph.get_embedding(node_id)
        self.turtle = TurtleND(self.graph.ndim)
        self.turtle.position = emb
        self._path = ReasoningPath(
            node_ids=[node_id],
            embeddings=[emb.copy()],
            edges=[],
            scores=[],
        )
        self._visited = {node_id}

    def step(self, edge_type: str | None = None) -> str | None:
        """Take one reasoning step.

        Discovers candidate edges from the current node, scores them using
        the steering strategy, and moves to the highest-scoring unvisited
        neighbor.

        Parameters
        ----------
        edge_type : str, optional
            If given, only follow edges of this type.

        Returns
        -------
        str or None
            The ID of the node moved to, or ``None`` if no candidates
            remain (dead end).
        """
        if not self._path.node_ids:
            raise RuntimeError("Must call start() before step()")

        current_id = self._path.node_ids[-1]
        current_emb = self.graph.get_embedding(current_id)
        heading = self.turtle.heading

        # Discover all edges from current node
        all_edges = self.graph.discover_neighbors(current_id, heading)

        # Score unvisited candidates
        scored: list[tuple[float, str, np.ndarray, SemanticEdge]] = []
        for edge in all_edges:
            if edge.target_id in self._visited:
                continue
            if edge.target_id not in self.graph:
                continue
            if edge_type is not None and edge.edge_type != edge_type:
                continue

            target_emb = self.graph.get_embedding(edge.target_id)
            score = self.strategy.score_candidate(
                current_emb, target_emb, heading, edge, self._path
            )
            scored.append((score, edge.target_id, target_emb, edge))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_id, best_emb, best_edge = scored[0]

        # Move turtle and align heading
        self.turtle.position = best_emb
        self._align_heading(best_emb - current_emb)

        # Record
        self._path.node_ids.append(best_id)
        self._path.embeddings.append(best_emb.copy())
        self._path.edges.append(best_edge)
        self._path.scores.append(best_score)
        self._visited.add(best_id)

        return best_id

    def reason(
        self,
        start_id: str,
        max_hops: int = 10,
        edge_type: str | None = None,
        min_score: float = -float("inf"),
    ) -> ReasoningPath:
        """Execute multi-hop reasoning from *start_id*.

        Follows the best edge at each step, maintaining semantic coherence
        via the turtle's heading.  Stops when *max_hops* is reached, a dead
        end is hit, or the step score falls below *min_score*.

        Parameters
        ----------
        start_id : str
            Starting node.
        max_hops : int
            Maximum number of edges to traverse.
        edge_type : str, optional
            Restrict to edges of this type.
        min_score : float
            Minimum acceptable step score.

        Returns
        -------
        ReasoningPath
            The complete reasoning trace.
        """
        self.start(start_id)

        for _ in range(max_hops):
            next_id = self.step(edge_type)
            if next_id is None:
                break
            if self._path.scores and self._path.scores[-1] < min_score:
                self.backtrack()
                break

        return self._path

    def reason_toward(
        self,
        start_id: str,
        target_id: str,
        max_hops: int = 20,
        edge_type: str | None = None,
    ) -> ReasoningPath:
        """Find a reasoning path from *start_id* to *target_id*.

        Temporarily installs a :class:`TargetSteering` strategy and orients
        the turtle's heading toward the target embedding.

        Parameters
        ----------
        start_id, target_id : str
            Source and destination nodes.
        max_hops : int
            Maximum path length.
        edge_type : str, optional
            Restrict to edges of this type.

        Returns
        -------
        ReasoningPath
            The discovered path (may or may not reach *target_id*).
        """
        target_emb = self.graph.get_embedding(target_id)
        old_strategy = self.strategy
        self.strategy = TargetSteering(target_emb)

        self.start(start_id)

        # Orient heading toward target
        direction = target_emb - self.graph.get_embedding(start_id)
        self._align_heading(direction)

        for _ in range(max_hops):
            next_id = self.step(edge_type)
            if next_id is None or next_id == target_id:
                break

        self.strategy = old_strategy
        return self._path

    def beam_reason(
        self,
        start_id: str,
        max_hops: int = 10,
        beam_width: int = 3,
        edge_type: str | None = None,
        min_score: float = -float("inf"),
    ) -> list[ReasoningPath]:
        """Multi-path beam search through the knowledge graph.

        Maintains *beam_width* hypothesis paths simultaneously, extending
        the best candidates at each hop.

        Parameters
        ----------
        start_id : str
            Starting node.
        max_hops : int
            Maximum path length.
        beam_width : int
            Number of paths to maintain.
        edge_type : str, optional
            Restrict to edges of this type.
        min_score : float
            Minimum acceptable step score.

        Returns
        -------
        list[ReasoningPath]
            Up to *beam_width* complete paths, sorted best-first.
        """
        initial_emb = self.graph.get_embedding(start_id)
        initial_path = ReasoningPath(
            node_ids=[start_id],
            edges=[],
            embeddings=[initial_emb.copy()],
            scores=[],
        )

        # Each beam: (cumulative_score, path, visited_set)
        beams: list[tuple[float, ReasoningPath, set[str]]] = [(0.0, initial_path, {start_id})]

        for _ in range(max_hops):
            candidates: list[tuple[float, ReasoningPath, set[str]]] = []

            for cum_score, path, visited in beams:
                current_id = path.node_ids[-1]
                current_emb = self.graph.get_embedding(current_id)

                # Compute heading from last two positions
                if len(path.embeddings) >= 2:
                    direction = path.embeddings[-1] - path.embeddings[-2]
                    norm = np.linalg.norm(direction)
                    heading = (
                        direction / norm if norm > 1e-10 else np.eye(1, self.graph.ndim, 0).ravel()
                    )
                else:
                    heading = np.eye(1, self.graph.ndim, 0).ravel()

                edges = self.graph.discover_neighbors(current_id, heading)

                for edge in edges:
                    if edge.target_id in visited:
                        continue
                    if edge.target_id not in self.graph:
                        continue
                    if edge_type is not None and edge.edge_type != edge_type:
                        continue

                    target_emb = self.graph.get_embedding(edge.target_id)
                    score = self.strategy.score_candidate(
                        current_emb, target_emb, heading, edge, path
                    )
                    if score < min_score:
                        continue

                    new_path = ReasoningPath(
                        node_ids=path.node_ids + [edge.target_id],
                        edges=path.edges + [edge],
                        embeddings=path.embeddings + [target_emb.copy()],
                        scores=path.scores + [score],
                    )
                    candidates.append((cum_score + score, new_path, visited | {edge.target_id}))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        return [path for _, path, _ in beams]

    def backtrack(self, n_steps: int = 1) -> str | None:
        """Back up *n_steps* along the reasoning path.

        Does not remove nodes from the visited set — the reasoner will not
        revisit backtracked nodes, forcing exploration of alternatives.

        Returns
        -------
        str or None
            The node ID after backtracking.
        """
        for _ in range(min(n_steps, len(self._path.node_ids) - 1)):
            self._path.node_ids.pop()
            self._path.embeddings.pop()
            if self._path.edges:
                self._path.edges.pop()
            if self._path.scores:
                self._path.scores.pop()

        if self._path.embeddings:
            self.turtle.position = self._path.embeddings[-1]

        return self._path.node_ids[-1] if self._path.node_ids else None

    # ----- Heading alignment -----

    def _align_heading(self, direction: np.ndarray) -> None:
        """Smoothly rotate the turtle's heading toward *direction*.

        Finds the rotation plane closest to the desired direction and
        applies a single Givens rotation.
        """
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return
        target = direction / norm
        current = self.turtle.heading
        dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
        angle = math.degrees(math.acos(dot))

        if angle < 1e-6:
            return

        # Project target onto the plane perpendicular to heading
        perp = target - dot * current
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-10:
            return
        perp = perp / perp_norm

        # Find which basis vector is closest to the perpendicular
        best_i, best_d = 1, 0.0
        for i in range(1, self.turtle.ndim):
            d = abs(float(np.dot(self.turtle.basis(i), perp)))
            if d > best_d:
                best_d = d
                best_i = i

        self.turtle.rotate(angle, 0, best_i)

    # ----- Repr -----

    def __repr__(self) -> str:
        pos = self.current_node or "unstarted"
        return (
            f"GraphReasoner(position={pos}, "
            f"path_length={self._path.length}, "
            f"mean_score={self._path.mean_score:.4f})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def graph_from_disulfides(sslist, edge_threshold: float = 10.0) -> KnowledgeGraph:
    """Build a :class:`KnowledgeGraph` from a DisulfideList.

    Each disulfide becomes a node with its 5D torsion-angle vector as the
    embedding.  A :class:`RadiusDiscoverer` with angular distance is
    installed for lazy neighbor discovery.

    Parameters
    ----------
    sslist : DisulfideList
        Source disulfide bonds.
    edge_threshold : float
        Angular distance threshold in degrees for edge discovery.

    Returns
    -------
    KnowledgeGraph
        A 5-dimensional graph ready for reasoning.
    """
    graph = KnowledgeGraph(ndim=5, name="disulfide_torsion")

    for ss in sslist:
        node_id = f"{ss.pdb_id}_{ss.proximal}_{ss.distal}"
        embedding = np.array(ss.torsion_array, dtype="d")
        graph.add_node(node_id, embedding, node=ss)

    graph.add_discoverer(
        RadiusDiscoverer(threshold=edge_threshold, edge_type="torsion", angular=True)
    )

    return graph


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "angular_distance",
    "euclidean_distance",
    "SemanticEdge",
    "ReasoningPath",
    "KnowledgeGraph",
    "EdgeDiscoverer",
    "RadiusDiscoverer",
    "KNNDiscoverer",
    "DirectedDiscoverer",
    "SteeringStrategy",
    "TargetSteering",
    "GradientSteering",
    "ExplorationSteering",
    "GraphReasoner",
    "graph_from_disulfides",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

# End of file
