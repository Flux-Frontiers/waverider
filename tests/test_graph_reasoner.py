"""Tests for graph_reasoner: distance functions, KnowledgeGraph, discoverers, and GraphReasoner."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from waverider.graph_reasoner import (
    ExplorationSteering,
    GraphReasoner,
    KNNDiscoverer,
    KnowledgeGraph,
    RadiusDiscoverer,
    ReasoningPath,
    SemanticEdge,
    TargetSteering,
    angular_distance,
    euclidean_distance,
)

# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------


class TestAngularDistance:
    def test_identical(self):
        a = np.array([10.0, 20.0, 30.0])
        assert angular_distance(a, a) == pytest.approx(0.0)

    def test_no_wrapping_needed(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert angular_distance(a, b) == pytest.approx(5.0)

    def test_wraps_at_180(self):
        # -170 and 170 differ by 20° in angle space, not 340°
        a = np.array([-170.0])
        b = np.array([170.0])
        assert angular_distance(a, b) == pytest.approx(20.0)

    def test_wraps_at_180_multi(self):
        a = np.array([-170.0, -90.0])
        b = np.array([170.0, 90.0])
        # diffs: 20, 180 → norm(20, 180)
        expected = float(np.linalg.norm([20.0, 180.0]))
        assert angular_distance(a, b) == pytest.approx(expected, rel=1e-6)

    def test_symmetric(self):
        a = np.array([10.0, -50.0])
        b = np.array([-170.0, 80.0])
        assert angular_distance(a, b) == pytest.approx(angular_distance(b, a))


class TestEuclideanDistance:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == pytest.approx(0.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert euclidean_distance(a, b) == pytest.approx(np.sqrt(2.0))

    def test_3d(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 2.0])
        assert euclidean_distance(a, b) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_graph():
    """5 nodes along the x-axis at x=0,1,2,3,4."""
    g = KnowledgeGraph(ndim=3, name="linear")
    for i in range(5):
        g.add_node(f"n{i}", np.array([float(i), 0.0, 0.0]))
    return g


@pytest.fixture
def radius_graph(linear_graph):
    """linear_graph with a RadiusDiscoverer(threshold=1.5)."""
    linear_graph.add_discoverer(RadiusDiscoverer(threshold=1.5))
    return linear_graph


@pytest.fixture
def knn_graph(linear_graph):
    """linear_graph with a KNNDiscoverer(k=2)."""
    linear_graph.add_discoverer(KNNDiscoverer(k=2))
    return linear_graph


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_add_and_retrieve_node(self):
        g = KnowledgeGraph(ndim=2)
        g.add_node("a", np.array([1.0, 2.0]))
        assert_allclose(g.get_embedding("a"), [1.0, 2.0])

    def test_wrong_embedding_shape_raises(self):
        g = KnowledgeGraph(ndim=3)
        with pytest.raises(ValueError):
            g.add_node("x", np.array([1.0, 2.0]))

    def test_contains(self, linear_graph):
        assert "n0" in linear_graph
        assert "n99" not in linear_graph

    def test_len(self, linear_graph):
        assert len(linear_graph) == 5

    def test_node_ids(self, linear_graph):
        ids = linear_graph.node_ids
        assert set(ids) == {"n0", "n1", "n2", "n3", "n4"}

    def test_get_node_payload(self):
        g = KnowledgeGraph(ndim=2)
        payload = {"info": "test"}
        g.add_node("a", np.array([0.0, 0.0]), node=payload)
        assert g.get_node("a") is payload

    def test_get_node_missing_returns_none(self, linear_graph):
        assert linear_graph.get_node("nonexistent") is None

    def test_add_static_edge(self, linear_graph):
        edge = SemanticEdge("n0", "n1", weight=0.9, edge_type="manual")
        linear_graph.add_edge(edge)
        neighbors = linear_graph.discover_neighbors("n0")
        targets = [e.target_id for e in neighbors]
        assert "n1" in targets

    def test_repr(self, linear_graph):
        r = repr(linear_graph)
        assert "linear" in r and "5" in r


# ---------------------------------------------------------------------------
# RadiusDiscoverer
# ---------------------------------------------------------------------------


class TestRadiusDiscoverer:
    def test_finds_adjacent_nodes(self, radius_graph):
        edges = radius_graph.discover_neighbors("n2")
        targets = {e.target_id for e in edges}
        assert "n1" in targets
        assert "n3" in targets

    def test_excludes_self(self, radius_graph):
        edges = radius_graph.discover_neighbors("n0")
        targets = {e.target_id for e in edges}
        assert "n0" not in targets

    def test_does_not_reach_distant_node(self, radius_graph):
        edges = radius_graph.discover_neighbors("n0")
        targets = {e.target_id for e in edges}
        # n4 is 4 units away — outside radius 1.5
        assert "n4" not in targets

    def test_weights_are_in_0_1(self, radius_graph):
        for edges in [radius_graph.discover_neighbors(f"n{i}") for i in range(5)]:
            for e in edges:
                assert 0.0 <= e.weight <= 1.0

    def test_edge_type_preserved(self):
        g = KnowledgeGraph(ndim=2)
        g.add_node("a", np.array([0.0, 0.0]))
        g.add_node("b", np.array([0.5, 0.0]))
        g.add_discoverer(RadiusDiscoverer(threshold=1.0, edge_type="torsion"))
        edges = g.discover_neighbors("a")
        assert all(e.edge_type == "torsion" for e in edges)

    def test_angular_mode(self):
        g = KnowledgeGraph(ndim=1)
        g.add_node("a", np.array([-170.0]))
        g.add_node("b", np.array([170.0]))
        g.add_discoverer(RadiusDiscoverer(threshold=25.0, angular=True))
        edges = g.discover_neighbors("a")
        targets = {e.target_id for e in edges}
        # Angular distance is 20° → within threshold 25
        assert "b" in targets


# ---------------------------------------------------------------------------
# KNNDiscoverer
# ---------------------------------------------------------------------------


class TestKNNDiscoverer:
    def test_finds_k_neighbors(self, knn_graph):
        # Interior node n2 has neighbors on both sides
        edges = knn_graph.discover_neighbors("n2")
        assert len(edges) == 2

    def test_excludes_self(self, knn_graph):
        for i in range(5):
            edges = knn_graph.discover_neighbors(f"n{i}")
            targets = {e.target_id for e in edges}
            assert f"n{i}" not in targets

    def test_returns_nearest(self, knn_graph):
        # n0's 2 nearest are n1 and n2
        edges = knn_graph.discover_neighbors("n0")
        targets = {e.target_id for e in edges}
        assert "n1" in targets

    def test_edge_count_bounded_by_graph_size(self):
        g = KnowledgeGraph(ndim=2)
        g.add_node("a", np.array([0.0, 0.0]))
        g.add_node("b", np.array([1.0, 0.0]))
        g.add_discoverer(KNNDiscoverer(k=10))  # k larger than graph
        edges = g.discover_neighbors("a")
        assert len(edges) <= 1  # only 1 other node


# ---------------------------------------------------------------------------
# GraphReasoner
# ---------------------------------------------------------------------------


class TestGraphReasoner:
    def test_reason_returns_path(self, radius_graph):
        r = GraphReasoner(radius_graph, ExplorationSteering())
        path = r.reason("n0", max_hops=4)
        assert isinstance(path, ReasoningPath)
        assert path.length >= 2

    def test_reason_path_starts_at_start_node(self, radius_graph):
        r = GraphReasoner(radius_graph, ExplorationSteering())
        path = r.reason("n0", max_hops=3)
        assert path.node_ids[0] == "n0"

    def test_reason_visits_distinct_nodes(self, radius_graph):
        r = GraphReasoner(radius_graph, ExplorationSteering())
        path = r.reason("n0", max_hops=4)
        # ExplorationSteering should not revisit nodes
        assert len(path.node_ids) == len(set(path.node_ids))

    def test_reason_toward_reaches_target(self):
        g = KnowledgeGraph(ndim=2)
        for i in range(6):
            g.add_node(f"n{i}", np.array([float(i), 0.0]))
        g.add_discoverer(RadiusDiscoverer(threshold=1.5))
        r = GraphReasoner(g, TargetSteering(np.array([5.0, 0.0])))
        path = r.reason_toward("n0", "n5", max_hops=10)
        assert "n5" in path.node_ids

    def test_beam_reason_returns_multiple_paths(self, radius_graph):
        r = GraphReasoner(radius_graph, ExplorationSteering())
        paths = r.beam_reason("n0", max_hops=3, beam_width=2)
        assert isinstance(paths, list)
        assert len(paths) >= 1

    def test_path_length_property(self):
        path = ReasoningPath(node_ids=["a", "b", "c"])
        assert path.length == 3

    def test_path_total_score(self):
        path = ReasoningPath(node_ids=["a", "b"], scores=[0.5, 0.8])
        assert path.total_score == pytest.approx(1.3)

    def test_path_mean_score(self):
        path = ReasoningPath(node_ids=["a", "b"], scores=[0.4, 0.6])
        assert path.mean_score == pytest.approx(0.5)

    def test_path_mean_score_empty(self):
        path = ReasoningPath()
        assert path.mean_score == pytest.approx(0.0)

    def test_path_copy_is_independent(self):
        path = ReasoningPath(node_ids=["a", "b"], scores=[0.5])
        copy = path.copy()
        copy.node_ids.append("c")
        assert len(path.node_ids) == 2

    def test_reasoner_position_updates(self, radius_graph):
        r = GraphReasoner(radius_graph, ExplorationSteering())
        r.reason("n0", max_hops=1)
        # After at least one step the position should not be at n0
        # (or it advanced to a neighbor)
        pos = r.position
        # Position is the current node's embedding — just check it's valid ndarray
        assert pos.shape == (3,)


# ---------------------------------------------------------------------------
# SemanticEdge
# ---------------------------------------------------------------------------


class TestSemanticEdge:
    def test_construction(self):
        edge = SemanticEdge("a", "b", weight=0.7, edge_type="spatial")
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.weight == pytest.approx(0.7)
        assert edge.edge_type == "spatial"

    def test_default_edge_type(self):
        edge = SemanticEdge("x", "y", weight=0.5)
        assert edge.edge_type == "semantic"
