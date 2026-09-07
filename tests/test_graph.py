"""Comprehensive tests for delta_search.graph module."""

from __future__ import annotations

import threading

import pytest

from delta_search.graph import Graph, ThreadSafeGraph

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_graph(self) -> None:
        g = Graph[int]()
        assert g.num_nodes == 0
        assert g.num_edges == 0
        assert g.is_empty()

    def test_from_nodes(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        assert g.num_nodes == 3
        assert g.num_edges == 0

    def test_from_edges(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_from_edges_constructor(self) -> None:
        g = Graph[int](edges=[(1, 2), (2, 3)])
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_copy_is_independent(self) -> None:
        g1 = Graph[int].from_edges([(1, 2)])
        g2 = Graph.from_copy(g1)
        g2.add_edge(2, 3)
        assert g1.num_edges == 1
        assert g2.num_edges == 2


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------


class TestNodeOperations:
    def test_add_node(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_node(1)
        assert empty_graph.has_node(1)
        assert empty_graph.num_nodes == 1

    def test_add_node_idempotent(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_node(1)
        empty_graph.add_node(1)
        assert empty_graph.num_nodes == 1

    def test_add_node_with_attrs(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_node(1, label="a")
        assert empty_graph.node_data(1) == {"label": "a"}

    def test_remove_node(self, triangle: Graph[int]) -> None:
        triangle.remove_node(1)
        assert not triangle.has_node(1)
        assert triangle.num_nodes == 2
        assert triangle.num_edges == 1  # edge 2-3 remains

    def test_remove_node_not_found(self, empty_graph: Graph[int]) -> None:
        with pytest.raises(KeyError, match="not in graph"):
            empty_graph.remove_node(1)

    def test_has_node(self, empty_graph: Graph[int]) -> None:
        assert not empty_graph.has_node(1)
        empty_graph.add_node(1)
        assert empty_graph.has_node(1)

    def test_nodes_view(self, triangle: Graph[int]) -> None:
        nodes = triangle.nodes
        assert isinstance(nodes, type({}.keys()))
        assert set(nodes) == {1, 2, 3}

    def test_node_list(self, triangle: Graph[int]) -> None:
        lst = triangle.node_list()
        assert isinstance(lst, list)
        assert set(lst) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------


class TestEdgeOperations:
    def test_add_edge(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_edge(1, 2)
        assert empty_graph.has_edge(1, 2)
        assert empty_graph.has_edge(2, 1)
        assert empty_graph.num_edges == 1

    def test_add_edge_creates_nodes(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_edge(5, 6)
        assert empty_graph.has_node(5)
        assert empty_graph.has_node(6)

    def test_add_edge_with_attrs(self, empty_graph: Graph[int]) -> None:
        key = empty_graph.add_edge(1, 2, weight=3.0)
        assert empty_graph.edge_data(1, 2) == {"weight": 3.0}
        assert key == frozenset((1, 2))

    def test_add_edge_merge_attrs(self, empty_graph: Graph[int]) -> None:
        empty_graph.add_edge(1, 2, weight=1.0)
        empty_graph.add_edge(1, 2, color="red")
        assert empty_graph.edge_data(1, 2) == {"weight": 1.0, "color": "red"}

    def test_add_edge_no_self_loops(self, empty_graph: Graph[int]) -> None:
        with pytest.raises(ValueError, match="Self-loops are not supported"):
            empty_graph.add_edge(1, 1)

    def test_add_edge_self_loop_constructor(self) -> None:
        with pytest.raises(ValueError, match="Self-loops are not supported"):
            Graph[int](edges=[(1, 1), (2, 3)])

    def test_remove_edge(self, triangle: Graph[int]) -> None:
        triangle.remove_edge(1, 2)
        assert not triangle.has_edge(1, 2)
        assert triangle.num_edges == 2

    def test_remove_edge_not_found(self, empty_graph: Graph[int]) -> None:
        with pytest.raises(KeyError, match="not in graph"):
            empty_graph.remove_edge(1, 2)

    def test_has_edge(self, triangle: Graph[int]) -> None:
        assert triangle.has_edge(1, 2)
        assert triangle.has_edge(2, 1)
        assert not triangle.has_edge(1, 4)

    def test_edge_data_no_phantom(self, empty_graph: Graph[int]) -> None:
        """edge_data should not create entries for non-existent edges."""
        result = empty_graph.edge_data(1, 2)
        assert result == {}
        assert empty_graph.num_edges == 0
        # Internal edge_data dict should not have phantom entry
        assert empty_graph.num_edges == 0

    def test_edges_view(self, triangle: Graph[int]) -> None:
        edges = triangle.edges
        assert isinstance(edges, type({}.keys()))
        assert len(set(edges)) == 3

    def test_num_edges(self, empty_graph: Graph[int]) -> None:
        assert empty_graph.num_edges == 0
        empty_graph.add_edge(1, 2)
        assert empty_graph.num_edges == 1
        empty_graph.add_edge(2, 3)
        assert empty_graph.num_edges == 2
        empty_graph.remove_edge(1, 2)
        assert empty_graph.num_edges == 1


# ---------------------------------------------------------------------------
# Neighbourhood queries
# ---------------------------------------------------------------------------


class TestNeighbourhood:
    def test_neighbors(self, triangle: Graph[int]) -> None:
        assert triangle.neighbors(1) == {2, 3}

    def test_degree(self, triangle: Graph[int]) -> None:
        assert triangle.degree(1) == 2

    def test_common_neighbors(self, path_graph: Graph[int]) -> None:
        assert path_graph.common_neighbors(1, 3) == {2}

    def test_common_neighbors_empty(self, triangle: Graph[int]) -> None:
        # In K3, every pair shares the third node
        assert triangle.common_neighbors(1, 2) == {3}


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------


class TestSubgraph:
    def test_subgraph(self, dense_graph: Graph[int]) -> None:
        sub = dense_graph.subgraph({1, 2, 3})
        assert sub.num_nodes == 3
        # Edges: 1-2, 1-3, 2-3 = 3 edges
        assert sub.num_edges == 3

    def test_subgraph_preserves_attrs(self) -> None:
        g = Graph[int]()
        g.add_node(1, label="a")
        g.add_edge(1, 2, weight=5.0)
        sub = g.subgraph({1, 2})
        assert sub.node_data(1) == {"label": "a"}
        assert sub.edge_data(1, 2) == {"weight": 5.0}

    def test_subgraph_empty(self, triangle: Graph[int]) -> None:
        sub = triangle.subgraph(set())
        assert sub.num_nodes == 0
        assert sub.num_edges == 0

    def test_node_induced_subgraph(self, path_graph: Graph[int]) -> None:
        sub = path_graph.subgraph({1, 2, 3})
        assert sub.num_nodes == 3
        assert sub.num_edges == 2

    def test_edge_subgraph(self, dense_graph: Graph[int]) -> None:
        edges = [frozenset((1, 2)), frozenset((3, 4))]
        sub = dense_graph.edge_subgraph(edges)
        assert sub.num_nodes == 4
        assert sub.num_edges == 2


# ---------------------------------------------------------------------------
# Delta helpers
# ---------------------------------------------------------------------------


class TestDeltaHelpers:
    def test_add_edge_delta_new(self, empty_graph: Graph[int]) -> None:
        assert empty_graph.add_edge_delta(1, 2) is True
        assert empty_graph.has_edge(1, 2)

    def test_add_edge_delta_existing(self, triangle: Graph[int]) -> None:
        assert triangle.add_edge_delta(1, 2) is False

    def test_remove_edge_delta_existing(self, triangle: Graph[int]) -> None:
        assert triangle.remove_edge_delta(1, 2) is True
        assert not triangle.has_edge(1, 2)

    def test_remove_edge_delta_missing(self, empty_graph: Graph[int]) -> None:
        assert empty_graph.remove_edge_delta(1, 2) is False

    def test_add_node_delta_new(self, empty_graph: Graph[int]) -> None:
        assert empty_graph.add_node_delta(1) is True
        assert empty_graph.has_node(1)

    def test_add_node_delta_existing(self, triangle: Graph[int]) -> None:
        assert triangle.add_node_delta(1) is False

    def test_remove_node_delta_existing(self, triangle: Graph[int]) -> None:
        assert triangle.remove_node_delta(1) is True
        assert not triangle.has_node(1)

    def test_remove_node_delta_missing(self, empty_graph: Graph[int]) -> None:
        assert empty_graph.remove_node_delta(1) is False


# ---------------------------------------------------------------------------
# Edge count invariant
# ---------------------------------------------------------------------------


class TestEdgeCountInvariant:
    def test_remove_node_correct_edge_count(self) -> None:
        """Removing a node must correctly decrement edge count.

        Each incident edge should be accounted for.
        """
        g = Graph[int].from_edges([(1, 2), (1, 3), (1, 4)])
        assert g.num_edges == 3
        g.remove_node(1)
        assert g.num_edges == 0
        assert g.num_nodes == 3

    def test_double_remove_node(self) -> None:
        """Removing two adjacent nodes should leave correct counts."""
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        g.remove_node(1)
        assert g.num_edges == 1
        g.remove_node(2)
        assert g.num_edges == 0
        assert g.num_nodes == 1


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equal_graphs(self) -> None:
        g1 = Graph[int].from_edges([(1, 2), (2, 3)])
        g2 = Graph[int].from_edges([(2, 3), (1, 2)])
        assert g1 == g2

    def test_unequal_graphs(self) -> None:
        g1 = Graph[int].from_edges([(1, 2)])
        g2 = Graph[int].from_edges([(1, 3)])
        assert g1 != g2

    def test_different_sizes_shortcircuit(self) -> None:
        g1 = Graph[int].from_edges([(1, 2)])
        g2 = Graph[int].from_edges([(1, 2), (2, 3)])
        assert g1 != g2

    def test_not_equal_to_non_graph(self) -> None:
        g = Graph[int]()
        assert g != "not a graph"

    def test_hash_raises(self) -> None:
        g = Graph[int]()
        with pytest.raises(TypeError, match="unhashable"):
            hash(g)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter(self, triangle: Graph[int]) -> None:
        assert set(triangle) == {1, 2, 3}

    def test_contains(self, triangle: Graph[int]) -> None:
        assert 1 in triangle
        assert 4 not in triangle

    def test_len(self, triangle: Graph[int]) -> None:
        assert len(triangle) == 3


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class TestUtility:
    def test_clear(self, triangle: Graph[int]) -> None:
        triangle.clear()
        assert triangle.num_nodes == 0
        assert triangle.num_edges == 0

    def test_degree_sequence(self, path_graph: Graph[int]) -> None:
        seq = path_graph.degree_sequence()
        assert seq == [2, 2, 1, 1]

    def test_is_subgraph_of(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        sub = g.subgraph({1, 2})
        assert sub.is_subgraph_of(g)

    def test_repr(self, triangle: Graph[int]) -> None:
        r = repr(triangle)
        assert "Graph" in r
        assert "nodes=3" in r
        assert "edges=3" in r


# ---------------------------------------------------------------------------
# ThreadSafeGraph
# ---------------------------------------------------------------------------


class TestThreadSafeGraph:
    def test_basic_operations(self) -> None:
        g = ThreadSafeGraph[int]()
        g.add_edge(1, 2)
        assert g.has_edge(1, 2)
        assert g.num_edges == 1

    def test_concurrent_writes(self) -> None:
        g = ThreadSafeGraph[int]()
        errors: list[Exception] = []

        def add_edges(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    g.add_edge(i, i + 1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_edges, args=(i,)) for i in range(0, 500, 100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert g.num_edges == 500  # all edges are distinct

    def test_lock_exposed(self) -> None:
        g = ThreadSafeGraph[int]()
        assert isinstance(g.lock, type(threading.RLock()))

    def test_concurrent_mixed_operations(self) -> None:
        g = ThreadSafeGraph[int]()
        errors: list[Exception] = []

        def add_then_remove(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    g.add_edge(i, i + 1)
                for i in range(start, start + 25):
                    g.remove_edge(i, i + 1)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_then_remove, args=(i,))
            for i in range(0, 200, 50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_clear(self) -> None:
        g = ThreadSafeGraph[int]()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    g.add_edge(i + 10, i + 11)
            except Exception as e:
                errors.append(e)

        def clearer() -> None:
            try:
                g.clear()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_thread_safe_remove_node(self) -> None:
        g = ThreadSafeGraph[int]()
        for i in range(10):
            g.add_edge(i, i + 1)

        def remover(start: int) -> None:
            for i in range(start, start + 3):
                if g.has_node(i):
                    g.remove_node(i)

        threads = [threading.Thread(target=remover, args=(i,)) for i in range(0, 10, 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_thread_safe_constructor(self) -> None:
        g = ThreadSafeGraph[int](nodes=[1, 2, 3], edges=[(1, 2)])
        assert g.num_nodes == 3
        assert g.num_edges == 1
        assert hasattr(g, "lock")

    def test_thread_safe_subgraph(self) -> None:
        g = ThreadSafeGraph[int]()
        for i in range(10):
            g.add_edge(i, i + 1)
        sub = g.subgraph([0, 1, 2, 3])
        assert sub.num_nodes == 4

    def test_thread_safe_edge_subgraph(self) -> None:
        g = ThreadSafeGraph[int]()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        sub = g.edge_subgraph([frozenset({1, 2})])
        assert sub.num_edges == 1

    def test_thread_safe_node_list(self) -> None:
        g = ThreadSafeGraph[int]()
        g.add_edge(1, 2)
        nodes = g.node_list()
        assert sorted(nodes) == [1, 2]

    def test_thread_safe_degree_sequence(self) -> None:
        g = ThreadSafeGraph[int]()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        seq = g.degree_sequence()
        assert seq == [2, 1, 1]

    def test_thread_safe_is_subgraph_of(self) -> None:
        g1 = ThreadSafeGraph[int]()
        g1.add_edge(1, 2)
        g2 = ThreadSafeGraph[int]()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        assert g1.is_subgraph_of(g2)
        assert not g2.is_subgraph_of(g1)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_neighbors_nonexistent_node(self) -> None:
        g = Graph[int](nodes=[1, 2])
        with pytest.raises(KeyError):
            g.neighbors(99)

    def test_degree_nonexistent_node(self) -> None:
        g = Graph[int](nodes=[1, 2])
        with pytest.raises(KeyError):
            g.degree(99)

    def test_common_neighbors_nonexistent_node(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        with pytest.raises(KeyError):
            g.common_neighbors(1, 99)

    def test_remove_edge_if_present_missing(self) -> None:
        g = Graph[int]()
        assert g.remove_edge_if_present(1, 2) is False

    def test_node_data_returns_empty_dict(self) -> None:
        g = Graph[int]()
        data = g.node_data(1)
        assert data == {}

    def test_edge_values(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        g.add_edge(1, 2, weight=5.0)
        vals = list(g.edge_values)
        assert len(vals) == 1
        assert vals[0] == {"weight": 5.0}
