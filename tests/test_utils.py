"""Tests for delta_search.utils module."""

from __future__ import annotations

from delta_search.graph import Graph
from delta_search.utils import (
    bfs_reachable,
    connected_components,
    is_connected,
    is_dominating_set,
    is_independent_set,
    is_planary,
    vertex_cover_cost,
)


class TestIsConnected:
    def test_empty_graph(self) -> None:
        assert is_connected(Graph[int]())

    def test_single_node(self) -> None:
        g = Graph[int](nodes=[1])
        assert is_connected(g)

    def test_connected_path(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
        assert is_connected(g)

    def test_disconnected(self) -> None:
        g = Graph[int].from_edges([(1, 2), (3, 4)])
        assert not is_connected(g)

    def test_triangle(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        assert is_connected(g)


class TestConnectedComponents:
    def test_empty_graph(self) -> None:
        assert connected_components(Graph[int]()) == []

    def test_single_component(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        comps = connected_components(g)
        assert len(comps) == 1
        assert comps[0] == {1, 2, 3}

    def test_two_components(self) -> None:
        g = Graph[int].from_edges([(1, 2), (3, 4)])
        comps = connected_components(g)
        assert len(comps) == 2
        assert {frozenset(c) for c in comps} == {frozenset({1, 2}), frozenset({3, 4})}

    def test_isolated_nodes(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        comps = connected_components(g)
        assert len(comps) == 3


class TestIsPlanary:
    def test_empty_graph(self) -> None:
        assert is_planary(Graph[int]())

    def test_single_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        assert is_planary(g)

    def test_triangle(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        assert is_planary(g)

    def test_complete_k5(self) -> None:
        g = Graph[int].from_edges(
            [
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
            ]
        )
        assert not is_planary(g)

    def test_tree(self) -> None:
        g = Graph[int].from_edges([(1, 2), (1, 3), (1, 4)])
        assert is_planary(g)

    def test_two_nodes(self) -> None:
        g = Graph[int](nodes=[1, 2])
        assert is_planary(g)


class TestIsDominatingSet:
    def test_empty_graph(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        assert not is_dominating_set(g, set())

    def test_complete_graph(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        assert is_dominating_set(g, {1})

    def test_path_graph(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
        assert is_dominating_set(g, {2, 3})
        assert not is_dominating_set(g, {1})

    def test_all_nodes(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert is_dominating_set(g, {1, 2, 3})


class TestIsIndependentSet:
    def test_empty_set(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert is_independent_set(g, set())

    def test_single_node(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert is_independent_set(g, {1})

    def test_non_adjacent(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
        assert is_independent_set(g, {1, 3})

    def test_adjacent_pair(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert not is_independent_set(g, {1, 2})

    def test_triangle(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        assert not is_independent_set(g, {1, 2})
        assert is_independent_set(g, {1})


class TestVertexCoverCost:
    def test_empty_set(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert vertex_cover_cost(g, set()) == 0

    def test_full_cover(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert vertex_cover_cost(g, {1, 2, 3}) == 3.0

    def test_partial_cover(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
        assert vertex_cover_cost(g, {2}) == 1

    def test_custom_costs(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        costs = {1: 5.0, 2: 3.0}
        assert vertex_cover_cost(g, {1, 2}, vertex_costs=costs) == 8.0

    def test_default_cost_override(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        assert vertex_cover_cost(g, {1}, default_cost=10.0) == 10.0


# ---------------------------------------------------------------------------
# BFS reachability
# ---------------------------------------------------------------------------


class TestBfsReachable:
    def test_empty_sources(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        assert bfs_reachable(g, set()) == set()

    def test_nonexistent_source(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        result = bfs_reachable(g, {99})
        assert result == set()

    def test_single_source(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        result = bfs_reachable(g, {1})
        assert result == {1, 2, 3}

    def test_disconnected_components(self) -> None:
        g = Graph[int]()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        result = bfs_reachable(g, {1})
        assert result == {1, 2}
        assert 3 not in result
        assert 4 not in result

    def test_multiple_sources_same_component(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        result = bfs_reachable(g, {1, 3})
        assert result == {1, 2, 3}


# ---------------------------------------------------------------------------
# Connected components edge cases
# ---------------------------------------------------------------------------


class TestConnectedComponentsEdgeCases:
    def test_isolated_nodes_mixed(self) -> None:
        g = Graph[int]()
        g.add_edge(1, 2)
        g.add_node(3)
        g.add_node(4)
        comps = connected_components(g)
        assert len(comps) == 3

    def test_single_node(self) -> None:
        g = Graph[int](nodes=[1])
        comps = connected_components(g)
        assert len(comps) == 1
        assert comps[0] == {1}


# ---------------------------------------------------------------------------
# Planarity edge cases
# ---------------------------------------------------------------------------


class TestPlanarityEdgeCases:
    def test_disconnected_planar_components(self) -> None:
        g = Graph[int]()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        assert is_planary(g)

    def test_empty_graph(self) -> None:
        g = Graph[int]()
        assert is_planary(g)

    def test_single_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        assert is_planary(g)


# ---------------------------------------------------------------------------
# Dominating set edge cases
# ---------------------------------------------------------------------------


class TestDominatingSetEdgeCases:
    def test_empty_graph_empty_set(self) -> None:
        g = Graph[int]()
        assert is_dominating_set(g, set()) is True

    def test_empty_set_nonempty_graph(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        assert is_dominating_set(g, set()) is False
