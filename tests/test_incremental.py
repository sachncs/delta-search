"""Tests for delta_search.incremental module."""

from __future__ import annotations

import pytest

from delta_search.graph import Graph
from delta_search.incremental import (
    ConnectivityTracker,
    DominationTracker,
    IndependentSetTracker,
    UnionFind,
)


class TestUnionFind:
    def test_add_and_find(self) -> None:
        uf = UnionFind()
        uf.add(1)
        assert uf.find(1) == 1

    def test_add_duplicate_raises(self) -> None:
        uf = UnionFind()
        uf.add(1)
        with pytest.raises(KeyError, match="already in UnionFind"):
            uf.add(1)

    def test_find_not_found_raises(self) -> None:
        uf = UnionFind()
        with pytest.raises(KeyError, match="not in UnionFind"):
            uf.find(1)

    def test_union_merges(self) -> None:
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        assert uf.union(1, 2) is True
        assert uf.connected(1, 2)

    def test_union_same_component(self) -> None:
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        uf.union(1, 2)
        assert uf.union(1, 2) is False

    def test_connected_false(self) -> None:
        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        assert uf.connected(1, 2) is False

    def test_num_components(self) -> None:
        uf = UnionFind()
        assert uf.num_components == 0
        uf.add(1)
        uf.add(2)
        assert uf.num_components == 2
        uf.union(1, 2)
        assert uf.num_components == 1

    def test_contains(self) -> None:
        uf = UnionFind()
        uf.add(1)
        assert 1 in uf
        assert 2 not in uf

    def test_len(self) -> None:
        uf = UnionFind()
        assert len(uf) == 0
        uf.add(1)
        uf.add(2)
        assert len(uf) == 2

    def test_path_compression(self) -> None:
        uf = UnionFind()
        for i in range(10):
            uf.add(i)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.connected(0, 3)
        root = uf.find(3)
        assert root == uf.find(0)

    def test_union_by_rank(self) -> None:
        uf = UnionFind()
        for i in range(5):
            uf.add(i)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)
        uf.union(3, 4)
        assert uf.num_components == 1

    def test_union_not_found_raises(self) -> None:
        uf = UnionFind()
        uf.add(1)
        with pytest.raises(KeyError):
            uf.union(1, 99)


class TestDominationTracker:
    def _make_graph(self) -> Graph[int]:
        return Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])

    def test_add_to_dominating_set(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        tracker.add_to_dominating_set(2)
        assert tracker.is_dominated(2)
        assert tracker.is_dominated(1)
        assert tracker.is_dominated(3)
        assert not tracker.is_dominated(4)

    def test_remove_from_dominating_set(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        tracker.add_to_dominating_set(2)
        tracker.remove_from_dominating_set(2)
        assert not tracker.is_dominated(2)

    def test_add_idempotent(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        tracker.add_to_dominating_set(2)
        tracker.add_to_dominating_set(2)
        assert tracker.num_selected == 1

    def test_remove_not_selected(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        tracker.remove_from_dominating_set(2)
        assert tracker.num_selected == 0

    def test_is_dominating_set(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        assert not tracker.is_dominating_set()
        tracker.add_to_dominating_set(2)
        tracker.add_to_dominating_set(4)
        assert tracker.is_dominating_set()

    def test_selected_property(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        tracker.add_to_dominating_set(2)
        selected = tracker.selected
        assert 2 in selected
        assert isinstance(selected, set)

    def test_num_selected(self) -> None:
        g = self._make_graph()
        tracker = DominationTracker(g)
        assert tracker.num_selected == 0
        tracker.add_to_dominating_set(2)
        assert tracker.num_selected == 1


class TestConnectivityTracker:
    def _make_graph(self) -> Graph[int]:
        return Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])

    def test_add_node(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        assert tracker.num_nodes == 1
        assert tracker.is_connected()

    def test_add_connected_nodes(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        tracker.add(2)
        assert tracker.connected(1, 2)

    def test_add_idempotent(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        tracker.add(1)
        assert tracker.num_nodes == 1

    def test_remove_node(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        tracker.add(2)
        tracker.remove(2)
        assert tracker.num_nodes == 1
        assert tracker.is_connected()

    def test_nodes_property(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        tracker.add(2)
        assert frozenset([1, 2]) == tracker.nodes

    def test_disconnected_set(self) -> None:
        g = Graph[int].from_edges([(1, 2), (3, 4)])
        tracker = ConnectivityTracker(g)
        tracker.add(1)
        tracker.add(3)
        assert not tracker.is_connected()

    def test_empty_set_connected(self) -> None:
        g = self._make_graph()
        tracker = ConnectivityTracker(g)
        assert tracker.is_connected()


class TestIndependentSetTracker:
    def _make_graph(self) -> Graph[int]:
        return Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])

    def test_can_add(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        assert tracker.can_add(1)
        assert tracker.can_add(3)

    def test_add(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        assert 1 in tracker.selected

    def test_add_violates_independence(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        with pytest.raises(ValueError, match="Cannot add"):
            tracker.add(2)

    def test_remove(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        tracker.remove(1)
        assert 1 not in tracker.selected

    def test_is_independent(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        tracker.add(3)
        assert tracker.is_independent()

    def test_num_selected(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        assert tracker.num_selected == 0
        tracker.add(1)
        assert tracker.num_selected == 1

    def test_can_add_neighbor_selected(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        assert not tracker.can_add(2)

    def test_remove_nonexistent(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.remove(99)
        assert tracker.num_selected == 0

    def test_selected_property(self) -> None:
        g = self._make_graph()
        tracker = IndependentSetTracker(g)
        tracker.add(1)
        tracker.add(3)
        selected = tracker.selected
        assert isinstance(selected, frozenset)
        assert 1 in selected
        assert 3 in selected
