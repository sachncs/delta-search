"""Shared fixtures for delta_search tests."""

from __future__ import annotations

import pytest

from delta_search.graph import Graph


@pytest.fixture
def empty_graph() -> Graph[int]:
    return Graph[int]()


@pytest.fixture
def triangle() -> Graph[int]:
    """Graph with 3 nodes and 3 edges (complete K3)."""
    return Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])


@pytest.fixture
def path_graph() -> Graph[int]:
    """Graph: 1-2-3-4."""
    return Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])


@pytest.fixture
def dense_graph() -> Graph[int]:
    """Graph with 5 nodes and 8 edges."""
    return Graph[int].from_edges(
        [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        ]
    )
