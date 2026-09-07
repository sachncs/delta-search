"""Graph utility functions for structural analysis.

Provides oracles used by concrete problem implementations:

- ``is_connected``: BFS connectivity check.
- ``connected_components``: BFS component enumeration.
- ``is_planary``: DFS-based planarity heuristic.
- ``is_dominating_set``: domination check.
- ``is_independent_set``: independence check.
- ``vertex_cover_cost``: cost computation.
- ``bfs_reachable``: BFS reachability from a set of source nodes.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import Graph, NodeT

__all__ = [
    "bfs_reachable",
    "is_connected",
    "connected_components",
    "is_planary",
    "is_dominating_set",
    "is_independent_set",
    "vertex_cover_cost",
]


def bfs_reachable(
    graph: Graph[NodeT],
    sources: set[NodeT],
) -> set[NodeT]:
    """Return all nodes reachable from any source via BFS.

    Args:
        graph: The graph to traverse.
        sources: The set of starting nodes.

    Returns:
        Set of all nodes reachable from at least one source.

    """
    if not sources:
        return set()
    visited: set[NodeT] = set()
    queue: deque[NodeT] = deque()
    for s in sources:
        if graph.has_node(s):
            visited.add(s)
            queue.append(s)
    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def is_connected(graph: Graph[NodeT]) -> bool:
    """Return True if the graph is connected (single component).

    Uses BFS from an arbitrary node.  O(V + E).

    Args:
        graph: The graph to check.

    Returns:
        True if the graph has a single connected component.
        Returns True for empty graphs (vacuously true).

    """
    if graph.num_nodes == 0:
        return True

    start = next(iter(graph))
    visited: set[NodeT] = {start}
    queue: deque[NodeT] = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == graph.num_nodes


def connected_components(graph: Graph[NodeT]) -> list[set[NodeT]]:
    """Enumerate connected components via BFS.

    Args:
        graph: The graph to decompose.

    Returns:
        List of sets, each containing the nodes of one connected component.

    """
    remaining: set[NodeT] = set(graph.nodes)
    components: list[set[NodeT]] = []

    while remaining:
        start = next(iter(remaining))
        component: set[NodeT] = {start}
        queue: deque[NodeT] = deque([start])
        remaining.discard(start)

        while queue:
            node = queue.popleft()
            for neighbor in graph.neighbors(node):
                if neighbor in remaining:
                    component.add(neighbor)
                    remaining.discard(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def is_planary(graph: Graph[NodeT]) -> bool:
    """Heuristic planarity check using Euler's formula.

    A connected planar graph satisfies ``|E| <= 3|V| - 6``.
    A disconnected graph is planar if every component is planar.
    This is a necessary but not sufficient condition -- it catches K5
    and K3,3 but may have false positives for some graphs.

    Args:
        graph: The graph to test.

    Returns:
        True if the graph satisfies the planarity necessary condition.

    """
    if graph.num_edges == 0:
        return True

    comps = connected_components(graph)
    for comp in comps:
        num_v = len(comp)
        num_e = sum(1 for u in comp for v in graph.neighbors(u) if v in comp and u < v)
        if num_v >= 3 and num_e > 3 * num_v - 6:
            return False
    return True


def is_dominating_set(
    graph: Graph[NodeT],
    selected: set[NodeT],
) -> bool:
    """Check if ``selected`` is a dominating set.

    A dominating set D has the property that every node not in D
    is adjacent to at least one node in D.

    Args:
        graph: The full input graph.
        selected: The candidate dominating set.

    Returns:
        True if every non-selected node has a neighbor in ``selected``.

    """
    if not selected:
        return graph.num_nodes == 0

    dominated: set[NodeT] = set(selected)
    for node in selected:
        dominated.update(graph.neighbors(node))

    return set(graph.nodes) <= dominated


def is_independent_set(
    graph: Graph[NodeT],
    selected: set[NodeT],
) -> bool:
    """Check if ``selected`` is an independent set.

    An independent set has no two adjacent nodes.

    Args:
        graph: The full input graph.
        selected: The candidate independent set.

    Returns:
        True if no two nodes in ``selected`` are adjacent.

    """
    return all(not graph.neighbors(node) & selected for node in selected)


def vertex_cover_cost(
    graph: Graph[NodeT],
    selected: set[NodeT],
    vertex_costs: dict[NodeT, float] | None = None,
    default_cost: float = 1.0,
) -> float:
    """Compute total cost of a vertex cover.

    Args:
        graph: The full input graph.
        selected: The candidate vertex cover.
        vertex_costs: Per-node cost mapping.  Defaults to ``default_cost``.
        default_cost: Cost for nodes not in ``vertex_costs``.

    Returns:
        Sum of costs for all nodes in ``selected``.

    """
    costs = vertex_costs or {}
    return sum(costs.get(n, default_cost) for n in selected)
