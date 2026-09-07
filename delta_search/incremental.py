"""Incremental data structures for delta search.

Provides efficient structures for maintaining graph properties incrementally:

- ``UnionFind``: Disjoint-set with path compression and union by rank.
- ``DominationTracker``: Tracks which vertices are dominated.
- ``ConnectivityTracker``: Incremental connectivity via Union-Find.
- ``IndependentSetTracker``: Incremental independence checking.

These are used by problem implementations to achieve O(1) or
O(degree) delta computation instead of full recomputation.

Usage::

    from delta_search.incremental import UnionFind, DominationTracker

    uf = UnionFind()
    uf.add(1)
    uf.add(2)
    uf.union(1, 2)
    assert uf.connected(1, 2)
"""

from __future__ import annotations

from typing import Generic

from .graph import Graph, NodeT

__all__ = [
    "UnionFind",
    "DominationTracker",
    "ConnectivityTracker",
    "IndependentSetTracker",
]


class UnionFind(Generic[NodeT]):
    """Disjoint-set with path compression and union by rank.

    O(alpha(n)) amortized per operation (effectively O(1)).

    Usage::

        uf = UnionFind()
        uf.add(1)
        uf.add(2)
        uf.union(1, 2)
        assert uf.connected(1, 2)
    """

    def __init__(self) -> None:
        """Initialize an empty union-find structure."""
        self._parent: dict[NodeT, NodeT] = {}
        self._rank: dict[NodeT, int] = {}
        self._num_components: int = 0

    def add(self, node: NodeT) -> None:
        """Add a node as a new singleton component.

        Args:
            node: The node to add.

        Raises:
            KeyError: If node already exists.

        """
        if node in self._parent:
            raise KeyError(f"Node {node!r} already in UnionFind")
        self._parent[node] = node
        self._rank[node] = 0
        self._num_components += 1

    def find(self, node: NodeT) -> NodeT:
        """Find the canonical representative of node's component.

        Args:
            node: The node to look up.

        Returns:
            The root representative of the component.

        Raises:
            KeyError: If node is not in the UnionFind.

        """
        if node not in self._parent:
            raise KeyError(f"Node {node!r} not in UnionFind")
        # Path compression
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        # Compress path
        while node != root:
            nxt = self._parent[node]
            self._parent[node] = root
            node = nxt
        return root

    def union(self, a: NodeT, b: NodeT) -> bool:
        """Merge the components containing a and b.

        Args:
            a: First node.
            b: Second node.

        Returns:
            True if a and b were in different components (merged).
            False if already in the same component.

        Raises:
            KeyError: If either node is not in the UnionFind.

        """
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        # Union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        self._num_components -= 1
        return True

    def connected(self, a: NodeT, b: NodeT) -> bool:
        """Check if a and b are in the same component.

        Args:
            a: First node.
            b: Second node.

        Returns:
            True if they share a component root.

        """
        return self.find(a) == self.find(b)

    @property
    def num_components(self) -> int:
        """Number of disjoint components."""
        return self._num_components

    def __contains__(self, node: NodeT) -> bool:
        return node in self._parent

    def __len__(self) -> int:
        return len(self._parent)


class DominationTracker(Generic[NodeT]):
    """Incremental tracker for dominating set membership.

    Maintains, for each vertex in the input graph, the count of its
    neighbors that are in the selected dominating set. A vertex is
    dominated when its count is > 0 (or when it is itself selected).

    All operations are O(degree(node)) in the input graph.

    Usage::

        tracker = DominationTracker(input_graph)
        tracker.add_to_dominating_set(node)
        assert tracker.is_dominating_set()
    """

    def __init__(self, input_graph: Graph[NodeT]) -> None:
        """Initialize the domination tracker.

        Args:
            input_graph: The underlying graph to track domination over.

        """
        self._graph: Graph[NodeT] = input_graph
        self._selected: set[NodeT] = set()
        self._dominated_count: dict[NodeT, int] = {}
        self._all_nodes: set[NodeT] = set(self._graph.nodes)

        # Initialize domination counts to 0 for all nodes
        for n in self._all_nodes:
            self._dominated_count[n] = 0

    def add_to_dominating_set(self, node: NodeT) -> None:
        """Add a node to the dominating set and update counts.

        Args:
            node: The node to add.

        """
        if node in self._selected:
            return
        self._selected.add(node)
        # This node dominates itself
        self._dominated_count[node] += 1
        # All neighbors are now dominated by this node
        for nb in self._graph.neighbors(node):
            self._dominated_count[nb] += 1

    def remove_from_dominating_set(self, node: NodeT) -> None:
        """Remove a node from the dominating set and update counts.

        Args:
            node: The node to remove.

        """
        if node not in self._selected:
            return
        self._selected.discard(node)
        self._dominated_count[node] = max(0, self._dominated_count[node] - 1)
        for nb in self._graph.neighbors(node):
            self._dominated_count[nb] = max(0, self._dominated_count[nb] - 1)

    def is_dominated(self, node: NodeT) -> bool:
        """Check if a node is dominated (has a neighbor in the set or is selected).

        Args:
            node: The node to check.

        Returns:
            True if the node is dominated.

        """
        return self._dominated_count.get(node, 0) > 0

    def is_dominating_set(self) -> bool:
        """Check if the current selected set is a dominating set.

        Returns:
            True if every node in the graph is dominated.

        """
        return all(self._dominated_count.get(n, 0) > 0 for n in self._all_nodes)

    @property
    def selected(self) -> set[NodeT]:
        """The current dominating set."""
        return set(self._selected)

    @property
    def num_selected(self) -> int:
        """Number of nodes in the dominating set."""
        return len(self._selected)


class ConnectivityTracker(Generic[NodeT]):
    """Incremental connectivity tracker using Union-Find.

    Tracks whether a growing set of nodes forms a connected subgraph.
    Adding a node unions it with all its neighbors already in the set.

    All operations are O(degree(node) * alpha(n)).

    Usage::

        tracker = ConnectivityTracker(input_graph)
        tracker.add(1)
        tracker.add(2)
        assert tracker.connected(1, 2)
    """

    def __init__(self, input_graph: Graph[NodeT]) -> None:
        """Initialize the connectivity tracker.

        Args:
            input_graph: The underlying graph to track connectivity over.

        """
        self._graph: Graph[NodeT] = input_graph
        self._uf: UnionFind[NodeT] = UnionFind()
        self._nodes: set[NodeT] = set()

    def add(self, node: NodeT) -> None:
        """Add a node and connect it to its neighbors in the set.

        Args:
            node: The node to add.

        """
        if node in self._nodes:
            return
        self._nodes.add(node)
        self._uf.add(node)
        for nb in self._graph.neighbors(node):
            if nb in self._nodes:
                self._uf.union(node, nb)

    def remove(self, node: NodeT) -> None:
        """Remove a node (invalidates connectivity — use with care).

        After removal, you should rebuild the tracker for accurate
        connectivity checks. This is a convenience for the undo path.

        Args:
            node: The node to remove.

        """
        self._nodes.discard(node)
        # Rebuild from scratch for correctness
        old_nodes = list(self._nodes)
        self._uf = UnionFind()
        self._nodes = set()
        for n in old_nodes:
            self.add(n)

    def is_connected(self) -> bool:
        """Check if all nodes in the set are mutually connected.

        Returns:
            True if the set is connected (or empty/single).

        """
        if len(self._nodes) <= 1:
            return True
        return self._uf.num_components == 1

    def connected(self, a: NodeT, b: NodeT) -> bool:
        """Check if two specific nodes are connected.

        Args:
            a: First node.
            b: Second node.

        Returns:
            True if they are in the same component.

        """
        return self._uf.connected(a, b)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the tracked set."""
        return len(self._nodes)

    @property
    def nodes(self) -> frozenset[NodeT]:
        """The current node set."""
        return frozenset(self._nodes)


class IndependentSetTracker(Generic[NodeT]):
    """Incremental tracker for independent set membership.

    Maintains the selected set and checks independence in O(degree(node))
    per operation by testing whether any neighbor of a candidate node is
    already in the set.

    Usage::

        tracker = IndependentSetTracker(input_graph)
        tracker.add(1)  # OK
        # tracker.add(2) would fail if 1-2 is an edge
    """

    def __init__(self, input_graph: Graph[NodeT]) -> None:
        """Initialize the independent set tracker.

        Args:
            input_graph: The underlying graph to track independence over.

        """
        self._graph: Graph[NodeT] = input_graph
        self._selected: set[NodeT] = set()

    def can_add(self, node: NodeT) -> bool:
        """Check if adding node would preserve independence.

        Args:
            node: The candidate node.

        Returns:
            True if no neighbor of node is in the selected set.

        """
        return not (self._graph.neighbors(node) & self._selected)

    def add(self, node: NodeT) -> None:
        """Add a node to the independent set.

        Args:
            node: The node to add.

        Raises:
            ValueError: If adding would violate independence.

        """
        if not self.can_add(node):
            raise ValueError(
                f"Cannot add {node!r}: neighbor already in independent set"
            )
        self._selected.add(node)

    def remove(self, node: NodeT) -> None:
        """Remove a node from the independent set.

        Args:
            node: The node to remove.

        """
        self._selected.discard(node)

    def is_independent(self) -> bool:
        """Check if the current set is independent.

        Returns:
            True if no two selected nodes are adjacent.

        """
        return all(self.can_add(n) for n in self._selected)

    @property
    def selected(self) -> frozenset[NodeT]:
        """The current independent set."""
        return frozenset(self._selected)

    @property
    def num_selected(self) -> int:
        """Number of nodes in the independent set."""
        return len(self._selected)
