"""Optimized graph data structures for subgraph extraction.

Provides a lightweight, cache-friendly graph representation optimized for:

- O(1) neighbor lookups via adjacency sets.
- O(1) edge existence checks via frozenset-based edge registry.
- Efficient incremental delta computation (add/remove single edges/nodes).
- Fast subgraph isolation for candidate evaluation.

Thread safety:
    ``Graph`` is not thread-safe.  All public methods mutate internal state
    without locking.  Use ``ThreadSafeGraph`` for concurrent access.

Self-loops:
    Self-loops (u, u) are rejected at the API boundary.  The frozenset edge
    key design cannot distinguish a self-loop from a singleton set, and
    self-loops have ambiguous semantics for undirected graphs.  Use
    ``node_attrs()`` to store self-referential attributes.
"""

from __future__ import annotations

import copy
import threading
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, KeysView, ValuesView

__all__ = [
    "Node",
    "NodeT",
    "Graph",
    "ThreadSafeGraph",
]


@runtime_checkable
class Node(Protocol):
    """Protocol for graph node identifiers.

    Nodes must be hashable for set/dict storage and support ordering
    via ``__lt__`` for deterministic edge canonicalisation.
    """

    def __hash__(self) -> int: ...
    def __lt__(self, other: Any) -> bool: ...


NodeT = TypeVar("NodeT", bound=Node)


class Graph(Generic[NodeT]):
    """Adjacency-set graph with O(1) membership tests and incremental mutation.

    Internal storage:

        adj: Adjacency sets.  Both u->v and v->u are stored so that
            ``v in adj[u]`` is always O(1).
        edge_attrs: Edge attributes keyed by canonical frozenset.  Using
            frozenset as key means (u, v) and (v, u) map to the same entry.
        node_attrs: Per-node attribute bag.
        edge_count: Cached edge count avoids summing len(adj[u]) on every
            access.

    The class is generic over ``NodeT`` (must be hashable and comparable).
    Edge and node payloads are plain dicts to stay lightweight; typed
    wrappers can be layered on top for specific problems.

    Not thread-safe -- use ``ThreadSafeGraph`` for concurrent access.
    """

    def __init__(
        self,
        nodes: Iterable[NodeT] | None = None,
        edges: Iterable[tuple[NodeT, NodeT]] | None = None,
    ) -> None:
        """Initialize the graph.

        Args:
            nodes: Optional iterable of nodes to add.
            edges: Optional iterable of (u, v) edges to add.

        """
        self.adj: dict[NodeT, set[NodeT]] = {}
        self.edge_attrs: dict[frozenset[NodeT], dict[str, Any]] = {}
        self.node_attrs: dict[NodeT, dict[str, Any]] = {}
        self.edge_count: int = 0

        if nodes is not None:
            for n in nodes:
                self.add_node(n)

        if edges is not None:
            for u, v in edges:
                self.add_edge(u, v)

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[NodeT, NodeT]]) -> Graph[NodeT]:
        """Create a graph from an edge list.

        Args:
            edges: Iterable of (u, v) tuples representing undirected edges.

        Returns:
            A new Graph containing all nodes and edges.

        """
        g: Graph[NodeT] = cls()
        for u, v in edges:
            g.add_edge(u, v)
        return g

    @classmethod
    def from_copy(cls, other: Graph[NodeT]) -> Graph[NodeT]:
        """Return a deep copy safe to mutate without affecting the source.

        Args:
            other: The graph to copy.

        Returns:
            An independent deep copy of ``other``.

        """
        return copy.deepcopy(other)

    def add_node(self, node: NodeT, **attrs: Any) -> None:
        """Insert a node (idempotent).  O(1).

        Args:
            node: The node identifier.
            **attrs: Arbitrary keyword attributes stored on the node.

        """
        if node not in self.adj:
            self.adj[node] = set()
        if attrs:
            self.node_attrs.setdefault(node, {}).update(attrs)

    def remove_node(self, node: NodeT) -> None:
        """Remove a node and all incident edges.  O(degree(node)).

        Args:
            node: The node to remove.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        if node not in self.adj:
            raise KeyError(f"Node {node!r} not in graph")

        for neighbor in list(self.adj[node]):
            self.remove_edge_if_present(node, neighbor)

        del self.adj[node]
        self.node_attrs.pop(node, None)

    def has_node(self, node: NodeT) -> bool:
        """Check membership.  O(1).

        Args:
            node: The node to look up.

        Returns:
            True if ``node`` is in the graph.

        """
        return node in self.adj

    @property
    def nodes(self) -> KeysView[NodeT]:
        """All nodes.  Returns a view (no copy, O(1))."""
        return self.adj.keys()

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.adj)

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return self.edge_count

    def node_data(self, node: NodeT) -> dict[str, Any]:
        """Return the attribute dict for node (creates if absent).

        Note: This method creates an entry in ``node_attrs`` for nodes
        not yet in the attribute bag.  This is intentional for undo
        support where node attributes must be restored before the node
        is re-added to ``adj``.

        Args:
            node: The node whose attributes to retrieve.

        Returns:
            Mutable attribute dictionary for the node.

        """
        return self.node_attrs.setdefault(node, {})

    def add_edge(self, u: NodeT, v: NodeT, **attrs: Any) -> frozenset[NodeT]:
        """Add an undirected edge {u, v}.  O(1).

        Self-loops raise ``ValueError``.  If the edge already exists,
        attrs are merged into the existing attribute dict.

        Args:
            u: First endpoint.
            v: Second endpoint.
            **attrs: Arbitrary keyword attributes stored on the edge.

        Returns:
            The canonical frozenset key for the edge.

        Raises:
            ValueError: If ``u == v`` (self-loop).

        """
        if u == v:
            raise ValueError(
                f"Self-loops are not supported.  Node: {u!r}.  "
                "Use node_attrs() to store self-referential attributes."
            )

        key = frozenset((u, v))

        self.add_node(u)
        self.add_node(v)

        is_new = u not in self.adj[v]
        self.adj[u].add(v)
        self.adj[v].add(u)

        if is_new:
            self.edge_count += 1
            self.edge_attrs[key] = dict(attrs) if attrs else {}
        elif attrs:
            self.edge_attrs.setdefault(key, {}).update(attrs)

        return key

    def remove_edge(self, u: NodeT, v: NodeT) -> None:
        """Remove edge {u, v}.  O(1).

        Args:
            u: First endpoint.
            v: Second endpoint.

        Raises:
            KeyError: If the edge is not in the graph.

        """
        if not self.has_edge(u, v):
            raise KeyError(f"Edge ({u!r}, {v!r}) not in graph")
        self.remove_edge_if_present(u, v)

    def remove_edge_if_present(self, u: NodeT, v: NodeT) -> bool:
        """Remove edge if present.  Returns True if the edge was removed.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge existed and was removed, False otherwise.

        """
        if not self.has_edge(u, v):
            return False
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        key = frozenset((u, v))
        self.edge_attrs.pop(key, None)
        self.edge_count -= 1
        return True

    def has_edge(self, u: NodeT, v: NodeT) -> bool:
        """O(1) edge existence check.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge {u, v} exists.

        """
        return u in self.adj and v in self.adj[u]

    @property
    def edges(self) -> KeysView[frozenset[NodeT]]:
        """All edge keys.  Returns a view (no copy, O(1)).

        .. note::
           Iteration order of ``edges`` follows ``edge_attrs.keys()``,
           which depends on ``frozenset`` hashing. For deterministic
           ordering across ``PYTHONHASHSEED``, use
           :meth:`sorted_edges` instead.
        """
        return self.edge_attrs.keys()

    def sorted_edges(self) -> list[tuple[NodeT, NodeT]]:
        """Return edges as a sorted list of ``(u, v)`` tuples.

        Each edge's endpoints are emitted in canonical order
        (``u <= v`` by ``Node.__lt__``). Sorting eliminates the
        hash-dependent iteration order of ``edge_attrs.keys()`` so
        output is reproducible across ``PYTHONHASHSEED``.

        Returns:
            Sorted list of ``(u, v)`` endpoint tuples.
        """
        ordered: list[tuple[NodeT, NodeT]] = []
        for key in self.edge_attrs:
            it = iter(key)
            u = next(it)
            v = next(it)
            ordered.append((u, v) if u <= v else (v, u))
        ordered.sort()
        return ordered

    @property
    def edge_values(self) -> ValuesView[dict[str, Any]]:
        """Edge attribute dicts.  Returns a view (no copy, O(1))."""
        return self.edge_attrs.values()

    def edge_data(self, u: NodeT, v: NodeT) -> dict[str, Any]:
        """Attribute dict for edge {u, v}.  Returns empty dict if absent.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            Attribute dictionary for the edge (may be empty).

        """
        key = frozenset((u, v))
        return self.edge_attrs.get(key, {})

    def neighbors(self, node: NodeT) -> set[NodeT]:
        """Return the open neighborhood of node.  O(degree).

        Args:
            node: The node whose neighbors to return.

        Returns:
            Set of adjacent nodes.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        return set(self.adj[node])

    def degree(self, node: NodeT) -> int:
        """Degree of node.

        Args:
            node: The node whose degree to return.

        Returns:
            Number of incident edges.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        return len(self.adj[node])

    def common_neighbors(self, u: NodeT, v: NodeT) -> set[NodeT]:
        """Intersection of neighborhoods -- useful for triangle counting.

        Args:
            u: First node.
            v: Second node.

        Returns:
            Set of nodes adjacent to both ``u`` and ``v``.

        """
        return self.adj[u] & self.adj[v]

    def subgraph(self, nodes: Iterable[NodeT]) -> Graph[NodeT]:
        """Induced subgraph on nodes.

        Returns a new ``Graph`` containing only the specified nodes and
        all edges between them.  Cost is ``O(V' + E')`` where
        ``V' = |nodes|`` and ``E'`` is the number of edges fully
        inside ``nodes``.

        Args:
            nodes: The node set to include.

        Returns:
            A new Graph induced by ``nodes``.

        """
        node_set: set[NodeT] = set(nodes)
        sub: Graph[NodeT] = Graph()

        for n in node_set:
            if n in self.adj:
                sub.add_node(n, **self.node_attrs.get(n, {}))

        for u in node_set:
            if u not in self.adj:
                continue
            for v in self.adj[u]:
                if v in node_set and u < v:
                    data = self.edge_attrs.get(frozenset((u, v)), {})
                    sub.add_edge(u, v, **data)

        return sub

    def edge_subgraph(self, edge_keys: Iterable[frozenset[NodeT]]) -> Graph[NodeT]:
        """Subgraph induced by a subset of edges (keeps all endpoints).

        Useful when the candidate action is "add/remove these edges".

        Args:
            edge_keys: Frozenset edge keys to include.

        Returns:
            A new Graph containing the given edges and their endpoints.

        """
        sub: Graph[NodeT] = Graph()
        seen: set[frozenset[NodeT]] = set()
        ordered: list[frozenset[NodeT]] = []
        for key in edge_keys:
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        ordered.sort(key=lambda k: tuple(sorted(k)))
        for key in ordered:
            nodes = sorted(key)
            u, v = nodes[0], nodes[1]
            sub.add_node(u, **self.node_attrs.get(u, {}))
            sub.add_node(v, **self.node_attrs.get(v, {}))
            data = self.edge_attrs.get(key, {})
            sub.add_edge(u, v, **data)
        return sub

    def add_edge_delta(self, u: NodeT, v: NodeT, **attrs: Any) -> bool:
        """Try to add edge; return True if the graph changed.

        This is the primitive used inside ``calculate_delta`` -- it avoids
        re-evaluating the whole graph when testing a single candidate move.

        Args:
            u: First endpoint.
            v: Second endpoint.
            **attrs: Arbitrary keyword attributes stored on the edge.

        Returns:
            True if the edge was newly added.

        """
        if self.has_edge(u, v):
            return False
        self.add_edge(u, v, **attrs)
        return True

    def remove_edge_delta(self, u: NodeT, v: NodeT) -> bool:
        """Try to remove edge; return True if the graph changed.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge existed and was removed.

        """
        if not self.has_edge(u, v):
            return False
        self.remove_edge(u, v)
        return True

    def add_node_delta(self, node: NodeT, **attrs: Any) -> bool:
        """Try to add node; return True if it was new.

        Args:
            node: The node identifier.
            **attrs: Arbitrary keyword attributes stored on the node.

        Returns:
            True if the node was newly added.

        """
        if node in self.adj:
            return False
        self.add_node(node, **attrs)
        return True

    def remove_node_delta(self, node: NodeT) -> bool:
        """Try to remove node; return True if it existed.

        Args:
            node: The node to remove.

        Returns:
            True if the node existed and was removed.

        """
        if node not in self.adj:
            return False
        self.remove_node(node)
        return True

    def __iter__(self) -> Iterator[NodeT]:
        """Iterate over nodes."""
        return iter(self.adj)

    def __contains__(self, node: NodeT) -> bool:
        """Check if node is in graph."""
        return node in self.adj

    def __repr__(self) -> str:
        """String representation."""
        return f"Graph(nodes={self.num_nodes}, edges={self.edge_count})"

    def __eq__(self, other: object) -> bool:
        """Equality check -- short-circuits on cheap checks first."""
        if not isinstance(other, Graph):
            return NotImplemented
        if self.edge_count != other.edge_count:
            return False
        if len(self.adj) != len(other.adj):
            return False
        return (
            self.adj == other.adj
            and self.edge_attrs == other.edge_attrs
            and self.node_attrs == other.node_attrs
        )

    def __hash__(self) -> int:
        """Graphs are mutable and therefore unhashable."""
        raise TypeError(f"unhashable type: '{type(self).__name__}'")

    def __len__(self) -> int:
        """Number of nodes."""
        return self.num_nodes

    def is_empty(self) -> bool:
        """True if the graph has no nodes.

        Returns:
            True when the node set is empty.

        """
        return self.num_nodes == 0

    def clear(self) -> None:
        """Remove all nodes and edges.  O(1) amortized."""
        self.adj.clear()
        self.edge_attrs.clear()
        self.node_attrs.clear()
        self.edge_count = 0

    def node_list(self) -> list[NodeT]:
        """Materialize nodes into an indexable list.

        Returns:
            List of all nodes in insertion order.

        """
        return list(self.adj.keys())

    def degree_sequence(self) -> list[int]:
        """Sorted descending degree sequence -- useful for heuristic seeding.

        Returns:
            List of degrees sorted from highest to lowest.

        """
        return sorted((len(nbrs) for nbrs in self.adj.values()), reverse=True)

    def is_subgraph_of(self, other: Graph[NodeT]) -> bool:
        """Check whether every node and edge of self exists in other.

        Args:
            other: The potential supergraph.

        Returns:
            True if ``self`` is a subgraph of ``other``.

        """
        for n in self.adj:
            if n not in other.adj:
                return False
        for u in self:
            for v in self.adj[u]:
                if not other.has_edge(u, v):
                    return False
        return True


class ThreadSafeGraph(Graph[NodeT]):
    """Thread-safe wrapper around ``Graph``.

    All public methods acquire a reentrant lock.  Read-only properties
    (``nodes``, ``edges``, ``num_nodes``, ``num_edges``) are not locked
    because they return views or cached integers -- but the caller must
    hold the lock if they need a consistent snapshot across multiple reads.

    Usage::

        g = ThreadSafeGraph[int]()
        with g.lock:
            g.add_edge(1, 2)
            g.add_edge(2, 3)
            logging.info(g.num_nodes)  # consistent read
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the thread-safe graph.

        Args:
            *args: Positional arguments forwarded to the parent Graph.__init__.
            **kwargs: Keyword arguments forwarded to the parent Graph.__init__.

        """
        self.lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def add_node(self, node: NodeT, **attrs: Any) -> None:
        """Thread-safe node insert.

        Args:
            node: The node identifier.
            **attrs: Arbitrary keyword attributes stored on the node.

        """
        with self.lock:
            super().add_node(node, **attrs)

    def remove_node(self, node: NodeT) -> None:
        """Thread-safe node removal.

        Args:
            node: The node to remove.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        with self.lock:
            super().remove_node(node)

    def add_edge(self, u: NodeT, v: NodeT, **attrs: Any) -> frozenset[NodeT]:
        """Thread-safe edge insert.

        Args:
            u: First endpoint.
            v: Second endpoint.
            **attrs: Arbitrary keyword attributes stored on the edge.

        Returns:
            The canonical frozenset key for the edge.

        Raises:
            ValueError: If ``u == v`` (self-loop).

        """
        with self.lock:
            return super().add_edge(u, v, **attrs)

    def remove_edge(self, u: NodeT, v: NodeT) -> None:
        """Thread-safe edge removal.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Raises:
            KeyError: If the edge is not in the graph.

        """
        with self.lock:
            super().remove_edge(u, v)

    def add_edge_delta(self, u: NodeT, v: NodeT, **attrs: Any) -> bool:
        """Thread-safe conditional edge insert.

        Args:
            u: First endpoint.
            v: Second endpoint.
            **attrs: Arbitrary keyword attributes stored on the edge.

        Returns:
            True if the edge was newly added.

        """
        with self.lock:
            return super().add_edge_delta(u, v, **attrs)

    def remove_edge_delta(self, u: NodeT, v: NodeT) -> bool:
        """Thread-safe conditional edge removal.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge existed and was removed.

        """
        with self.lock:
            return super().remove_edge_delta(u, v)

    def add_node_delta(self, node: NodeT, **attrs: Any) -> bool:
        """Thread-safe conditional node insert.

        Args:
            node: The node identifier.
            **attrs: Arbitrary keyword attributes stored on the node.

        Returns:
            True if the node was newly added.

        """
        with self.lock:
            return super().add_node_delta(node, **attrs)

    def remove_node_delta(self, node: NodeT) -> bool:
        """Thread-safe conditional node removal.

        Args:
            node: The node to remove.

        Returns:
            True if the node existed and was removed.

        """
        with self.lock:
            return super().remove_node_delta(node)

    def remove_edge_if_present(self, u: NodeT, v: NodeT) -> bool:
        """Thread-safe conditional edge removal.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge existed and was removed.

        """
        with self.lock:
            return super().remove_edge_if_present(u, v)

    def has_node(self, node: NodeT) -> bool:
        """Thread-safe node membership check.

        Args:
            node: The node to look up.

        Returns:
            True if ``node`` is in the graph.

        """
        with self.lock:
            return super().has_node(node)

    def has_edge(self, u: NodeT, v: NodeT) -> bool:
        """Thread-safe edge existence check.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            True if the edge {u, v} exists.

        """
        with self.lock:
            return super().has_edge(u, v)

    def neighbors(self, node: NodeT) -> set[NodeT]:
        """Thread-safe neighbor lookup.

        Args:
            node: The node whose neighbors to return.

        Returns:
            Set of adjacent nodes.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        with self.lock:
            return super().neighbors(node)

    def degree(self, node: NodeT) -> int:
        """Thread-safe degree lookup.

        Args:
            node: The node whose degree to return.

        Returns:
            Number of incident edges.

        Raises:
            KeyError: If ``node`` is not in the graph.

        """
        with self.lock:
            return super().degree(node)

    def common_neighbors(self, u: NodeT, v: NodeT) -> set[NodeT]:
        """Thread-safe common neighbor lookup.

        Args:
            u: First node.
            v: Second node.

        Returns:
            Set of nodes adjacent to both ``u`` and ``v``.

        """
        with self.lock:
            return super().common_neighbors(u, v)

    def clear(self) -> None:
        """Thread-safe graph clear."""
        with self.lock:
            super().clear()

    def subgraph(self, nodes: Iterable[NodeT]) -> Graph[NodeT]:
        """Thread-safe induced subgraph."""
        with self.lock:
            return super().subgraph(nodes)

    def edge_subgraph(self, edge_keys: Iterable[frozenset[NodeT]]) -> Graph[NodeT]:
        """Thread-safe edge-induced subgraph."""
        with self.lock:
            return super().edge_subgraph(edge_keys)

    def node_list(self) -> list[NodeT]:
        """Thread-safe node materialization."""
        with self.lock:
            return super().node_list()

    def degree_sequence(self) -> list[int]:
        """Thread-safe degree sequence."""
        with self.lock:
            return super().degree_sequence()

    def is_subgraph_of(self, other: Graph[NodeT]) -> bool:
        """Thread-safe subgraph check."""
        with self.lock:
            return super().is_subgraph_of(other)
