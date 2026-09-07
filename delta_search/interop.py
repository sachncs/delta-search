"""NetworkX interoperability.

Provides bidirectional conversion between delta search graphs and
NetworkX graphs.  NetworkX is an optional dependency -- import errors
are raised at call time if not installed.

Usage::

    from delta_search.interop import from_networkx, to_networkx

    import networkx as nx
    G = nx.complete_graph(5)
    graph = from_networkx(G)

    G2 = to_networkx(graph)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .graph import Graph

__all__ = [
    "from_networkx",
    "to_networkx",
]


def _require_networkx() -> None:
    """Check that networkx is installed.

    Raises:
        ImportError: If networkx is not installed.

    """
    import importlib.util

    if importlib.util.find_spec("networkx") is None:
        raise ImportError(
            "NetworkX is required for interop.  Install it with: pip install networkx"
        )


def from_networkx(nx_graph: Any) -> Graph[Any]:
    """Convert a NetworkX graph to a delta search Graph.

    Node and edge attributes are preserved.  Note that multigraphs
    with parallel edges will only retain one edge per pair.

    Args:
        nx_graph: A NetworkX Graph (or subclass).

    Returns:
        An equivalent delta search Graph.

    Raises:
        ImportError: If NetworkX is not installed.
        TypeError: If ``nx_graph`` is not a ``networkx.Graph``.

    """
    _require_networkx()
    import networkx

    if not isinstance(nx_graph, networkx.Graph):
        raise TypeError(f"Expected a networkx.Graph, got {type(nx_graph).__name__}")

    from .graph import Graph

    g: Graph[Any] = Graph()

    for node, attrs in nx_graph.nodes(data=True):
        g.add_node(node, **attrs)

    for u, v, attrs in nx_graph.edges(data=True):
        g.add_edge(u, v, **attrs)

    return g


def to_networkx(graph: Graph[Any]) -> Any:
    """Convert a delta search Graph to a NetworkX graph.

    Node and edge attributes are preserved.

    Args:
        graph: A delta search Graph.

    Returns:
        An equivalent NetworkX Graph.

    Raises:
        ImportError: If NetworkX is not installed.

    """
    _require_networkx()
    import networkx

    nx_graph: Any = networkx.Graph()

    for node in graph:
        attrs = dict(graph.node_data(node))
        nx_graph.add_node(node, **attrs)

    for u, v in graph.edges:
        attrs = dict(graph.edge_data(u, v))
        nx_graph.add_edge(u, v, **attrs)

    return nx_graph
