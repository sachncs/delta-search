"""Graph file I/O utilities.

Supports loading and saving graphs in JSON format.

Format::

    {
        "nodes": [{"id": 1, "label": "a"}, {"id": 2}],
        "edges": [{"source": 1, "target": 2, "weight": 3.0}]
    }

Usage::

    from delta_search.io import load_graph, save_graph

    graph = load_graph("input.json")
    save_graph(graph, "output.json")
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .graph import Graph, NodeT

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "save_graph",
    "load_graph",
]


def save_graph(graph: Graph[NodeT], path: str | Path) -> None:
    """Save a graph to JSON format.

    Args:
        graph: The graph to save.
        path: Output file path.

    """
    nodes: list[dict[str, Any]] = []
    for node in graph:
        entry: dict[str, Any] = {"id": node}
        attrs = dict(graph.node_data(node))
        if attrs:
            entry.update(attrs)
        nodes.append(entry)

    edges: list[dict[str, Any]] = []
    for u, v in graph.edges:
        entry = {"source": u, "target": v}
        attrs = dict(graph.edge_data(u, v))
        if attrs:
            entry.update(attrs)
        edges.append(entry)

    data = {"nodes": nodes, "edges": edges}

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_graph(path: str | Path) -> Graph[Any]:
    """Load a graph from JSON format.

    Args:
        path: Input file path.

    Returns:
        The loaded graph.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields (``id``, ``source``, ``target``) are missing.
        ValueError: If the JSON structure is not a dict with ``nodes`` and
            ``edges`` arrays.

    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at top level, got {type(data).__name__}"
        )
    if "nodes" not in data and "edges" not in data:
        raise ValueError("Graph JSON must contain 'nodes' and/or 'edges' keys")

    graph: Graph[Any] = Graph()

    for node_entry in data.get("nodes", []):
        if not isinstance(node_entry, dict):
            raise ValueError(
                f"Each node entry must be a dict, got {type(node_entry).__name__}"
            )
        node_id = node_entry["id"]
        attrs = {k: v for k, v in node_entry.items() if k != "id"}
        graph.add_node(node_id, **attrs)

    for edge_entry in data.get("edges", []):
        if not isinstance(edge_entry, dict):
            raise ValueError(
                f"Each edge entry must be a dict, got {type(edge_entry).__name__}"
            )
        source = edge_entry["source"]
        target = edge_entry["target"]
        attrs = {k: v for k, v in edge_entry.items() if k not in ("source", "target")}
        graph.add_edge(source, target, **attrs)

    return graph
