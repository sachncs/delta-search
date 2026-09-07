"""Visualization utilities for delta search.

Provides plotting and export functions for solver results.  Matplotlib
is an optional dependency -- import errors are raised at call time if
not installed.

Usage::

    from delta_search.visualization import (
        plot_convergence,
        plot_benchmark_comparison,
        export_solution_graph,
    )

    plot_convergence(objectives, save_path="convergence.png")
    export_solution_graph(graph, solution_graph, "solution.json")
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from .benchmarks import BenchmarkResult
    from .graph import Graph, NodeT

__all__ = [
    "plot_convergence",
    "plot_benchmark_comparison",
    "plot_multi_start_distribution",
    "export_solution_graph",
    "solution_summary",
]


def _require_matplotlib() -> Any:
    """Check that matplotlib is installed.

    Returns:
        The matplotlib.pyplot module.

    Raises:
        ImportError: If matplotlib is not installed.

    """
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            "matplotlib is required for visualization.  "
            "Install it with: pip install matplotlib"
        )
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(
    objectives: list[float],
    save_path: str | Path | None = None,
    title: str = "Objective Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Objective",
    figsize: tuple[int, int] = (8, 5),
    show: bool = False,
) -> Any:
    """Plot the objective value over solver iterations.

    Args:
        objectives: Objective value at each iteration.
        save_path: Path to save the plot.  None to skip saving.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size as (width, height) in inches.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure object.

    """
    plt = _require_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    iterations = list(range(len(objectives)))
    ax.plot(iterations, objectives, linewidth=2, color="#2563eb")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_benchmark_comparison(
    results: list[BenchmarkResult],
    save_path: str | Path | None = None,
    title: str = "Benchmark Comparison",
    figsize: tuple[int, int] = (10, 6),
    show: bool = False,
) -> Any:
    """Plot a bar chart comparing benchmark results.

    Args:
        results: List of benchmark results.
        save_path: Path to save the plot.  None to skip saving.
        title: Plot title.
        figsize: Figure size as (width, height) in inches.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure object.

    """
    plt = _require_matplotlib()

    names = [r.case_name for r in results]
    objectives = [r.objective for r in results]
    times = [r.elapsed_ms for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.Set2(range(len(names)))

    ax1.barh(names, objectives, color=colors)
    ax1.set_xlabel("Objective Value")
    ax1.set_title("Solution Quality")
    ax1.invert_yaxis()

    ax2.barh(names, times, color=colors)
    ax2.set_xlabel("Time (ms)")
    ax2.set_title("Runtime")
    ax2.invert_yaxis()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_multi_start_distribution(
    all_objectives: list[list[float]],
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Multi-Start Objective Distribution",
    figsize: tuple[int, int] = (8, 5),
    show: bool = False,
) -> Any:
    """Plot box plots of multi-start objective distributions.

    Args:
        all_objectives: List of objective lists (one per problem).
        labels: Optional labels for each problem.
        save_path: Path to save the plot.  None to skip saving.
        title: Plot title.
        figsize: Figure size as (width, height) in inches.
        show: Whether to call plt.show().

    Returns:
        The matplotlib Figure object.

    """
    plt = _require_matplotlib()

    if labels is None:
        labels = [f"Problem {i}" for i in range(len(all_objectives))]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(all_objectives, labels=labels, patch_artist=True)

    colors = plt.cm.Set2(range(len(all_objectives)))
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def export_solution_graph(
    input_graph: Graph[NodeT],
    solution_graph: Graph[NodeT],
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export a solution subgraph with metadata.

    Writes a JSON file containing the original graph structure, the
    solution subgraph, and optional metadata (objective, timing, etc.).

    Args:
        input_graph: The original full graph.
        solution_graph: The solution subgraph.
        path: Output file path.
        metadata: Optional metadata dict to include.

    """
    nodes_data: list[dict[str, Any]] = []
    for node in solution_graph:
        node_entry: dict[str, Any] = {"id": node}
        attrs = dict(solution_graph.node_data(node))
        if attrs:
            node_entry.update(attrs)
        nodes_data.append(node_entry)

    edges_data: list[dict[str, Any]] = []
    for u, v in solution_graph.edges:
        edge_entry: dict[str, Any] = {"source": u, "target": v}
        attrs = dict(solution_graph.edge_data(u, v))
        if attrs:
            edge_entry.update(attrs)
        edges_data.append(edge_entry)

    data: dict[str, Any] = {
        "input_graph": {
            "num_nodes": input_graph.num_nodes,
            "num_edges": input_graph.num_edges,
        },
        "solution": {
            "nodes": nodes_data,
            "edges": edges_data,
        },
    }
    if metadata:
        data["metadata"] = metadata

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def solution_summary(
    solution_graph: Graph[NodeT],
) -> dict[str, Any]:
    """Return a dict summarizing a solution subgraph.

    Args:
        solution_graph: The solution subgraph.

    Returns:
        Dict with num_nodes, num_edges, nodes, and edges.

    """
    return {
        "num_nodes": solution_graph.num_nodes,
        "num_edges": solution_graph.num_edges,
        "nodes": list(solution_graph.nodes),
        "edges": [{"source": u, "target": v} for u, v in solution_graph.edges],
    }
