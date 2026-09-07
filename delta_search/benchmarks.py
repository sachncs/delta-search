"""Benchmark suite for comparing delta search against paper results.

Generates standard test graphs (complete, grid, random) and runs all
six problem types, producing structured result tables suitable for
comparison with published results.

Usage::

    from delta_search.benchmarks import BenchmarkSuite
    suite = BenchmarkSuite(sizes=[10, 20, 50])
    results = suite.run()
    suite.print_table(results)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from .graph import Graph, NodeT
from .problems import (
    MaximumPlanarSubgraphProblem,
    MaximumWeightedIndependentSetProblem,
    MinimumConnectedDominatingSetProblem,
    MinimumWeightedSteinerTreeProblem,
    PrizeCollectingVertexCoverProblem,
    UncapacitatedFacilityLocationProblem,
)
from .solver import GreedySolver

if TYPE_CHECKING:
    from .problem import SubgraphExtractionProblem

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSuite",
    "generate_complete_graph",
    "generate_grid_graph",
    "generate_random_graph",
]


def generate_complete_graph(n: int) -> Graph[int]:
    """Generate a complete graph on n nodes.

    Args:
        n: Number of nodes.

    Returns:
        A complete Graph[int] with nodes 0..n-1.

    """
    g: Graph[int] = Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def generate_grid_graph(rows: int, cols: int) -> Graph[int]:
    """Generate a 2D grid graph.

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A grid Graph[int] with nodes numbered row-major.

    """
    g: Graph[int] = Graph()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            g.add_node(node)
            if c + 1 < cols:
                g.add_edge(node, node + 1)
            if r + 1 < rows:
                g.add_edge(node, node + cols)
    return g


def generate_random_graph(
    n: int,
    edge_prob: float = 0.3,
    seed: int | None = None,
) -> Graph[int]:
    """Generate a random Erdos-Renyi graph.

    Args:
        n: Number of nodes.
        edge_prob: Probability of each edge existing.
        seed: Random seed for reproducibility.

    Returns:
        A random Graph[int].

    """
    rng = random.Random(seed)
    g: Graph[int] = Graph()
    for i in range(n):
        g.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_prob:
                g.add_edge(i, j)
    return g


@dataclass
class BenchmarkCase(Generic[NodeT]):
    """A single benchmark configuration.

    Attributes:
        name: Human-readable label (e.g. "complete_n10").
        problem_class: The problem class to instantiate.
        graph: The input graph.
        problem_kwargs: Extra kwargs for the problem constructor.

    """

    name: str
    problem_class: type[SubgraphExtractionProblem[NodeT]]
    graph: Graph[NodeT]
    problem_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        case_name: The name of the benchmark case.
        problem_class: Name of the problem class.
        graph_nodes: Number of nodes in the input graph.
        graph_edges: Number of edges in the input graph.
        objective: Best objective found.
        iterations: Number of iterations.
        evaluations: Total action evaluations.
        elapsed_ms: Wall-clock time in milliseconds.
        converged: Whether the solver converged.
        convergence_reason: Reason for convergence.

    """

    case_name: str = ""
    problem_class: str = ""
    graph_nodes: int = 0
    graph_edges: int = 0
    objective: float = 0.0
    iterations: int = 0
    evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


class BenchmarkSuite:
    """Configurable benchmark suite for delta search problems.

    Generates standard test graphs and runs each configured problem
    against them, collecting timing and quality metrics.

    Args:
        sizes: List of graph sizes to test (node counts).
        seed: Random seed for graph generation and solver.
        max_iterations: Iteration cap per benchmark run.
        edge_prob: Edge probability for random graph generation.

    """

    def __init__(
        self,
        sizes: list[int] | None = None,
        seed: int = 42,
        max_iterations: int = 500,
        edge_prob: float = 0.3,
    ) -> None:
        """Initialize the benchmark suite.

        Args:
            sizes: List of graph sizes to test (node counts).
            seed: Random seed for graph generation and solver.
            max_iterations: Iteration cap per benchmark run.
            edge_prob: Edge probability for random graph generation.

        """
        self.sizes = sizes or [10, 20, 50]
        self.seed = seed
        self.max_iterations = max_iterations
        self.edge_prob = edge_prob

    def generate_cases(self) -> list[BenchmarkCase[int]]:
        """Generate the standard benchmark cases.

        For each size, creates:
        - Complete graph with MPS
        - Grid graph with MCDS
        - Random graph with MWIS
        - Random graph with PCVC
        - Random graph with UFLP
        - Random graph with MWST (terminals = first 2 nodes)

        Returns:
            List of BenchmarkCase instances.

        """
        cases: list[BenchmarkCase[int]] = []

        for n in self.sizes:
            side = max(2, int(n**0.5))

            cases.append(
                BenchmarkCase(
                    name=f"mps_complete_n{n}",
                    problem_class=MaximumPlanarSubgraphProblem,
                    graph=generate_complete_graph(n),
                )
            )

            cases.append(
                BenchmarkCase(
                    name=f"mcds_grid_{side}x{side}",
                    problem_class=MinimumConnectedDominatingSetProblem,
                    graph=generate_grid_graph(side, side),
                )
            )

            cases.append(
                BenchmarkCase(
                    name=f"mwis_random_n{n}",
                    problem_class=MaximumWeightedIndependentSetProblem,
                    graph=generate_random_graph(n, self.edge_prob, self.seed),
                )
            )

            cases.append(
                BenchmarkCase(
                    name=f"pcvc_random_n{n}",
                    problem_class=PrizeCollectingVertexCoverProblem,
                    graph=generate_random_graph(n, self.edge_prob, self.seed),
                )
            )

            cases.append(
                BenchmarkCase(
                    name=f"uflp_random_n{n}",
                    problem_class=UncapacitatedFacilityLocationProblem,
                    graph=generate_random_graph(n, self.edge_prob, self.seed),
                )
            )

            g = generate_random_graph(n, self.edge_prob, self.seed)
            nodes = list(g.nodes)
            terminals = set(nodes[:2]) if len(nodes) >= 2 else set(nodes[:1])
            cases.append(
                BenchmarkCase(
                    name=f"mwst_random_n{n}",
                    problem_class=MinimumWeightedSteinerTreeProblem,
                    graph=g,
                    problem_kwargs={"terminals": terminals},
                )
            )

        return cases

    def run_case(self, case: BenchmarkCase[int]) -> BenchmarkResult:
        """Run a single benchmark case.

        Args:
            case: The benchmark case to run.

        Returns:
            BenchmarkResult with timing and quality metrics.

        """
        problem = case.problem_class(case.graph, **case.problem_kwargs)
        solver = GreedySolver(problem)

        start = time.monotonic()
        result = solver.solve(max_iterations=self.max_iterations)
        elapsed = (time.monotonic() - start) * 1000

        return BenchmarkResult(
            case_name=case.name,
            problem_class=type(problem).__name__,
            graph_nodes=case.graph.num_nodes,
            graph_edges=case.graph.num_edges,
            objective=result.best_objective,
            iterations=result.iteration,
            evaluations=result.total_evaluations,
            elapsed_ms=elapsed,
            converged=result.converged,
            convergence_reason=result.convergence_reason,
        )

    def run(
        self,
        cases: list[BenchmarkCase[int]] | None = None,
    ) -> list[BenchmarkResult]:
        """Run all benchmark cases.

        Args:
            cases: Optional pre-generated cases.  If None, calls
                generate_cases().

        Returns:
            List of BenchmarkResult, one per case.

        """
        if cases is None:
            cases = self.generate_cases()
        return [self.run_case(c) for c in cases]

    @staticmethod
    def print_table(results: list[BenchmarkResult]) -> str:
        """Format results as a readable table.

        Args:
            results: List of benchmark results.

        Returns:
            Formatted table string.

        """
        header = (
            f"{'Case':<30} {'Problem':<38} "
            f"{'V':>4} {'E':>5} {'Obj':>10} "
            f"{'Iter':>6} {'Evals':>8} {'Time(ms)':>10}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for r in results:
            line = (
                f"{r.case_name:<30} {r.problem_class:<38} "
                f"{r.graph_nodes:>4} {r.graph_edges:>5} {r.objective:>10.2f} "
                f"{r.iterations:>6} {r.evaluations:>8} {r.elapsed_ms:>10.2f}"
            )
            lines.append(line)

        return "\n".join(lines)
