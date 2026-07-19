"""Ablation study and scaling analysis framework.

Provides systematic comparison of solver variants across graph sizes
to measure the impact of each optimization component and characterize
scaling behavior.

Usage::

    from delta_search.ablation import AblationStudy, ScalingStudy

    study = AblationStudy(
        problem_factory=lambda g: PrizeCollectingVertexCoverProblem(g),
        graph_sizes=[100, 500, 1000, 5000],
    )
    results = study.run(num_trials=3)
    study.print_report(results)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from .graph import Graph, NodeT
from .solver import EarlyTerminationCondition, GreedySolver

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .problem import SubgraphExtractionProblem

__all__ = [
    "SolverConfig",
    "AblationResult",
    "AblationStudy",
    "ScalingResult",
    "ScalingStudy",
]


@dataclass
class SolverConfig(Generic[NodeT]):
    """Configuration for a solver variant in ablation study.

    Attributes:
        name: Human-readable name (e.g., "greedy", "beam-5").
        solver_factory: Callable that takes a problem and returns a solver.
        early_stop: Optional termination conditions.

    """

    name: str
    solver_factory: Callable[
        [SubgraphExtractionProblem[NodeT]],
        Any,
    ]
    early_stop: EarlyTerminationCondition[NodeT] | None = None


@dataclass
class AblationResult:
    """Result from one solver variant on one graph.

    Attributes:
        config_name: Name of the solver variant.
        graph_size: Number of nodes in the graph.
        graph_edges: Number of edges in the graph.
        best_objective: Best objective found.
        total_iterations: Iterations completed.
        total_evaluations: Action evaluations.
        elapsed_ms: Wall-clock time in milliseconds.
        converged: Whether the solver converged.
        convergence_reason: Reason for termination.
        trial: Trial number (0-indexed).

    """

    config_name: str = ""
    graph_size: int = 0
    graph_edges: int = 0
    best_objective: float = float("-inf")
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""
    trial: int = 0


@dataclass
class AblationStudy(Generic[NodeT]):
    """Systematic ablation study across solver variants.

    Runs multiple solver configurations on graphs of specified sizes
    and collects statistics.

    Args:
        problem_factory: Callable that takes a Graph and returns a problem.
        graph_sizes: List of graph sizes (num nodes) to test.
        max_iterations: Iteration cap per solve.
        graph_density: Edge probability for Erdos-Renyi graphs.
        seed: RNG seed for reproducible graph generation.

    """

    problem_factory: Callable[
        [Graph[int]],
        SubgraphExtractionProblem[int],
    ]
    graph_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000])
    max_iterations: int = 100
    graph_density: float = 0.05
    seed: int = 42

    def _generate_graph(self, n: int) -> Graph[int]:
        """Generate a random Erdos-Renyi graph.

        Args:
            n: Number of nodes.
            density: Edge probability.

        Returns:
            A random graph.

        """
        import random

        rng = random.Random(self.seed)
        graph: Graph[int] = Graph()
        for i in range(n):
            graph.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < self.graph_density:
                    graph.add_edge(i, j)
        return graph

    def run(
        self,
        configs: list[SolverConfig[int]],
        num_trials: int = 1,
        observer: Any | None = None,
    ) -> list[AblationResult]:
        """Run ablation study.

        Args:
            configs: Solver configurations to compare.
            num_trials: Number of random graphs per (config, size).
            observer: Optional solver observer.

        Returns:
            List of AblationResult for each (config, size, trial).

        """
        results: list[AblationResult] = []

        for size in self.graph_sizes:
            for trial in range(num_trials):
                graph = self._generate_graph(size)
                problem = self.problem_factory(graph)

                for config in configs:
                    start = time.monotonic()

                    if config.early_stop is not None:
                        solver = GreedySolver(
                            problem,
                            early_stop=config.early_stop,
                        )
                    else:
                        solver = config.solver_factory(problem)

                    result_dict = solver.solve(
                        max_iterations=self.max_iterations,
                        observer=observer,
                    )

                    elapsed_ms = (time.monotonic() - start) * 1000

                    # Extract results from the solver return value
                    if isinstance(result_dict, dict):
                        best_obj = result_dict.get("best_objective", float("-inf"))
                        total_iter = result_dict.get("total_iterations", 0)
                        total_eval = result_dict.get("total_evaluations", 0)
                        converged = result_dict.get("converged", False)
                        reason = result_dict.get("convergence_reason", "")
                    else:
                        # GreedySolver returns SolverState
                        best_obj = result_dict.best_objective
                        total_iter = result_dict.iteration + 1
                        total_eval = result_dict.total_evaluations
                        converged = result_dict.converged
                        reason = result_dict.convergence_reason

                    ablation_result: AblationResult = AblationResult(
                        config_name=config.name,
                        graph_size=size,
                        graph_edges=graph.num_edges,
                        best_objective=best_obj,
                        total_iterations=total_iter,
                        total_evaluations=total_eval,
                        elapsed_ms=elapsed_ms,
                        converged=converged,
                        convergence_reason=reason,
                        trial=trial,
                    )
                    results.append(ablation_result)

        return results

    def print_report(self, results: list[AblationResult]) -> None:
        """Print a formatted ablation report.

        Args:
            results: Results from run().

        """
        logger.info(f"\n{'=' * 80}")
        logger.info("ABLATION STUDY REPORT")
        logger.info(f"{'=' * 80}\n")

        # Group by graph size
        sizes = sorted(set(r.graph_size for r in results))
        configs = sorted(set(r.config_name for r in results))

        for size in sizes:
            logger.info(f"Graph size: {size} nodes")
            logger.info(
                f"  {'Config':<20} {'Objective':>12} {'Time (ms)':>12} "
                f"{'Evaluations':>12} {'Converged':>10}"
            )
            logger.info(f"  {'-' * 66}")

            for config in configs:
                trial_results = [
                    r
                    for r in results
                    if r.graph_size == size and r.config_name == config
                ]
                if not trial_results:
                    continue

                avg_obj = sum(r.best_objective for r in trial_results) / len(
                    trial_results
                )
                avg_time = sum(r.elapsed_ms for r in trial_results) / len(trial_results)
                avg_eval = sum(r.total_evaluations for r in trial_results) / len(
                    trial_results
                )
                conv_rate = sum(1 for r in trial_results if r.converged) / len(
                    trial_results
                )

                logger.info(
                    f"  {config:<20} {avg_obj:>12.4f} {avg_time:>12.1f} "
                    f"{avg_eval:>12.0f} {conv_rate:>9.0%}"
                )
            logger.info("")


@dataclass
class ScalingResult:
    """Scaling result for one (config, size) pair.

    Attributes:
        config_name: Solver variant name.
        graph_size: Number of nodes.
        avg_objective: Average objective across trials.
        avg_time_ms: Average wall-clock time.
        avg_evaluations: Average action evaluations.
        std_objective: Std dev of objective.
        std_time_ms: Std dev of time.
        time_per_eval_ms: Average time per evaluation.
        convergence_rate: Fraction of trials that converged.
        num_trials: Number of trials.

    """

    config_name: str = ""
    graph_size: int = 0
    avg_objective: float = 0.0
    avg_time_ms: float = 0.0
    avg_evaluations: float = 0.0
    std_objective: float = 0.0
    std_time_ms: float = 0.0
    time_per_eval_ms: float = 0.0
    convergence_rate: float = 0.0
    num_trials: int = 0


@dataclass
class ScalingStudy(Generic[NodeT]):
    """Scaling analysis across graph sizes.

    Measures how solver performance scales with graph size.
    Useful for characterizing algorithmic complexity empirically.

    Args:
        problem_factory: Callable that takes a Graph and returns a problem.
        graph_sizes: List of graph sizes to test.
        max_iterations: Iteration cap per solve.
        graph_density: Edge probability for random graphs.
        seed: RNG seed for reproducible graph generation.

    """

    problem_factory: Callable[
        [Graph[int]],
        SubgraphExtractionProblem[int],
    ]
    graph_sizes: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    max_iterations: int = 100
    graph_density: float = 0.05
    seed: int = 42

    def _generate_graph(self, n: int) -> Graph[int]:
        """Generate a random Erdos-Renyi graph."""
        import random

        rng = random.Random(self.seed)
        graph: Graph[int] = Graph()
        for i in range(n):
            graph.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < self.graph_density:
                    graph.add_edge(i, j)
        return graph

    def run(
        self,
        configs: list[SolverConfig[int]],
        num_trials: int = 3,
    ) -> list[ScalingResult]:
        """Run scaling study.

        Args:
            configs: Solver configurations to test.
            num_trials: Number of trials per (config, size).

        Returns:
            List of ScalingResult for each (config, size).

        """
        ablation: AblationStudy[int] = AblationStudy(
            problem_factory=self.problem_factory,
            graph_sizes=self.graph_sizes,
            max_iterations=self.max_iterations,
            graph_density=self.graph_density,
        )
        ablation_results = ablation.run(configs, num_trials=num_trials)

        # Aggregate by (config, size)
        import math

        scaling_results: list[ScalingResult] = []
        sizes = sorted(set(r.graph_size for r in ablation_results))
        config_names = sorted(set(r.config_name for r in ablation_results))

        for config_name in config_names:
            for size in sizes:
                trials = [
                    r
                    for r in ablation_results
                    if r.config_name == config_name and r.graph_size == size
                ]
                if not trials:
                    continue

                n = len(trials)
                obj_vals = [r.best_objective for r in trials]
                time_vals = [r.elapsed_ms for r in trials]
                eval_vals = [float(r.total_evaluations) for r in trials]

                avg_obj = sum(obj_vals) / n
                avg_time = sum(time_vals) / n
                avg_eval = sum(eval_vals) / n

                std_obj = (
                    math.sqrt(sum((x - avg_obj) ** 2 for x in obj_vals) / n)
                    if n > 1
                    else 0.0
                )
                std_time = (
                    math.sqrt(sum((x - avg_time) ** 2 for x in time_vals) / n)
                    if n > 1
                    else 0.0
                )

                time_per_eval = avg_time / avg_eval if avg_eval > 0 else 0.0
                conv_rate = sum(1 for r in trials if r.converged) / n

                scaling_results.append(
                    ScalingResult(
                        config_name=config_name,
                        graph_size=size,
                        avg_objective=avg_obj,
                        avg_time_ms=avg_time,
                        avg_evaluations=avg_eval,
                        std_objective=std_obj,
                        std_time_ms=std_time,
                        time_per_eval_ms=time_per_eval,
                        convergence_rate=conv_rate,
                        num_trials=n,
                    )
                )

        return scaling_results

    def print_report(self, results: list[ScalingResult]) -> None:
        """Print scaling analysis report.

        Args:
            results: Results from run().

        """
        logger.info(f"\n{'=' * 90}")
        logger.info("SCALING STUDY REPORT")
        logger.info(f"{'=' * 90}\n")

        config_names = sorted(set(r.config_name for r in results))

        for config_name in config_names:
            config_results = [r for r in results if r.config_name == config_name]
            config_results.sort(key=lambda r: r.graph_size)

            logger.info(f"Solver: {config_name}")
            logger.info(
                f"  {'Size':>8} {'Objective':>12} {'Time (ms)':>12} "
                f"{'Evaluations':>12} {'Time/Eval':>12} {'Conv %':>8}"
            )
            logger.info(f"  {'-' * 64}")

            for r in config_results:
                logger.info(
                    f"  {r.graph_size:>8} {r.avg_objective:>12.4f} "
                    f"{r.avg_time_ms:>12.1f} {r.avg_evaluations:>12.0f} "
                    f"{r.time_per_eval_ms:>12.4f} {r.convergence_rate:>7.0%}"
                )

            # Estimate scaling exponent: time ~ size^alpha
            if len(config_results) >= 2:
                sizes = [r.graph_size for r in config_results]
                times = [r.avg_time_ms for r in config_results if r.avg_time_ms > 0]
                if len(times) >= 2 and all(t > 0 for t in times):
                    log_sizes = [math.log(s) for s in sizes[: len(times)]]
                    log_times = [math.log(t) for t in times]
                    # Simple linear regression on log-log
                    n_pts = len(log_sizes)
                    mean_x = sum(log_sizes) / n_pts
                    mean_y = sum(log_times) / n_pts
                    ss_xy = sum(
                        (x - mean_x) * (y - mean_y)
                        for x, y in zip(log_sizes, log_times, strict=False)
                    )
                    ss_xx = sum((x - mean_x) ** 2 for x in log_sizes)
                    alpha = ss_xy / ss_xx if ss_xx > 0 else 0.0
                    logger.info(f"  Estimated scaling: O(n^{alpha:.2f})")

            logger.info("")

    def export_csv(
        self,
        results: list[ScalingResult],
        path: str,
    ) -> None:
        """Export scaling results to CSV.

        Args:
            results: Results from run().
            path: Output file path.

        """
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "config",
                    "graph_size",
                    "avg_objective",
                    "std_objective",
                    "avg_time_ms",
                    "std_time_ms",
                    "avg_evaluations",
                    "time_per_eval_ms",
                    "convergence_rate",
                    "num_trials",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.config_name,
                        r.graph_size,
                        r.avg_objective,
                        r.std_objective,
                        r.avg_time_ms,
                        r.std_time_ms,
                        r.avg_evaluations,
                        r.time_per_eval_ms,
                        r.convergence_rate,
                        r.num_trials,
                    ]
                )
