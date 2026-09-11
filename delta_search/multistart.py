"""Multi-start solver for improved solution quality.

Runs the greedy solver from multiple random initial states and returns
the best result found across all starts.  This is useful for
non-monotone problems where solution quality depends heavily on the
starting point.

Usage::

    from delta_search import Graph, UncapacitatedFacilityLocationProblem
    from delta_search.multistart import MultiStartSolver

    graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
    problem = UncapacitatedFacilityLocationProblem(graph)
    solver = MultiStartSolver(problem, num_starts=20, seed=42)
    result = solver.solve(max_iterations=100)
    logging.info(f"Best objective: {result.best_objective}")
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from .graph import Graph, NodeT
from .problem import (
    Action,
    DefaultState,
    DeltaResult,
    SubgraphExtractionProblem,
    SubgraphState,
)
from .solver import EarlyTerminationCondition, GreedySolver

if TYPE_CHECKING:
    from .problem import SolverObserver

__all__ = [
    "MultiStartResult",
    "MultiStartSolver",
]


@dataclass
class MultiStartResult(Generic[NodeT]):
    """Aggregated result from multiple solver runs.

    Attributes:
        best_objective: Best objective value found across all starts.
        best_state: State corresponding to best_objective.
        best_start_index: Zero-indexed start that produced the best result.
        all_objectives: List of final objectives from each start.
        total_iterations: Sum of iterations across all starts.
        total_evaluations: Sum of action evaluations across all starts.
        elapsed_ms: Total wall-clock time for all starts.
        num_starts: Number of starts actually completed.
        converged: Whether the best run converged.
        convergence_reason: Convergence reason from the best run.

    """

    best_objective: float = float("-inf")
    best_state: SubgraphState[NodeT] | None = None
    best_start_index: int = 0
    all_objectives: list[float] = field(default_factory=list)
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    num_starts: int = 0
    converged: bool = False
    convergence_reason: str = ""


def _generate_random_initial_state(
    graph: Graph[NodeT],
    rng: random.Random,
    fraction: float = 0.1,
) -> DefaultState[NodeT]:
    """Generate a random initial state by sampling a subset of nodes/edges.

    Args:
        graph: The full input graph.
        rng: Random number generator for reproducibility.
        fraction: Approximate fraction of nodes to include.

    Returns:
        A DefaultState with a random subgraph.

    """
    nodes = list(graph.nodes)
    k = max(1, int(len(nodes) * fraction))
    selected_nodes: set[NodeT] = set(rng.sample(nodes, min(k, len(nodes))))

    sub = Graph[NodeT]()
    for n in selected_nodes:
        sub.add_node(n)
    for u, v in graph.sorted_edges():
        if u in selected_nodes and v in selected_nodes:
            sub.add_edge(u, v)

    return DefaultState[NodeT](graph=sub)


class _RandomStartProblem(SubgraphExtractionProblem[NodeT]):
    """Wrapper that overrides evaluate_initial_state with a fixed initial state.

    Delegates all abstract methods to the underlying problem but replaces
    evaluate_initial_state to return the provided initial state.
    """

    def __init__(
        self,
        inner: SubgraphExtractionProblem[NodeT],
        initial_state: SubgraphState[NodeT],
    ) -> None:
        super().__init__(inner.graph, defensive_copy=False)
        self._inner = inner
        self._initial_state = initial_state

    @property
    def observer(self) -> SolverObserver:
        return self._inner.observer

    def set_observer(self, observer: SolverObserver) -> None:
        self._inner.set_observer(observer)

    def add_observer(self, observer: SolverObserver) -> None:
        self._inner.add_observer(observer)

    def remove_observer(self, observer: SolverObserver) -> None:
        self._inner.remove_observer(observer)

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> SubgraphState[NodeT]:
        return self._initial_state

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        return self._inner.calculate_delta(current_state, candidate_action)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        return self._inner.compute_reward(state)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        return self._inner.compute_penalty(state)

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        return self._inner.is_feasible(state)

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        return self._inner.enumerate_actions(state)

    def generate_composite_actions(
        self,
        state: SubgraphState[NodeT],
    ) -> list[Action]:
        return self._inner.generate_composite_actions(state)

    def objective(self, state: SubgraphState[NodeT]) -> float:
        return self._inner.objective(state)

    def on_iteration_start(self, state: SubgraphState[NodeT], iteration: int) -> None:
        self._inner.on_iteration_start(state, iteration)

    def on_iteration_end(self, state: SubgraphState[NodeT], iteration: int) -> None:
        self._inner.on_iteration_end(state, iteration)


class MultiStartSolver(Generic[NodeT]):
    """Multi-start solver that runs greedy optimization from random starts.

    At each start, a random initial state is generated by sampling a
    fraction of the input graph's nodes.  The greedy solver is then run
    from this initial state.  After all starts, the best result is
    returned.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        num_starts: Number of random starts to run.
        seed: Random seed for reproducibility.  None for non-deterministic.
        initial_fraction: Fraction of nodes to include in random starts.
        early_stop: Optional termination conditions (shared across starts).

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        num_starts: int = 10,
        seed: int | None = None,
        initial_fraction: float = 0.1,
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the multi-start solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            num_starts: Number of random starts to run.
            seed: Random seed for reproducibility. None for non-deterministic.
            initial_fraction: Fraction of nodes to include in random starts.
            early_stop: Optional termination conditions (shared across starts).

        """
        if num_starts <= 0:
            raise ValueError(f"num_starts must be positive, got {num_starts}")
        self.problem = problem
        self.num_starts = num_starts
        self.seed = seed
        self.initial_fraction = initial_fraction
        self.early_stop = early_stop or EarlyTerminationCondition()

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
    ) -> MultiStartResult[NodeT]:
        """Run the multi-start optimization loop.

        Args:
            max_iterations: Iteration cap per start (overridden by
                early_stop.max_iterations if set).
            observer: Optional observer for lifecycle events.  Attached
                to each solver run.

        Returns:
            MultiStartResult with the best solution found across all starts.

        Raises:
            ValueError: If ``max_iterations <= 0``.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        rng = random.Random(self.seed)
        overall_start = time.monotonic()

        best_objective = float("-inf")
        best_state: SubgraphState[NodeT] | None = None
        best_start_index = 0
        all_objectives: list[float] = []
        total_iterations = 0
        total_evaluations = 0
        best_converged = False
        best_convergence_reason = ""

        for start_idx in range(self.num_starts):
            initial = _generate_random_initial_state(
                self.problem.graph,
                rng,
                self.initial_fraction,
            )
            wrapped = _RandomStartProblem(self.problem, initial)

            solver = GreedySolver(wrapped, early_stop=self.early_stop)
            result = solver.solve(
                max_iterations=max_iterations,
                observer=observer,
            )

            all_objectives.append(result.best_objective)
            total_iterations += result.iteration
            total_evaluations += result.total_evaluations

            if result.best_objective > best_objective:
                best_objective = result.best_objective
                best_state = result.best_state
                best_start_index = start_idx
                best_converged = result.converged
                best_convergence_reason = result.convergence_reason

        elapsed_ms = (time.monotonic() - overall_start) * 1000

        return MultiStartResult(
            best_objective=best_objective,
            best_state=best_state,
            best_start_index=best_start_index,
            all_objectives=all_objectives,
            total_iterations=total_iterations,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            num_starts=self.num_starts,
            converged=best_converged,
            convergence_reason=best_convergence_reason,
        )
