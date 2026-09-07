"""Multi-objective delta search — Pareto frontier optimization.

Optimizes multiple objectives simultaneously and returns a Pareto
frontier of non-dominated solutions.

Usage::

    from delta_search import Graph, PrizeCollectingVertexCoverProblem
    from delta_search.multi_objective import MultiObjectiveSolver, ObjectiveWeights

    graph = Graph[int].from_edges([(1, 2), (2, 3)])
    problem = PrizeCollectingVertexCoverProblem(graph)

    weights = ObjectiveWeights(
        objectives=["cost", "coverage"],
        weights=[0.5, 0.5],
    )
    solver = MultiObjectiveSolver(problem, objective_weights=weights)
    result = solver.solve(max_iterations=100)
    logging.info(f"Pareto front size: {len(result.pareto_front)}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from .graph import NodeT
from .solver import EarlyTerminationCondition

if TYPE_CHECKING:
    from .problem import (
        Action,
        SolverObserver,
        SubgraphExtractionProblem,
        SubgraphState,
    )

__all__ = [
    "ObjectiveWeights",
    "ParetoPoint",
    "MultiObjectiveResult",
    "MultiObjectiveSolver",
]


@dataclass
class ObjectiveWeights:
    """Configuration for multi-objective optimization.

    Attributes:
        objectives: Names of the objectives.
        weights: Scalarization weights (one per objective).

    """

    objectives: list[str]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.objectives) != len(self.weights):
            raise ValueError(
                f"Length mismatch: {len(self.objectives)} objectives "
                f"vs {len(self.weights)} weights"
            )
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]


@dataclass
class ParetoPoint(Generic[NodeT]):
    """A point on the Pareto frontier.

    Attributes:
        state: The solution state.
        objectives: Dict mapping objective name to value.
        scalarized: Weighted sum scalar value.

    """

    state: SubgraphState[NodeT]
    objectives: dict[str, float]
    scalarized: float


@dataclass
class MultiObjectiveResult(Generic[NodeT]):
    """Result from multi-objective optimization.

    Attributes:
        pareto_front: List of non-dominated solutions.
        best_scalarized: Best scalarized objective found.
        best_state: State corresponding to best_scalarized.
        total_iterations: Iterations completed.
        total_evaluations: Action evaluations.
        elapsed_ms: Wall-clock time.
        converged: Whether the solver converged.
        convergence_reason: Reason for termination.

    """

    pareto_front: list[ParetoPoint[NodeT]] = field(default_factory=list)
    best_scalarized: float = float("-inf")
    best_state: SubgraphState[NodeT] | None = None
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """Check if solution a dominates solution b.

    A dominates B if A is at least as good in all objectives
    and strictly better in at least one.
    """
    at_least_as_good = all(a[k] >= b[k] for k in a)
    strictly_better = any(a[k] > b[k] for k in a)
    return at_least_as_good and strictly_better


def _update_pareto(
    front: list[ParetoPoint[NodeT]],
    candidate: ParetoPoint[NodeT],
) -> list[ParetoPoint[NodeT]]:
    """Add a candidate to the Pareto front, removing dominated points.

    Args:
        front: Current Pareto front.
        candidate: New candidate point.

    Returns:
        Updated Pareto front.

    """
    new_front: list[ParetoPoint[NodeT]] = []
    for point in front:
        if not _dominates(candidate.objectives, point.objectives):
            new_front.append(point)
    # Check if candidate is dominated by any existing point
    dominated = any(
        _dominates(point.objectives, candidate.objectives) for point in front
    )
    if not dominated:
        new_front.append(candidate)
    return new_front


class MultiObjectiveSolver(Generic[NodeT]):
    """Multi-objective solver using scalarization + Pareto tracking.

    At each iteration, evaluates all candidate actions using a
    weighted scalarization.  Additionally tracks a Pareto front of
    non-dominated solutions encountered during search.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        objective_weights: Scalarization weights for combining objectives.
        early_stop: Optional termination conditions.
        max_pareto_size: Cap on Pareto front size (None = unlimited).

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        objective_weights: ObjectiveWeights | None = None,
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
        max_pareto_size: int | None = None,
    ) -> None:
        """Initialize the multi-objective solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            objective_weights: Scalarization weights for combining objectives.
            early_stop: Optional termination conditions.
            max_pareto_size: Cap on Pareto front size (None = unlimited).

        """
        self.problem = problem
        self.objective_weights = objective_weights or ObjectiveWeights(
            objectives=["objective"],
            weights=[1.0],
        )
        self.early_stop = early_stop or EarlyTerminationCondition()
        self.max_pareto_size = max_pareto_size

    def _compute_objectives(
        self,
        state: SubgraphState[NodeT],
    ) -> dict[str, float]:
        """Compute all objective values for a state.

        Override this method to define custom objectives.
        Default uses reward and penalty as two objectives.

        Args:
            state: The solution state.

        Returns:
            Dict mapping objective name to value.

        """
        return {
            "reward": self.problem.compute_reward(state),
            "penalty": -self.problem.compute_penalty(state),
        }

    def _scalarize(self, objectives: dict[str, float]) -> float:
        """Compute weighted scalar from objective values.

        Args:
            objectives: Dict of objective name to value.

        Returns:
            Weighted sum scalar.

        """
        total = 0.0
        for name, weight in zip(
            self.objective_weights.objectives,
            self.objective_weights.weights,
            strict=False,
        ):
            total += weight * objectives.get(name, 0.0)
        return total

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
    ) -> MultiObjectiveResult[NodeT]:
        """Run multi-objective optimization.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.

        Returns:
            MultiObjectiveResult with Pareto front and best scalarized solution.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        state = self.problem.evaluate_initial_state(self.problem.graph)
        current_objective = self.problem.objective(state)

        best_objective = current_objective
        total_evaluations = 0
        stall_count = 0
        start_time = time.monotonic()

        # Initialize Pareto front
        init_objs = self._compute_objectives(state)
        init_scalar = self._scalarize(init_objs)
        pareto_front: list[ParetoPoint[NodeT]] = [
            ParetoPoint(state=state, objectives=init_objs, scalarized=init_scalar),
        ]

        limit = (
            self.early_stop.max_iterations
            if self.early_stop.max_iterations is not None
            else max_iterations
        )

        converged = False
        convergence_reason = ""

        for iteration in range(limit):
            self.problem.on_iteration_start(state, iteration)

            actions = self.problem.enumerate_actions(state)
            best_action: Action | None = None
            best_action_obj = float("-inf")

            for action in actions:
                delta = self.problem.calculate_delta(state, action)
                if not delta.feasible:
                    continue

                obj = current_objective + delta.reward_change - delta.penalty_change
                total_evaluations += 1

                elapsed = (time.monotonic() - start_time) * 1000
                self.problem.observer.on_action_evaluated(
                    action,
                    delta,
                    elapsed,
                )

                if obj > best_action_obj:
                    best_action_obj = obj
                    best_action = action

            if best_action is None:
                converged = True
                convergence_reason = "no actions available"
                break

            state = self.problem.apply_action(state, best_action)
            current_objective = best_action_obj

            # Update Pareto front
            objs = self._compute_objectives(state)
            scalar = self._scalarize(objs)
            point: ParetoPoint[NodeT] = ParetoPoint(
                state=state,
                objectives=objs,
                scalarized=scalar,
            )
            pareto_front = _update_pareto(pareto_front, point)
            if (
                self.max_pareto_size is not None
                and len(pareto_front) > self.max_pareto_size
            ):
                pareto_front.sort(key=lambda p: p.scalarized, reverse=True)
                del pareto_front[self.max_pareto_size :]

            elapsed_total = (time.monotonic() - start_time) * 1000

            if current_objective > best_objective:
                best_objective = current_objective
                stall_count = 0
            else:
                stall_count += 1

            self.problem.observer.on_iteration_complete(
                iteration,
                best_action,
                best_action_obj,
            )
            self.problem.on_iteration_end(state, iteration)

            # Check termination
            if (
                self.early_stop.max_time_ms is not None
                and elapsed_total >= self.early_stop.max_time_ms
            ):
                converged = True
                convergence_reason = "time limit"
                break
            if (
                self.early_stop.stall_iterations is not None
                and stall_count >= self.early_stop.stall_iterations
            ):
                converged = True
                convergence_reason = (
                    f"stalled for {self.early_stop.stall_iterations} iterations"
                )
                break
            if (
                self.early_stop.max_evaluations is not None
                and total_evaluations >= self.early_stop.max_evaluations
            ):
                converged = True
                convergence_reason = "evaluation limit"
                break
            if (
                self.early_stop.objective_target is not None
                and best_objective >= self.early_stop.objective_target
            ):
                converged = True
                convergence_reason = "objective target reached"
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Find best scalarized in Pareto front
        best_pareto = max(pareto_front, key=lambda p: p.scalarized)

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return MultiObjectiveResult(
            pareto_front=pareto_front,
            best_scalarized=best_pareto.scalarized,
            best_state=best_pareto.state,
            total_iterations=iteration + 1 if iteration >= 0 else 0,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            converged=converged,
            convergence_reason=convergence_reason,
        )
