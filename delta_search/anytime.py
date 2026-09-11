"""Anytime delta search — returns progressively improving solutions.

Continuously tracks the best solution found so far and can return
intermediate results at any point.  Useful for anytime optimization
where you want to compare against competitors at equal time budgets.

Usage::

    from delta_search import Graph, PrizeCollectingVertexCoverProblem
    from delta_search.anytime import AnytimeSolver

    graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
    problem = PrizeCollectingVertexCoverProblem(graph)
    solver = AnytimeSolver(problem)
    result = solver.solve(max_iterations=200)

    for snapshot in result.progress_history:
        logging.info(f"iter {snapshot.iteration}: obj={snapshot.objective}")
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
    "ProgressSnapshot",
    "AnytimeResult",
    "AnytimeSolver",
]


@dataclass
class ProgressSnapshot(Generic[NodeT]):
    """A point-in-time snapshot of solver progress.

    Attributes:
        iteration: Iteration number.
        objective: Best objective at this point.
        state: Best state at this point.
        elapsed_ms: Wall-clock time at this point.

    """

    iteration: int
    objective: float
    state: SubgraphState[NodeT]
    elapsed_ms: float


@dataclass
class AnytimeResult(Generic[NodeT]):
    """Result from anytime search with full progress history.

    Attributes:
        best_objective: Best objective found overall.
        best_state: State corresponding to best_objective.
        best_iteration: Iteration where best was found.
        progress_history: List of snapshots showing improvement over time.
        total_iterations: Total iterations completed.
        total_evaluations: Total action evaluations.
        elapsed_ms: Total wall-clock time.
        converged: Whether the search converged.
        convergence_reason: Reason for termination.

    """

    best_objective: float = float("-inf")
    best_state: SubgraphState[NodeT] | None = None
    best_iteration: int = 0
    progress_history: list[ProgressSnapshot[NodeT]] = field(default_factory=list)
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


class AnytimeSolver(Generic[NodeT]):
    """Anytime solver that tracks progressive improvement.

    Wraps the greedy evaluation loop but continuously maintains the
    best solution found so far and records progress snapshots.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        early_stop: Optional termination conditions.
        snapshot_interval: Record a snapshot every N iterations (default: 1).
        max_history: Cap on number of snapshots kept (None = unlimited).

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
        snapshot_interval: int = 1,
        max_history: int | None = None,
    ) -> None:
        """Initialize the anytime solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            early_stop: Optional termination conditions.
            snapshot_interval: Record a snapshot every N iterations (default: 1).
            max_history: Cap on number of snapshots kept (None = unlimited).

        """
        self.problem = problem
        self.early_stop = early_stop or EarlyTerminationCondition()
        self.snapshot_interval = max(1, snapshot_interval)
        self.max_history = max_history

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
    ) -> AnytimeResult[NodeT]:
        """Run anytime search, recording progress at each snapshot interval.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.

        Returns:
            AnytimeResult with full progress history.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        state = self.problem.evaluate_initial_state(self.problem.graph)
        current_objective = self.problem.objective(state)

        best_state = state
        best_objective = current_objective
        best_iteration = 0

        progress: list[ProgressSnapshot[NodeT]] = []
        start_time = time.monotonic()

        # Record initial snapshot
        progress.append(
            ProgressSnapshot(
                iteration=0,
                objective=current_objective,
                state=state,
                elapsed_ms=0.0,
            )
        )

        limit = (
            self.early_stop.max_iterations
            if self.early_stop.max_iterations is not None
            else max_iterations
        )
        stall_count = 0
        total_evaluations = 0

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

            elapsed_total = (time.monotonic() - start_time) * 1000

            if current_objective > best_objective:
                best_objective = current_objective
                best_state = state
                best_iteration = iteration + 1
                stall_count = 0
            else:
                stall_count += 1

            # Record snapshot at intervals
            if (iteration + 1) % self.snapshot_interval == 0:
                progress.append(
                    ProgressSnapshot(
                        iteration=iteration + 1,
                        objective=best_objective,
                        state=best_state,
                        elapsed_ms=elapsed_total,
                    )
                )
                if self.max_history is not None and len(progress) > self.max_history:
                    progress.pop(0)  # drop oldest snapshot to honour the cap

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

        # Final snapshot
        progress.append(
            ProgressSnapshot(
                iteration=iteration + 1 if iteration >= 0 else 0,
                objective=best_objective,
                state=best_state,
                elapsed_ms=elapsed_ms,
            )
        )
        if self.max_history is not None and len(progress) > self.max_history:
            progress.pop(0)  # drop oldest snapshot to honour the cap

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return AnytimeResult(
            best_objective=best_objective,
            best_state=best_state,
            best_iteration=best_iteration,
            progress_history=progress,
            total_iterations=iteration + 1 if iteration >= 0 else 0,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            converged=converged,
            convergence_reason=convergence_reason,
        )
