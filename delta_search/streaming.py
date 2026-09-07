"""Dynamic and streaming graph support for delta search.

Allows the input graph to be modified live (edge/node insertions
and deletions) while the solver is running.  The streaming solver
processes graph updates and incrementally maintains solution quality.

Usage::

    from delta_search import Graph, MaximumWeightedIndependentSetProblem
    from delta_search.streaming import StreamingSolver

    graph = Graph[int].from_edges([(1, 2), (2, 3)])
    problem = MaximumWeightedIndependentSetProblem(graph)
    solver = StreamingSolver(problem)

    # Process initial graph
    result = solver.solve(max_iterations=50)

    # Graph changes — solver adapts
    graph.add_edge(3, 4)
    result = solver.solve(max_iterations=50, resume=True)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    "GraphUpdate",
    "StreamingResult",
    "StreamingSolver",
]


@dataclass
class GraphUpdate(Generic[NodeT]):
    """A pending graph mutation.

    Attributes:
        update_type: One of 'add_edge', 'remove_edge', 'add_node', 'remove_node'.
        targets: The node(s) affected.

    """

    update_type: str
    targets: tuple[NodeT, ...]


@dataclass
class StreamingResult(Generic[NodeT]):
    """Result from a streaming solve pass.

    Attributes:
        best_objective: Best objective found in this pass.
        best_state: State corresponding to best_objective.
        updates_applied: Number of graph updates applied.
        total_iterations: Iterations completed in this pass.
        total_evaluations: Action evaluations in this pass.
        elapsed_ms: Wall-clock time for this pass.
        converged: Whether the solver converged.
        convergence_reason: Reason for termination.

    """

    best_objective: float = float("-inf")
    best_state: SubgraphState[NodeT] | None = None
    updates_applied: int = 0
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


class StreamingSolver(Generic[NodeT]):
    """Solver that handles live graph mutations.

    Maintains the best solution across graph updates.  When the graph
    changes, the solver can resume from the previous best state,
    re-evaluating only affected actions.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        early_stop: Optional termination conditions.

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the streaming solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            early_stop: Optional termination conditions.

        """
        self.problem = problem
        self.early_stop = early_stop or EarlyTerminationCondition()
        self._pending_updates: list[GraphUpdate[NodeT]] = []
        self._best_state: SubgraphState[NodeT] | None = None
        self._best_objective: float = float("-inf")

    def enqueue_update(self, update: GraphUpdate[NodeT]) -> None:
        """Queue a graph mutation to be applied before the next solve.

        Args:
            update: The graph update to apply.

        """
        self._pending_updates.append(update)

    def add_edge(self, u: NodeT, v: NodeT) -> None:
        """Queue an edge addition.

        Args:
            u: First endpoint.
            v: Second endpoint.

        """
        self._pending_updates.append(
            GraphUpdate(update_type="add_edge", targets=(u, v)),
        )

    def remove_edge(self, u: NodeT, v: NodeT) -> None:
        """Queue an edge removal.

        Args:
            u: First endpoint.
            v: Second endpoint.

        """
        self._pending_updates.append(
            GraphUpdate(update_type="remove_edge", targets=(u, v)),
        )

    def add_node(self, node: NodeT) -> None:
        """Queue a node addition.

        Args:
            node: The node to add.

        """
        self._pending_updates.append(
            GraphUpdate(update_type="add_node", targets=(node,)),
        )

    def remove_node(self, node: NodeT) -> None:
        """Queue a node removal.

        Args:
            node: The node to remove.

        """
        self._pending_updates.append(
            GraphUpdate(update_type="remove_node", targets=(node,)),
        )

    def _apply_updates(self) -> int:
        """Apply all pending graph updates.

        Returns:
            Number of updates applied.

        """
        count = 0
        for update in self._pending_updates:
            if update.update_type == "add_edge":
                u, v = update.targets
                self.problem.graph.add_edge(u, v)
            elif update.update_type == "remove_edge":
                u, v = update.targets
                self.problem.graph.remove_edge(u, v)
            elif update.update_type == "add_node":
                self.problem.graph.add_node(update.targets[0])
            elif update.update_type == "remove_node":
                self.problem.graph.remove_node(update.targets[0])
            count += 1
        self._pending_updates.clear()
        return count

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
        resume: bool = False,
    ) -> StreamingResult[NodeT]:
        """Run the solver, applying any pending graph updates first.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.
            resume: If True, start from the previous best state.

        Returns:
            StreamingResult with the best solution found.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        updates_applied = self._apply_updates()

        # Determine starting state
        if resume and self._best_state is not None:
            state = self._best_state
            current_objective = self._best_objective
        else:
            state = self.problem.evaluate_initial_state(self.problem.graph)
            current_objective = self.problem.objective(state)

        best_state = state
        best_objective = current_objective
        total_evaluations = 0
        stall_count = 0
        start_time = time.monotonic()

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

            elapsed_total = (time.monotonic() - start_time) * 1000

            if current_objective > best_objective:
                best_objective = current_objective
                best_state = state
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

        # Persist best state for resume
        self._best_state = best_state
        self._best_objective = best_objective

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return StreamingResult(
            best_objective=best_objective,
            best_state=best_state,
            updates_applied=updates_applied,
            total_iterations=iteration + 1 if iteration >= 0 else 0,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            converged=converged,
            convergence_reason=convergence_reason,
        )
