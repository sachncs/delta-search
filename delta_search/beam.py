"""Beam search variant of delta search.

Maintains top-K candidate solutions simultaneously, expands all K
at each iteration, and retains the best K.  This provides better
exploration than single-trajectory greedy search.

Usage::

    from delta_search import Graph, MaximumWeightedIndependentSetProblem
    from delta_search.beam import BeamSearchSolver

    graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
    problem = MaximumWeightedIndependentSetProblem(graph)
    solver = BeamSearchSolver(problem, beam_width=5)
    result = solver.solve(max_iterations=100)
    logging.info(f"Best objective: {result.best_objective}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from .graph import NodeT
from .solver import EarlyTerminationCondition

if TYPE_CHECKING:
    from .problem import (
        SolverObserver,
        SubgraphExtractionProblem,
        SubgraphState,
    )

__all__ = [
    "BeamSearchResult",
    "BeamSearchSolver",
]


@dataclass
class BeamSearchResult(Generic[NodeT]):
    """Result from beam search.

    Attributes:
        best_objective: Best objective found across all beam candidates.
        best_state: State corresponding to best_objective.
        best_beam_index: Which beam candidate produced the best result.
        beam_objectives: Objective of each beam candidate at the end.
        total_iterations: Number of iterations completed.
        total_evaluations: Total action evaluations across all beams.
        elapsed_ms: Wall-clock time.
        converged: Whether the search converged.
        convergence_reason: Reason for termination.

    """

    best_objective: float = float("-inf")
    best_state: SubgraphState[NodeT] | None = None
    best_beam_index: int = 0
    beam_objectives: list[float] = field(default_factory=list)
    total_iterations: int = 0
    total_evaluations: int = 0
    elapsed_ms: float = 0.0
    converged: bool = False
    convergence_reason: str = ""


@dataclass
class _BeamCandidate(Generic[NodeT]):
    """Internal representation of a beam candidate."""

    state: SubgraphState[NodeT]
    objective: float


class BeamSearchSolver(Generic[NodeT]):
    """Beam search solver for delta search problems.

    At each iteration, for each of the K beam candidates:
    1. Enumerate candidate actions.
    2. Evaluate each action via calculate_delta.
    3. Generate successor states for the top actions.
    4. Retain the best K successors across all beams.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        beam_width: Number of candidates to maintain (K).
        early_stop: Optional termination conditions.

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        beam_width: int = 5,
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the beam search solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            beam_width: Number of candidates to maintain (K).
            early_stop: Optional termination conditions.

        """
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        self.problem = problem
        self.beam_width = beam_width
        self.early_stop = early_stop or EarlyTerminationCondition()

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
    ) -> BeamSearchResult[NodeT]:
        """Run beam search.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.

        Returns:
            BeamSearchResult with the best solution found.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        # Initialize beam with the initial state
        initial_state = self.problem.evaluate_initial_state(self.problem.graph)
        initial_obj = self.problem.objective(initial_state)
        beam: list[_BeamCandidate[NodeT]] = [
            _BeamCandidate(state=initial_state, objective=initial_obj),
        ]

        best_objective = initial_obj
        best_state: SubgraphState[NodeT] | None = initial_state
        best_beam_index = 0
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
            successors: list[_BeamCandidate[NodeT]] = []

            for beam_idx, candidate in enumerate(beam):
                actions = self.problem.enumerate_actions(candidate.state)
                for action in actions:
                    delta = self.problem.calculate_delta(candidate.state, action)
                    if not delta.feasible:
                        continue

                    obj = (
                        candidate.objective + delta.reward_change - delta.penalty_change
                    )
                    total_evaluations += 1

                    # Create successor state
                    new_state = self.problem.apply_action(
                        candidate.state,
                        action,
                    )
                    successors.append(
                        _BeamCandidate(state=new_state, objective=obj),
                    )

                    if obj > best_objective:
                        best_objective = obj
                        best_state = new_state
                        best_beam_index = beam_idx

            if not successors:
                converged = True
                convergence_reason = "no feasible successors"
                break

            # Retain top-K successors
            successors.sort(key=lambda c: c.objective, reverse=True)
            new_beam = successors[: self.beam_width]

            # Check for improvement
            new_best = max(c.objective for c in new_beam)
            if new_best <= best_objective:
                stall_count += 1
            else:
                stall_count = 0

            beam = new_beam

            elapsed = (time.monotonic() - start_time) * 1000
            if observer:
                best_in_beam = max(beam, key=lambda c: c.objective)
                observer.on_iteration_complete(
                    iteration,
                    None,
                    best_in_beam.objective,
                )

            # Check termination
            if (
                self.early_stop.max_time_ms is not None
                and elapsed >= self.early_stop.max_time_ms
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

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return BeamSearchResult(
            best_objective=best_objective,
            best_state=best_state,
            best_beam_index=best_beam_index,
            beam_objectives=[c.objective for c in beam],
            total_iterations=iteration + 1 if not converged or iteration > 0 else 0,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            converged=converged,
            convergence_reason=convergence_reason,
        )
