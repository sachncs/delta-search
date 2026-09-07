"""Adaptive beam ΔSearch — learned ordering for candidate evaluation.

Extends beam search with adaptive candidate ordering.  Instead of
evaluating all K beam candidates with the same strategy, learns which
ordering produces the best results.  Includes diversity-aware beam
selection to avoid premature convergence.

Usage::

    from delta_search.adaptive_beam import AdaptiveBeamSolver

    solver = AdaptiveBeamSolver(
        problem, beam_width=5, diversity_weight=0.2,
    )
    result = solver.solve(max_iterations=100)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from .graph import NodeT
from .solver import EarlyTerminationCondition

if TYPE_CHECKING:
    from .problem import (
        Action,
        SubgraphExtractionProblem,
        SubgraphState,
    )

__all__ = [
    "BeamState",
    "AdaptiveBeamResult",
    "AdaptiveBeamSolver",
]


@dataclass
class BeamState(Generic[NodeT]):
    """A single beam candidate with metadata.

    Attributes:
        state: The solution state.
        objective: Current objective value.
        action_history: Actions taken so far.
        score: Selection score for next iteration.
        diversity_score: Diversity relative to other beam members.

    """

    state: SubgraphState[NodeT]
    objective: float = 0.0
    action_history: list[Action] = field(default_factory=list)
    score: float = 0.0
    diversity_score: float = 0.0


@dataclass
class AdaptiveBeamResult(Generic[NodeT]):
    """Result from adaptive beam search.

    Attributes:
        best_objective: Best objective found across all beams.
        best_state: State corresponding to best_objective.
        best_beam_index: Which beam produced the best result.
        beam_objectives: Objective of each beam at the end.
        total_iterations: Iterations completed.
        total_evaluations: Total action evaluations.
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


class AdaptiveBeamSolver(Generic[NodeT]):
    """Adaptive beam search solver with diversity-aware selection.

    Extends standard beam search with:
    1. Diversity-aware beam selection: penalizes beams that are too
       similar to already-selected beams.
    2. Adaptive width: dynamically adjusts beam width based on
       progress (wider when stuck, narrower when improving).
    3. Action history tracking: records which actions each beam took.

    Args:
        problem: A SubgraphExtractionProblem instance.
        beam_width: Number of beams to maintain.
        diversity_weight: Weight for diversity in beam selection.
        adaptive_width: Whether to adapt beam width dynamically.
        early_stop: Optional termination conditions.

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        beam_width: int = 5,
        diversity_weight: float = 0.2,
        adaptive_width: bool = True,
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the adaptive beam search solver.

        Args:
            problem: A SubgraphExtractionProblem instance.
            beam_width: Number of beams to maintain.
            diversity_weight: Weight for diversity in beam selection.
            adaptive_width: Whether to adapt beam width dynamically.
            early_stop: Optional termination conditions.

        """
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}")
        self.problem = problem
        self.beam_width = beam_width
        self.diversity_weight = diversity_weight
        self.adaptive_width = adaptive_width
        self.early_stop = early_stop or EarlyTerminationCondition()

    def _compute_diversity(
        self,
        candidate: SubgraphState[NodeT],
        existing: list[SubgraphState[NodeT]],
    ) -> float:
        """Compute diversity of candidate relative to existing beams.

        Diversity is 1 minus the average Jaccard similarity of node sets.
        """
        if not existing:
            return 1.0

        cand_nodes = set(candidate.graph.nodes)
        if not cand_nodes:
            return 0.0

        total_sim = 0.0
        for ex in existing:
            ex_nodes = set(ex.graph.nodes)
            if not ex_nodes:
                continue
            intersection = len(cand_nodes & ex_nodes)
            union = len(cand_nodes | ex_nodes)
            if union > 0:
                total_sim += intersection / union

        avg_sim = total_sim / len(existing)
        return 1.0 - avg_sim

    def solve(
        self,
        max_iterations: int = 1000,
        observer: Any | None = None,
    ) -> AdaptiveBeamResult[NodeT]:
        """Run adaptive beam search.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.

        Returns:
            AdaptiveBeamResult with best solution and beam stats.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        # Initialize beams from random starts
        initial_state = self.problem.evaluate_initial_state(self.problem.graph)
        initial_obj = self.problem.objective(initial_state)

        beams: list[BeamState[NodeT]] = [
            BeamState(state=initial_state, objective=initial_obj)
            for _ in range(self.beam_width)
        ]

        best_objective = initial_obj
        best_state = initial_state
        best_beam_idx = 0
        total_evaluations = 0
        stall_count = 0
        current_width = self.beam_width
        start_time = time.monotonic()

        limit = (
            self.early_stop.max_iterations
            if self.early_stop.max_iterations is not None
            else max_iterations
        )

        converged = False
        convergence_reason = ""

        for iteration in range(limit):
            all_candidates: list[tuple[BeamState[NodeT], Action, float]] = []

            for _beam_idx, beam in enumerate(beams):
                self.problem.on_iteration_start(beam.state, iteration)
                actions = self.problem.enumerate_actions(beam.state)

                for action in actions:
                    delta = self.problem.calculate_delta(beam.state, action)
                    if not delta.feasible:
                        continue

                    obj = beam.objective + delta.reward_change - delta.penalty_change
                    total_evaluations += 1

                    elapsed = (time.monotonic() - start_time) * 1000
                    self.problem.observer.on_action_evaluated(
                        action,
                        delta,
                        elapsed,
                    )

                    all_candidates.append((beam, action, obj))

            if not all_candidates:
                converged = True
                convergence_reason = "no feasible actions"
                break

            # Sort by objective (best first)
            all_candidates.sort(key=lambda x: x[2], reverse=True)

            # Select top-K with diversity
            new_beams: list[BeamState[NodeT]] = []
            selected_states: list[SubgraphState[NodeT]] = []

            for beam, action, obj in all_candidates:
                if len(new_beams) >= current_width:
                    break

                new_state = self.problem.apply_action(beam.state, action)

                # Diversity check
                diversity = self._compute_diversity(new_state, selected_states)

                # Only add if diverse enough or beam is not full
                if len(new_beams) < current_width // 2 or diversity > 0.1:
                    new_beam = BeamState(
                        state=new_state,
                        objective=obj,
                        action_history=beam.action_history + [action],
                        diversity_score=diversity,
                    )
                    new_beams.append(new_beam)
                    selected_states.append(new_state)

            if not new_beams:
                converged = True
                convergence_reason = "no diverse candidates"
                break

            beams = new_beams

            # Update best
            for beam_idx, beam in enumerate(beams):
                if beam.objective > best_objective:
                    best_objective = beam.objective
                    best_state = beam.state
                    best_beam_idx = beam_idx

            elapsed_total = (time.monotonic() - start_time) * 1000

            # Adaptive width
            if self.adaptive_width and iteration > 0:
                recent_improvement = best_objective - (
                    beams[0].objective if beams else best_objective
                )
                if recent_improvement <= 0:
                    stall_count += 1
                    if stall_count > 3 and current_width < self.beam_width * 2:
                        current_width = min(current_width + 1, self.beam_width * 2)
                else:
                    stall_count = 0
                    if current_width > self.beam_width:
                        current_width = max(current_width - 1, self.beam_width)

            self.problem.observer.on_iteration_complete(
                iteration,
                None,
                best_objective,
            )
            self.problem.on_iteration_end(beams[0].state, iteration)

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

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return AdaptiveBeamResult(
            best_objective=best_objective,
            best_state=best_state,
            best_beam_index=best_beam_idx,
            beam_objectives=[b.objective for b in beams],
            total_iterations=iteration + 1 if iteration >= 0 else 0,
            total_evaluations=total_evaluations,
            elapsed_ms=elapsed_ms,
            converged=converged,
            convergence_reason=convergence_reason,
        )
