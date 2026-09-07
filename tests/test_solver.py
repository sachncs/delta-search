"""Tests for delta_search.solver module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from delta_search.graph import Graph
from delta_search.problem import (
    Action,
    ActionType,
    DeltaResult,
    SubgraphExtractionProblem,
)
from delta_search.solver import EarlyTerminationCondition, GreedySolver, SolverState


@dataclass
class SimpleState:
    graph: Graph[int]
    undo: object = None


class SimpleProblem(SubgraphExtractionProblem[int]):
    def evaluate_initial_state(self, graph: Graph[int]) -> SimpleState:
        return SimpleState(graph=Graph[int]())

    def calculate_delta(
        self,
        current_state: SimpleState,
        candidate_action: Action,
    ) -> DeltaResult:
        if candidate_action.action_type is ActionType.ADD_EDGE:
            return DeltaResult(reward_change=1.0, penalty_change=0.0, feasible=True)
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=True)

    def compute_reward(self, state: SimpleState) -> float:
        return float(state.graph.num_edges)

    def compute_penalty(self, state: SimpleState) -> float:
        return 0.0

    def is_feasible(self, state: SimpleState) -> bool:
        return True


class TestGreedySolver:
    def test_basic_solve(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)
        result = solver.solve(max_iterations=5)
        assert isinstance(result, SolverState)
        assert result.iteration > 0
        assert result.best_objective >= 0

    def test_convergence_no_actions(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)
        result = solver.solve(max_iterations=10)
        assert result.converged
        assert "no actions" in result.convergence_reason

    def test_early_stop_max_evaluations(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(max_evaluations=3)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.total_evaluations <= 3

    def test_early_stop_stall(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(stall_iterations=2)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.converged

    def test_early_stop_objective_target(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(objective_target=2.0)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.best_objective >= 2.0

    def test_best_state_preserved(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)
        result = solver.solve(max_iterations=10)
        assert result.best_state is not None
        assert result.best_state.graph.num_edges > 0

    def test_with_observer(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)

        events: list[str] = []

        class TrackingObserver:
            def on_action_evaluated(self, action, delta, elapsed_ms):
                events.append("evaluated")

            def on_iteration_complete(self, iteration, best_action, objective):
                events.append("iteration")

            def on_convergence(self, iterations, final_objective):
                events.append("converged")

        result = solver.solve(max_iterations=3, observer=TrackingObserver())
        assert "converged" in events
        assert len(events) >= 2
        assert result.best_objective >= 0


class TestSolverState:
    def test_defaults(self) -> None:
        s = SolverState()
        assert s.iteration == 0
        assert s.best_objective == float("-inf")
        assert s.best_state is None
        assert s.total_evaluations == 0
        assert s.elapsed_ms == 0.0
        assert not s.converged


class TestEarlyTerminationCondition:
    def test_defaults(self) -> None:
        c = EarlyTerminationCondition()
        assert c.max_iterations is None
        assert c.max_evaluations is None
        assert c.max_time_ms is None
        assert c.objective_target is None
        assert c.stall_iterations is None


class TestSolverEdgeCases:
    def test_max_iterations_zero_raises(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_max_iterations_negative_raises(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=-5)

    def test_max_time_ms_termination(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(max_time_ms=0.001)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100000)
        assert result.converged
        assert "time limit" in result.convergence_reason

    def test_convergence_reason_evaluations(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(max_evaluations=2)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.converged
        assert "evaluations" in result.convergence_reason

    def test_convergence_reason_objective_target(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(objective_target=1.0)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.converged
        assert "target" in result.convergence_reason

    def test_convergence_reason_stall(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        problem = SimpleProblem(g)
        stop = EarlyTerminationCondition(stall_iterations=2)
        solver = GreedySolver(problem, early_stop=stop)
        result = solver.solve(max_iterations=100)
        assert result.converged
        assert "stalled" in result.convergence_reason

    def test_observer_callback_count(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        solver = GreedySolver(problem)

        eval_count = 0
        iter_count = 0

        class CountingObserver:
            def on_action_evaluated(self, action, delta, elapsed_ms):
                nonlocal eval_count
                eval_count += 1

            def on_iteration_complete(self, iteration, best_action, objective):
                nonlocal iter_count
                iter_count += 1

            def on_convergence(self, iterations, final_objective):
                pass

        result = solver.solve(max_iterations=3, observer=CountingObserver())
        assert eval_count > 0
        assert iter_count == result.iteration
