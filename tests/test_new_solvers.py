"""Tests for beam, anytime, streaming, multi-objective, and learned solvers."""

from __future__ import annotations

import pytest

from delta_search import (
    Graph,
    PrizeCollectingVertexCoverProblem,
)
from delta_search.anytime import AnytimeResult, AnytimeSolver
from delta_search.beam import BeamSearchResult, BeamSearchSolver
from delta_search.learned import LearnedGuidanceSolver, _extract_features
from delta_search.multi_objective import (
    MultiObjectiveResult,
    MultiObjectiveSolver,
    ObjectiveWeights,
    ParetoPoint,
    _dominates,
    _update_pareto,
)
from delta_search.streaming import StreamingResult, StreamingSolver


class TestBeamSearch:
    """Tests for BeamSearchSolver."""

    def _make_problem(self) -> PrizeCollectingVertexCoverProblem:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        return PrizeCollectingVertexCoverProblem(graph)

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        solver = BeamSearchSolver(problem, beam_width=2)
        result = solver.solve(max_iterations=100)
        assert isinstance(result, BeamSearchResult)
        assert result.best_state is not None
        assert result.total_iterations > 0

    def test_beam_width_1(self) -> None:
        problem = self._make_problem()
        solver = BeamSearchSolver(problem, beam_width=1)
        result = solver.solve(max_iterations=50)
        assert result.best_state is not None

    def test_invalid_beam_width(self) -> None:
        problem = self._make_problem()
        with pytest.raises(ValueError, match="beam_width must be positive"):
            BeamSearchSolver(problem, beam_width=0)

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = BeamSearchSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_convergence(self) -> None:
        problem = self._make_problem()
        solver = BeamSearchSolver(problem, beam_width=2)
        result = solver.solve(max_iterations=5)
        assert result.total_iterations <= 5


class TestAnytimeSolver:
    """Tests for AnytimeSolver."""

    def _make_problem(self) -> PrizeCollectingVertexCoverProblem:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        return PrizeCollectingVertexCoverProblem(graph)

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        solver = AnytimeSolver(problem)
        result = solver.solve(max_iterations=100)
        assert isinstance(result, AnytimeResult)
        assert result.best_state is not None
        assert result.total_iterations > 0
        assert len(result.progress_history) > 0

    def test_snapshots_track_progress(self) -> None:
        problem = self._make_problem()
        solver = AnytimeSolver(problem)
        result = solver.solve(max_iterations=20)
        for snap in result.progress_history:
            assert snap.iteration >= 0
            assert snap.objective is not None

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = AnytimeSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_max_history_caps_snapshots(self) -> None:
        problem = self._make_problem()
        solver = AnytimeSolver(problem, snapshot_interval=1, max_history=5)
        result = solver.solve(max_iterations=50)
        # Cap applies to interval snapshots; initial + final are always added
        assert len(result.progress_history) <= 7


class TestStreamingSolver:
    """Tests for StreamingSolver."""

    def _make_graph(self) -> Graph[int]:
        return Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])

    def _make_problem(self) -> PrizeCollectingVertexCoverProblem:
        return PrizeCollectingVertexCoverProblem(self._make_graph())

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        solver = StreamingSolver(problem)
        result = solver.solve(max_iterations=100)
        assert isinstance(result, StreamingResult)
        assert result.best_state is not None

    def test_apply_updates(self) -> None:
        problem = self._make_problem()
        solver = StreamingSolver(problem)
        solver.add_edge(4, 5)
        solver.add_node(6)
        result = solver.solve(max_iterations=20)
        assert result.best_state is not None

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = StreamingSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)


class TestMultiObjective:
    """Tests for MultiObjectiveSolver and Pareto utilities."""

    def _make_problem(self) -> PrizeCollectingVertexCoverProblem:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        return PrizeCollectingVertexCoverProblem(graph)

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        weights = ObjectiveWeights(
            objectives=["reward", "penalty"],
            weights=[0.5, 0.5],
        )
        solver = MultiObjectiveSolver(problem, objective_weights=weights)
        result = solver.solve(max_iterations=100)
        assert isinstance(result, MultiObjectiveResult)
        assert result.best_state is not None
        assert len(result.pareto_front) >= 1

    def test_pareto_dominates(self) -> None:
        a = {"obj1": 10.0, "obj2": 5.0}
        b = {"obj1": 8.0, "obj2": 4.0}
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_pareto_equal_not_dominating(self) -> None:
        a = {"obj1": 10.0, "obj2": 5.0}
        b = {"obj1": 10.0, "obj2": 5.0}
        assert not _dominates(a, b)

    def test_update_pareto_adds_and_removes(self) -> None:
        from delta_search.problem import DefaultState

        front: list[ParetoPoint[int]] = [
            ParetoPoint(
                state=DefaultState[int](),
                objectives={"x": 1.0, "y": 2.0},
                scalarized=1.5,
            ),
        ]
        candidate: ParetoPoint[int] = ParetoPoint(
            state=DefaultState[int](),
            objectives={"x": 2.0, "y": 3.0},
            scalarized=2.5,
        )
        front = _update_pareto(front, candidate)
        assert len(front) == 1
        assert front[0].objectives == candidate.objectives

    def test_objective_weights_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            ObjectiveWeights(objectives=["a", "b"], weights=[1.0])

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = MultiObjectiveSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_max_pareto_size_caps_front(self) -> None:
        problem = self._make_problem()
        weights = ObjectiveWeights(
            objectives=["reward", "penalty"],
            weights=[0.5, 0.5],
        )
        solver = MultiObjectiveSolver(
            problem,
            objective_weights=weights,
            max_pareto_size=2,
        )
        result = solver.solve(max_iterations=100)
        assert len(result.pareto_front) <= 2


class TestLearnedGuidance:
    """Tests for LearnedGuidanceSolver."""

    def _make_problem(self) -> PrizeCollectingVertexCoverProblem:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        return PrizeCollectingVertexCoverProblem(graph)

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        solver = LearnedGuidanceSolver(problem, min_samples=5)
        result = solver.solve(max_iterations=100)
        assert result["best_state"] is not None
        assert result["total_iterations"] > 0

    def test_extract_features(self) -> None:
        problem = self._make_problem()
        state = problem.evaluate_initial_state(problem.graph)
        from delta_search.problem import Action, ActionType

        action = Action(
            action_type=ActionType.ADD_NODE,
            targets=(1,),
        )
        features = _extract_features(problem, state, action, 0.0)
        assert "degree" in features
        assert "is_add" in features
        assert features["is_add"] == 1.0
        assert features["is_node"] == 1.0

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = LearnedGuidanceSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_custom_sklearn_params(self) -> None:
        problem = self._make_problem()
        solver = LearnedGuidanceSolver(
            problem,
            min_samples=5,
            n_estimators=5,
            max_depth=2,
        )
        result = solver.solve(max_iterations=100)
        assert result["best_state"] is not None
