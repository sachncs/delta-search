"""Tests for delta_search.multistart module."""

from __future__ import annotations

import pytest

from delta_search.graph import Graph
from delta_search.multistart import (
    MultiStartResult,
    MultiStartSolver,
    _generate_random_initial_state,
)
from delta_search.problems import (
    PrizeCollectingVertexCoverProblem,
)


class TestGenerateRandomInitialState:
    def test_produces_nonempty_state(self) -> None:
        import random

        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        rng = random.Random(42)
        state = _generate_random_initial_state(g, rng, fraction=0.5)
        assert state.graph.num_nodes > 0

    def test_subset_of_input(self) -> None:
        import random

        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1)])
        rng = random.Random(0)
        state = _generate_random_initial_state(g, rng, fraction=0.3)
        assert state.graph.num_nodes <= g.num_nodes

    def test_edges_are_subgraph(self) -> None:
        import random

        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        rng = random.Random(7)
        state = _generate_random_initial_state(g, rng, fraction=1.0)
        for u, v in state.graph.edges:
            assert g.has_edge(u, v)

    def test_fraction_1_returns_all(self) -> None:
        import random

        g = Graph[int].from_edges([(1, 2), (2, 3)])
        rng = random.Random(0)
        state = _generate_random_initial_state(g, rng, fraction=1.0)
        assert state.graph.num_nodes == 3


class TestMultiStartSolver:
    def test_basic_solve(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=3, seed=42)
        result = solver.solve(max_iterations=5)
        assert isinstance(result, MultiStartResult)
        assert result.num_starts == 3
        assert len(result.all_objectives) == 3

    def test_best_objective(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=5, seed=42)
        result = solver.solve(max_iterations=10)
        assert result.best_objective >= max(result.all_objectives) - 1e-9

    def test_best_start_index_valid(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=5, seed=42)
        result = solver.solve(max_iterations=10)
        assert 0 <= result.best_start_index < result.num_starts

    def test_total_iterations(self) -> None:
        g = Graph[int].from_edges(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3), (2, 4)],
        )
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(
            problem,
            num_starts=5,
            seed=42,
            initial_fraction=0.5,
        )
        result = solver.solve(max_iterations=20)
        assert result.total_iterations > 0

    def test_reproducible_with_seed(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver1 = MultiStartSolver(problem, num_starts=3, seed=42)
        r1 = solver1.solve(max_iterations=5)
        solver2 = MultiStartSolver(problem, num_starts=3, seed=42)
        r2 = solver2.solve(max_iterations=5)
        assert r1.all_objectives == r2.all_objectives

    def test_num_starts_zero_raises(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        with pytest.raises(ValueError, match="positive"):
            MultiStartSolver(problem, num_starts=0)

    def test_max_iterations_zero_raises(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=2, seed=42)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)

    def test_with_nonmonotone_problem(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=5, seed=42)
        result = solver.solve(max_iterations=10)
        assert result.num_starts == 5
        assert len(result.all_objectives) == 5

    def test_elapsed_ms_positive(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = MultiStartSolver(problem, num_starts=2, seed=42)
        result = solver.solve(max_iterations=5)
        assert result.elapsed_ms >= 0
