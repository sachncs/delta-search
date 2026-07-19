"""Tests for delta_search.progress module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from delta_search.graph import Graph
from delta_search.problems import PrizeCollectingVertexCoverProblem
from delta_search.progress import CallbackObserver, StreamingObserver
from delta_search.solver import GreedySolver

if TYPE_CHECKING:
    from delta_search.problem import Action, DeltaResult


class TestStreamingObserver:
    def test_basic_output(self, caplog) -> None:
        caplog.set_level("INFO")
        observer = StreamingObserver()
        observer.start()

        g = Graph[int].from_edges(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)],
        )
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        result = solver.solve(max_iterations=10, observer=observer)

        assert "[iter" in caplog.text
        assert "[done]" in caplog.text
        assert result.iteration > 0

    def test_verbose_output(self, caplog) -> None:
        caplog.set_level("INFO")
        observer = StreamingObserver(verbose=True)
        observer.start()

        g = Graph[int].from_edges(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)],
        )
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        solver.solve(max_iterations=10, observer=observer)

        assert "[eval]" in caplog.text

    def test_log_file(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "test.log"
        observer = StreamingObserver(log_file=str(path))
        observer.start()

        g = Graph[int].from_edges(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)],
        )
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        solver.solve(max_iterations=10, observer=observer)

        content = path.read_text()
        assert "[iter" in content

    def test_context_manager(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "ctx.log"
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)

        with StreamingObserver(log_file=str(path)) as observer:
            solver.solve(max_iterations=5, observer=observer)

        content = path.read_text()
        assert "[done]" in content

    def test_double_start_closes_previous(self, tmp_path: object) -> None:
        import pathlib

        path = pathlib.Path(str(tmp_path)) / "double.log"
        observer = StreamingObserver(log_file=str(path))
        observer.start()
        observer.start()  # should not leak
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        solver.solve(max_iterations=3, observer=observer)
        assert path.read_text() != ""


class TestCallbackObserver:
    def test_callbacks_are_called(self) -> None:
        eval_count = 0
        iter_count = 0
        done_called = False

        def on_eval(action: Action, delta: DeltaResult, elapsed: float) -> None:
            nonlocal eval_count
            eval_count += 1

        def on_iter(
            iteration: int, best_action: Action | None, objective: float
        ) -> None:
            nonlocal iter_count
            iter_count += 1

        def on_done(iterations: int, final_objective: float) -> None:
            nonlocal done_called
            done_called = True

        observer = CallbackObserver(on_eval=on_eval, on_iter=on_iter, on_done=on_done)

        g = Graph[int].from_edges(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)],
        )
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        solver.solve(max_iterations=10, observer=observer)

        assert eval_count > 0
        assert iter_count > 0
        assert done_called

    def test_no_callbacks(self) -> None:
        observer = CallbackObserver()
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        solver = GreedySolver(problem)
        result = solver.solve(max_iterations=2, observer=observer)
        assert isinstance(result.best_objective, float)
