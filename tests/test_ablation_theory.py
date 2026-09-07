"""Tests for ablation study and theoretical guarantees."""

from __future__ import annotations

from delta_search import (
    Graph,
    GreedySolver,
    PrizeCollectingVertexCoverProblem,
)
from delta_search.ablation import (
    AblationResult,
    AblationStudy,
    ScalingResult,
    ScalingStudy,
    SolverConfig,
)
from delta_search.theory import (
    ApproximationBound,
    ComplexityBounds,
    ConvergenceAnalysis,
    SubmodularAnalyzer,
)


class TestAblationStudy:
    """Tests for AblationStudy."""

    def _make_problem_factory(self):
        def factory(graph):
            return PrizeCollectingVertexCoverProblem(graph)

        return factory

    def test_basic_run(self) -> None:
        study = AblationStudy(
            problem_factory=self._make_problem_factory(),
            graph_sizes=[10, 20],
            max_iterations=10,
        )
        configs = [
            SolverConfig(
                name="greedy",
                solver_factory=lambda p: GreedySolver(p),
            ),
        ]
        results = study.run(configs, num_trials=1)
        assert len(results) == 2  # 2 sizes * 1 trial * 1 config
        assert all(isinstance(r, AblationResult) for r in results)
        assert all(r.graph_size in (10, 20) for r in results)

    def test_multiple_configs(self) -> None:
        study = AblationStudy(
            problem_factory=self._make_problem_factory(),
            graph_sizes=[15],
            max_iterations=5,
        )
        configs = [
            SolverConfig(
                name="greedy-5",
                solver_factory=lambda p: GreedySolver(p),
            ),
            SolverConfig(
                name="greedy-10",
                solver_factory=lambda p: GreedySolver(p),
            ),
        ]
        results = study.run(configs, num_trials=1)
        assert len(results) == 2  # 1 size * 1 trial * 2 configs
        names = {r.config_name for r in results}
        assert names == {"greedy-5", "greedy-10"}

    def test_print_report(self, caplog) -> None:
        caplog.set_level("INFO")
        study = AblationStudy(
            problem_factory=self._make_problem_factory(),
            graph_sizes=[10],
            max_iterations=5,
        )
        configs = [
            SolverConfig(
                name="greedy",
                solver_factory=lambda p: GreedySolver(p),
            ),
        ]
        results = study.run(configs, num_trials=1)
        study.print_report(results)
        assert "ABLATION STUDY REPORT" in caplog.text


class TestScalingStudy:
    """Tests for ScalingStudy."""

    def _make_problem_factory(self):
        def factory(graph):
            return PrizeCollectingVertexCoverProblem(graph)

        return factory

    def test_basic_run(self) -> None:
        study = ScalingStudy(
            problem_factory=self._make_problem_factory(),
            graph_sizes=[10, 20],
            max_iterations=10,
        )
        configs = [
            SolverConfig(
                name="greedy",
                solver_factory=lambda p: GreedySolver(p),
            ),
        ]
        results = study.run(configs, num_trials=1)
        assert len(results) == 2
        assert all(isinstance(r, ScalingResult) for r in results)

    def test_print_report(self, caplog) -> None:
        caplog.set_level("INFO")
        study = ScalingStudy(
            problem_factory=self._make_problem_factory(),
            graph_sizes=[10, 20],
            max_iterations=5,
        )
        configs = [
            SolverConfig(
                name="greedy",
                solver_factory=lambda p: GreedySolver(p),
            ),
        ]
        results = study.run(configs, num_trials=1)
        study.print_report(results)
        assert "SCALING STUDY REPORT" in caplog.text


class TestSubmodularAnalyzer:
    """Tests for SubmodularAnalyzer."""

    def test_pcvc_classification(self) -> None:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        is_monotone, is_submodular = analyzer.classify_submodularity()
        assert is_monotone is True
        assert is_submodular is True

    def test_approximation_bound(self) -> None:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        bound = analyzer.compute_bound()
        assert isinstance(bound, ApproximationBound)
        assert bound.is_monotone is True
        assert bound.is_submodular is True
        assert bound.ratio > 0
        assert bound.bound_type == "greedy-1-1/e"

    def test_convergence_analysis(self) -> None:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        conv = analyzer.convergence_analysis(epsilon=0.1)
        assert isinstance(conv, ConvergenceAnalysis)
        assert conv.has_convergence is True
        assert conv.max_iterations > 0

    def test_complexity_bounds(self) -> None:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        comp = analyzer.complexity_bounds()
        assert isinstance(comp, ComplexityBounds)
        assert comp.delta_time_complexity != ""
        assert comp.overall_complexity != ""

    def test_full_analysis(self) -> None:
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        result = analyzer.full_analysis()
        assert result["problem_name"] == "PrizeCollectingVertexCoverProblem"
        assert result["is_monotone"] is True
        assert result["is_submodular"] is True
        assert "approximation_bound" in result
        assert "convergence" in result
        assert "complexity" in result

    def test_print_report(self, caplog) -> None:
        caplog.set_level("INFO")
        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = PrizeCollectingVertexCoverProblem(graph)
        analyzer = SubmodularAnalyzer(problem)
        analyzer.print_report()
        assert "THEORETICAL ANALYSIS REPORT" in caplog.text
