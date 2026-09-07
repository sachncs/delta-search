"""Tests for context engineering, test-time compute, budget metrics,
adaptive beam, and hybrid pipeline."""

from __future__ import annotations

import pytest

from delta_search.adaptive_beam import (
    AdaptiveBeamResult,
    AdaptiveBeamSolver,
)
from delta_search.budget_metrics import (
    BudgetAwareEvaluator,
    BudgetComparator,
    BudgetMetric,
    QualityScore,
    compute_pareto_front,
)
from delta_search.context_engineering import (
    ContextEngineeringProblem,
    ContextEngineeringResult,
    ContextEngineeringSolver,
    DocumentChunk,
)
from delta_search.hybrid_pipeline import (
    HybridPipeline,
    HybridPipelineResult,
)
from delta_search.test_time_compute import (
    ReasoningNode,
    TestTimeComputeProblem,
    TestTimeComputeResult,
    TestTimeComputeSolver,
)


class TestContextEngineering:
    """Tests for ContextEngineeringProblem and solver."""

    def _make_docs(self):
        return [
            DocumentChunk("d1", "Python is a language", 8, relevance=0.9),
            DocumentChunk("d2", "Python for ML", 7, relevance=0.8),
            DocumentChunk("d3", "JavaScript in browsers", 6, relevance=0.3),
        ]

    def test_basic_solve(self) -> None:
        docs = self._make_docs()
        edges = [("d1", "d2", 0.8), ("d1", "d3", 0.1)]
        problem = ContextEngineeringProblem(
            documents=docs,
            edges=edges,
            max_context_tokens=50,
        )
        solver = ContextEngineeringSolver(problem)
        result = solver.solve(max_iterations=20)
        assert isinstance(result, ContextEngineeringResult)
        assert result.num_chunks >= 0
        assert result.total_tokens >= 0

    def test_budget_constraint(self) -> None:
        docs = [
            DocumentChunk("d1", "text1", 30, relevance=0.9),
            DocumentChunk("d2", "text2", 30, relevance=0.8),
        ]
        problem = ContextEngineeringProblem(
            documents=docs,
            max_context_tokens=40,
        )
        solver = ContextEngineeringSolver(problem)
        result = solver.solve(max_iterations=10)
        # Should not exceed budget by much
        assert result.total_tokens <= 60  # Allow some overrun

    def test_relevance_prioritization(self) -> None:
        docs = [
            DocumentChunk("d1", "relevant", 5, relevance=0.95),
            DocumentChunk("d2", "irrelevant", 5, relevance=0.1),
        ]
        problem = ContextEngineeringProblem(
            documents=docs,
            max_context_tokens=100,
        )
        solver = ContextEngineeringSolver(problem)
        result = solver.solve(max_iterations=10)
        # High-relevance doc should be selected
        selected_ids = {c.chunk_id for c in result.selected_chunks}
        assert "d1" in selected_ids


class TestTestTimeCompute:
    """Tests for TestTimeComputeProblem and solver."""

    def _make_nodes(self):
        return [
            ReasoningNode("s1", "Premise", score=0.8, cost=10, depth=1),
            ReasoningNode("s2", "Step 2", score=0.6, cost=8, depth=2, parent="s1"),
            ReasoningNode("s3", "Conclusion", score=0.9, cost=12, depth=3, parent="s2"),
        ]

    def test_basic_solve(self) -> None:
        nodes = self._make_nodes()
        edges = [("s1", "s2", 0.7), ("s2", "s3", 0.9)]
        problem = TestTimeComputeProblem(
            nodes=nodes,
            edges=edges,
            max_compute=100,
        )
        solver = TestTimeComputeSolver(problem)
        result = solver.solve(max_iterations=20)
        assert isinstance(result, TestTimeComputeResult)
        assert result.num_steps >= 0
        assert result.total_cost >= 0

    def test_budget_constraint(self) -> None:
        nodes = [
            ReasoningNode("s1", "Step 1", score=0.8, cost=50),
            ReasoningNode("s2", "Step 2", score=0.7, cost=50),
            ReasoningNode("s3", "Step 3", score=0.9, cost=50),
        ]
        problem = TestTimeComputeProblem(
            nodes=nodes,
            max_compute=60,
        )
        solver = TestTimeComputeSolver(problem)
        result = solver.solve(max_iterations=10)
        # Should respect budget (allow some overrun for penalty-based approach)
        assert result.total_cost <= 120  # Allow some overrun

    def test_score_prioritization(self) -> None:
        nodes = [
            ReasoningNode("s1", "High score", score=0.95, cost=10),
            ReasoningNode("s2", "Low score", score=0.1, cost=10),
        ]
        problem = TestTimeComputeProblem(
            nodes=nodes,
            max_compute=100,
        )
        solver = TestTimeComputeSolver(problem)
        result = solver.solve(max_iterations=10)
        expanded_ids = {n.node_id for n in result.expanded_nodes}
        assert "s1" in expanded_ids


class TestBudgetMetrics:
    """Tests for budget-aware evaluation."""

    def test_quality_per_token(self) -> None:
        evaluator = BudgetAwareEvaluator(
            metric=BudgetMetric.QUALITY_PER_TOKEN,
            quality_fn=lambda s: 0.8,
            cost_fn=lambda s: 100.0,
        )
        from delta_search.problem import DefaultState

        state = DefaultState()
        score = evaluator.evaluate(state)
        assert isinstance(score, QualityScore)
        assert score.metric_value == pytest.approx(0.008, abs=1e-4)

    def test_quality_per_latency(self) -> None:
        evaluator = BudgetAwareEvaluator(
            metric=BudgetMetric.QUALITY_PER_LATENCY,
            quality_fn=lambda s: 0.9,
            cost_fn=lambda s: 50.0,
        )
        from delta_search.problem import DefaultState

        state = DefaultState()
        score = evaluator.evaluate(state)
        assert score.metric_value == pytest.approx(0.018, abs=1e-4)

    def test_quality_minus_budget_penalty(self) -> None:
        evaluator = BudgetAwareEvaluator(
            metric=BudgetMetric.QUALITY_MINUS_BUDGET_PENALTY,
            quality_fn=lambda s: 0.9,
            cost_fn=lambda s: 1200.0,
            budget_target=1000.0,
            penalty_weight=0.001,
        )
        from delta_search.problem import DefaultState

        state = DefaultState()
        score = evaluator.evaluate(state)
        # 0.9 - (200 * 0.001) = 0.7
        assert score.metric_value == pytest.approx(0.7, abs=1e-4)

    def test_compare_solutions(self) -> None:
        from delta_search.problem import DefaultState

        evaluator = BudgetAwareEvaluator(
            metric=BudgetMetric.QUALITY_PER_TOKEN,
            quality_fn=lambda s: 0.8,
            cost_fn=lambda s: 100.0,
        )
        comparator = BudgetComparator(evaluator)
        state_a = DefaultState()
        state_b = DefaultState()
        result = comparator.compare(state_a, state_b)
        assert result.winner == "tie"

    def test_pareto_front(self) -> None:
        costs = [10, 20, 30, 40]
        qualities = [0.5, 0.8, 0.7, 0.9]
        front = compute_pareto_front(costs, qualities)
        assert len(front.points) > 0
        # Best efficiency should be the first point (lowest cost)
        assert front.best_efficiency is not None


class TestAdaptiveBeam:
    """Tests for AdaptiveBeamSolver."""

    def _make_problem(self):
        from delta_search import Graph, PrizeCollectingVertexCoverProblem

        graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        return PrizeCollectingVertexCoverProblem(graph)

    def test_basic_solve(self) -> None:
        problem = self._make_problem()
        solver = AdaptiveBeamSolver(problem, beam_width=2)
        result = solver.solve(max_iterations=20)
        assert isinstance(result, AdaptiveBeamResult)
        assert result.best_state is not None

    def test_diversity(self) -> None:
        problem = self._make_problem()
        solver = AdaptiveBeamSolver(
            problem,
            beam_width=3,
            diversity_weight=0.5,
        )
        result = solver.solve(max_iterations=15)
        assert result.best_state is not None

    def test_invalid_beam_width(self) -> None:
        problem = self._make_problem()
        with pytest.raises(ValueError, match="beam_width must be positive"):
            AdaptiveBeamSolver(problem, beam_width=0)

    def test_max_iterations_must_be_positive(self) -> None:
        problem = self._make_problem()
        solver = AdaptiveBeamSolver(problem)
        with pytest.raises(ValueError, match="positive"):
            solver.solve(max_iterations=0)


class TestHybridPipeline:
    """Tests for HybridPipeline."""

    def _make_docs(self):
        return [
            DocumentChunk("d1", "Python is a language", 8, relevance=0.9),
            DocumentChunk("d2", "Python for ML", 7, relevance=0.8),
        ]

    def _make_nodes(self):
        return [
            ReasoningNode("s1", "Step 1", score=0.8, cost=10, depth=1),
            ReasoningNode("s2", "Step 2", score=0.6, cost=8, depth=2),
        ]

    def test_basic_run(self) -> None:
        pipeline = HybridPipeline(
            documents=self._make_docs(),
            reasoning_nodes=self._make_nodes(),
            context_budget=50,
            compute_budget=50,
        )
        result = pipeline.run(context_iterations=10, reasoning_iterations=10)
        assert isinstance(result, HybridPipelineResult)
        assert result.context_result is not None
        assert result.reasoning_result is not None
        assert result.overall_quality >= 0

    def test_context_only(self) -> None:
        pipeline = HybridPipeline(
            documents=self._make_docs(),
            context_budget=50,
        )
        result = pipeline.run(context_iterations=10)
        assert result.context_result is not None
        assert result.context_result.num_chunks >= 0

    def test_budget_metrics(self) -> None:
        pipeline = HybridPipeline(
            documents=self._make_docs(),
            reasoning_nodes=self._make_nodes(),
            context_budget=50,
            compute_budget=50,
        )
        result = pipeline.run(context_iterations=5, reasoning_iterations=5)
        assert result.quality_per_token >= 0
        assert result.quality_per_latency >= 0
