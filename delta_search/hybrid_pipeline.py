"""Hybrid pipeline for retrieval and reasoning using ΔSearch.

A two-stage system: first ΔSearch selects context (retrieval),
then a second ΔSearch stage selects reasoning paths.  This gives
the paper a coherent systems architecture rather than a single
isolated trick.

Usage::

    from delta_search.hybrid_pipeline import HybridPipeline

    pipeline = HybridPipeline(
        documents=docs, edges=doc_edges,
        reasoning_nodes=reason_nodes, reasoning_edges=reason_edges,
        context_budget=500, compute_budget=100,
    )
    result = pipeline.run()
    logging.info(f"Answer quality: {result.quality_score}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from .context_engineering import (
    ContextEngineeringProblem,
    ContextEngineeringResult,
    ContextEngineeringSolver,
    DocumentChunk,
)
from .graph import NodeT
from .test_time_compute import (
    ReasoningNode,
    TestTimeComputeProblem,
    TestTimeComputeResult,
    TestTimeComputeSolver,
)

if TYPE_CHECKING:
    from .solver import EarlyTerminationCondition

__all__ = [
    "PipelineConfig",
    "PipelineStageResult",
    "HybridPipelineResult",
    "HybridPipeline",
]


@dataclass
class PipelineConfig:
    """Configuration for the hybrid pipeline.

    Attributes:
        context_budget: Token budget for context selection.
        compute_budget: Compute budget for reasoning.
        context_iterations: Max iterations for context stage.
        reasoning_iterations: Max iterations for reasoning stage.
        context_relevance_weight: Weight for relevance in context stage.
        context_coverage_weight: Weight for coverage in context stage.
        reasoning_score_weight: Weight for reasoning quality.
        end_to_end_optimize: Whether to jointly optimize both stages.

    """

    context_budget: int = 4096
    compute_budget: int = 1000
    context_iterations: int = 50
    reasoning_iterations: int = 50
    context_relevance_weight: float = 1.0
    context_coverage_weight: float = 0.3
    reasoning_score_weight: float = 1.0
    end_to_end_optimize: bool = False


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage.

    Attributes:
        stage_name: Name of the stage.
        selected_items: IDs of selected items.
        total_cost: Cost incurred by this stage.
        quality: Quality achieved by this stage.
        elapsed_ms: Wall-clock time.

    """

    stage_name: str = ""
    selected_items: list[Any] = field(default_factory=list)
    total_cost: float = 0.0
    quality: float = 0.0
    elapsed_ms: float = 0.0


@dataclass
class HybridPipelineResult(Generic[NodeT]):
    """Full result from the hybrid pipeline.

    Attributes:
        context_result: Result from context selection stage.
        reasoning_result: Result from reasoning stage.
        overall_quality: Combined quality score.
        total_tokens: Total tokens used.
        total_compute: Total compute used.
        quality_per_token: Quality per token (budget metric).
        quality_per_latency: Quality per millisecond.
        elapsed_ms: Total wall-clock time.

    """

    context_result: ContextEngineeringResult | None = None
    reasoning_result: TestTimeComputeResult | None = None
    overall_quality: float = 0.0
    total_tokens: int = 0
    total_compute: int = 0
    quality_per_token: float = 0.0
    quality_per_latency: float = 0.0
    elapsed_ms: float = 0.0


class HybridPipeline(Generic[NodeT]):
    """Two-stage hybrid pipeline: ΔSearch context + ΔSearch reasoning.

    Stage 1 (Retrieval): ΔSearch selects which documents/chunks enter
    the prompt, maximizing relevance and coverage under a token budget.

    Stage 2 (Reasoning): ΔSearch expands reasoning branches using the
    selected context, maximizing quality under a compute budget.

    The pipeline can run stages independently (sequential) or
    jointly optimize across both stages (end-to-end).

    Args:
        documents: Document chunks for context selection.
        doc_edges: Semantic links between documents.
        reasoning_nodes: Reasoning steps / candidate answers.
        reasoning_edges: Logical connections between reasoning steps.
        context_budget: Token budget for context.
        compute_budget: Compute budget for reasoning.
        context_relevance_weight: Weight for relevance.
        context_coverage_weight: Weight for coverage.
        reasoning_score_weight: Weight for reasoning quality.
        context_early_stop: Early stop for context stage.
        reasoning_early_stop: Early stop for reasoning stage.

    """

    def __init__(
        self,
        documents: list[DocumentChunk[NodeT]],
        doc_edges: list[tuple[NodeT, NodeT, float]] | None = None,
        reasoning_nodes: list[ReasoningNode[NodeT]] | None = None,
        reasoning_edges: list[tuple[NodeT, NodeT, float]] | None = None,
        context_budget: int = 4096,
        compute_budget: int = 1000,
        context_relevance_weight: float = 1.0,
        context_coverage_weight: float = 0.3,
        reasoning_score_weight: float = 1.0,
        context_early_stop: EarlyTerminationCondition[Any] | None = None,
        reasoning_early_stop: EarlyTerminationCondition[Any] | None = None,
    ) -> None:
        """Initialize the hybrid pipeline.

        Args:
            documents: Document chunks for context selection.
            doc_edges: Semantic links between documents.
            reasoning_nodes: Reasoning steps / candidate answers.
            reasoning_edges: Logical connections between reasoning steps.
            context_budget: Token budget for context.
            compute_budget: Compute budget for reasoning.
            context_relevance_weight: Weight for relevance.
            context_coverage_weight: Weight for coverage.
            reasoning_score_weight: Weight for reasoning quality.
            context_early_stop: Early stop for context stage.
            reasoning_early_stop: Early stop for reasoning stage.

        """
        self.documents = documents
        self.doc_edges = doc_edges or []
        self.reasoning_nodes = reasoning_nodes or []
        self.reasoning_edges = reasoning_edges or []
        self.context_budget = context_budget
        self.compute_budget = compute_budget
        self.context_relevance_weight = context_relevance_weight
        self.context_coverage_weight = context_coverage_weight
        self.reasoning_score_weight = reasoning_score_weight
        self.context_early_stop = context_early_stop
        self.reasoning_early_stop = reasoning_early_stop

    def _run_context_stage(
        self,
        context_iterations: int,
    ) -> ContextEngineeringResult:
        """Run the context selection stage.

        Args:
            context_iterations: Max iterations.

        Returns:
            ContextEngineeringResult.

        """
        problem = ContextEngineeringProblem(
            documents=self.documents,
            edges=self.doc_edges,
            max_context_tokens=self.context_budget,
            relevance_weight=self.context_relevance_weight,
            coverage_weight=self.context_coverage_weight,
        )
        solver = ContextEngineeringSolver(
            problem,
            early_stop=self.context_early_stop,
        )
        return solver.solve(max_iterations=context_iterations)

    def _run_reasoning_stage(
        self,
        reasoning_iterations: int,
        context_result: ContextEngineeringResult | None = None,
    ) -> TestTimeComputeResult:
        """Run the reasoning stage.

        Args:
            reasoning_iterations: Max iterations.
            context_result: Output from context stage (for joint optimization).

        Returns:
            TestTimeComputeResult.

        """
        if not self.reasoning_nodes:
            return TestTimeComputeResult()

        # Optionally filter reasoning nodes based on context
        nodes = self.reasoning_nodes
        if context_result is not None:
            selected_ids = {c.chunk_id for c in context_result.selected_chunks}
            # Keep nodes that connect to selected context
            nodes = [n for n in nodes if n.parent in selected_ids or n.parent is None]
            if not nodes:
                nodes = self.reasoning_nodes[:5]  # Fallback

        problem = TestTimeComputeProblem(
            nodes=nodes,
            edges=self.reasoning_edges,
            max_compute=self.compute_budget,
            score_weight=self.reasoning_score_weight,
        )
        solver = TestTimeComputeSolver(
            problem,
            early_stop=self.reasoning_early_stop,
        )
        return solver.solve(max_iterations=reasoning_iterations)

    def run(
        self,
        context_iterations: int = 50,
        reasoning_iterations: int = 50,
        observer: Any | None = None,
    ) -> HybridPipelineResult[NodeT]:
        """Run the full hybrid pipeline.

        Args:
            context_iterations: Max iterations for context stage.
            reasoning_iterations: Max iterations for reasoning stage.
            observer: Optional observer for lifecycle events.

        Returns:
            HybridPipelineResult with results from both stages.

        """
        import time

        start = time.monotonic()

        # Stage 1: Context selection
        context_result = self._run_context_stage(context_iterations)

        # Stage 2: Reasoning with selected context
        reasoning_result = self._run_reasoning_stage(
            reasoning_iterations,
            context_result,
        )

        # Compute budget metrics
        total_tokens = context_result.total_tokens
        total_compute = reasoning_result.total_cost
        overall_quality = (
            context_result.relevance_score * 0.5 + reasoning_result.total_score * 0.5
        )

        quality_per_token = overall_quality / max(total_tokens, 1)

        elapsed_ms = (time.monotonic() - start) * 1000
        quality_per_latency = overall_quality / max(elapsed_ms, 1.0)

        return HybridPipelineResult(
            context_result=context_result,
            reasoning_result=reasoning_result,
            overall_quality=overall_quality,
            total_tokens=total_tokens,
            total_compute=total_compute,
            quality_per_token=quality_per_token,
            quality_per_latency=quality_per_latency,
            elapsed_ms=elapsed_ms,
        )
