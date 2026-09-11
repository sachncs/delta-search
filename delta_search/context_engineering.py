"""Context engineering using ΔSearch — RAG context selection.

Models documents/chunks as nodes and semantic links as edges, then uses
ΔSearch to maximize relevance, coverage, and novelty under a token budget.
Each document is a node, each semantic similarity above threshold is an edge.

Usage::

    from delta_search.context_engineering import (
        ContextEngineeringProblem,
        ContextEngineeringSolver,
    )

    docs = [
        {"id": "d1", "text": "Python is a programming language", "tokens": 8},
        {"id": "d2", "text": "Python is used for machine learning", "tokens": 7},
        {"id": "d3", "text": "JavaScript runs in browsers", "tokens": 6},
    ]
    edges = [("d1", "d2", 0.8), ("d1", "d3", 0.1), ("d2", "d3", 0.05)]
    problem = ContextEngineeringProblem(
        documents=docs, edges=edges, query_tokens=10, max_context_tokens=50,
    )
    solver = ContextEngineeringSolver(problem)
    result = solver.solve(max_iterations=20)
    logging.info(f"Selected {result.num_chunks} chunks, {result.total_tokens} tokens")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from .graph import Graph, NodeT
from .problem import (
    Action,
    ActionType,
    DefaultState,
    DeltaResult,
    SubgraphExtractionProblem,
    SubgraphState,
)
from .solver import EarlyTerminationCondition, GreedySolver

__all__ = [
    "DocumentChunk",
    "ContextEngineeringProblem",
    "ContextEngineeringResult",
    "ContextEngineeringSolver",
]


@dataclass
class DocumentChunk(Generic[NodeT]):
    """A document chunk with metadata.

    Attributes:
        chunk_id: Unique identifier.
        text: The chunk text.
        tokens: Token count.
        relevance: Query relevance score [0, 1].
        embedding: Optional embedding vector for semantic linking.
        metadata: Additional metadata.

    """

    chunk_id: NodeT
    text: str
    tokens: int
    relevance: float = 0.0
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEngineeringProblem(SubgraphExtractionProblem[NodeT]):
    """ΔSearch problem for context selection in RAG.

    Maximize relevance, coverage, and novelty under a token budget.

    Reward: Σ(relevance) + α * coverage + β * novelty
    Penalty: Σ(tokens) when exceeding budget, or as cost term

    Coverage measures how many topics/aspects are covered.
    Novelty penalizes redundancy (semantic similarity within selected set).

    Args:
        documents: List of document chunks.
        edges: List of (id1, id2, similarity) triples for semantic links.
        max_context_tokens: Maximum token budget for context.
        relevance_weight: Weight for relevance in objective.
        coverage_weight: Weight for coverage bonus.
        novelty_weight: Weight for novelty (penalizes redundancy).
        cost_per_token: Cost per token in the penalty.
        similarity_threshold: Minimum similarity for semantic edges.
        query_tokens: Tokens consumed by the query itself.

    """

    def __init__(
        self,
        documents: list[DocumentChunk[NodeT]],
        edges: list[tuple[NodeT, NodeT, float]] | None = None,
        max_context_tokens: int = 4096,
        relevance_weight: float = 1.0,
        coverage_weight: float = 0.3,
        novelty_weight: float = 0.2,
        cost_per_token: float = 0.001,
        similarity_threshold: float = 0.3,
        query_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the context engineering problem.

        Args:
            documents: List of document chunks.
            edges: List of (id1, id2, similarity) triples for semantic links.
            max_context_tokens: Maximum token budget for context.
            relevance_weight: Weight for relevance in objective.
            coverage_weight: Weight for coverage bonus.
            novelty_weight: Weight for novelty (penalizes redundancy).
            cost_per_token: Cost per token in the penalty.
            similarity_threshold: Minimum similarity for semantic edges.
            query_tokens: Tokens consumed by the query itself.
            **kwargs: Additional keyword arguments passed to the parent class.

        """
        graph: Graph[NodeT] = Graph()
        self._documents: dict[NodeT, DocumentChunk[NodeT]] = {}
        self._max_tokens = max_context_tokens
        self._relevance_weight = relevance_weight
        self._coverage_weight = coverage_weight
        self._novelty_weight = novelty_weight
        self._cost_per_token = cost_per_token
        self._similarity_threshold = similarity_threshold
        self._query_tokens = query_tokens

        for doc in documents:
            graph.add_node(doc.chunk_id)
            self._documents[doc.chunk_id] = doc

        if edges:
            for u, v, sim in edges:
                if (
                    sim >= similarity_threshold
                    and graph.has_node(u)
                    and graph.has_node(v)
                ):
                    graph.add_edge(u, v)

        super().__init__(graph, **kwargs)

    @property
    def documents(self) -> dict[NodeT, DocumentChunk[NodeT]]:
        """Map of chunk_id to DocumentChunk."""
        return self._documents

    @property
    def max_tokens(self) -> int:
        """Token budget for context."""
        return self._max_tokens

    def evaluate_initial_state(
        self,
        graph: Graph[NodeT],
    ) -> DefaultState[NodeT]:
        """Start with an empty context."""
        state = DefaultState[NodeT](graph=Graph())
        state.metrics["selected"] = set()
        state.metrics["total_tokens"] = self._query_tokens
        state.metrics["relevance_sum"] = 0.0
        state.metrics["coverage_set"] = set()
        return state

    def enumerate_actions(
        self,
        state: SubgraphState[NodeT],
    ) -> list[Action]:
        """Enumerate add/remove for each document not yet selected."""
        selected: set[NodeT] = state.metrics.get("selected", set())
        actions: list[Action] = []

        for node in self.graph.nodes:
            if node not in selected:
                actions.append(
                    Action(
                        action_type=ActionType.ADD_NODE,
                        targets=(node,),
                    )
                )
            else:
                actions.append(
                    Action(
                        action_type=ActionType.REMOVE_NODE,
                        targets=(node,),
                    )
                )

        return actions

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if state is feasible (within token budget or can be corrected)."""
        return True

    def _compute_novelty(
        self,
        state: SubgraphState[NodeT],
        candidate: NodeT,
    ) -> float:
        """Compute novelty bonus for adding a candidate.

        Novelty is 1 minus the average similarity to already-selected docs.
        High novelty = low redundancy.
        """
        selected: set[NodeT] = state.metrics.get("selected", set())
        if not selected:
            return 1.0

        total_sim = 0.0
        count = 0
        for s in selected:
            if self.graph.has_edge(candidate, s):
                sim = self.graph.edge_data(candidate, s).get("weight", 0.0)
                total_sim += sim
                count += 1

        if count == 0:
            return 1.0
        return 1.0 - (total_sim / count)

    def compute_coverage(self, state: SubgraphState[NodeT]) -> float:
        """Compute coverage score.

        Coverage is the fraction of graph components represented.
        """
        selected: set[NodeT] = state.metrics.get("selected", set())
        if not self.graph.nodes:
            return 0.0

        return len(selected) / len(self.graph.nodes)

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute incremental change for adding/removing a document.

        Returns DeltaResult with reward_change and penalty_change.
        """
        selected: set[NodeT] = set(current_state.metrics.get("selected", set()))
        total_tokens: int = current_state.metrics.get(
            "total_tokens", self._query_tokens
        )

        node = candidate_action.targets[0]
        doc = self._documents[node]

        if candidate_action.action_type == ActionType.ADD_NODE:
            if node in selected:
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=0.0,
                    feasible=False,
                )

            new_tokens = total_tokens + doc.tokens
            if new_tokens > self._max_tokens:
                penalty = (new_tokens - self._max_tokens) * self._cost_per_token
            else:
                penalty = 0.0

            relevance_gain = doc.relevance * self._relevance_weight
            novelty = self._compute_novelty(current_state, node) * self._novelty_weight
            old_coverage = self.compute_coverage(current_state)
            new_coverage = (
                len(selected | {node}) / len(self.graph.nodes)
                if self.graph.nodes
                else 0.0
            )
            coverage_gain = (new_coverage - old_coverage) * self._coverage_weight

            reward = relevance_gain + novelty + coverage_gain

            return DeltaResult(
                reward_change=reward,
                penalty_change=penalty,
                feasible=True,
            )

        else:  # REMOVE_NODE
            if node not in selected:
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=0.0,
                    feasible=False,
                )

            relevance_loss = doc.relevance * self._relevance_weight

            penalty_reduction = 0.0
            if total_tokens > self._max_tokens:
                freed = min(doc.tokens, total_tokens - self._max_tokens)
                penalty_reduction = freed * self._cost_per_token

            return DeltaResult(
                reward_change=-relevance_loss,
                penalty_change=-penalty_reduction,
                feasible=True,
            )

    def apply_action(
        self,
        state: SubgraphState[NodeT],
        action: Action,
        *,
        incremental: bool = False,
    ) -> SubgraphState[NodeT]:
        """Apply add/remove to state."""
        import copy

        new_state = copy.deepcopy(state)
        selected: set[NodeT] = new_state.metrics["selected"]
        node = action.targets[0]
        doc = self._documents[node]

        if action.action_type == ActionType.ADD_NODE:
            selected.add(node)
            new_state.metrics["total_tokens"] += doc.tokens
            new_state.metrics["relevance_sum"] += doc.relevance
            new_state.graph.add_node(node)
        else:
            selected.discard(node)
            new_state.metrics["total_tokens"] -= doc.tokens
            new_state.metrics["relevance_sum"] -= doc.relevance
            if new_state.graph.has_node(node):
                new_state.graph.remove_node(node)

        new_state.undo = None
        return new_state

    def objective(self, state: SubgraphState[NodeT]) -> float:
        """Combined objective: relevance + coverage + novelty - cost."""
        relevance = state.metrics.get("relevance_sum", 0.0) * self._relevance_weight
        coverage = self.compute_coverage(state) * self._coverage_weight
        total_tokens = state.metrics.get("total_tokens", 0)

        penalty = 0.0
        if total_tokens > self._max_tokens:
            penalty = (total_tokens - self._max_tokens) * self._cost_per_token

        return float(relevance + coverage - penalty)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Reward: relevance + coverage."""
        relevance = state.metrics.get("relevance_sum", 0.0) * self._relevance_weight
        coverage = self.compute_coverage(state) * self._coverage_weight
        return float(relevance + coverage)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Penalty: token budget overrun."""
        total_tokens = state.metrics.get("total_tokens", 0)
        if total_tokens > self._max_tokens:
            return float((total_tokens - self._max_tokens) * self._cost_per_token)
        return 0.0


@dataclass
class ContextEngineeringResult:
    """Result from context engineering.

    Attributes:
        selected_chunks: List of selected document chunks.
        num_chunks: Number of chunks selected.
        total_tokens: Total tokens in selected context.
        relevance_score: Sum of relevance scores.
        coverage_score: Coverage fraction.
        objective: Final objective value.
        iterations: Iterations used.
        evaluations: Action evaluations.
        elapsed_ms: Wall-clock time.

    """

    selected_chunks: list[DocumentChunk[Any]] = field(default_factory=list)
    num_chunks: int = 0
    total_tokens: int = 0
    relevance_score: float = 0.0
    coverage_score: float = 0.0
    objective: float = 0.0
    iterations: int = 0
    evaluations: int = 0
    elapsed_ms: float = 0.0


class ContextEngineeringSolver(Generic[NodeT]):
    """Solver wrapper for context engineering.

    Provides a convenient interface that returns ContextEngineeringResult.

    Args:
        problem: A ContextEngineeringProblem instance.
        early_stop: Optional termination conditions.

    """

    def __init__(
        self,
        problem: ContextEngineeringProblem[NodeT],
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the context engineering solver.

        Args:
            problem: A ContextEngineeringProblem instance.
            early_stop: Optional termination conditions.

        """
        self.problem = problem
        self._greedy = GreedySolver(problem, early_stop=early_stop)

    def solve(
        self,
        max_iterations: int = 100,
        observer: Any | None = None,
    ) -> ContextEngineeringResult:
        """Run context engineering.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer.

        Returns:
            ContextEngineeringResult with selected chunks and metrics.

        """
        state = self._greedy.solve(
            max_iterations=max_iterations,
            observer=observer,
        )

        best = state.best_state
        if best is None:
            return ContextEngineeringResult(
                objective=state.best_objective,
                iterations=state.iteration + 1,
                evaluations=state.total_evaluations,
                elapsed_ms=state.elapsed_ms,
            )

        selected: set[NodeT] = best.metrics.get("selected", set())
        chunks = [
            self.problem.documents[n] for n in selected if n in self.problem.documents
        ]

        return ContextEngineeringResult(
            selected_chunks=chunks,
            num_chunks=len(chunks),
            total_tokens=best.metrics.get("total_tokens", 0),
            relevance_score=best.metrics.get("relevance_sum", 0.0),
            coverage_score=self.problem.compute_coverage(best),
            objective=state.best_objective,
            iterations=state.iteration + 1,
            evaluations=state.total_evaluations,
            elapsed_ms=state.elapsed_ms,
        )
