"""Test-time compute scaling using ΔSearch — reasoning tree expansion.

Applies ΔSearch to reasoning trees, candidate answers, or search traces
so that the model expands only the most promising branches under a
compute budget.  Nodes are reasoning steps, edges are logical derivations.

Usage::

    from delta_search.test_time_compute import (
        ReasoningNode,
        TestTimeComputeProblem,
        TestTimeComputeSolver,
    )

    nodes = [
        ReasoningNode("step1", "Premise A", score=0.8, cost=10),
        ReasoningNode("step2", "A -> B", score=0.6, cost=8),
        ReasoningNode("step3", "B -> C", score=0.9, cost=12),
    ]
    edges = [("step1", "step2", 0.7), ("step2", "step3", 0.9)]
    problem = TestTimeComputeProblem(
        nodes=nodes, edges=edges, max_compute=100,
    )
    solver = TestTimeComputeSolver(problem)
    result = solver.solve(max_iterations=20)
    logging.info(f"Expanded {result.num_steps} steps, {result.total_cost} compute")
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
    "ReasoningNode",
    "TestTimeComputeProblem",
    "TestTimeComputeResult",
    "TestTimeComputeSolver",
]


@dataclass
class ReasoningNode(Generic[NodeT]):
    """A reasoning step in the tree.

    Attributes:
        node_id: Unique identifier.
        content: The reasoning content / text.
        score: Confidence or quality score [0, 1].
        cost: Compute cost (tokens, FLOPs, latency).
        depth: Depth in the reasoning tree.
        parent: Parent node ID (None for root).
        metadata: Additional metadata.

    """

    node_id: NodeT
    content: str
    score: float = 0.0
    cost: int = 1
    depth: int = 0
    parent: NodeT | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestTimeComputeProblem(SubgraphExtractionProblem[NodeT]):
    __test__ = False
    """ΔSearch problem for test-time compute allocation.

    Maximize reasoning quality under a compute budget.

    Reward: Σ(score) + depth_bonus
    Penalty: Σ(cost) when exceeding budget, or as cost term

    Args:
        nodes: List of reasoning nodes.
        edges: List of (id1, id2, strength) triples.
        max_compute: Maximum compute budget.
        score_weight: Weight for node scores.
        depth_bonus_weight: Bonus for deeper reasoning chains.
        cost_weight: Cost per compute unit.
        coherence_weight: Weight for edge coherence (logical consistency).

    """

    def __init__(
        self,
        nodes: list[ReasoningNode[NodeT]],
        edges: list[tuple[NodeT, NodeT, float]] | None = None,
        max_compute: int = 1000,
        score_weight: float = 1.0,
        depth_bonus_weight: float = 0.1,
        cost_weight: float = 0.01,
        coherence_weight: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize the test-time compute problem.

        Args:
            nodes: List of reasoning nodes.
            edges: List of (id1, id2, strength) triples.
            max_compute: Maximum compute budget.
            score_weight: Weight for node scores.
            depth_bonus_weight: Bonus for deeper reasoning chains.
            cost_weight: Cost per compute unit.
            coherence_weight: Weight for edge coherence (logical consistency).
            **kwargs: Additional keyword arguments passed to the parent class.

        """
        graph: Graph[NodeT] = Graph()
        self._nodes: dict[NodeT, ReasoningNode[NodeT]] = {}
        self._max_compute = max_compute
        self._score_weight = score_weight
        self._depth_bonus_weight = depth_bonus_weight
        self._cost_weight = cost_weight
        self._coherence_weight = coherence_weight

        for node in nodes:
            graph.add_node(node.node_id)
            self._nodes[node.node_id] = node

        if edges:
            for u, v, _strength in edges:
                if graph.has_node(u) and graph.has_node(v):
                    graph.add_edge(u, v)

        super().__init__(graph, **kwargs)

    @property
    def nodes_map(self) -> dict[NodeT, ReasoningNode[NodeT]]:
        """Map of node_id to ReasoningNode."""
        return self._nodes

    @property
    def max_compute(self) -> int:
        """Maximum compute budget."""
        return self._max_compute

    def evaluate_initial_state(
        self,
        graph: Graph[NodeT],
    ) -> DefaultState[NodeT]:
        """Start with no reasoning steps expanded."""
        state = DefaultState[NodeT](graph=Graph())
        state.metrics["expanded"] = set()
        state.metrics["total_cost"] = 0
        state.metrics["total_score"] = 0.0
        state.metrics["max_depth"] = 0
        return state

    def enumerate_actions(
        self,
        state: SubgraphState[NodeT],
    ) -> list[Action]:
        """Enumerate expand/prune for each reasoning node."""
        expanded: set[NodeT] = state.metrics.get("expanded", set())
        actions: list[Action] = []

        for node_id in self.graph.nodes:
            if node_id not in expanded:
                actions.append(
                    Action(
                        action_type=ActionType.ADD_NODE,
                        targets=(node_id,),
                    )
                )
            else:
                actions.append(
                    Action(
                        action_type=ActionType.REMOVE_NODE,
                        targets=(node_id,),
                    )
                )

        return actions

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if state is feasible (within compute budget or can be corrected)."""
        return True

    def _compute_coherence(
        self,
        state: SubgraphState[NodeT],
        candidate: NodeT,
    ) -> float:
        """Compute coherence bonus for expanding a node.

        Coherence is the average edge strength to already-expanded nodes.
        """
        expanded: set[NodeT] = state.metrics.get("expanded", set())
        if not expanded:
            return 1.0

        total_strength = 0.0
        count = 0
        for e in expanded:
            if self.graph.has_edge(candidate, e):
                strength = self.graph.edge_data(candidate, e).get("weight", 0.0)
                total_strength += strength
                count += 1

        if count == 0:
            return 0.5  # Neutral if no connections
        return total_strength / count

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute incremental change for expanding/pruning a reasoning step."""
        expanded: set[NodeT] = set(current_state.metrics.get("expanded", set()))
        total_cost: int = current_state.metrics.get("total_cost", 0)
        max_depth: int = current_state.metrics.get("max_depth", 0)

        node_id = candidate_action.targets[0]
        node = self._nodes[node_id]

        if candidate_action.action_type == ActionType.ADD_NODE:
            if node_id in expanded:
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=0.0,
                    feasible=False,
                )

            new_cost = total_cost + node.cost
            if new_cost > self._max_compute:
                penalty = (new_cost - self._max_compute) * self._cost_weight
            else:
                penalty = 0.0

            score_gain = node.score * self._score_weight
            new_depth = max_depth + 1
            depth_gain = (new_depth - max_depth) * self._depth_bonus_weight
            coherence = (
                self._compute_coherence(current_state, node_id) * self._coherence_weight
            )

            reward = score_gain + depth_gain + coherence

            return DeltaResult(
                reward_change=reward,
                penalty_change=penalty,
                feasible=True,
            )

        else:  # REMOVE_NODE
            if node_id not in expanded:
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=0.0,
                    feasible=False,
                )

            score_loss = node.score * self._score_weight

            cost_reduction = 0.0
            new_cost = total_cost - node.cost
            if total_cost > self._max_compute:
                freed = min(node.cost, total_cost - self._max_compute)
                cost_reduction = freed * self._cost_weight

            return DeltaResult(
                reward_change=-score_loss,
                penalty_change=-cost_reduction,
                feasible=True,
            )

    def apply_action(
        self,
        state: SubgraphState[NodeT],
        action: Action,
    ) -> SubgraphState[NodeT]:
        """Apply expand/prune to state."""
        import copy

        new_state = copy.deepcopy(state)
        expanded: set[NodeT] = new_state.metrics["expanded"]
        node_id = action.targets[0]
        node = self._nodes[node_id]

        if action.action_type == ActionType.ADD_NODE:
            expanded.add(node_id)
            new_state.metrics["total_cost"] += node.cost
            new_state.metrics["total_score"] += node.score
            new_state.metrics["max_depth"] = max(
                new_state.metrics["max_depth"],
                node.depth,
            )
            new_state.graph.add_node(node_id)
        else:
            expanded.discard(node_id)
            new_state.metrics["total_cost"] -= node.cost
            new_state.metrics["total_score"] -= node.score
            if new_state.graph.has_node(node_id):
                new_state.graph.remove_node(node_id)

        new_state.undo = None
        return new_state

    def objective(self, state: SubgraphState[NodeT]) -> float:
        """Combined objective: score + depth + coherence - cost."""
        score = state.metrics.get("total_score", 0.0) * self._score_weight
        depth = state.metrics.get("max_depth", 0) * self._depth_bonus_weight
        total_cost = state.metrics.get("total_cost", 0)

        penalty = 0.0
        if total_cost > self._max_compute:
            penalty = (total_cost - self._max_compute) * self._cost_weight

        return float(score + depth - penalty)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Reward: score + depth."""
        score = state.metrics.get("total_score", 0.0) * self._score_weight
        depth = state.metrics.get("max_depth", 0) * self._depth_bonus_weight
        return float(score + depth)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Penalty: compute budget overrun."""
        total_cost = state.metrics.get("total_cost", 0)
        if total_cost > self._max_compute:
            return float((total_cost - self._max_compute) * self._cost_weight)
        return 0.0


@dataclass
class TestTimeComputeResult:
    __test__ = False
    """Result from test-time compute optimization.

    Attributes:
        expanded_nodes: List of expanded reasoning nodes.
        num_steps: Number of steps expanded.
        total_cost: Total compute cost.
        total_score: Sum of reasoning scores.
        max_depth: Maximum reasoning depth.
        objective: Final objective value.
        iterations: Iterations used.
        evaluations: Action evaluations.
        elapsed_ms: Wall-clock time.

    """

    expanded_nodes: list[ReasoningNode[Any]] = field(default_factory=list)
    num_steps: int = 0
    total_cost: int = 0
    total_score: float = 0.0
    max_depth: int = 0
    objective: float = 0.0
    iterations: int = 0
    evaluations: int = 0
    elapsed_ms: float = 0.0


class TestTimeComputeSolver(Generic[NodeT]):
    __test__ = False
    """Solver wrapper for test-time compute allocation.

    Args:
        problem: A TestTimeComputeProblem instance.
        early_stop: Optional termination conditions.

    """

    def __init__(
        self,
        problem: TestTimeComputeProblem[NodeT],
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
    ) -> None:
        """Initialize the test-time compute solver.

        Args:
            problem: A TestTimeComputeProblem instance.
            early_stop: Optional termination conditions.

        """
        self.problem = problem
        self._greedy = GreedySolver(problem, early_stop=early_stop)

    def solve(
        self,
        max_iterations: int = 100,
        observer: Any | None = None,
    ) -> TestTimeComputeResult:
        """Run test-time compute optimization.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer.

        Returns:
            TestTimeComputeResult with expanded nodes and metrics.

        """
        state = self._greedy.solve(
            max_iterations=max_iterations,
            observer=observer,
        )

        best = state.best_state
        if best is None:
            return TestTimeComputeResult(
                objective=state.best_objective,
                iterations=state.iteration + 1,
                evaluations=state.total_evaluations,
                elapsed_ms=state.elapsed_ms,
            )

        expanded: set[NodeT] = best.metrics.get("expanded", set())
        nodes = [
            self.problem.nodes_map[n] for n in expanded if n in self.problem.nodes_map
        ]

        return TestTimeComputeResult(
            expanded_nodes=nodes,
            num_steps=len(nodes),
            total_cost=best.metrics.get("total_cost", 0),
            total_score=best.metrics.get("total_score", 0.0),
            max_depth=best.metrics.get("max_depth", 0),
            objective=state.best_objective,
            iterations=state.iteration + 1,
            evaluations=state.total_evaluations,
            elapsed_ms=state.elapsed_ms,
        )
