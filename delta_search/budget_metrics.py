"""Budget-aware selection metrics for ΔSearch.

Introduces quality-per-token and quality-per-latency as evaluation
objectives, matching actual compute constraints of deployed systems.

Usage::

    from delta_search.budget_metrics import BudgetMetric, BudgetAwareEvaluator

    evaluator = BudgetAwareEvaluator(
        metric=BudgetMetric.QUALITY_PER_TOKEN,
        quality_fn=lambda state: compute_answer_quality(state),
        cost_fn=lambda state: count_tokens(state),
    )
    score = evaluator.evaluate(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Generic

from .graph import NodeT

if TYPE_CHECKING:
    from collections.abc import Callable

    from .problem import SubgraphState

__all__ = [
    "BudgetMetric",
    "QualityScore",
    "BudgetAwareEvaluator",
    "BudgetComparisonResult",
    "BudgetComparator",
]


class BudgetMetric(Enum):
    """Types of budget-aware metrics.

    Attributes:
        QUALITY_PER_TOKEN: Answer quality divided by token count.
        QUALITY_PER_LATENCY: Answer quality divided by latency (ms).
        QUALITY_PER_COST: Answer quality divided by monetary cost.
        QUALITY_MINUS_BUDGET_PENALTY: Quality minus linear budget penalty.

    """

    QUALITY_PER_TOKEN = auto()
    QUALITY_PER_LATENCY = auto()
    QUALITY_PER_COST = auto()
    QUALITY_MINUS_BUDGET_PENALTY = auto()


@dataclass
class QualityScore(Generic[NodeT]):
    """A quality score with budget metadata.

    Attributes:
        state: The solution state scored.
        quality: Raw quality value.
        cost: Budget cost (tokens, ms, $).
        metric_value: Budget-aware metric value.
        metric_type: Which metric was used.

    """

    state: SubgraphState[NodeT]
    quality: float = 0.0
    cost: float = 0.0
    metric_value: float = 0.0
    metric_type: BudgetMetric = BudgetMetric.QUALITY_PER_TOKEN


class BudgetAwareEvaluator(Generic[NodeT]):
    """Evaluates solutions using budget-aware metrics.

    Instead of raw accuracy, measures quality per unit of compute.
    This matches real deployed systems where every token/second/dollar
    has a cost.

    Args:
        metric: Which budget metric to use.
        quality_fn: Function that maps state to quality [0, 1].
        cost_fn: Function that maps state to cost (tokens, ms, etc).
        budget_target: Target budget (for penalty-based metrics).
        penalty_weight: Weight for budget penalty (QUALITY_MINUS_BUDGET_PENALTY).

    """

    def __init__(
        self,
        metric: BudgetMetric = BudgetMetric.QUALITY_PER_TOKEN,
        quality_fn: Callable[[SubgraphState[NodeT]], float] | None = None,
        cost_fn: Callable[[SubgraphState[NodeT]], float] | None = None,
        budget_target: float = 1000.0,
        penalty_weight: float = 0.001,
    ) -> None:
        """Initialize the budget-aware evaluator.

        Args:
            metric: Which budget metric to use.
            quality_fn: Function that maps state to quality [0, 1].
            cost_fn: Function that maps state to cost (tokens, ms, etc).
            budget_target: Target budget (for penalty-based metrics).
            penalty_weight: Weight for budget penalty (QUALITY_MINUS_BUDGET_PENALTY).

        """
        self.metric = metric
        self.quality_fn = quality_fn or (lambda s: 0.0)
        self.cost_fn = cost_fn or (lambda s: 1.0)
        self.budget_target = budget_target
        self.penalty_weight = penalty_weight

    def evaluate(self, state: SubgraphState[NodeT]) -> QualityScore[NodeT]:
        """Evaluate a state with the configured budget metric.

        Args:
            state: The solution state to evaluate.

        Returns:
            QualityScore with the computed metric value.

        """
        quality = self.quality_fn(state)
        cost = self.cost_fn(state)

        if (
            self.metric == BudgetMetric.QUALITY_PER_TOKEN
            or self.metric == BudgetMetric.QUALITY_PER_LATENCY
        ):
            metric_value = quality / max(cost, 1.0)

        elif self.metric == BudgetMetric.QUALITY_PER_COST:
            metric_value = quality / max(cost, 0.0001)

        elif self.metric == BudgetMetric.QUALITY_MINUS_BUDGET_PENALTY:
            overrun = max(0.0, cost - self.budget_target)
            penalty = overrun * self.penalty_weight
            metric_value = quality - penalty

        else:
            metric_value = quality

        return QualityScore(
            state=state,
            quality=quality,
            cost=cost,
            metric_value=metric_value,
            metric_type=self.metric,
        )


@dataclass
class BudgetComparisonResult(Generic[NodeT]):
    """Result from comparing two solutions on budget metrics.

    Attributes:
        winner: Which solution won ('a' or 'b' or 'tie').
        score_a: Score for solution A.
        score_b: Score for solution B.
        quality_diff: Quality difference (A - B).
        cost_diff: Cost difference (A - B).
        efficiency_diff: Efficiency difference (A - B).

    """

    winner: str = "tie"
    score_a: QualityScore[NodeT] | None = None
    score_b: QualityScore[NodeT] | None = None
    quality_diff: float = 0.0
    cost_diff: float = 0.0
    efficiency_diff: float = 0.0


class BudgetComparator(Generic[NodeT]):
    """Compares solutions using budget-aware metrics.

    Useful for ablation studies and comparing ΔSearch against baselines
    on the metric that matters for deployment.

    Args:
        evaluator: The BudgetAwareEvaluator to use.

    """

    def __init__(self, evaluator: BudgetAwareEvaluator[NodeT]) -> None:
        """Initialize the comparator.

        Args:
            evaluator: The BudgetAwareEvaluator to use for scoring.

        """
        self.evaluator = evaluator

    def compare(
        self,
        state_a: SubgraphState[NodeT],
        state_b: SubgraphState[NodeT],
    ) -> BudgetComparisonResult[NodeT]:
        """Compare two solutions.

        Args:
            state_a: First solution.
            state_b: Second solution.

        Returns:
            BudgetComparisonResult with winner and details.

        """
        score_a = self.evaluator.evaluate(state_a)
        score_b = self.evaluator.evaluate(state_b)

        quality_diff = score_a.quality - score_b.quality
        cost_diff = score_a.cost - score_b.cost
        efficiency_diff = score_a.metric_value - score_b.metric_value

        if efficiency_diff > 1e-6:
            winner = "a"
        elif efficiency_diff < -1e-6:
            winner = "b"
        else:
            winner = "tie"

        return BudgetComparisonResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            quality_diff=quality_diff,
            cost_diff=cost_diff,
            efficiency_diff=efficiency_diff,
        )

    def rank(
        self,
        states: list[SubgraphState[NodeT]],
    ) -> list[tuple[int, QualityScore[NodeT]]]:
        """Rank multiple solutions by budget metric.

        Args:
            states: List of solution states.

        Returns:
            List of (index, score) sorted by metric value descending.

        """
        scored = [(i, self.evaluator.evaluate(s)) for i, s in enumerate(states)]
        scored.sort(key=lambda x: x[1].metric_value, reverse=True)
        return scored


@dataclass
class ParetoBudgetFront:
    """Pareto front of quality vs cost tradeoffs.

    Attributes:
        points: List of (cost, quality) points on the front.
        best_efficiency: Point with highest quality/cost ratio.

    """

    points: list[tuple[float, float]] = field(default_factory=list)
    best_efficiency: tuple[float, float] | None = None


def compute_pareto_front(
    costs: list[float],
    qualities: list[float],
) -> ParetoBudgetFront:
    """Compute the Pareto front of cost vs quality tradeoffs.

    A point is on the Pareto front if no other point has both
    lower cost and higher quality.

    Args:
        costs: List of costs.
        qualities: List of quality values.

    Returns:
        ParetoBudgetFront with the Pareto-optimal points.

    """
    if len(costs) != len(qualities):
        raise ValueError("costs and qualities must have same length")

    points = list(zip(costs, qualities, strict=False))

    # Find Pareto front: no other point has lower cost AND higher quality
    pareto: list[tuple[float, float]] = []
    for i, (c_i, q_i) in enumerate(points):
        dominated = False
        for j, (c_j, q_j) in enumerate(points):
            if i != j and c_j <= c_i and q_j >= q_i and (c_j < c_i or q_j > q_i):
                dominated = True
                break
        if not dominated:
            pareto.append((c_i, q_i))

    pareto.sort(key=lambda x: x[0])

    # Find best efficiency (highest quality/cost)
    best_eff = None
    best_ratio = -1.0
    for c, q in pareto:
        if c > 0:
            ratio = q / c
            if ratio > best_ratio:
                best_ratio = ratio
                best_eff = (c, q)

    return ParetoBudgetFront(points=pareto, best_efficiency=best_eff)
