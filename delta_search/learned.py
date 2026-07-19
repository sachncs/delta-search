"""Learned search guidance using lightweight ML.

Trains a gradient boosting model on search traces to predict which
actions will lead to the best objective improvement.  Uses features
like degree, centrality, and constraint contribution.

Usage::

    from delta_search import Graph, PrizeCollectingVertexCoverProblem
    from delta_search.learned import LearnedGuidanceSolver

    graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
    problem = PrizeCollectingVertexCoverProblem(graph)
    solver = LearnedGuidanceSolver(problem)
    result = solver.solve(max_iterations=100)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from .graph import NodeT
from .problem import (
    Action,
    ActionType,
    SubgraphExtractionProblem,
    SubgraphState,
)
from .solver import EarlyTerminationCondition

if TYPE_CHECKING:
    from .problem import SolverObserver

__all__ = [
    "ActionFeatures",
    "LearnedGuidanceSolver",
]


@dataclass
class ActionFeatures:
    """Feature vector for an action.

    Attributes:
        action: The action being evaluated.
        features: Dict of feature name to value.

    """

    action: Action
    features: dict[str, float]


def _extract_features(
    problem: SubgraphExtractionProblem[NodeT],
    state: SubgraphState[NodeT],
    action: Action,
    current_objective: float,
) -> dict[str, float]:
    """Extract features for an action.

    Features:
        - degree: Degree of the target node in the input graph.
        - current_degree: Degree in the current subgraph (if applicable).
        - is_add: 1.0 if action is ADD, 0.0 if REMOVE.
        - is_node: 1.0 if node action, 0.0 if edge action.
        - num_nodes: Number of nodes in current subgraph.
        - num_edges: Number of edges in current subgraph.
        - graph_density: Edge density of the input graph.
        - objective: Current objective value.

    Args:
        problem: The problem instance.
        state: The current state.
        action: The action to evaluate.
        current_objective: Current objective value.

    Returns:
        Dict of feature name to float value.

    """
    g = problem.state_graph(state)
    input_g = problem.graph
    features: dict[str, float] = {}

    is_add = action.action_type in (ActionType.ADD_NODE, ActionType.ADD_EDGE)
    is_node = action.action_type in (ActionType.ADD_NODE, ActionType.REMOVE_NODE)

    features["is_add"] = 1.0 if is_add else 0.0
    features["is_node"] = 1.0 if is_node else 0.0
    features["num_nodes"] = float(g.num_nodes)
    features["num_edges"] = float(g.num_edges)
    features["objective"] = current_objective

    total_possible = input_g.num_nodes * (input_g.num_nodes - 1) / 2
    features["graph_density"] = (
        input_g.num_edges / total_possible if total_possible > 0 else 0.0
    )

    if is_node:
        node = action.targets[0]
        features["degree"] = (
            float(input_g.degree(node)) if input_g.has_node(node) else 0.0
        )
        features["current_degree"] = float(g.degree(node)) if g.has_node(node) else 0.0
    else:
        u, v = action.targets[0], action.targets[1]
        features["degree"] = float(input_g.degree(u) + input_g.degree(v)) / 2
        features["current_degree"] = 0.0

    return features


def _train_model(
    training_data: list[tuple[dict[str, float], float]],
    n_estimators: int = 10,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> Any:
    """Train a lightweight gradient boosting model.

    Args:
        training_data: List of (features, target) pairs.

    Returns:
        Trained model, or None if sklearn is not available.

    """
    if not training_data:
        return None

    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None

    # Convert dict features to sorted numerical lists
    features_list = [[d[0].get(k, 0.0) for k in sorted(d[0])] for d in training_data]
    targets = [d[1] for d in training_data]

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )
    model.fit(features_list, targets)
    return model


class LearnedGuidanceSolver(Generic[NodeT]):
    """Solver with learned action scoring.

    Uses a lightweight gradient boosting model to predict action quality.
    The model is trained online from search traces: each iteration's
    best action and its objective improvement become training data.

    When the model is not yet trained (or sklearn is unavailable),
    falls back to the standard greedy selection.

    Args:
        problem: A concrete SubgraphExtractionProblem instance.
        early_stop: Optional termination conditions.
        train_every: Retrain the model every N iterations.
        min_samples: Minimum training samples before using the model.
        exploration_rate: Probability of exploring a random action.
        n_estimators: Number of boosting stages for gradient boosting.
        max_depth: Maximum depth of individual regression estimators.
        learning_rate: Learning rate shrinks the contribution of each tree.

    """

    def __init__(
        self,
        problem: SubgraphExtractionProblem[NodeT],
        early_stop: EarlyTerminationCondition[NodeT] | None = None,
        train_every: int = 10,
        min_samples: int = 20,
        exploration_rate: float = 0.1,
        n_estimators: int = 10,
        max_depth: int = 3,
        learning_rate: float = 0.1,
    ) -> None:
        """Initialize the learned guidance solver.

        Args:
            problem: A concrete SubgraphExtractionProblem instance.
            early_stop: Optional termination conditions.
            train_every: Retrain the model every N iterations.
            min_samples: Minimum training samples before using the model.
            exploration_rate: Probability of exploring a random action.
            n_estimators: Number of boosting stages for gradient boosting.
            max_depth: Maximum depth of individual regression estimators.
            learning_rate: Learning rate shrinks the contribution of each tree.

        """
        self.problem = problem
        self.early_stop = early_stop or EarlyTerminationCondition()
        self.train_every = train_every
        self.min_samples = min_samples
        self.exploration_rate = exploration_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._training_data: list[tuple[dict[str, float], float]] = []
        self._model: Any = None

    def solve(
        self,
        max_iterations: int = 1000,
        observer: SolverObserver | None = None,
    ) -> dict[str, Any]:
        """Run the learned guidance solver.

        Args:
            max_iterations: Iteration cap.
            observer: Optional observer for lifecycle events.

        Returns:
            Dict with best_objective, best_state, total_iterations,
            total_evaluations, elapsed_ms, model_trained.

        """
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if observer:
            self.problem.set_observer(observer)

        state = self.problem.evaluate_initial_state(self.problem.graph)
        current_objective = self.problem.objective(state)

        best_state = state
        best_objective = current_objective
        total_evaluations = 0
        stall_count = 0
        start_time = time.monotonic()
        model_trained = False

        limit = (
            self.early_stop.max_iterations
            if self.early_stop.max_iterations is not None
            else max_iterations
        )

        converged = False
        convergence_reason = ""

        for iteration in range(limit):
            self.problem.on_iteration_start(state, iteration)

            actions = self.problem.enumerate_actions(state)
            if not actions:
                converged = True
                convergence_reason = "no actions available"
                break

            # Score actions
            if self._model is not None and random.random() >= self.exploration_rate:
                scored = self._score_actions(
                    state,
                    actions,
                    current_objective,
                )
                scored.sort(key=lambda x: x[1], reverse=True)
                best_action = scored[0][0]
                delta = self.problem.calculate_delta(state, best_action)
                best_action_obj = (
                    current_objective + delta.reward_change - delta.penalty_change
                )
                total_evaluations += 1
            else:
                # Greedy fallback or exploration
                best_action = None
                best_action_obj = float("-inf")
                for action in actions:
                    delta = self.problem.calculate_delta(state, action)
                    if not delta.feasible:
                        continue
                    obj = current_objective + delta.reward_change - delta.penalty_change
                    total_evaluations += 1
                    if obj > best_action_obj:
                        best_action_obj = obj
                        best_action = action

            if best_action is None:
                converged = True
                convergence_reason = "no feasible actions"
                break

            # Record training data
            features = _extract_features(
                self.problem,
                state,
                best_action,
                current_objective,
            )
            improvement = best_action_obj - current_objective
            self._training_data.append((features, improvement))

            # Retrain model periodically
            if (
                iteration % self.train_every == 0
                and len(self._training_data) >= self.min_samples
            ):
                self._model = _train_model(
                    self._training_data,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                )
                model_trained = self._model is not None

            state = self.problem.apply_action(state, best_action)
            current_objective = best_action_obj

            elapsed_total = (time.monotonic() - start_time) * 1000

            if current_objective > best_objective:
                best_objective = current_objective
                best_state = state
                stall_count = 0
            else:
                stall_count += 1

            self.problem.observer.on_iteration_complete(
                iteration,
                best_action,
                best_action_obj,
            )
            self.problem.on_iteration_end(state, iteration)

            # Check termination
            if (
                self.early_stop.max_time_ms is not None
                and elapsed_total >= self.early_stop.max_time_ms
            ):
                converged = True
                convergence_reason = "time limit"
                break
            if (
                self.early_stop.stall_iterations is not None
                and stall_count >= self.early_stop.stall_iterations
            ):
                converged = True
                convergence_reason = (
                    f"stalled for {self.early_stop.stall_iterations} iterations"
                )
                break
            if (
                self.early_stop.max_evaluations is not None
                and total_evaluations >= self.early_stop.max_evaluations
            ):
                converged = True
                convergence_reason = "evaluation limit"
                break
            if (
                self.early_stop.objective_target is not None
                and best_objective >= self.early_stop.objective_target
            ):
                converged = True
                convergence_reason = "objective target reached"
                break

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if observer:
            observer.on_convergence(iteration + 1, best_objective)

        return {
            "best_objective": best_objective,
            "best_state": best_state,
            "total_iterations": iteration + 1 if iteration >= 0 else 0,
            "total_evaluations": total_evaluations,
            "elapsed_ms": elapsed_ms,
            "converged": converged,
            "convergence_reason": convergence_reason,
            "model_trained": model_trained,
            "training_samples": len(self._training_data),
        }

    def _score_actions(
        self,
        state: SubgraphState[NodeT],
        actions: list[Action],
        current_objective: float,
    ) -> list[tuple[Action, float]]:
        """Score actions using the trained model.

        Args:
            state: The current state.
            actions: Candidate actions.
            current_objective: Current objective value.

        Returns:
            List of (action, predicted_score) sorted by score descending.

        """
        if self._model is None:
            return [(a, 0.0) for a in actions]

        scores: list[tuple[Action, float]] = []
        for action in actions:
            features = _extract_features(
                self.problem,
                state,
                action,
                current_objective,
            )
            feature_values = [features.get(k, 0.0) for k in sorted(features)]
            prediction = self._model.predict([feature_values])[0]
            scores.append((action, float(prediction)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
