"""Comprehensive tests for delta_search.problem module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from delta_search.graph import Graph
from delta_search.problem import (
    Action,
    ActionType,
    DeltaResult,
    NullObserver,
    SolverObserver,
    SubgraphExtractionProblem,
    SubgraphState,
)

# ---------------------------------------------------------------------------
# Concrete test problem
# ---------------------------------------------------------------------------


@dataclass
class SimpleState:
    """Test state with a graph and a counter."""

    graph: Graph[int]
    counter: int = 0
    undo: object = None
    metrics: dict[str, Any] = field(default_factory=dict)


class SimpleProblem(SubgraphExtractionProblem[int]):
    """Toy problem: maximize edge count in subgraph."""

    def evaluate_initial_state(self, graph: Graph[int]) -> SimpleState:
        return SimpleState(graph=Graph[int]())

    def calculate_delta(
        self,
        current_state: SimpleState,
        candidate_action: Action,
    ) -> DeltaResult:
        if candidate_action.action_type is ActionType.ADD_EDGE:
            return DeltaResult(reward_change=1.0, penalty_change=0.0, feasible=True)
        elif candidate_action.action_type is ActionType.REMOVE_EDGE:
            return DeltaResult(reward_change=-1.0, penalty_change=0.0, feasible=True)
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=True)

    def compute_reward(self, state: SimpleState) -> float:
        return float(state.graph.num_edges)

    def compute_penalty(self, state: SimpleState) -> float:
        return 0.0

    def is_feasible(self, state: SimpleState) -> bool:
        return True


# ---------------------------------------------------------------------------
# Test SubgraphState protocol
# ---------------------------------------------------------------------------


class TestSubgraphState:
    def test_protocol_accepted(self) -> None:
        s = SimpleState(graph=Graph[int]())
        assert isinstance(s, SubgraphState)

    def test_plain_graph_rejected(self) -> None:
        g = Graph[int]()
        # Graph doesn't have .graph attribute as a property
        assert not isinstance(g, SubgraphState)


# ---------------------------------------------------------------------------
# Test ABC enforcement
# ---------------------------------------------------------------------------


class TestABCEnforcement:
    def test_cannot_instantiate(self) -> None:
        g = Graph[int]()
        with pytest.raises(TypeError, match="abstract method"):
            type("PartialProblem", (SubgraphExtractionProblem,), {})(g)


# ---------------------------------------------------------------------------
# Test SubgraphExtractionProblem
# ---------------------------------------------------------------------------


class TestSubgraphExtractionProblem:
    def test_init_defensive_copy(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        # Modifying original should not affect problem's graph
        g.add_edge(2, 3)
        assert problem.graph.num_edges == 1

    def test_init_no_copy(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g, defensive_copy=False)
        g.add_edge(2, 3)
        assert problem.graph.num_edges == 2

    def test_graph_property(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        assert problem.graph.num_nodes == 2
        assert problem.graph.num_edges == 1

    def test_repr(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        assert "SimpleProblem" in repr(problem)
        assert "nodes=2" in repr(problem)
        assert "edges=1" in repr(problem)


# ---------------------------------------------------------------------------
# Test objective
# ---------------------------------------------------------------------------


class TestObjective:
    def test_objective(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        assert problem.objective(state) == 0.0


# ---------------------------------------------------------------------------
# Test enumerate_actions
# ---------------------------------------------------------------------------


class TestEnumerateActions:
    def test_empty_graph(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        # No actions possible on empty input graph
        actions = problem.enumerate_actions(state)
        assert actions == []

    def test_node_additions(self) -> None:
        g = Graph[int](nodes=[1, 2, 3])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        add_actions = [a for a in actions if a.action_type is ActionType.ADD_NODE]
        assert len(add_actions) == 3

    def test_edge_additions_no_duplicates(self) -> None:
        """Edge {1,2} should produce exactly one ADD_EDGE action, not two."""
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1, 2, 3]))
        actions = problem.enumerate_actions(state)
        edge_adds = [a for a in actions if a.action_type is ActionType.ADD_EDGE]
        # Should have exactly 2 unique edges: {1,2} and {2,3}
        assert len(edge_adds) == 2
        # Verify canonical ordering
        for a in edge_adds:
            u, v = a.targets
            assert u < v

    def test_edge_removals(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2)]))
        actions = problem.enumerate_actions(state)
        edge_rems = [a for a in actions if a.action_type is ActionType.REMOVE_EDGE]
        assert len(edge_rems) == 1

    def test_node_removals(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2), (2, 3)]))
        actions = problem.enumerate_actions(state)
        node_rems = [a for a in actions if a.action_type is ActionType.REMOVE_NODE]
        assert len(node_rems) == 3

    def test_composite_actions_hook(self) -> None:
        """generate_composite_actions should be called."""

        class ProblemWithComposite(SimpleProblem):
            def generate_composite_actions(self, state: Any) -> list[Action]:
                return [Action(ActionType.ADD_EDGE, (99, 100))]

        g = Graph[int](nodes=[99, 100])
        problem = ProblemWithComposite(g)
        state = SimpleState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        composites = [a for a in actions if a.targets == (99, 100)]
        assert len(composites) == 1


# ---------------------------------------------------------------------------
# Test apply_action / undo_action
# ---------------------------------------------------------------------------


class TestApplyAction:
    def test_add_node(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        action = Action(ActionType.ADD_NODE, (1,))
        new_state = problem.apply_action(state, action)
        assert new_state.graph.has_node(1)
        assert not state.graph.has_node(1)  # original unchanged

    def test_remove_node(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2), (2, 3)]))
        action = Action(ActionType.REMOVE_NODE, (2,))
        new_state = problem.apply_action(state, action)
        assert not new_state.graph.has_node(2)
        assert new_state.graph.num_edges == 0

    def test_add_edge(self) -> None:
        g = Graph[int](nodes=[1, 2])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1, 2]))
        action = Action(ActionType.ADD_EDGE, (1, 2))
        new_state = problem.apply_action(state, action)
        assert new_state.graph.has_edge(1, 2)
        assert new_state.graph.num_edges == 1

    def test_remove_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2)]))
        action = Action(ActionType.REMOVE_EDGE, (1, 2))
        new_state = problem.apply_action(state, action)
        assert not new_state.graph.has_edge(1, 2)
        assert new_state.graph.num_edges == 0

    def test_undo_add_node(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        action = Action(ActionType.ADD_NODE, (1,))
        new_state = problem.apply_action(state, action)
        restored = problem.undo_action(new_state)
        assert not restored.graph.has_node(1)

    def test_undo_remove_node(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2)]))
        action = Action(ActionType.REMOVE_NODE, (1,))
        new_state = problem.apply_action(state, action)
        restored = problem.undo_action(new_state)
        assert restored.graph.has_node(1)
        assert restored.graph.has_edge(1, 2)

    def test_undo_add_edge(self) -> None:
        g = Graph[int](nodes=[1, 2])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1, 2]))
        action = Action(ActionType.ADD_EDGE, (1, 2))
        new_state = problem.apply_action(state, action)
        restored = problem.undo_action(new_state)
        assert not restored.graph.has_edge(1, 2)

    def test_undo_remove_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2)]))
        action = Action(ActionType.REMOVE_EDGE, (1, 2))
        new_state = problem.apply_action(state, action)
        restored = problem.undo_action(new_state)
        assert restored.graph.has_edge(1, 2)

    def test_undo_no_info_raises(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        with pytest.raises(RuntimeError, match="No undo information"):
            problem.undo_action(state)


# ---------------------------------------------------------------------------
# Test _state_graph error handling
# ---------------------------------------------------------------------------


class TestStateGraph:
    def test_bad_state_type(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        bad_state = "not a state"
        with pytest.raises(TypeError, match="has no 'graph' attribute"):
            problem.state_graph(bad_state)


# ---------------------------------------------------------------------------
# Test SolverObserver
# ---------------------------------------------------------------------------


class TestSolverObserver:
    def test_null_observer(self) -> None:
        obs = NullObserver()
        obs.on_action_evaluated(
            Action(ActionType.ADD_EDGE, (1, 2)), DeltaResult(1.0, 0.0, True), 1.0
        )
        obs.on_iteration_complete(0, None, 0.0)
        obs.on_convergence(0, 0.0)

    def test_set_observer(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        obs = MagicMock(spec=SolverObserver)
        problem.set_observer(obs)
        assert problem.observer is obs

    def test_protocol_check(self) -> None:
        obs = NullObserver()
        assert isinstance(obs, SolverObserver)


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_add_edge_self_loop_via_action(self) -> None:
        """apply_action should handle self-loop rejection gracefully."""
        g = Graph[int](nodes=[1])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1]))
        action = Action(ActionType.ADD_EDGE, (1, 1))
        with pytest.raises(ValueError, match="Self-loops"):
            problem.apply_action(state, action)

    def test_remove_nonexistent_node_via_action(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        action = Action(ActionType.REMOVE_NODE, (99,))
        with pytest.raises(KeyError):
            problem.apply_action(state, action)

    def test_large_graph_action_count(self) -> None:
        """Ensure action count is O(V + E) not O(V^2)."""
        g = Graph[int](nodes=list(range(100)))
        for i in range(99):
            g.add_edge(i, i + 1)
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=list(range(100))))
        actions = problem.enumerate_actions(state)
        # 100 node removals + 99 edge removals + 0 additions = 199
        assert len(actions) == 199


# ---------------------------------------------------------------------------
# Undo round-trips
# ---------------------------------------------------------------------------


class TestUndoRoundTrip:
    def test_apply_undo_reapply(self) -> None:
        g = Graph[int](nodes=[1, 2])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1, 2]))
        action = Action(ActionType.ADD_EDGE, (1, 2))

        state2 = problem.apply_action(state, action)
        assert state2.graph.has_edge(1, 2)

        state3 = problem.undo_action(state2)
        assert not state3.graph.has_edge(1, 2)

        state4 = problem.apply_action(state3, action)
        assert state4.graph.has_edge(1, 2)

    def test_undo_remove_node_restores_edges(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2), (2, 3)]))
        action = Action(ActionType.REMOVE_NODE, (2,))

        new_state = problem.apply_action(state, action)
        assert not new_state.graph.has_node(2)
        assert new_state.graph.num_edges == 0

        restored = problem.undo_action(new_state)
        assert restored.graph.has_node(2)
        assert restored.graph.has_edge(1, 2)
        assert restored.graph.has_edge(2, 3)

    def test_undo_with_no_info_raises(self) -> None:
        g = Graph[int](nodes=[1, 2])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int](nodes=[1, 2]))
        action = Action(ActionType.ADD_EDGE, (1, 2))

        new_state = problem.apply_action(state, action)
        new_state.undo = None

        with pytest.raises(RuntimeError, match="No undo information"):
            problem.undo_action(new_state)


# ---------------------------------------------------------------------------
# Observer tests
# ---------------------------------------------------------------------------


class TestObserverManagement:
    def test_add_observer(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        obs = MagicMock(spec=SolverObserver)
        problem.add_observer(obs)
        assert len(problem.observers) == 2
        assert obs in problem.observers

    def test_remove_observer(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        obs = MagicMock(spec=SolverObserver)
        problem.add_observer(obs)
        problem.remove_observer(obs)
        assert obs not in problem.observers

    def test_multiple_observers(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        obs1 = MagicMock(spec=SolverObserver)
        obs2 = MagicMock(spec=SolverObserver)
        problem.add_observer(obs1)
        problem.add_observer(obs2)
        assert len(problem.observers) == 3


# ---------------------------------------------------------------------------
# DefaultState
# ---------------------------------------------------------------------------


class TestDefaultState:
    def test_default_fields(self) -> None:
        from delta_search.problem import DefaultState

        s = DefaultState()
        assert s.graph.num_nodes == 0
        assert s.metrics == {}
        assert s.undo is None

    def test_custom_fields(self) -> None:
        from delta_search.problem import DefaultState

        g = Graph[int].from_edges([(1, 2)])
        s = DefaultState(graph=g, metrics={"key": "val"}, undo="some_entry")
        assert s.graph.num_edges == 1
        assert s.metrics == {"key": "val"}
        assert s.undo == "some_entry"

    def test_satisfies_protocol(self) -> None:
        from delta_search.problem import DefaultState

        s = DefaultState()
        assert isinstance(s, SubgraphState)

    def test_equality(self) -> None:
        from delta_search.problem import DefaultState

        s1 = DefaultState()
        s2 = DefaultState()
        assert s1 == s2

    def test_immutability_of_undo(self) -> None:
        from delta_search.problem import DefaultState

        s = DefaultState()
        s.undo = "test"
        assert s.undo == "test"


# ---------------------------------------------------------------------------
# Objective with penalty
# ---------------------------------------------------------------------------


class TestObjectiveWithPenalty:
    def test_objective_formula(self) -> None:
        g = Graph[int]()
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int]())
        assert problem.objective(state) == 0.0

    def test_objective_with_edges(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = SimpleProblem(g)
        state = SimpleState(graph=Graph[int].from_edges([(1, 2)]))
        assert problem.objective(state) == 1.0
