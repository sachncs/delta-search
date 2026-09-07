"""Tests for delta_search.problems module."""

from __future__ import annotations

from delta_search.graph import Graph
from delta_search.problem import DefaultState
from delta_search.problems import (
    MaximumPlanarSubgraphProblem,
    MaximumWeightedIndependentSetProblem,
    MinimumConnectedDominatingSetProblem,
    MinimumWeightedSteinerTreeProblem,
    PrizeCollectingVertexCoverProblem,
    ProblemType,
    UncapacitatedFacilityLocationProblem,
)


class TestState:
    def test_empty_state(self) -> None:
        s = DefaultState()
        assert s.graph.num_nodes == 0
        assert s.metrics == {}

    def test_state_with_graph(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        s = DefaultState(graph=g)
        assert s.graph.num_edges == 1


class TestMaximumPlanarSubgraph:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = MaximumPlanarSubgraphProblem(g)
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.MONOTONE

    def test_add_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumPlanarSubgraphProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1, 2, 3]))
        actions = problem.enumerate_actions(state)
        assert len(actions) > 0
        assert all(a.action_type.name == "ADD_EDGE" for a in actions)

    def test_feasibility(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = MaximumPlanarSubgraphProblem(g)
        state = DefaultState(graph=Graph[int].from_edges([(1, 2), (2, 3), (3, 1)]))
        assert problem.is_feasible(state)

    def test_objective(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumPlanarSubgraphProblem(g)
        state = DefaultState(graph=Graph[int].from_edges([(1, 2), (2, 3)]))
        assert problem.compute_reward(state) == 2.0


class TestMinimumConnectedDominatingSet:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = MinimumConnectedDominatingSetProblem(g)
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.MONOTONE

    def test_add_node(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MinimumConnectedDominatingSetProblem(g)
        state = DefaultState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        assert len(actions) == 3

    def test_single_node_dominates_complete(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = MinimumConnectedDominatingSetProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1]))
        assert problem.is_feasible(state)

    def test_penalty_when_not_dominating(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 4)])
        problem = MinimumConnectedDominatingSetProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1]))
        assert problem.compute_penalty(state) > 0


class TestMaximumWeightedIndependentSet:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumWeightedIndependentSetProblem(g)
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.MONOTONE

    def test_add_node(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumWeightedIndependentSetProblem(g)
        state = DefaultState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        assert len(actions) == 3

    def test_independence_check(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumWeightedIndependentSetProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1, 3]))
        assert problem.is_feasible(state)

    def test_not_independent(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MaximumWeightedIndependentSetProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1, 2]))
        assert not problem.is_feasible(state)

    def test_weighted(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = MaximumWeightedIndependentSetProblem(
            g,
            vertex_weights={1: 10.0, 2: 5.0},
        )
        state = DefaultState(graph=Graph[int](nodes=[1]))
        assert problem.compute_reward(state) == 10.0


class TestPrizeCollectingVertexCover:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = PrizeCollectingVertexCoverProblem(g)
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.NON_MONOTONE

    def test_add_and_remove(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = PrizeCollectingVertexCoverProblem(g)
        state = DefaultState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        add_actions = [a for a in actions if a.action_type.name == "ADD_NODE"]
        assert len(add_actions) == 3

    def test_empty_cover_cost(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        state = DefaultState(graph=Graph[int]())
        reward = problem.compute_reward(state)
        assert reward < 0

    def test_full_cover(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = PrizeCollectingVertexCoverProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1, 2]))
        reward = problem.compute_reward(state)
        assert reward < 0


class TestUncapacitatedFacilityLocation:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = UncapacitatedFacilityLocationProblem(g)
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.NON_MONOTONE

    def test_add_facility(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = UncapacitatedFacilityLocationProblem(g)
        state = DefaultState(graph=Graph[int]())
        actions = problem.enumerate_actions(state)
        assert len(actions) == 3

    def test_total_cost(self) -> None:
        g = Graph[int].from_edges([(1, 2)])
        problem = UncapacitatedFacilityLocationProblem(g)
        state = DefaultState(graph=Graph[int](nodes=[1]))
        reward = problem.compute_reward(state)
        assert reward < 0


class TestMinimumWeightedSteinerTree:
    def test_basic(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
        problem = MinimumWeightedSteinerTreeProblem(
            g,
            terminals={1, 3},
        )
        state = problem.evaluate_initial_state(g)
        assert state.graph.num_nodes == 0
        assert problem.problem_type is ProblemType.NON_MONOTONE

    def test_add_edge(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MinimumWeightedSteinerTreeProblem(
            g,
            terminals={1, 3},
        )
        state = DefaultState(graph=Graph[int](nodes=[1, 2, 3]))
        actions = problem.enumerate_actions(state)
        edge_actions = [a for a in actions if a.action_type.name == "ADD_EDGE"]
        assert len(edge_actions) > 0

    def test_connected_terminals(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MinimumWeightedSteinerTreeProblem(
            g,
            terminals={1, 3},
        )
        state = DefaultState(graph=Graph[int].from_edges([(1, 2), (2, 3)]))
        assert problem.is_feasible(state)

    def test_disconnected_terminals(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MinimumWeightedSteinerTreeProblem(
            g,
            terminals={1, 3},
        )
        state = DefaultState(graph=Graph[int](nodes=[1, 3]))
        assert not problem.is_feasible(state)

    def test_terminal_not_removable(self) -> None:
        g = Graph[int].from_edges([(1, 2), (2, 3)])
        problem = MinimumWeightedSteinerTreeProblem(
            g,
            terminals={1, 3},
        )
        state = DefaultState(graph=Graph[int](nodes=[1, 2, 3]))
        actions = problem.enumerate_actions(state)
        remove_actions = [a for a in actions if a.action_type.name == "REMOVE_NODE"]
        assert len(remove_actions) == 1
        assert remove_actions[0].targets[0] == 2
