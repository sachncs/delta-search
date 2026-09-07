"""Concrete problem implementations for delta search.

Each class implements ``SubgraphExtractionProblem`` for a specific
NP-hard subgraph extraction problem from the paper (arXiv:2606.13834).

Monotone problems (add-only):

- MaximumPlanarSubgraphProblem
- MinimumConnectedDominatingSetProblem
- MaximumWeightedIndependentSetProblem

Non-monotone problems (add + remove):

- PrizeCollectingVertexCoverProblem
- UncapacitatedFacilityLocationProblem
- MinimumWeightedSteinerTreeProblem
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from .graph import Graph, NodeT
from .problem import (
    Action,
    ActionType,
    DefaultState,
    DeltaResult,
    SubgraphExtractionProblem,
    SubgraphState,
)
from .utils import (
    bfs_reachable,
    is_connected,
    is_dominating_set,
    is_independent_set,
    is_planary,
)

__all__ = [
    "ProblemType",
    "extract_graph",
    "extract_node",
    "extract_edge",
    "node_add_remove_actions",
    "edge_add_actions",
    "MaximumPlanarSubgraphProblem",
    "MinimumConnectedDominatingSetProblem",
    "MaximumWeightedIndependentSetProblem",
    "PrizeCollectingVertexCoverProblem",
    "UncapacitatedFacilityLocationProblem",
    "MinimumWeightedSteinerTreeProblem",
]


class ProblemType(Enum):
    """Classification of problems by monotonicity."""

    MONOTONE = auto()
    NON_MONOTONE = auto()


def extract_graph(state: SubgraphState[Any]) -> Graph[Any]:
    """Extract the graph from a state object.

    Args:
        state: The state containing a graph attribute.

    Returns:
        The graph embedded in the state.

    """
    return state.graph


def extract_node(action: Action) -> Any:
    """Extract the single node target from an action.

    Args:
        action: An ADD_NODE or REMOVE_NODE action.

    Returns:
        The targeted node identifier.

    """
    return action.targets[0]


def extract_edge(action: Action) -> tuple[Any, Any]:
    """Extract the edge target (u, v) from an action.

    Args:
        action: An ADD_EDGE or REMOVE_EDGE action.

    Returns:
        A tuple of the two endpoint nodes.

    """
    return action.targets[0], action.targets[1]


def node_add_remove_actions(
    input_graph: Graph[Any],
    state_graph: Graph[Any],
) -> list[Action]:
    """Generate ADD_NODE and REMOVE_NODE actions.

    Shared by monotone node-only and non-monotone node-only problems.

    Args:
        input_graph: The full input graph.
        state_graph: The current candidate subgraph.

    Returns:
        List of node add/remove actions.

    """
    actions: list[Action] = []
    current_nodes: set[Any] = set(state_graph.nodes)
    all_nodes: set[Any] = set(input_graph.nodes)

    for n in all_nodes - current_nodes:
        actions.append(Action(ActionType.ADD_NODE, (n,)))
    for n in current_nodes:
        actions.append(Action(ActionType.REMOVE_NODE, (n,)))

    return actions


def edge_add_actions(
    input_graph: Graph[Any],
    state_graph: Graph[Any],
) -> list[Action]:
    """Generate ADD_EDGE actions for edges not yet in state_graph.

    Args:
        input_graph: The full input graph.
        state_graph: The current candidate subgraph.

    Returns:
        List of edge add actions with canonical ordering.

    """
    actions: list[Action] = []
    seen_edges: set[frozenset[Any]] = set()
    current_nodes: set[Any] = set(state_graph.nodes)
    current_edges: frozenset[frozenset[Any]] = frozenset(state_graph.edges)

    for n in current_nodes:
        for nb in input_graph.neighbors(n):
            if nb in current_nodes:
                u, v = (n, nb) if n < nb else (nb, n)
                ekey = frozenset((u, v))
                if ekey not in current_edges and ekey not in seen_edges:
                    actions.append(Action(ActionType.ADD_EDGE, (u, v)))
                    seen_edges.add(ekey)

    return actions


# ---------------------------------------------------------------------------
# Monotone problems (add-only)
# ---------------------------------------------------------------------------


class MaximumPlanarSubgraphProblem(SubgraphExtractionProblem[NodeT]):
    """Find a maximum-edge planar subgraph.

    Monotone: only edge additions.  The reward is the number of edges
    in the planar subgraph.  Infeasible actions (those that would
    create a non-planar graph) are rejected.
    """

    problem_type = ProblemType.MONOTONE

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding a single edge.

        Uses incremental planarity checking: tests the Euler bound
        on the affected component without copying the entire graph.
        O(V_component + E_component) for the BFS component detection.

        Args:
            current_state: The current candidate state.
            candidate_action: An ADD_EDGE action to evaluate.

        Returns:
            DeltaResult with reward_change=1.0 if planar, 0.0 otherwise.

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_EDGE:
            u, v = extract_edge(candidate_action)
            if g.has_edge(u, v):
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=0.0,
                    feasible=False,
                )
            # Find the component containing u (and v if connected)
            from .utils import bfs_reachable

            component_nodes = bfs_reachable(g, {u})
            # Count edges in this component
            component_edges = 0
            for n in component_nodes:
                for nb in g.neighbors(n):
                    if nb in component_nodes and n < nb:
                        component_edges += 1
            num_v = len(component_nodes)
            num_e = component_edges + 1  # adding the new edge
            feasible = num_v < 3 or num_e <= 3 * num_v - 6
            return DeltaResult(
                reward_change=1.0 if feasible else 0.0,
                penalty_change=0.0,
                feasible=feasible,
            )
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Count of edges in the candidate subgraph.

        Args:
            state: The candidate state.

        Returns:
            Number of edges as a float.

        """
        return float(extract_graph(state).num_edges)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Always zero -- planarity is enforced via is_feasible.

        Args:
            state: The candidate state.

        Returns:
            0.0 (no soft penalty).

        """
        return 0.0

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if the current subgraph is planar.

        Args:
            state: The candidate state.

        Returns:
            True if the subgraph is planar.

        """
        return is_planary(extract_graph(state))

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Only edge additions are allowed (monotone).

        Args:
            state: The current candidate state.

        Returns:
            List of ADD_EDGE actions for non-present edges.

        """
        return edge_add_actions(self._input_graph, extract_graph(state))


class MinimumConnectedDominatingSetProblem(SubgraphExtractionProblem[NodeT]):
    """Find a minimum connected dominating set.

    Monotone: only vertex additions.  Reward is negative vertex count
    (minimize).  Penalty is 0 when dominating, large otherwise.
    """

    problem_type = ProblemType.MONOTONE

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        penalty_weight: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.

        Args:
            graph: The input graph.
            penalty_weight: Weight for the domination penalty term.

        """
        super().__init__(graph, **kwargs)
        self.penalty_weight = penalty_weight

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding a single node.

        Uses incremental domination checking: O(degree(node)) to test
        if the new node helps dominate un-dominated vertices.  Connectivity
        is checked via BFS on the candidate subgraph.

        Args:
            current_state: The current candidate state.
            candidate_action: An ADD_NODE action to evaluate.

        Returns:
            DeltaResult with reward_change=-1.0 (one more node).

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_NODE:
            node = extract_node(candidate_action)
            selected = set(g.nodes) | {node}
            dominating = is_dominating_set(self._input_graph, selected)
            # Build subgraph incrementally: add node and edges to selected neighbors
            test = Graph.from_copy(g)
            test.add_node(node)
            for nb in self._input_graph.neighbors(node):
                if nb in selected:
                    test.add_edge(node, nb)
            connected = is_connected(test) if test.num_nodes > 1 else True
            feasible = dominating and connected
            return DeltaResult(
                reward_change=-1.0,
                penalty_change=0.0 if feasible else self.penalty_weight,
                feasible=True,
            )
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Negative node count (fewer nodes is better).

        Args:
            state: The candidate state.

        Returns:
            Negative count of nodes in the subgraph.

        """
        return -float(extract_graph(state).num_nodes)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Penalty when the set is not a connected dominating set.

        Args:
            state: The candidate state.

        Returns:
            0.0 if connected and dominating, penalty_weight otherwise.

        """
        g = extract_graph(state)
        if g.num_nodes == 0:
            return self.penalty_weight
        dominating = is_dominating_set(self._input_graph, set(g.nodes))
        connected = is_connected(g) if g.num_nodes > 1 else True
        if dominating and connected:
            return 0.0
        return self.penalty_weight

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if the set is a connected dominating set.

        Args:
            state: The candidate state.

        Returns:
            True if the set is non-empty, connected, and dominating.

        """
        g = extract_graph(state)
        if g.num_nodes == 0:
            return False
        return is_dominating_set(self._input_graph, set(g.nodes)) and (
            is_connected(g) if g.num_nodes > 1 else True
        )

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Only node additions (monotone).

        Args:
            state: The current candidate state.

        Returns:
            List of ADD_NODE actions for nodes not in the current set.

        """
        g = extract_graph(state)
        actions: list[Action] = []
        current_nodes: set[NodeT] = set(g.nodes)
        for n in set(self._input_graph.nodes) - current_nodes:
            actions.append(Action(ActionType.ADD_NODE, (n,)))
        return actions


class MaximumWeightedIndependentSetProblem(SubgraphExtractionProblem[NodeT]):
    """Find a maximum weight independent set.

    Monotone: only vertex additions.  Reward is sum of vertex weights.
    Penalty is large when independence is violated.
    """

    problem_type = ProblemType.MONOTONE

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        vertex_weights: dict[NodeT, float] | None = None,
        penalty_weight: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.

        Args:
            graph: The input graph.
            vertex_weights: Optional dict mapping nodes to weights.
                Defaults to unit weight for all nodes.
            penalty_weight: Weight for the independence-violation penalty.

        """
        super().__init__(graph, **kwargs)
        self.vertex_weights = vertex_weights or {}
        self.penalty_weight = penalty_weight

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding a single node.

        O(degree(node)) — tests whether any neighbor of the candidate
        node is already in the selected set.  No graph copy needed.

        Args:
            current_state: The current candidate state.
            candidate_action: An ADD_NODE action to evaluate.

        Returns:
            DeltaResult with reward_change equal to node weight if feasible.

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_NODE:
            node = extract_node(candidate_action)
            w = self.vertex_weights.get(node, 1.0)
            # O(degree) check: is any neighbor already selected?
            selected = set(g.nodes)
            feasible = not (self._input_graph.neighbors(node) & selected)
            return DeltaResult(
                reward_change=w if feasible else 0.0,
                penalty_change=0.0,
                feasible=feasible,
            )
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Sum of vertex weights in the independent set.

        Args:
            state: The candidate state.

        Returns:
            Total weight of selected vertices.

        """
        total = 0.0
        for n in extract_graph(state).nodes:
            total += self.vertex_weights.get(n, 1.0)
        return total

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Penalty when the set is not independent.

        Args:
            state: The candidate state.

        Returns:
            penalty_weight if independence is violated, 0.0 otherwise.

        """
        if not is_independent_set(self._input_graph, set(extract_graph(state).nodes)):
            return self.penalty_weight
        return 0.0

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if the selected nodes form an independent set.

        Args:
            state: The candidate state.

        Returns:
            True if no two selected nodes are adjacent.

        """
        return is_independent_set(self._input_graph, set(extract_graph(state).nodes))

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Only node additions (monotone).

        Args:
            state: The current candidate state.

        Returns:
            List of ADD_NODE actions for nodes not in the current set.

        """
        g = extract_graph(state)
        actions: list[Action] = []
        current_nodes: set[NodeT] = set(g.nodes)
        for n in set(self._input_graph.nodes) - current_nodes:
            actions.append(Action(ActionType.ADD_NODE, (n,)))
        return actions


# ---------------------------------------------------------------------------
# Non-monotone problems (add + remove)
# ---------------------------------------------------------------------------


class PrizeCollectingVertexCoverProblem(SubgraphExtractionProblem[NodeT]):
    """Prize Collecting Vertex Cover.

    Minimize: (sum of vertex costs in cover) + (sum of penalties for
    uncovered edges).  Non-monotone: vertices can be added or removed.
    """

    problem_type = ProblemType.NON_MONOTONE

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        vertex_costs: dict[NodeT, float] | None = None,
        edge_penalties: dict[frozenset[NodeT], float] | None = None,
        default_penalty: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.

        Args:
            graph: The input graph.
            vertex_costs: Optional dict mapping nodes to inclusion costs.
            edge_penalties: Optional dict mapping edges to penalties for
                being uncovered.  Defaults to ``default_penalty``.
            default_penalty: Penalty for uncovered edges not in
                ``edge_penalties``.

        """
        super().__init__(graph, **kwargs)
        self.vertex_costs = vertex_costs or {}
        self.edge_penalties = edge_penalties or {}
        self.default_penalty = default_penalty

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def penalty_for(self, u: NodeT, v: NodeT) -> float:
        """Look up the penalty for an uncovered edge.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            The penalty for leaving edge {u, v} uncovered.

        """
        key = frozenset((u, v))
        return self.edge_penalties.get(key, self.default_penalty)

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding or removing a single node.

        For ADD_NODE: reward = (penalties saved by covering new edges) - cost.
        For REMOVE_NODE: reward = cost - (penalties incurred by uncovering edges).

        Args:
            current_state: The current candidate state.
            candidate_action: An ADD_NODE or REMOVE_NODE action.

        Returns:
            DeltaResult with the computed reward change.

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_NODE:
            node = extract_node(candidate_action)
            cost = self.vertex_costs.get(node, 1.0)
            covered_savings = 0.0
            for nb in self._input_graph.neighbors(node):
                if nb not in g.nodes:
                    covered_savings += self.penalty_for(node, nb)
            return DeltaResult(
                reward_change=covered_savings - cost,
                penalty_change=0.0,
                feasible=True,
            )

        if candidate_action.action_type is ActionType.REMOVE_NODE:
            node = extract_node(candidate_action)
            cost = self.vertex_costs.get(node, 1.0)
            actual_uncovered = 0.0
            for nb in self._input_graph.neighbors(node):
                if nb not in g.nodes:
                    actual_uncovered += self.penalty_for(node, nb)
            return DeltaResult(
                reward_change=cost - actual_uncovered,
                penalty_change=0.0,
                feasible=True,
            )

        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Negative total cost (vertex costs + uncovered edge penalties).

        Args:
            state: The candidate state.

        Returns:
            Negative combined cost of the vertex cover.

        """
        g = extract_graph(state)
        cover = set(g.nodes)
        cost = 0.0
        for n in cover:
            cost += self.vertex_costs.get(n, 1.0)
        uncovered = 0.0
        for u, v in self._input_graph.edges:
            if u not in cover and v not in cover:
                uncovered += self.penalty_for(u, v)
        return -(cost + uncovered)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Always zero -- no soft constraints.

        Args:
            state: The candidate state.

        Returns:
            0.0.

        """
        return 0.0

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """All states are feasible for prize collecting vertex cover.

        Args:
            state: The candidate state.

        Returns:
            Always True.

        """
        return True

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Both node additions and removals (non-monotone).

        Args:
            state: The current candidate state.

        Returns:
            List of ADD_NODE and REMOVE_NODE actions.

        """
        return node_add_remove_actions(self._input_graph, extract_graph(state))


class UncapacitatedFacilityLocationProblem(SubgraphExtractionProblem[NodeT]):
    """Uncapacitated Facility Location Problem.

    Minimize: (facility opening costs) + (customer assignment costs).
    Non-monotone: facilities can be opened or closed.
    """

    problem_type = ProblemType.NON_MONOTONE

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        facility_costs: dict[NodeT, float] | None = None,
        edge_costs: dict[frozenset[NodeT], float] | None = None,
        default_edge_cost: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.

        Args:
            graph: The input graph.
            facility_costs: Optional dict mapping nodes to opening costs.
            edge_costs: Optional dict mapping edges to assignment costs.
                Defaults to ``default_edge_cost``.
            default_edge_cost: Assignment cost for edges not in
                ``edge_costs``.

        """
        super().__init__(graph, **kwargs)
        self.facility_costs = facility_costs or {}
        self.edge_costs = edge_costs or {}
        self.default_edge_cost = default_edge_cost

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def cost_for_edge(self, u: NodeT, v: NodeT) -> float:
        """Look up the assignment cost between two nodes.

        Args:
            u: First node.
            v: Second node.

        Returns:
            The edge cost, or default_edge_cost if not specified.

        """
        key = frozenset((u, v))
        return self.edge_costs.get(key, self.default_edge_cost)

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding or removing a facility.

        Args:
            current_state: The current candidate state.
            candidate_action: An ADD_NODE or REMOVE_NODE action.

        Returns:
            DeltaResult with the computed reward change.

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_NODE:
            node = extract_node(candidate_action)
            opening_cost = self.facility_costs.get(node, 1.0)
            customers = set(self._input_graph.nodes) - set(g.nodes) - {node}
            assignment_savings = 0.0
            for c in customers:
                if self._input_graph.has_edge(node, c):
                    assignment_savings += self.cost_for_edge(node, c)
            return DeltaResult(
                reward_change=assignment_savings - opening_cost,
                penalty_change=0.0,
                feasible=True,
            )

        if candidate_action.action_type is ActionType.REMOVE_NODE:
            node = extract_node(candidate_action)
            opening_cost = self.facility_costs.get(node, 1.0)
            facilities = set(g.nodes) - {node}
            customers = set(self._input_graph.nodes) - facilities
            assignment_increase = 0.0
            for c in customers:
                if not self._input_graph.has_edge(node, c):
                    continue
                old_cost = self.cost_for_edge(node, c)
                alternatives = [
                    self.cost_for_edge(f, c)
                    for f in facilities
                    if self._input_graph.has_edge(f, c)
                ]
                if not alternatives:
                    return DeltaResult(
                        reward_change=0.0,
                        penalty_change=0.0,
                        feasible=False,
                    )
                new_cost = min(alternatives)
                assignment_increase += new_cost - old_cost
            return DeltaResult(
                reward_change=opening_cost - assignment_increase,
                penalty_change=0.0,
                feasible=True,
            )

        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Negative total cost (opening + assignment).

        Args:
            state: The candidate state.

        Returns:
            Negative combined opening and assignment cost.

        """
        g = extract_graph(state)
        facilities = set(g.nodes)
        customers = set(self._input_graph.nodes) - facilities
        opening = 0.0
        for f in facilities:
            opening += self.facility_costs.get(f, 1.0)

        def min_assignment_cost(c: NodeT) -> float:
            """Find minimum assignment cost for customer c to open facilities.

            Args:
                c: The customer node.

            Returns:
                Minimum edge cost to an open facility, or inf if unreachable.

            """
            costs = [
                self.cost_for_edge(f, c)
                for f in facilities
                if self._input_graph.has_edge(f, c)
            ]
            return min(costs) if costs else float("inf")

        assignment = sum(min_assignment_cost(c) for c in customers)
        return -(opening + assignment)

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Always zero -- no soft constraints.

        Args:
            state: The candidate state.

        Returns:
            0.0.

        """
        return 0.0

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """All states are feasible for uncapacitated facility location.

        Args:
            state: The candidate state.

        Returns:
            Always True.

        """
        return True

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Both facility open and close (non-monotone).

        Args:
            state: The current candidate state.

        Returns:
            List of ADD_NODE and REMOVE_NODE actions.

        """
        return node_add_remove_actions(self._input_graph, extract_graph(state))


class MinimumWeightedSteinerTreeProblem(SubgraphExtractionProblem[NodeT]):
    """Minimum Weighted Steiner Tree.

    Minimize: (total edge weight in selected subgraph) while
    connecting all terminals.  Non-monotone: edges and Steiner
    vertices can be added or removed.
    """

    problem_type = ProblemType.NON_MONOTONE

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        terminals: set[NodeT] | None = None,
        edge_weights: dict[frozenset[NodeT], float] | None = None,
        default_weight: float = 1.0,
        connectivity_penalty: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.

        Args:
            graph: The input graph.
            terminals: Optional set of terminal nodes that must be
                connected.
            edge_weights: Optional dict mapping edges to weights.
                Defaults to ``default_weight``.
            default_weight: Edge weight for edges not in ``edge_weights``.
            connectivity_penalty: Penalty when terminals are not fully
                connected.

        """
        super().__init__(graph, **kwargs)
        self.terminals = terminals or set()
        self.edge_weights = edge_weights or {}
        self.default_weight = default_weight
        self.connectivity_penalty = connectivity_penalty

    def evaluate_initial_state(self, graph: Graph[NodeT]) -> DefaultState[Any]:
        """Generate an empty starting state.

        Args:
            graph: The full input graph.

        Returns:
            An empty DefaultState.

        """
        return DefaultState[Any](graph=Graph())

    def weight_of(self, u: NodeT, v: NodeT) -> float:
        """Look up the weight of an edge.

        Args:
            u: First endpoint.
            v: Second endpoint.

        Returns:
            The edge weight, or default_weight if not specified.

        """
        key = frozenset((u, v))
        return self.edge_weights.get(key, self.default_weight)

    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the change from adding/removing an edge or node.

        ADD_EDGE and ADD_NODE are O(1).  REMOVE_EDGE uses BFS that
        avoids the removed edge (no graph copy).  REMOVE_NODE is O(1).

        Args:
            current_state: The current candidate state.
            candidate_action: The action to evaluate.

        Returns:
            DeltaResult with the computed reward change.

        """
        g = extract_graph(current_state)
        if candidate_action.action_type is ActionType.ADD_EDGE:
            u, v = extract_edge(candidate_action)
            weight = self.weight_of(u, v)
            return DeltaResult(
                reward_change=-weight,
                penalty_change=0.0,
                feasible=True,
            )

        if candidate_action.action_type is ActionType.REMOVE_EDGE:
            u, v = extract_edge(candidate_action)
            weight = self.weight_of(u, v)
            terminal_nodes = self.terminals & set(g.nodes)
            if len(terminal_nodes) <= 1:
                still_connected = True
            else:
                # BFS from one terminal, avoiding the removed edge
                start = next(iter(terminal_nodes))
                visited: set[NodeT] = {start}
                queue: list[NodeT] = [start]
                removed_key = frozenset((u, v))
                while queue:
                    node = queue.pop()
                    for nb in g.neighbors(node):
                        if nb in visited:
                            continue
                        edge_key = frozenset((node, nb))
                        if edge_key == removed_key:
                            continue
                        visited.add(nb)
                        queue.append(nb)
                still_connected = terminal_nodes <= visited
            return DeltaResult(
                reward_change=weight if still_connected else 0.0,
                penalty_change=0.0 if still_connected else self.connectivity_penalty,
                feasible=True,
            )

        if candidate_action.action_type is ActionType.ADD_NODE:
            return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=True)

        if candidate_action.action_type is ActionType.REMOVE_NODE:
            node = extract_node(candidate_action)
            if node in self.terminals:
                return DeltaResult(
                    reward_change=0.0,
                    penalty_change=self.connectivity_penalty,
                    feasible=True,
                )
            return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=True)

        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=False)

    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Negative total edge weight.

        Args:
            state: The candidate state.

        Returns:
            Negative sum of edge weights in the subgraph.

        """
        g = extract_graph(state)
        total = 0.0
        for u, v in g.edges:
            total += self.weight_of(u, v)
        return -total

    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Penalty if terminals are not connected.

        Args:
            state: The candidate state.

        Returns:
            connectivity_penalty if terminals are disconnected, 0.0 otherwise.

        """
        if self.is_feasible(state):
            return 0.0
        return self.connectivity_penalty

    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Check if all terminals are mutually connected.

        Uses BFS from one terminal to verify all terminals are reachable.

        Args:
            state: The candidate state.

        Returns:
            True if all terminals in the subgraph are connected.

        """
        g = extract_graph(state)
        terminal_nodes = self.terminals & set(g.nodes)
        if len(terminal_nodes) <= 1:
            return True
        start = next(iter(terminal_nodes))
        reachable = bfs_reachable(g, {start})
        return terminal_nodes <= reachable

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Edge and node additions/removals (non-monotone).

        Non-terminal nodes can be removed; all input graph nodes can be
        added.  Terminal nodes are never removable to prevent infeasible
        states where terminals are absent.

        Args:
            state: The current candidate state.

        Returns:
            List of add/remove actions for nodes and edges.

        """
        actions: list[Action] = []
        g = extract_graph(state)
        current_nodes: set[NodeT] = set(g.nodes)
        current_edges: frozenset[frozenset[NodeT]] = frozenset(g.edges)

        for n in set(self._input_graph.nodes) - current_nodes:
            actions.append(Action(ActionType.ADD_NODE, (n,)))

        for n in current_nodes:
            if n not in self.terminals:
                actions.append(Action(ActionType.REMOVE_NODE, (n,)))

        actions.extend(edge_add_actions(self._input_graph, g))

        for ekey in current_edges:
            u, v = tuple(ekey)
            actions.append(Action(ActionType.REMOVE_EDGE, (u, v)))

        return actions
