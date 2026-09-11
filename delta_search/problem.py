"""Abstract base class for subgraph extraction problems.

The delta search framework (arXiv:2606.13834) solves NP-hard subgraph
extraction problems via a Reward-Penalty optimization loop:

1. Start from an initial candidate subgraph (``evaluate_initial_state``).
2. At each iteration, enumerate candidate actions (add/remove node/edge).
3. For each candidate, compute the incremental objective change
   (``calculate_delta``) -- this is the key performance lever because
   it avoids full re-evaluation.
4. The solver selects the action that maximizes the combined
   ``compute_reward(state) - compute_penalty(state)`` signal.
5. ``is_feasible(state)`` gates which candidates survive.

Subclass this ABC and implement every ``@abstractmethod`` to define a
concrete problem (e.g. Maximum Planar Subgraph, UFLP, PCVC).

Thread safety:
    The framework itself is single-threaded.  If you run parallel delta
    search instances, each must have its own ``SubgraphExtractionProblem``
    instance.  Use ``ThreadSafeGraph`` if the underlying graph is shared.

Typical usage::

    class MaxPlanarSubgraph(SubgraphExtractionProblem[int]):
        ...

    problem = MaxPlanarSubgraph(input_graph)
    state = problem.evaluate_initial_state(input_graph)
    while not converged:
        actions = problem.enumerate_actions(state)
        best = max(actions, key=lambda a: problem.objective(
            problem.apply_action(state, a)))
        state = problem.apply_action(state, best)
"""

from __future__ import annotations

import abc
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

from .graph import Graph, NodeT

__all__ = [
    "SubgraphState",
    "DefaultState",
    "ActionType",
    "Action",
    "DeltaResult",
    "UndoEntry",
    "SolverObserver",
    "NullObserver",
    "FanoutObserver",
    "SubgraphExtractionProblem",
    "attach_observer",
]


@runtime_checkable
class SubgraphState(Protocol[NodeT]):
    """Protocol that all delta search state objects must satisfy.

    The state must expose a ``graph`` attribute containing the current
    candidate subgraph, an ``undo`` attribute that stores reversal
    information for the last applied action, and a ``metrics`` dict for
    caching problem-specific incremental data.
    """

    graph: Graph[NodeT]

    undo: UndoEntry | None

    metrics: dict[str, Any]


@dataclass
class DefaultState(Generic[NodeT]):
    """Default mutable state for problem implementations.

    Attributes:
        graph: The current candidate subgraph.
        metrics: Cached problem-specific metrics.
        undo: Reversal information for the last applied action.

    """

    graph: Graph[NodeT] = field(default_factory=Graph)
    metrics: dict[str, Any] = field(default_factory=dict)
    undo: UndoEntry | None = field(default=None, repr=False)


class ActionType(Enum):
    """Primitive mutations the solver can propose."""

    ADD_NODE = auto()
    REMOVE_NODE = auto()
    ADD_EDGE = auto()
    REMOVE_EDGE = auto()


class Action(NamedTuple):
    """A single candidate mutation to the current subgraph.

    ``action_type`` determines how ``targets`` is interpreted:

    - ``ADD_NODE`` / ``REMOVE_NODE``: targets = (node,)
    - ``ADD_EDGE`` / ``REMOVE_EDGE``: targets = (u, v)
    """

    action_type: ActionType
    targets: tuple[Any, ...]
    metadata: dict[str, Any] | None = None


class DeltaResult(NamedTuple):
    """Returned by ``calculate_delta``.

    Attributes:
        reward_change: Signed change in the objective function.  Positive
            means improvement.
        penalty_change: Signed change in constraint violation cost.
            Negative means worse.
        feasible: Whether the resulting state satisfies all hard constraints.

    """

    reward_change: float
    penalty_change: float
    feasible: bool


class UndoEntry(NamedTuple):
    """Internal record for reversing an action.

    Attributes:
        action_type: The inverse action type to restore the prior state.
        targets: Node identifiers targeted by the original action.
        edge_data: Edge attributes to restore (for edge actions).
        node_data: Node attributes to restore (for node removal).
        neighbor_data: Per-neighbor edge attributes to restore (for node removal).

    """

    action_type: ActionType
    targets: tuple[Any, ...]
    edge_data: dict[str, Any] | None = None
    node_data: dict[str, Any] | None = None
    neighbor_data: dict[Any, dict[str, Any]] | None = None


@runtime_checkable
class SolverObserver(Protocol):
    """Observer for solver lifecycle events.

    Implement this protocol to hook into the delta search loop for logging,
    metrics, tracing, or debugging.  All methods have default no-op
    implementations in ``NullObserver``.
    """

    def on_action_evaluated(
        self,
        action: Action,
        delta: DeltaResult,
        elapsed_ms: float,
    ) -> None:
        """Called after each candidate action is evaluated.

        Args:
            action: The action that was evaluated.
            delta: The delta result from ``calculate_delta``.
            elapsed_ms: Wall-clock time for this evaluation in milliseconds.

        """

    def on_iteration_complete(
        self,
        iteration: int,
        best_action: Action | None,
        objective: float,
    ) -> None:
        """Called at the end of each solver iteration.

        Args:
            iteration: Zero-indexed iteration number.
            best_action: The action selected for application, or None.
            objective: The objective value of the best action.

        """

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """Called when the solver terminates.

        Args:
            iterations: Total iterations completed.
            final_objective: Best objective value found.

        """


class NullObserver:
    """Default no-op observer that silently discards all events."""

    def on_action_evaluated(
        self,
        action: Action,
        delta: DeltaResult,
        elapsed_ms: float,
    ) -> None:
        """No-op implementation.

        Args:
            action: The action that was evaluated.
            delta: The delta result from ``calculate_delta``.
            elapsed_ms: Wall-clock time for this evaluation in milliseconds.

        """

    def on_iteration_complete(
        self,
        iteration: int,
        best_action: Action | None,
        objective: float,
    ) -> None:
        """No-op implementation.

        Args:
            iteration: Zero-indexed iteration number.
            best_action: The action selected for application, or None.
            objective: The objective value of the best action.

        """

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """No-op implementation.

        Args:
            iterations: Total iterations completed.
            final_objective: Best objective value found.

        """


class FanoutObserver:
    """Composite observer that fans events out to multiple child observers.

    Solvers address a single observer via ``problem.observer``. When the
    user attaches multiple observers via ``problem.add_observer`` this
    composite is returned, ensuring every registered observer receives
    every lifecycle event.

    Args:
        observers: Child observers to invoke on each event.

    """

    def __init__(self, observers: list[SolverObserver]) -> None:
        """Initialize the fan-out observer.

        Args:
            observers: Child observers to invoke on each event.
        """
        self._observers: list[SolverObserver] = list(observers)

    def add(self, observer: SolverObserver) -> None:
        """Register a child observer.

        Args:
            observer: The observer to add.
        """
        self._observers.append(observer)

    def remove(self, observer: SolverObserver) -> None:
        """Unregister a child observer.

        Args:
            observer: The observer to remove.
        """
        self._observers = [o for o in self._observers if o is not observer]

    @property
    def observers(self) -> list[SolverObserver]:
        """Read-only list of child observers."""
        return list(self._observers)

    def on_action_evaluated(
        self,
        action: Action,
        delta: DeltaResult,
        elapsed_ms: float,
    ) -> None:
        """Dispatch to every child observer."""
        for obs in self._observers:
            obs.on_action_evaluated(action, delta, elapsed_ms)

    def on_iteration_complete(
        self,
        iteration: int,
        best_action: Action | None,
        objective: float,
    ) -> None:
        """Dispatch to every child observer."""
        for obs in self._observers:
            obs.on_iteration_complete(iteration, best_action, objective)

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """Dispatch to every child observer."""
        for obs in self._observers:
            obs.on_convergence(iterations, final_objective)


class SubgraphExtractionProblem(abc.ABC, Generic[NodeT]):
    """Abstract interface every delta search-compatible problem must implement.

    Args:
        graph: The input (complete) graph from which subgraphs are extracted.
        defensive_copy: If True (default), the input graph is deep-copied to
            prevent accidental mutation of the original.  Set to False for
            read-only problems where the copy overhead matters.

    """

    def __init__(
        self,
        graph: Graph[NodeT],
        *,
        defensive_copy: bool = True,
    ) -> None:
        """Initialize the subgraph extraction problem.

        Args:
            graph: The input (complete) graph from which subgraphs are extracted.
            defensive_copy: If True (default), the input graph is deep-copied to
                prevent accidental mutation of the original. Set to False for
                read-only problems where the copy overhead matters.

        """
        self._input_graph: Graph[NodeT] = (
            Graph.from_copy(graph) if defensive_copy else graph
        )
        self._observers: list[SolverObserver] = [NullObserver()]

    @property
    def graph(self) -> Graph[NodeT]:
        """The full input graph.

        Returns:
            The immutable input graph used for action enumeration.

        """
        return self._input_graph

    @property
    def observer(self) -> SolverObserver:
        """An observer that fans events to all registered observers.

        Solvers address a single ``observer`` instance.  This property
        returns a :class:`FanoutObserver` wrapping every registered
        observer (including the default ``NullObserver``), so every
        attached observer receives every lifecycle event.

        Returns:
            A fan-out observer that dispatches each event to all
            registered observers.
        """
        return FanoutObserver(self._observers)

    @property
    def observers(self) -> list[SolverObserver]:
        """Read-only access to the observer list.

        Returns:
            A copy of the observer list.

        """
        return list(self._observers)

    def set_observer(self, observer: SolverObserver) -> None:
        """Replace all observers with a single observer.

        Args:
            observer: The new primary observer.

        """
        self._observers = [observer]

    def add_observer(self, observer: SolverObserver) -> None:
        """Add an observer to the observer list.

        Multiple observers can be attached simultaneously.  All
        registered observers receive lifecycle events.

        Args:
            observer: The observer to append.

        """
        self._observers.append(observer)

    def remove_observer(self, observer: SolverObserver) -> None:
        """Remove an observer from the observer list.

        Args:
            observer: The observer to remove.

        """
        self._observers = [o for o in self._observers if o is not observer]

    @abc.abstractmethod
    def evaluate_initial_state(self, graph: Graph[NodeT]) -> SubgraphState[NodeT]:
        """Generate the starting subgraph / candidate set.

        Args:
            graph: The full input graph.

        Returns:
            The initial candidate state.

        """

    @abc.abstractmethod
    def calculate_delta(
        self,
        current_state: SubgraphState[NodeT],
        candidate_action: Action,
    ) -> DeltaResult:
        """Compute the incremental change caused by applying candidate_action.

        Critical: this method must not re-evaluate the whole graph.
        It should only touch the nodes/edges directly affected by the
        action (plus any cached bookkeeping needed to update metrics).
        This is where the bulk of runtime is spent, so efficiency is key.

        Additive invariant (important for solver correctness):

        All built-in solvers reconstruct the objective of a candidate
        successor state as ``current_objective + delta.reward_change -
        delta.penalty_change``. Subclasses MUST therefore ensure that
        ``compute_reward(next_state) - compute_penalty(next_state) ==
        compute_reward(state) + delta.reward_change - delta.penalty_change``
        whenever ``delta = calculate_delta(state, action)`` and
        ``next_state = apply_action(state, action)``. Problems whose
        reward or penalty does not decompose this way must override
        :meth:`additive_objective_decomposes` to return ``False`` so
        the solver falls back to the full-recompute path.

        Args:
            current_state: The current candidate state.
            candidate_action: The action to evaluate.

        Returns:
            A DeltaResult with reward_change, penalty_change, and feasible.

        """

    def additive_objective_decomposes(self) -> bool:
        """Whether the solver's ``current + delta`` shortcut is safe.

        Override and return ``False`` for problems whose
        ``compute_reward`` or ``compute_penalty`` is not a simple
        additive decomposition of the objective.  When ``False``,
        solvers fall back to computing the full objective on the
        candidate state instead of relying on :meth:`calculate_delta`
        arithmetic alone.

        Returns:
            ``True`` for built-in additive problems.
        """
        return True

    @abc.abstractmethod
    def compute_reward(self, state: SubgraphState[NodeT]) -> float:
        """Scalar reward for state (higher = better).

        Examples: number of edges in a planar subgraph, total facility
        profit, number of covered vertices, etc.

        Args:
            state: The candidate state to evaluate.

        Returns:
            The scalar reward value.

        """

    @abc.abstractmethod
    def compute_penalty(self, state: SubgraphState[NodeT]) -> float:
        """Scalar penalty for state (lower = better).

        Penalty encodes soft constraint violations (e.g. exceeding a
        budget, violating connectivity targets).  Hard constraints should
        be enforced via ``is_feasible`` instead.

        Args:
            state: The candidate state to evaluate.

        Returns:
            The scalar penalty value.

        """

    @abc.abstractmethod
    def is_feasible(self, state: SubgraphState[NodeT]) -> bool:
        """Return True if state satisfies all hard constraints.

        The solver will discard any candidate action whose resulting
        state is infeasible.  Keep this fast -- it is called for every
        candidate at every iteration.

        Args:
            state: The candidate state to check.

        Returns:
            True if the state satisfies all hard constraints.

        """

    def enumerate_actions(self, state: SubgraphState[NodeT]) -> list[Action]:
        """Generate the candidate actions to consider at this iteration.

        Default: enumerate all single-node add/remove and all single-edge
        add/remove that are applicable.  Edge additions are deduplicated
        via canonical ordering so that {u, v} and {v, u} produce exactly
        one ``Action``.

        Override to restrict the action space (e.g. only edge additions
        for planar subgraph problems).

        Args:
            state: The current candidate state.

        Returns:
            List of candidate actions to evaluate.

        """
        actions: list[Action] = []
        seen_edges: set[frozenset[NodeT]] = set()

        current_nodes: set[NodeT] = set(self.state_graph(state).nodes)
        current_edges: frozenset[frozenset[NodeT]] = frozenset(
            self.state_graph(state).edges
        )
        all_nodes: set[NodeT] = set(self._input_graph.nodes)

        for n in all_nodes - current_nodes:
            actions.append(Action(ActionType.ADD_NODE, (n,)))

        for n in current_nodes:
            actions.append(Action(ActionType.REMOVE_NODE, (n,)))

        for n in current_nodes:
            for nb in self._input_graph.neighbors(n):
                if nb in current_nodes:
                    u, v = (n, nb) if n < nb else (nb, n)
                    ekey = frozenset((u, v))
                    if ekey not in current_edges and ekey not in seen_edges:
                        actions.append(Action(ActionType.ADD_EDGE, (u, v)))
                        seen_edges.add(ekey)

        for ekey in current_edges:
            u, v = sorted(ekey)
            actions.append(Action(ActionType.REMOVE_EDGE, (u, v)))

        actions.extend(self.generate_composite_actions(state))
        actions.sort(key=lambda a: (a.action_type.name, tuple(map(str, a.targets))))
        return actions

    def generate_composite_actions(
        self,
        state: SubgraphState[NodeT],
    ) -> list[Action]:
        """Override to add compound actions (e.g. add node + edge together).

        Called at the end of ``enumerate_actions``.  Default returns
        an empty list (atomic moves only).

        Args:
            state: The current candidate state.

        Returns:
            List of composite actions, or empty for atomic-only moves.

        """
        return []

    def apply_action(
        self,
        state: SubgraphState[NodeT],
        action: Action,
        *,
        incremental: bool = False,
    ) -> SubgraphState[NodeT]:
        """Return a new state with action applied.

        Default uses a copy-graph pattern: the state is shallow-copied and
        the graph inside it is deep-copied (``copy.deepcopy`` via
        ``Graph.from_copy``), so mutations do not affect the original.
        Enough information is stored to reverse the action via
        ``undo_action``.

        Performance: the default copy-of-graph cost is ``O(|V|+|E|)`` per
        call. For beam-shaped solvers that call ``apply_action`` for many
        actions per iteration, pass ``incremental=True`` to skip the
        deepcopy. With ``incremental=True`` the new state's graph *aliases*
        the previous one; subsequent ``undo_action`` will restore the
        pre-action graph via the stored ``UndoEntry`` instead of a copy.

        Override if your State has a more efficient structural update.

        Args:
            state: The current candidate state.
            action: The action to apply.
            incremental: If True, share the graph reference and rely on
                ``undo_action`` to roll back via ``UndoEntry``. Use this
                only inside solver loops that always call ``undo_action``
                before branching.

        Returns:
            A new state with the action applied.

        """
        new_state = copy.copy(state)
        if incremental:
            graph = self.state_graph(state)
            self.set_state_graph(new_state, graph)
        else:
            original_graph: Graph[NodeT] = self.state_graph(state)
            graph_copy: Graph[NodeT] = Graph.from_copy(original_graph)
            self.set_state_graph(new_state, graph_copy)
            graph = graph_copy

        undo: UndoEntry | None = None

        if action.action_type is ActionType.ADD_NODE:
            graph.add_node(action.targets[0])
            undo = UndoEntry(
                action_type=ActionType.REMOVE_NODE,
                targets=action.targets,
            )

        elif action.action_type is ActionType.REMOVE_NODE:
            rm_node = action.targets[0]
            nbr_data: dict[Any, dict[str, Any]] = {}
            for nb in graph.neighbors(rm_node):
                nbr_data[nb] = dict(graph.edge_data(rm_node, nb))
            nd = dict(self.node_data_snapshot(graph, rm_node))
            graph.remove_node(rm_node)
            undo = UndoEntry(
                action_type=ActionType.ADD_NODE,
                targets=action.targets,
                node_data=nd,
                neighbor_data=nbr_data,
            )

        elif action.action_type is ActionType.ADD_EDGE:
            eu, ev = action.targets[0], action.targets[1]
            existing = dict(graph.edge_data(eu, ev))
            graph.add_edge(eu, ev)
            undo = UndoEntry(
                action_type=ActionType.REMOVE_EDGE,
                targets=action.targets,
                edge_data=existing if existing else None,
            )

        elif action.action_type is ActionType.REMOVE_EDGE:
            ru, rv = action.targets[0], action.targets[1]
            ed = dict(graph.edge_data(ru, rv))
            graph.remove_edge(ru, rv)
            undo = UndoEntry(
                action_type=ActionType.ADD_EDGE,
                targets=action.targets,
                edge_data=ed,
            )

        new_state.undo = undo
        return new_state

    def undo_action(
        self,
        state: SubgraphState[NodeT],
    ) -> SubgraphState[NodeT]:
        """Reverse the last ``apply_action`` using the stored undo entry.

        First checks for undo info on the state itself, then falls back
        to the undo stack.  Returns the state restored to its prior
        condition.

        Args:
            state: The state to reverse.

        Returns:
            The state with the last action undone.

        Raises:
            RuntimeError: If no undo information is available.

        """
        undo: UndoEntry | None = state.undo
        if undo is None:
            raise RuntimeError("No undo information available on state")

        new_state = copy.copy(state)
        graph: Graph[NodeT] = self.state_graph(new_state)

        if undo.action_type is ActionType.REMOVE_NODE:
            graph.remove_node(undo.targets[0])

        elif undo.action_type is ActionType.ADD_NODE:
            graph.add_node(undo.targets[0], **(undo.node_data or {}))
            if undo.neighbor_data:
                for nb, ed in undo.neighbor_data.items():
                    if graph.has_node(nb):
                        graph.add_edge(undo.targets[0], nb, **ed)

        elif undo.action_type is ActionType.REMOVE_EDGE:
            u, v = undo.targets[0], undo.targets[1]
            graph.remove_edge(u, v)

        elif undo.action_type is ActionType.ADD_EDGE:
            u, v = undo.targets[0], undo.targets[1]
            graph.add_edge(u, v, **(undo.edge_data or {}))

        new_state.undo = None
        return new_state

    def on_iteration_start(self, state: SubgraphState[NodeT], iteration: int) -> None:
        """Hook called before each solver iteration.  No-op by default.

        Args:
            state: The current candidate state.
            iteration: Zero-indexed iteration number.

        """

    def on_iteration_end(self, state: SubgraphState[NodeT], iteration: int) -> None:
        """Hook called after each solver iteration.  No-op by default.

        Args:
            state: The current candidate state.
            iteration: Zero-indexed iteration number.

        """

    def state_graph(self, state: Any) -> Graph[NodeT]:
        """Return the Graph object embedded in state.

        Args:
            state: The state object to extract the graph from.

        Returns:
            The graph embedded in the state.

        Raises:
            TypeError: If the state has no ``.graph`` attribute.

        """
        if not hasattr(state, "graph"):
            raise TypeError(
                f"State of type {type(state).__name__} has no 'graph' "
                f"attribute.  Override state_graph() in your subclass to "
                f"extract the graph from your custom state type."
            )
        result: Graph[NodeT] = state.graph
        return result

    @staticmethod
    def set_state_graph(state: SubgraphState[NodeT], graph: Graph[NodeT]) -> None:
        """Replace the graph reference in state.

        Assumes state has a mutable ``.graph`` attribute (e.g. a
        dataclass field).  Override if your state stores the graph
        differently.

        Args:
            state: The state to modify.
            graph: The new graph to embed.

        """
        state.graph = graph

    @staticmethod
    def node_data_snapshot(
        graph: Graph[NodeT],
        node: NodeT,
    ) -> dict[str, Any]:
        """Snapshot node attributes for undo support.

        Args:
            graph: The graph containing the node.
            node: The node to snapshot.

        Returns:
            Copy of the node's attribute dictionary.

        """
        return dict(graph.node_data(node))

    def objective(self, state: SubgraphState[NodeT]) -> float:
        """Combined scalar: reward - penalty.

        The solver maximizes this.  Override only if your problem needs
        a non-linear combination.

        Args:
            state: The candidate state to evaluate.

        Returns:
            The combined objective value.

        """
        return self.compute_reward(state) - self.compute_penalty(state)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{type(self).__name__}("
            f"nodes={self._input_graph.num_nodes}, "
            f"edges={self._input_graph.num_edges})"
        )


@contextmanager
def attach_observer(
    problem: SubgraphExtractionProblem[NodeT],
    observer: SolverObserver | None,
) -> Iterator[None]:
    """Attach ``observer`` for the duration of the block, then restore.

    Preserves the prior observer list so transient solvers do not mutate
    the problem's observer configuration across calls. The helper is a
    public utility so every solver can wrap its ``solve`` body and
    guarantee observer-list hygiene.

    Args:
        problem: Problem whose observers to manage.
        observer: Observer to attach for the block, or ``None``.

    Yields:
        Nothing.

    """
    if observer is None:
        yield
        return
    prior = problem.observers
    problem.add_observer(observer)
    try:
        yield
    finally:
        problem._observers = prior
