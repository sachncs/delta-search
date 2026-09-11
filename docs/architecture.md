# Architecture

This document describes the internal architecture of ΔSearch.

## Overview

ΔSearch implements the Reward-Penalty optimization framework from [arXiv:2606.13834](https://arxiv.org/abs/2606.13834). The framework decomposes the objective into reward (maximize) and penalty (minimize) components, enabling efficient incremental evaluation.

## Design Principles

1. **O(1) Incremental Deltas** — evaluate candidate moves in constant time
2. **Undo-Stack Pattern** — efficient state mutations with rollback support
3. **Protocol-Based Abstraction** — structural typing for flexibility
4. **Defensive Copies** — prevent accidental state corruption

## Module Structure

### `delta_search/graph.py`

The core graph data structure with O(1) lookups.

```
Graph[NodeT]
├── adj: dict[NodeT, set[NodeT]]
├── edge_attrs: dict[frozenset[NodeT], dict[str, Any]]
├── node_attrs: dict[NodeT, dict[str, Any]]
└── edge_count: int
```

Node types must satisfy the `Node` protocol: `__hash__` + `__lt__`.

**Key operations:**
- `add_node(node)` — O(1)
- `add_edge(u, v)` — O(1)
- `remove_node(node)` — O(degree)
- `remove_edge(u, v)` — O(1)
- `neighbors(node)` — O(1)
- `has_edge(u, v)` — O(1)
- `degree(node)` — O(1)
- `subgraph(nodes)` — O(nodes + edges)
- `delta_add_edge(u, v)` — O(1)
- `delta_remove_edge(u, v)` — O(1)
- `delta_add_node(node)` — O(1)
- `delta_remove_node(node)` — O(degree)

**Thread-safe variant:** `ThreadSafeGraph[NodeT]` wraps `Graph` with `RLock` for concurrent access. All public methods (including reads like `subgraph`, `node_list`, `degree_sequence`, `is_subgraph_of`) acquire the lock.

### `delta_search/problem.py`

The abstract problem interface and supporting types.

**Protocols:**

```python
class SubgraphState(Generic[NodeT]):
    graph: Graph[NodeT]
    undo: UndoEntry | None
```

**Named tuples:**

- `Action(action_type, targets)` — a candidate mutation
- `DeltaResult(reward_change, penalty_change, feasible)` — incremental objective change
- `UndoEntry(action, snapshot, node_attrs, edge_attrs)` — rollback record

**Abstract methods (must implement):**

- `calculate_delta(state, action)` → `DeltaResult`
- `compute_reward(state)` → `float`
- `compute_penalty(state)` → `float`
- `is_feasible(state)` → `bool`

**Optional override:**

- `evaluate_initial_state(graph)` → `SubgraphState` (default: returns `DefaultState(graph=Graph())`)
- `generate_composite_actions(state)` → `list[Action]` (default: returns `[]`)

**Concrete methods:**

- `enumerate_actions(state)` → `list[Action]`
- `apply_action(state, action)` → `SubgraphState`
- `undo_action(state)` → `SubgraphState`
- `objective(state)` → `float`
- `add_observer(observer)` / `remove_observer(observer)`

### `delta_search/problems.py`

Six concrete problem implementations:

| Problem | Class | ProblemType |
|---------|-------|-------------|
| Maximum Planar Subgraph | `MaximumPlanarSubgraphProblem` | MONOTONE |
| Uncapacitated Facility Location | `UncapacitatedFacilityLocationProblem` | NON_MONOTONE |
| Prize Collecting Vertex Cover | `PrizeCollectingVertexCoverProblem` | NON_MONOTONE |
| Minimum Connected Dominating Set | `MinimumConnectedDominatingSetProblem` | MONOTONE |
| Maximum Weight Independent Set | `MaximumWeightedIndependentSetProblem` | MONOTONE |
| Minimum Weighted Steiner Tree | `MinimumWeightedSteinerTreeProblem` | MONOTONE |

Shared helpers:
- `node_add_remove_actions(state)` — enumerate add/remove node actions
- `edge_add_actions(state, candidates)` — enumerate add edge actions

### `delta_search/solver.py`

The greedy optimization loop.

```
GreedySolver[NodeT]
├── problem: SubgraphExtractionProblem[NodeT]
├── early_stop: EarlyTerminationCondition[NodeT]
└── solver_state: SolverState[NodeT]
```

**SolverState:** Tracks iterations, evaluations, best objective, convergence.

**EarlyTerminationCondition:** Configurable stopping criteria:
- `max_iterations` / `max_evaluations` — budget limits
- `max_time_ms` — wall-clock timeout
- `objective_target` — stop when objective reaches target
- `stall_iterations` — stop after N iterations with no improvement

**GreedySolver.solve()** returns a `SolverState` with:
- `best_objective` — best objective value found
- `iteration` — number of iterations completed
- `converged` — whether a stopping condition was met
- `convergence_reason` — human-readable explanation

### `delta_search/interop.py`

NetworkX conversion utilities:

- `from_networkx(nx_graph)` → `Graph`
- `to_networkx(graph)` → `nx.Graph`

### `delta_search/io.py`

JSON serialization:

- `save_graph(graph, path)` — write graph to JSON
- `load_graph(path)` → `Graph` — read graph from JSON

### `delta_search/cli.py`

Command-line interface:

- `build_parser()` → `argparse.ArgumentParser`
- `cmd_solve(args)` → `int`
- `cmd_validate(args)` → `int`
- `main(argv)` → `int`
- `import_problem(name)` → `type`

## Data Flow

```
Input Graph
    │
    ▼
evaluate_initial_state()
    │
    ▼
Initial State ─────────────────────────────┐
    │                                       │
    ▼                                       │
enumerate_actions()                        │
    │                                       │
    ▼                                       │
list[Action]                               │
    │                                       │
    ▼                                       │
apply_action()                             │
    │                                       │
    ▼                                       │
New State ──► calculate_delta() ──► DeltaResult
    │                                       │
    ▼                                       │
undo_action() ◄── backtracking ────────────┘
```

## Undo-Stack Pattern

When `apply_action` is called, the framework:

1. Copies the current state graph
2. Applies the mutation to the copy
3. Pushes an `UndoEntry(action, snapshot, node_attrs, edge_attrs)` onto `state.undo`
4. Notifies observers
5. Returns the new state

When `undo_action` is called:

1. Pops the `UndoEntry` from `state.undo`
2. Restores the graph snapshot, node attrs, and edge attrs
3. Notifies observers
4. Returns the restored state

This avoids catastrophic `deepcopy` operations while maintaining correctness.

## Observer Protocol

The `SolverObserver` protocol allows external code to monitor solver behavior:

```python
class SolverObserver(Protocol):
    def on_action_evaluated(self, action: Action, delta: DeltaResult, elapsed_ms: float) -> None: ...
    def on_iteration_complete(self, iteration: int, best_action: Action | None, objective: float) -> None: ...
    def on_convergence(self, iterations: int, final_objective: float) -> None: ...
```

Observers are notified on every action evaluation and iteration, enabling:
- Logging and debugging
- Metrics collection
- Early termination
- Custom heuristics

## Edge Ordering

To prevent duplicate action enumeration, edges are stored in canonical order:

```python
frozenset({min(u, v), max(u, v)})
```

This ensures `(1, 2)` and `(2, 1)` are treated as the same edge.

## Self-Loop Prevention

The API rejects self-loops to maintain consistency with the `frozenset` edge key design:

```python
if u == v:
    raise ValueError(f"Self-loops are not supported: {u} -> {v}")
```

## Thread Safety

`ThreadSafeGraph` provides thread-safe access via `RLock`:

- All public methods acquire the lock before accessing internal state
- No deadlocks possible (single lock, no nested locking)

For single-threaded use, prefer `Graph` directly for better performance.

## Performance Characteristics

| Operation | Graph | ThreadSafeGraph |
|-----------|-------|-----------------|
| Add node | O(1) | O(1) + lock |
| Add edge | O(1) | O(1) + lock |
| Remove node | O(d) | O(d) + lock |
| Remove edge | O(1) | O(1) + lock |
| Has edge | O(1) | O(1) + lock |
| Neighbors | O(1) | O(1) + lock |
| Degree | O(1) | O(1) + lock |
| Subgraph | O(V+E) | O(V+E) + lock |
| Node list | O(V) | O(V) + lock |
| Degree sequence | O(V) | O(V) + lock |
| Is subgraph of | O(V+E) | O(V+E) + lock |

Where `d` = degree of node, `V` = vertices, `E` = edges.
