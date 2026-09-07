# Getting Started with ΔSearch

This guide walks you through the basics of using ΔSearch to solve subgraph extraction problems.

## Prerequisites

- Python 3.10 or later
- Basic understanding of graph theory concepts

## Installation

```bash
git clone https://github.com/delta-search/delta-search.git
cd delta-search
pip install -e ".[dev]"
```

## Core Concepts

### Graph

The `Graph[Node]` class represents an undirected graph with O(1) neighbor and edge lookups.
Nodes must be hashable and comparable (`<`) — built-in types like `int`, `str`, and `tuple` satisfy this.

```python
from delta_search import Graph

# Create an empty graph
g: Graph[int] = Graph()

# Add edges (automatically adds nodes)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 1)

# Check properties
print(g.num_nodes)      # 3
print(g.num_edges)      # 3
print(g.neighbors(1))   # frozenset({2, 3})
print(g.has_edge(1, 2)) # True
```

### Action

An `Action` represents a single candidate mutation to the graph state.

```python
from delta_search import Action, ActionType

# Add an edge between nodes 1 and 2
action = Action(action_type=ActionType.ADD_EDGE, targets=(1, 2))

# Remove a node
action = Action(action_type=ActionType.REMOVE_NODE, targets=(3,))
```

### DeltaResult

A `DeltaResult` captures the incremental change in objective value.

```python
from delta_search import DeltaResult

result = DeltaResult(reward_change=1.0, penalty_change=0.0, feasible=True)
```

## Building Your First Solver

### Step 1: Define your state

State objects must satisfy the `SubgraphState` protocol — they need a `graph` attribute and an optional `undo` field.

```python
from dataclasses import dataclass, field
from delta_search import Graph, SubgraphState

@dataclass
class MaxEdgesState:
    """State tracking the current subgraph and its properties."""
    graph: Graph[int]
    undo: tuple | None = None
```

Or use the built-in `DefaultState`:

```python
from delta_search import DefaultState

state = DefaultState(graph=Graph[int]())
```

### Step 2: Implement the problem interface

```python
from delta_search import (
    SubgraphExtractionProblem,
    Action,
    ActionType,
    DeltaResult,
)

class MaxEdgesProblem(SubgraphExtractionProblem[int]):
    """Find subgraph with maximum edges."""

    def evaluate_initial_state(self, graph: Graph[int]) -> MaxEdgesState:
        """Generate starting candidate subgraph."""
        return MaxEdgesState(graph=Graph[int]())

    def calculate_delta(self, state: MaxEdgesState, action: Action) -> DeltaResult:
        """Compute incremental objective change."""
        if action.action_type is ActionType.ADD_EDGE:
            return DeltaResult(reward_change=1.0, penalty_change=0.0, feasible=True)
        elif action.action_type is ActionType.REMOVE_EDGE:
            return DeltaResult(reward_change=-1.0, penalty_change=0.0, feasible=True)
        return DeltaResult(reward_change=0.0, penalty_change=0.0, feasible=True)

    def compute_reward(self, state: MaxEdgesState) -> float:
        """Scalar reward (higher = better)."""
        return float(state.graph.num_edges)

    def compute_penalty(self, state: MaxEdgesState) -> float:
        """Scalar penalty (lower = better)."""
        return 0.0

    def is_feasible(self, state: MaxEdgesState) -> bool:
        """Check hard constraints."""
        return True
```

### Step 3: Run the solver

```python
from delta_search import GreedySolver

input_graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1), (3, 4)])
problem = MaxEdgesProblem(input_graph)

solver = GreedySolver(problem)
result = solver.solve(max_iterations=100)

print(f"Found {result.best_objective:.0f} edges")
print(f"Converged: {result.converged} ({result.convergence_reason})")
```

## Using Undo Support

ΔSearch supports efficient undo operations for backtracking:

```python
# Apply an action
new_state = problem.apply_action(state, action)

# Undo the last action (returns previous state)
previous_state = problem.undo_action(new_state)
```

## Adding Observers

Monitor solver behavior with the observer protocol:

```python
from delta_search import SolverObserver, Action, DeltaResult

class LoggingObserver:
    def on_action_evaluated(self, action: Action, delta: DeltaResult, elapsed_ms: float) -> None:
        print(f"Evaluated {action}: Δreward={delta.reward_change:+.1f}")

    def on_iteration_complete(self, iteration: int, best_action: Action | None, objective: float) -> None:
        print(f"Iteration {iteration}: objective={objective:.2f}")

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        print(f"Converged after {iterations} iterations: {final_objective:.2f}")

solver = GreedySolver(problem)
result = solver.solve(max_iterations=100, observer=LoggingObserver())
```

## Early Termination

Configure stopping criteria with `EarlyTerminationCondition`:

```python
from delta_search import EarlyTerminationCondition

stop = EarlyTerminationCondition(
    max_iterations=200,
    max_evaluations=10000,
    max_time_ms=5000.0,
    objective_target=10.0,
    stall_iterations=20,
)

solver = GreedySolver(problem, early_stop=stop)
result = solver.solve()
```

## Thread-Safe Access

For concurrent graph access, use `ThreadSafeGraph`:

```python
from delta_search import ThreadSafeGraph

g = ThreadSafeGraph[int]()
g.add_edge(1, 2)  # Thread-safe
g.add_edge(2, 3)  # Thread-safe
print(g.neighbors(1))  # Thread-safe
```

## NetworkX Interoperability

Convert between ΔSearch and NetworkX graphs:

```python
from delta_search.interop import from_networkx, to_networkx
import networkx as nx

nx_graph = nx.complete_graph(5)
ds_graph = from_networkx(nx_graph)  # → Graph[int]

nx_back = to_networkx(ds_graph)     # → nx.Graph
```

## CLI Usage

ΔSearch includes a command-line interface:

```bash
# Solve a problem
delta-search solve --problem mps --graph input.json --output result.json

# Validate a graph
delta-search validate --graph input.json
```

## Next Steps

- Read [Architecture](architecture.md) for design details
- Check [FAQ](faq.md) for common questions
- Explore the test suite in `tests/` for more examples
