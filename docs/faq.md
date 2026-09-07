# Frequently Asked Questions

## General

### What is ΔSearch?

ΔSearch is a general, fast heuristic framework for solving NP-hard subgraph extraction problems via Reward-Penalty optimization. It implements the framework described in [arXiv:2606.13834](https://arxiv.org/abs/2606.13834).

### What problems can ΔSearch solve?

ΔSearch can solve any subgraph extraction problem that can be formulated as a Reward-Penalty optimization problem, including:
- Maximum Planar Subgraph (MPS)
- Uncapacitated Facility Location Problem (UFLP)
- Prize Collecting Vertex Cover (PCVC)
- Minimum Connected Dominating Set (MCDS)
- Maximum Weight Independent Set (MWIS)
- Minimum Weight Spanning Tree (MWST)

### Is ΔSearch guaranteed to find optimal solutions?

No. ΔSearch is a heuristic framework. It provides high-quality solutions efficiently but does not guarantee optimality. For exact solutions, consider integer linear programming solvers.

### What Python versions are supported?

Python 3.10 and later.

## Usage

### How do I define my own problem?

Implement the `SubgraphExtractionProblem` interface with the four required methods:
1. `calculate_delta(state, action)` — compute incremental objective change
2. `compute_reward(state)` — scalar reward (higher = better)
3. `compute_penalty(state)` — scalar penalty (lower = better)
4. `is_feasible(state)` — check hard constraints

Optionally override `evaluate_initial_state(graph)` to customize the initial state (the default returns `DefaultState(graph=Graph())`).

See [Getting Started](getting-started.md) for a complete example.

### How do I add custom actions?

Override `generate_composite_actions(state)` in your problem subclass to add compound moves beyond the default add/remove node/edge actions.

### How do I monitor solver behavior?

Implement the observer protocol and pass it to `GreedySolver.solve()`:

```python
class MyObserver:
    def on_action_evaluated(self, action, delta, elapsed_ms):
        print(f"Evaluated {action}: Δreward={delta.reward_change:+.1f}")

    def on_iteration_complete(self, iteration, best_action, objective):
        print(f"Iteration {iteration}: objective={objective:.2f}")

    def on_convergence(self, iterations, final_objective):
        print(f"Converged after {iterations} iterations")

solver = GreedySolver(problem)
result = solver.solve(max_iterations=100, observer=MyObserver())
```

### When should I use ThreadSafeGraph?

Use `ThreadSafeGraph` when:
- Multiple threads access the same graph concurrently
- You're using a thread pool for parallel evaluation
- You need shared state between solver instances

For single-threaded use, `Graph` is faster and simpler.

### How do I convert from NetworkX?

```python
from delta_search.interop import from_networkx, to_networkx
import networkx as nx

# From NetworkX to ΔSearch
nx_graph = nx.complete_graph(5)
ds_graph = from_networkx(nx_graph)

# From ΔSearch to NetworkX
nx_back = to_networkx(ds_graph)
```

### How do I stop the solver early?

Use `EarlyTerminationCondition`:

```python
from delta_search import EarlyTerminationCondition, GreedySolver

stop = EarlyTerminationCondition(
    max_iterations=200,      # iteration budget
    max_evaluations=10000,   # evaluation budget
    max_time_ms=5000.0,      # wall-clock timeout
    objective_target=10.0,   # stop when objective reaches target
    stall_iterations=20,     # stop after N iterations with no improvement
)

solver = GreedySolver(problem, early_stop=stop)
result = solver.solve()
```

## Performance

### How fast is ΔSearch?

ΔSearch uses O(1) incremental deltas, so each action evaluation is constant time. The overall performance depends on:
- Number of actions enumerated per iteration
- Number of iterations
- Graph size and density

### How does ΔSearch compare to NetworkX?

ΔSearch is optimized for the subgraph extraction use case:
- O(1) incremental deltas vs. O(V+E) full evaluation
- Undo-stack avoids deepcopy overhead
- Protocol-based design for minimal overhead

NetworkX is more general-purpose and has a larger feature set.

### Can I use ΔSearch with large graphs?

Yes. ΔSearch uses O(V+E) memory for the graph structure. The undo stack grows linearly with the number of actions applied. For very large graphs, consider:
- Using `ThreadSafeGraph` for parallel processing
- Implementing custom `enumerate_actions` to limit candidate moves
- Using the `generate_composite_actions` hook for problem-specific optimizations

## Troubleshooting

### "Self-loops are not supported"

ΔSearch rejects self-loops because the `frozenset` edge key design cannot distinguish self-loops from singleton sets. Remove self-loops from your input graph.

### "State graph is empty"

Call `evaluate_initial_state(graph)` before `enumerate_actions(state)` to ensure the state graph is initialized.

### "Action not found in undo stack"

This can happen if:
- You call `undo_action` without a prior `apply_action`
- You modified the state outside of `apply_action`

Always use `apply_action` and `undo_action` to manage state.

## Contributing

### How do I report a bug?

Open a [bug report](https://github.com/delta-search/delta-search/issues/new?template=bug_report.md) with:
- Description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details

### How do I request a feature?

Open a [feature request](https://github.com/delta-search/delta-search/issues/new?template=feature_request.md) with:
- Problem statement
- Proposed solution
- Use case

### How do I contribute code?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full guide.
