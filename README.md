<p align="center">
  <h1 align="center">delta-search</h1>
  <p align="center">A general, fast heuristic framework for NP-hard subgraph extraction via Reward-Penalty optimization.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/delta-search/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/delta-search/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/delta-search/"><img src="https://img.shields.io/pypi/v/delta-search" alt="PyPI"></a>
    <a href="https://github.com/sachncs/delta-search/stargazers"><img src="https://img.shields.io/github/stars/sachncs/delta-search" alt="Stars"></a>
  </p>
</p>

**ΔSearch** is an independent implementation of the algorithm from the paper
["Solving Subgraph Extraction Problems Using ΔSearch"](https://arxiv.org/abs/2606.13834)
by Rebin Silva Valan Arasu and Rajiv Gupta at UC Riverside. A general, fast
heuristic framework for solving NP-hard subgraph extraction problems via
Reward-Penalty optimization.

---

## Features

- **General-purpose** — solves 6 NP-hard problems with a single framework
- **O(1) incremental deltas** — evaluates candidate moves without re-evaluating the entire graph
- **Undo-stack actions** — efficient state mutations with rollback support
- **Thread-safe graph** — concurrent access via `ThreadSafeGraph` with `RLock`
- **Observer protocol** — hook into solver lifecycle for logging, metrics, and tracing
- **Zero dependencies** — pure Python standard library; optional NetworkX interop
- **Fully typed** — `mypy --strict` compliant with `py.typed` marker
- **Production-ready** — 320+ tests, CI/CD, security scanning, 80%+ coverage

---

## Installation

### From PyPI

```bash
pip install delta-search
```

### From source

```bash
git clone https://github.com/sachncs/delta-search.git
cd delta-search
pip install -e ".[dev]"
```

---

## Quick Start

### CLI

```bash
# Solve a problem
delta-search solve --problem mps --graph input.json --output result.json

# Validate a graph
delta-search validate --graph input.json

# Available problems: mps, mcds, mwis, pcvc, uflp, mwst
```

### Python API

```python
from delta_search import (
    Graph,
    GreedySolver,
    MaximumPlanarSubgraphProblem,
)

# Create input graph
graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1), (3, 4), (1, 4)])

# Initialize problem and solver
problem = MaximumPlanarSubgraphProblem(graph)
solver = GreedySolver(problem)

# Run the solver
result = solver.solve(max_iterations=100)

print(f"Objective: {result.best_objective}")
print(f"Converged: {result.converged} ({result.convergence_reason})")
print(f"Iterations: {result.iteration}")
```

See [docs/getting-started.md](docs/getting-started.md) for a detailed walkthrough.

---

## Configuration

ΔSearch has zero runtime dependencies and requires no configuration. All settings are passed via constructor arguments.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DELTA_SEARCH_LOG_LEVEL` | `WARNING` | Log verbosity |
| `DELTA_SEARCH_SEED` | *(unset)* | Default RNG seed for all solvers |

No environment variables are required for core functionality.

---

## API

### Core Types

| Symbol | Type | Description |
|--------|------|-------------|
| `Graph[Node]` | class | Adjacency-set graph with O(1) lookups |
| `ThreadSafeGraph[Node]` | class | Thread-safe wrapper with `RLock` |
| `Action` | class | A single candidate mutation (add/remove node/edge) |
| `ActionType` | enum | `ADD_NODE`, `REMOVE_NODE`, `ADD_EDGE`, `REMOVE_EDGE` |
| `DeltaResult` | dataclass | `(reward_change, penalty_change, feasible)` |
| `SubgraphState` | protocol | State objects (must have `.graph`, `.metrics`) |
| `SolverObserver` | protocol | Observer protocol for solver lifecycle events |
| `GreedySolver` | class | Greedy optimization loop |
| `SolverState` | dataclass | Solver progress snapshot |
| `EarlyTerminationCondition` | class | Configurable stopping criteria |

### SubgraphExtractionProblem Methods

| Method | Required | Description |
|--------|----------|-------------|
| `evaluate_initial_state(graph)` | Yes | Generate starting candidate subgraph |
| `calculate_delta(state, action)` | Yes | Compute incremental objective change |
| `compute_reward(state)` | Yes | Scalar reward (higher = better) |
| `compute_penalty(state)` | Yes | Scalar penalty (lower = better) |
| `is_feasible(state)` | Yes | Check hard constraints |
| `enumerate_actions(state)` | No | Generate candidate actions |
| `generate_composite_actions(state)` | No | Add compound actions |
| `apply_action(state, action)` | No | Apply mutation with undo support |
| `undo_action(state)` | No | Reverse last action |

### Built-in Problems

| Problem | Class | Type |
|---------|-------|------|
| Maximum Planar Subgraph | `MaximumPlanarSubgraphProblem` | Monotone |
| Minimum Connected Dominating Set | `MinimumConnectedDominatingSetProblem` | Monotone |
| Maximum Weight Independent Set | `MaximumWeightedIndependentSetProblem` | Monotone |
| Prize Collecting Vertex Cover | `PrizeCollectingVertexCoverProblem` | Non-monotone |
| Uncapacitated Facility Location | `UncapacitatedFacilityLocationProblem` | Non-monotone |
| Minimum Weighted Steiner Tree | `MinimumWeightedSteinerTreeProblem` | Non-monotone |

### Advanced Solvers

| Solver | Description |
|--------|-------------|
| `MultiStartSolver` | Runs multiple random starts, returns best result |
| `BeamSearchSolver` | Maintains top-k candidates per iteration |
| `AnytimeSolver` | Tracks progress over time for anytime algorithms |
| `LearnedGuidanceSolver` | Uses online ML to guide search |
| `AdaptiveBeamSolver` | Diversity-aware beam selection |
| `MultiObjectiveSolver` | Pareto-optimal multi-objective optimization |

---

## Examples

### Maximum Planar Subgraph

```python
from delta_search import Graph, GreedySolver, MaximumPlanarSubgraphProblem

graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1), (3, 4), (1, 4)])
problem = MaximumPlanarSubgraphProblem(graph)
result = GreedySolver(problem).solve(max_iterations=100)
print(f"Best objective: {result.best_objective}")
```

### Multi-start with adaptive beam search

```python
from delta_search import Graph, AdaptiveBeamSolver, MaximumPlanarSubgraphProblem

graph = Graph[str].from_edges([("a", "b"), ("b", "c"), ("c", "a")])
problem = MaximumPlanarSubgraphProblem(graph)
solver = AdaptiveBeamSolver(problem, beam_width=8, diversity_weight=0.3)
result = solver.solve(max_iterations=200)
```

### NetworkX interop

```python
import networkx as nx
from delta_search.interop import from_networkx

g_nx = nx.erdos_renyi_graph(50, 0.1, seed=0)
graph = from_networkx(g_nx)
```

---

## Project Structure

```
delta-search/
├── delta_search/              # Main package
│   ├── __init__.py            # Public API exports
│   ├── graph.py               # Graph data structures
│   ├── problem.py             # Abstract problem interface
│   ├── problems.py            # 6 concrete problem implementations
│   ├── solver.py              # Greedy solver engine
│   ├── incremental.py         # Incremental data structures
│   ├── multistart.py          # Multi-start solver
│   ├── beam.py                # Beam search solver
│   ├── anytime.py             # Anytime solver
│   ├── learned.py             # Learned guidance solver
│   ├── adaptive_beam.py       # Adaptive beam search
│   ├── multi_objective.py     # Multi-objective optimization
│   ├── streaming.py           # Streaming graph mutations
│   ├── benchmarks.py          # Benchmark suite
│   ├── visualization.py       # Plotting and export utilities
│   ├── progress.py            # Progress bar and streaming output
│   ├── context_engineering.py # RAG context selection
│   ├── test_time_compute.py   # Reasoning tree expansion
│   ├── budget_metrics.py      # Budget-aware evaluation
│   ├── hybrid_pipeline.py     # Two-stage retrieval + reasoning
│   ├── ablation.py            # Ablation study utilities
│   ├── theory.py              # Theoretical analysis
│   ├── utils.py               # Graph utility functions
│   ├── interop.py             # NetworkX conversion
│   ├── io.py                  # JSON file I/O
│   ├── cli.py                 # Command-line interface
│   └── py.typed               # PEP 561 marker
├── tests/                     # Test suite (320+ tests)
├── docs/                      # Documentation
├── .github/                   # GitHub configuration
│   ├── workflows/
│   │   ├── ci.yml             # CI pipeline
│   │   └── release.yml        # PyPI release automation
│   └── ISSUE_TEMPLATE/
├── .pre-commit-config.yaml
└── pyproject.toml             # Package configuration
```

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
mypy delta_search/
ruff check .
ruff format .
pip-audit
```

---

## Testing

```bash
pytest                                  # Full suite
pytest tests/test_graph.py -v           # Specific module
pytest --cov=delta_search               # With coverage
```

---

## Build

```bash
python -m build
```

---

## Release

1. Bump version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit with a `version:X.Y.Z` message
4. Tag and push — CI publishes to PyPI

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Type Checker | mypy (strict mode) |
| Linter | ruff |
| Test Framework | pytest |
| CI/CD | GitHub Actions |
| Dependencies | None (pure Python) |

---

## Roadmap

- [x] Implement all 6 problem types from the paper
- [x] GreedySolver with early termination conditions
- [x] NetworkX graph interop
- [x] Thread-safe graph variant
- [x] Observer protocol for solver lifecycle
- [x] JSON file I/O
- [x] Command-line interface
- [x] PEP 561 typed package
- [x] CI with lint, typecheck, security scan, tests
- [x] PyPI release automation
- [x] Benchmark suite against paper results
- [x] Multi-start / randomized solver
- [x] Visualization utilities
- [x] Progress bar / streaming output
- [x] Context engineering for RAG
- [x] Test-time compute for reasoning
- [x] Multi-objective optimization
- [x] Learned heuristic guidance
- [x] Adaptive beam search
- [x] Hybrid pipeline
- [ ] GPU acceleration for large graphs
- [ ] Distributed computing support
- [ ] Web-based visualization dashboard

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to uphold its standards.

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

## Citation

If you use this **algorithm** in your research, please cite the original paper by Arasu and Gupta:

```bibtex
@article{arasu2026deltasearch,
  title={Solving Subgraph Extraction Problems Using $\Delta$Search},
  author={Arasu, Rebin Silva Valan and Gupta, Rajiv},
  journal={arXiv preprint arXiv:2606.13834},
  year={2026}
}
```

If you use this **software** (the Python implementation), please also credit [Sachin (sachncs)](https://github.com/sachncs).

## License

[MIT](LICENSE) © 2026 Sachin