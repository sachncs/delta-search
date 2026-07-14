# Deployment Guide

This guide covers deploying ΔSearch in various environments.

## Installation

### From PyPI

```bash
pip install delta-search
```

### From Source

```bash
git clone https://github.com/sachncs/delta-search.git
cd delta-search
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
pre-commit install
```

## Production Considerations

### Memory Management

ΔSearch uses adjacency sets for O(1) lookups. For very large graphs:

- Monitor memory usage with `tracemalloc` or system tools
- Consider chunking extremely large graphs (100K+ nodes)
- Use `ThreadSafeGraph` only when concurrent access is required

### Thread Safety

```python
from delta_search import ThreadSafeGraph, GreedySolver

# ThreadSafeGraph wraps all mutations with RLock
graph = ThreadSafeGraph[int]()
graph.add_edge(1, 2)

# Each solver instance must have its own problem instance
# Graph can be shared via ThreadSafeGraph
```

### Performance Tuning

```python
from delta_search import GreedySolver, EarlyTerminationCondition

# Configure early stopping for production workloads
early_stop = EarlyTerminationCondition(
    max_iterations=1000,
    max_evaluations=10000,
    max_time_ms=30000,  # 30 second timeout
    stall_iterations=50,  # Stop if no improvement for 50 iterations
)

solver = GreedySolver(problem, early_stop=early_stop)
```

## Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir delta-search

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "from delta_search import Graph; print('Ready')"]
```

## Cloud Deployment

### AWS Lambda

```python
import json
from delta_search import Graph, MaximumPlanarSubgraphProblem, GreedySolver

def lambda_handler(event, context):
    graph = Graph[int]()
    for edge in event.get("edges", []):
        graph.add_edge(edge[0], edge[1])

    problem = MaximumPlanarSubgraphProblem(graph)
    solver = GreedySolver(problem)
    result = solver.solve(max_iterations=100)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "objective": result.best_objective,
            "iterations": result.iteration,
        })
    }
```

### Google Cloud Functions

Similar to AWS Lambda, deploy as a Python function with HTTP trigger.

## Monitoring

### Logging

```python
import logging
from delta_search import GreedySolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ΔSearch uses Python logging throughout
solver = GreedySolver(problem)
result = solver.solve(max_iterations=100)
logger.info(f"Completed: objective={result.best_objective}")
```

### Metrics Collection

```python
from delta_search import SolverObserver, DeltaResult, Action

class MetricsObserver:
    def __init__(self):
        self.evaluations = 0
        self.improvements = 0

    def on_action_evaluated(self, action, delta, elapsed_ms):
        self.evaluations += 1
        if delta.reward_change > 0:
            self.improvements += 1

    def on_iteration_complete(self, iteration, best_action, objective):
        pass

    def on_convergence(self, iterations, final_objective):
        pass

observer = MetricsObserver()
solver.solve(max_iterations=100, observer=observer)
print(f"Total evaluations: {observer.evaluations}")
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `RecursionError` on large graphs | Increase `sys.setrecursionlimit()` or use iterative approaches |
| Slow performance | Ensure you're using `calculate_delta` (incremental) not full re-evaluation |
| Type errors with `mypy --strict` | Use `DefaultState[NodeT]` instead of `SubgraphState` directly |
| `ImportError` for optional deps | Install with `pip install delta-search[all]` |

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed solver progress
from delta_search import StreamingObserver
observer = StreamingObserver(verbose=True)
solver.solve(max_iterations=100, observer=observer)
```
