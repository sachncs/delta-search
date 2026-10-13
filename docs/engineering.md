# Engineering standards

DeltaSearch follows a small set of enforceable engineering rules.

## Design principles

1. **Make invalid states unrepresentable.** Validate public inputs at the boundary and use domain-specific exceptions.
2. **Separate policy from mechanism.** Problems define domain behavior; solvers define search policy; graphs define storage and mutation.
3. **Preserve reversibility.** Every mutation must either be immutable or have a tested, exact inverse.
4. **Make ownership explicit.** Results and states returned to callers must not alias internal solver state unless documented.
5. **Prefer deterministic behavior.** Stable ordering and explicit seeds are required for reproducible experiments.
6. **Measure before optimizing.** Performance claims must have a benchmark and must distinguish delta evaluation from state-copy cost.
7. **Keep optional integrations optional.** NetworkX, matplotlib, and scikit-learn code paths must fail with actionable errors and have dedicated tests.

## Python conventions

- Public names use `snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE` according to their role.
- A single leading underscore marks semi-private implementation details: `_problem`, `_state`, `_score_actions`.
- Double leading underscores are reserved for name-mangling cases; they are not a general privacy mechanism.
- Public APIs require Google-style docstrings with `Args`, `Returns`, and `Raises` sections where applicable.
- Mutable state is kept behind methods or explicit dataclasses; callers must not receive live internal dictionaries or views without documentation.
- New code must pass Ruff lint/format, strict mypy, tests, and the coverage threshold.

## Extension model

`SubgraphExtractionProblem` is the domain polymorphism boundary. New problems implement the abstract contract and should not duplicate solver loops. Search behavior belongs in a solver strategy, mutation behavior belongs in graph/state commands, and observability belongs in observers.
