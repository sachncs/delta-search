# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `attach_observer` context manager so solvers do not mutate the
  problem's observer list across calls
- `FanoutObserver` so multiple observers attached via `add_observer`
  all receive `on_iteration_complete` / `on_action_evaluated` /
  `on_convergence` events
- `apply_action(state, action, incremental=True)` opt-in to skip the
  default deepcopy and rely on the existing `undo_action` rollback
- `configure_logging()` and `default_seed()` honour
  `DELTA_SEARCH_LOG_LEVEL` and `DELTA_SEARCH_SEED` env vars
- `LearnedGuidanceSolver(..., random_state=...)` exposes the seed and
  logs a warning when sklearn is missing
- `requirements.txt` / `requirements-dev.txt` for resolver reproducibility
- `CODEOWNERS` for `/delta_search/`, `/tests/`, `/.github/workflows/`
- TqdmObserver gained `__enter__` / `__exit__` so its lifecycle matches
  StreamingObserver
- `__test__ = False` on `TestTimeCompute*` classes to silence pytest
  collection warnings
- `social-preview.png` asset in `.github/assets/` (upload via repo
  settings to apply)

### Changed

- `actions/setup-python` bumped from v6 to v7 so Python 3.14 builds
- `Graph.subgraph` rewritten to take advantage of set-membership to
  drop the per-node inner intersection from O(V'^2) to O(V' + E')
- `MWST` problem picks the BFS source from `min(terminals)` for
  hash-seed determinism
- `learned.py` uses an instance-level RNG seeded from `random_state`
- The library no longer mutates the root logger; CLI honours
  `--verbose` to enable DEBUG

### Fixed

- Removed `__del__` IO cleanup from `StreamingObserver` (unsafe at
  interpreter shutdown)
- All `open()` calls now pass `encoding="utf-8"`
- Replaced misleading PyPI badge (no release published) with install
  badge; cut v0.1.0 tag and release to unblock downstream
- Renamed `is_planary` to `is_planar` (correct spelling) and kept
  `is_planary` as a deprecated alias for backward compatibility
- `apply_action` docstring now spells out the deepcopy cost; opt-in
  `incremental=True` skips it
- LICENSE preamble reduced to standard MIT text; attribution moved to
  a new NOTICE file so GitHub correctly classifies the license as MIT
- Many docs URL / typo fixes (architecture, faq, getting-started)
- `Graph.save_graph` no longer mutates `node_attrs` of the source
  graph during serialization

- Context engineering module for RAG context selection under token budgets
- Test-time compute module for reasoning tree expansion under compute budgets
- Multi-objective optimization with Pareto frontier tracking
- Learned heuristic guidance with online model training
- Adaptive beam search with diversity-aware selection
- Hybrid pipeline for two-stage retrieval + reasoning
- Budget metrics for quality-per-token and quality-latency analysis
- Ablation study and scaling analysis utilities
- Theoretical approximation bounds and convergence analysis
- Progress bar (tqdm) and streaming observer output
- Streaming graph mutations with resume support
- Multi-start solver with random initial state generation
- `max_history` param for `AnytimeSolver` to cap snapshot memory
- `max_pareto_size` param for `MultiObjectiveSolver` to cap Pareto front
- Configurable sklearn hyperparams for `LearnedGuidanceSolver`
- Configurable seed for `AblationStudy` and `ScalingStudy` graph generation
- Context manager support for `StreamingObserver`
- Thread-safe wrappers for `ThreadSafeGraph.subgraph`, `edge_subgraph`,
  `node_list`, `degree_sequence`, `is_subgraph_of`

### Changed

- Refactored `SubgraphState` protocol to include `metrics` dict
- Made `DefaultState` generic over `NodeT` for type safety
- Replaced all `print()` calls with `logging` throughout codebase
- Improved encapsulation: private fields with controlled accessors
- Updated CI workflows to latest GitHub Actions versions

### Fixed

- CLI logging now configured with `basicConfig` so output is visible
- File handle leak in `StreamingObserver` guarded against double-start
- Removed unused `logger` definitions from 8 modules
- Moved inline `import random` to module top level in `learned.py`
- Removed hardcoded `Random(42)` in ablation graph generation

## [0.1.0] - 2026-06-15

### Added

- Initial release of the ΔSearch framework
- Core graph data structures with O(1) lookups and incremental mutation
- `ThreadSafeGraph` wrapper with `RLock` for concurrent access
- `SubgraphExtractionProblem` abstract base class with 5 required methods
- 6 concrete problem implementations:
  - Maximum Planar Subgraph (MPS)
  - Minimum Connected Dominating Set (MCDS)
  - Maximum Weighted Independent Set (MWIS)
  - Prize Collecting Vertex Cover (PCVC)
  - Uncapacitated Facility Location (UFLP)
  - Minimum Weighted Steiner Tree (MWST)
- `GreedySolver` with early termination conditions
- Undo-stack pattern with full rollback support
- `SolverObserver` protocol for lifecycle observability
- NetworkX graph interop via `to_networkx()` / `from_networkx()`
- JSON file I/O for graph serialization
- Command-line interface with solve and validate commands
- Full test suite (320 tests) with 80%+ coverage
- CI/CD pipeline with lint, typecheck, security scan, tests, and build
- PyPI release automation via GitHub Actions
- Documentation: getting-started, architecture, FAQ
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- PEP 561 typed package with `py.typed` marker
- Zero runtime dependencies
