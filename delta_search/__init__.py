"""Delta search -- a general heuristic framework for subgraph extraction.

This package provides the foundational data structures and abstract
interfaces described in "Solving Subgraph Extraction Problems Using
Delta Search" (arXiv:2606.13834).

Public API:
    Graph: Optimized adjacency-set graph representation.
    ThreadSafeGraph: Thread-safe wrapper around Graph.
    Action: A single candidate mutation (add/remove node/edge).
    ActionType: Enum of primitive mutation kinds.
    DeltaResult: Return type of calculate_delta.
    SubgraphExtractionProblem: Abstract base class for concrete problems.
    SubgraphState: Protocol for state objects.
    SolverObserver: Observer protocol for solver lifecycle events.
    GreedySolver: Greedy optimization loop.
    MultiStartSolver: Multi-start solver for improved solution quality.
    BeamSearchSolver: Beam search solver for parallel candidate evaluation.
    AnytimeSolver: Anytime solver with best-so-far tracking.
    StreamingSolver: Dynamic graph streaming solver.
    MultiObjectiveSolver: Multi-objective Pareto frontier solver.
    LearnedGuidanceSolver: ML-guided search solver.
    AdaptiveBeamSolver: Adaptive ordering beam ΔSearch.
    AblationStudy: Ablation study framework.
    ScalingStudy: Scaling analysis framework.
    SubmodularAnalyzer: Theoretical guarantees analyzer.
    ContextEngineeringSolver: ΔSearch for RAG context selection.
    TestTimeComputeSolver: ΔSearch for reasoning tree expansion.
    BudgetAwareEvaluator: Quality-per-token metrics.
    HybridPipeline: Two-stage retrieval + reasoning pipeline.
    SolverState: Solver progress snapshot.
    EarlyTerminationCondition: Configurable stopping criteria.
    DefaultState: Generic mutable state for problems.
    ProblemType: Monotone vs non-monotone classification.

Submodules:
    benchmarks: Benchmark suite for comparing against paper results.
    visualization: Plotting and export utilities.
    progress: Progress bar and streaming output observers.
    beam: Beam search solver.
    anytime: Anytime search solver.
    streaming: Dynamic graph streaming solver.
    multi_objective: Multi-objective Pareto frontier solver.
    learned: ML-guided search solver.
    incremental: Incremental computation data structures.
    ablation: Ablation study and scaling analysis.
    theory: Theoretical guarantees for submodular problems.
    context_engineering: ΔSearch for RAG context selection.
    test_time_compute: ΔSearch for reasoning tree expansion.
    budget_metrics: Budget-aware quality metrics.
    adaptive_beam: Adaptive ordering beam ΔSearch.
    hybrid_pipeline: Two-stage retrieval + reasoning pipeline.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .ablation import AblationStudy, ScalingStudy
from .adaptive_beam import AdaptiveBeamSolver
from .anytime import AnytimeResult, AnytimeSolver
from .beam import BeamSearchResult, BeamSearchSolver
from .budget_metrics import BudgetAwareEvaluator, BudgetMetric
from .context_engineering import ContextEngineeringSolver
from .graph import Graph, ThreadSafeGraph
from .hybrid_pipeline import HybridPipeline
from .learned import LearnedGuidanceSolver
from .multi_objective import MultiObjectiveSolver, ObjectiveWeights
from .multistart import MultiStartResult, MultiStartSolver
from .problem import (
    Action,
    ActionType,
    DefaultState,
    DeltaResult,
    NullObserver,
    SolverObserver,
    SubgraphExtractionProblem,
    SubgraphState,
    UndoEntry,
)
from .problems import (
    MaximumPlanarSubgraphProblem,
    MaximumWeightedIndependentSetProblem,
    MinimumConnectedDominatingSetProblem,
    MinimumWeightedSteinerTreeProblem,
    PrizeCollectingVertexCoverProblem,
    ProblemType,
    UncapacitatedFacilityLocationProblem,
)
from .solver import EarlyTerminationCondition, GreedySolver, SolverState
from .streaming import StreamingResult, StreamingSolver
from .test_time_compute import TestTimeComputeSolver
from .theory import SubmodularAnalyzer

__all__ = [
    "Graph",
    "ThreadSafeGraph",
    "Action",
    "ActionType",
    "DeltaResult",
    "NullObserver",
    "SolverObserver",
    "SubgraphExtractionProblem",
    "SubgraphState",
    "UndoEntry",
    "GreedySolver",
    "MultiStartSolver",
    "MultiStartResult",
    "BeamSearchSolver",
    "BeamSearchResult",
    "AnytimeSolver",
    "AnytimeResult",
    "StreamingSolver",
    "StreamingResult",
    "MultiObjectiveSolver",
    "ObjectiveWeights",
    "LearnedGuidanceSolver",
    "AdaptiveBeamSolver",
    "BudgetAwareEvaluator",
    "BudgetMetric",
    "ContextEngineeringSolver",
    "TestTimeComputeSolver",
    "HybridPipeline",
    "AblationStudy",
    "ScalingStudy",
    "SubmodularAnalyzer",
    "SolverState",
    "EarlyTerminationCondition",
    "DefaultState",
    "ProblemType",
    "MaximumPlanarSubgraphProblem",
    "MinimumConnectedDominatingSetProblem",
    "MaximumWeightedIndependentSetProblem",
    "PrizeCollectingVertexCoverProblem",
    "UncapacitatedFacilityLocationProblem",
    "MinimumWeightedSteinerTreeProblem",
]

try:
    __version__ = version("delta-search")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
