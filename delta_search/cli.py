"""Command-line interface for delta search.

Usage::

    python -m delta_search.cli solve --problem mps --graph input.json
    python -m delta_search.cli solve --problem mcds --graph input.json --max-iter 500
    python -m delta_search.cli validate --graph input.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from .io import load_graph
from .solver import EarlyTerminationCondition, GreedySolver

logger = logging.getLogger(__name__)

__all__ = [
    "import_problem",
    "build_parser",
    "add_solve_arguments",
    "add_validate_arguments",
    "cmd_solve",
    "cmd_validate",
    "main",
]

PROBLEM_REGISTRY: dict[str, str] = {
    "mps": "delta_search.problems.MaximumPlanarSubgraphProblem",
    "mcds": "delta_search.problems.MinimumConnectedDominatingSetProblem",
    "mwis": "delta_search.problems.MaximumWeightedIndependentSetProblem",
    "pcvc": "delta_search.problems.PrizeCollectingVertexCoverProblem",
    "uflp": "delta_search.problems.UncapacitatedFacilityLocationProblem",
    "mwst": "delta_search.problems.MinimumWeightedSteinerTreeProblem",
}


def import_problem(name: str) -> type:
    """Dynamically import a problem class by short name.

    Args:
        name: Short problem identifier (e.g. ``"mps"``).

    Returns:
        The problem class.

    Raises:
        ValueError: If ``name`` is not in the registry.
        ImportError: If the module cannot be imported.

    """
    if name not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem: {name!r}.  "
            f"Available: {', '.join(sorted(PROBLEM_REGISTRY))}"
        )
    module_path, class_name = PROBLEM_REGISTRY[name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise ImportError(f"{PROBLEM_REGISTRY[name]} is not a class")
    return cls


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        Configured ArgumentParser with solve and validate subcommands.

    """
    parser = argparse.ArgumentParser(
        prog="delta-search",
        description="Delta search: solve NP-hard subgraph extraction problems.",
    )
    subparsers = parser.add_subparsers(dest="command")

    add_solve_arguments(subparsers)
    add_validate_arguments(subparsers)

    return parser


def add_solve_arguments(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the solve subcommand arguments.

    Args:
        subparsers: The subparsers action to add to.

    """
    solve_parser = subparsers.add_parser(
        "solve",
        help="Run the solver on a graph.",
    )
    solve_parser.add_argument(
        "--problem",
        "-p",
        required=True,
        choices=list(PROBLEM_REGISTRY.keys()),
        help="Problem type to solve.",
    )
    solve_parser.add_argument(
        "--graph",
        "-g",
        required=True,
        help="Path to input graph (JSON).",
    )
    solve_parser.add_argument(
        "--max-iterations",
        "-n",
        type=int,
        default=1000,
        help="Maximum iterations (default: 1000).",
    )
    solve_parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Maximum action evaluations.",
    )
    solve_parser.add_argument(
        "--stall",
        type=int,
        default=None,
        help="Stop after N iterations with no improvement.",
    )
    solve_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for solution (JSON).",
    )
    solve_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress to stderr.",
    )


def add_validate_arguments(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the validate subcommand arguments.

    Args:
        subparsers: The subparsers action to add to.

    """
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a graph file.",
    )
    validate_parser.add_argument(
        "--graph",
        "-g",
        required=True,
        help="Path to input graph (JSON).",
    )


def cmd_solve(args: argparse.Namespace) -> int:
    """Execute the solve command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for success).

    """
    graph = load_graph(args.graph)
    problem_cls = import_problem(args.problem)
    problem = problem_cls(graph)

    early_stop: EarlyTerminationCondition[int] = EarlyTerminationCondition(
        max_evaluations=args.max_evaluations,
        stall_iterations=args.stall,
    )

    solver = GreedySolver(problem, early_stop=early_stop)

    if args.verbose:
        logger.info(
            f"Problem: {args.problem}  "
            f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges",
        )

    result = solver.solve(max_iterations=args.max_iterations)

    output = {
        "problem": args.problem,
        "iterations": result.iteration,
        "objective": result.best_objective,
        "converged": result.converged,
        "convergence_reason": result.convergence_reason,
        "evaluations": result.total_evaluations,
        "elapsed_ms": round(result.elapsed_ms, 2),
    }

    if result.best_state is not None:
        best_graph = result.best_state.graph
        output["solution"] = {
            "nodes": list(best_graph.nodes),
            "num_edges": best_graph.num_edges,
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Solution written to {args.output}")
    else:
        logger.info(json.dumps(output, indent=2, default=str))

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the validate command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 for valid graph, 1 for invalid).

    """
    try:
        graph = load_graph(args.graph)
        logger.info(
            f"Valid graph: {graph.num_nodes} nodes, {graph.num_edges} edges",
        )
        return 0
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error(f"Invalid graph: {e}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.  Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error or no command).

    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        return cmd_solve(args)
    if args.command == "validate":
        return cmd_validate(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
