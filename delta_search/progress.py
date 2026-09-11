"""Progress bar and streaming output for the delta search solver.

Provides observer implementations that display real-time progress during
solver execution.  Supports both tqdm-based progress bars and plain-text
streaming output.

Usage::

    from delta_search import Graph, GreedySolver, MaximumPlanarSubgraphProblem
    from delta_search.progress import TqdmObserver, StreamingObserver

    graph = Graph[int].from_edges([(1, 2), (2, 3), (3, 1)])
    problem = MaximumPlanarSubgraphProblem(graph)
    solver = GreedySolver(problem)

    # tqdm progress bar
    result = solver.solve(max_iterations=100, observer=TqdmObserver())

    # or plain-text streaming
    result = solver.solve(max_iterations=100, observer=StreamingObserver())
"""

from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .problem import Action, DeltaResult

logger = logging.getLogger(__name__)

__all__ = [
    "TqdmObserver",
    "StreamingObserver",
    "CallbackObserver",
]


def _require_tqdm() -> Any:
    """Check that tqdm is installed.

    Returns:
        The tqdm module.

    Raises:
        ImportError: If tqdm is not installed.

    """
    import importlib.util

    if importlib.util.find_spec("tqdm") is None:
        raise ImportError(
            "tqdm is required for progress bar display.  "
            "Install it with: pip install tqdm"
        )
    import tqdm

    return tqdm


class TqdmObserver:
    """Observer that displays a tqdm progress bar.

    Requires the ``tqdm`` package.  Shows iteration progress with
    objective value and timing information.

    Args:
        max_iterations: Total iterations for the progress bar.
        file: Output stream (default: stderr).
        desc: Description prefix for the progress bar.

    """

    def __init__(
        self,
        max_iterations: int = 1000,
        file: Any = None,
        desc: str = "Solving",
    ) -> None:
        """Initialize the tqdm progress bar observer.

        Args:
            max_iterations: Total iterations for the progress bar.
            file: Output stream (default: stderr).
            desc: Description prefix for the progress bar.

        """
        self.max_iterations = max_iterations
        self.file = file or sys.stderr
        self.desc = desc
        self._pbar: Any = None
        self._start_time: float = 0.0

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
        if self._pbar is not None:
            elapsed = time.monotonic() - self._start_time
            self._pbar.update(1)
            self._pbar.set_postfix(
                {
                    "obj": f"{objective:.2f}",
                    "time": f"{elapsed:.1f}s",
                }
            )

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """Called when the solver terminates.

        Args:
            iterations: Total iterations completed.
            final_objective: Best objective value found.

        """
        if self._pbar is not None:
            self._pbar.set_postfix(
                {
                    "obj": f"{final_objective:.2f}",
                    "status": "done",
                }
            )
            self._pbar.close()

    def start(self) -> None:
        """Initialize the progress bar.  Call before solve()."""
        tqdm = _require_tqdm()
        self._start_time = time.monotonic()
        self._pbar = tqdm.tqdm(
            total=self.max_iterations,
            desc=self.desc,
            file=self.file,
            unit="iter",
        )

    def close(self) -> None:
        """Close the progress bar."""
        if self._pbar is not None:
            self._pbar.close()


class StreamingObserver:
    """Observer that prints solver progress to a stream.

    Outputs one line per iteration with iteration number, objective,
    and elapsed time.  Useful for logging or non-tty environments.

    Args:
        file: Output stream (default: stderr).
        verbose: If True, also print action evaluation details.
        log_file: Optional file path to write log output.

    """

    def __init__(
        self,
        file: Any = None,
        verbose: bool = False,
        log_file: str | None = None,
    ) -> None:
        """Initialize the streaming observer.

        Args:
            file: Output stream (default: stderr).
            verbose: If True, also print action evaluation details.
            log_file: Optional file path to write log output.

        """
        self.file = file or sys.stderr
        self.verbose = verbose
        self._start_time: float = 0.0
        self._log_handle: Any = None
        self._log_file = log_file

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
        if self.verbose:
            msg = (
                f"  [eval] {action.action_type.name} "
                f"{action.targets} -> "
                f"delta={delta.reward_change:+.2f} "
                f"feasible={delta.feasible}"
            )
            self._write(msg)

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
        elapsed = time.monotonic() - self._start_time
        action_str = (
            f"{best_action.action_type.name} {best_action.targets}"
            if best_action
            else "none"
        )
        msg = (
            f"[iter {iteration:>5}] "
            f"obj={objective:>10.4f}  "
            f"action={action_str:<20}  "
            f"time={elapsed:.2f}s"
        )
        self._write(msg)

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """Called when the solver terminates.

        Args:
            iterations: Total iterations completed.
            final_objective: Best objective value found.

        """
        elapsed = time.monotonic() - self._start_time
        msg = (
            f"[done] iterations={iterations}  "
            f"objective={final_objective:.4f}  "
            f"elapsed={elapsed:.2f}s"
        )
        self._write(msg)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def start(self) -> None:
        """Initialize timing.  Call before solve()."""
        self._start_time = time.monotonic()
        if self._log_file is not None:
            self.close()  # close any prior handle before opening a new log file
            self._log_handle = open(self._log_file, "w", encoding="utf-8")  # noqa: SIM115

    def _write(self, msg: str) -> None:
        """Write message to output stream and optional log file.

        Args:
            msg: The message to write.

        """
        logger.info(msg)
        if self._log_handle is not None:
            self._log_handle.write(msg + "\n")
            self._log_handle.flush()

    def close(self) -> None:
        """Close the log file handle if open."""
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> StreamingObserver:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit — close log handle."""
        self.close()

    def __del__(self) -> None:
        """Ensure log file handle is closed on garbage collection."""
        self.close()


class CallbackObserver:
    """Observer that calls user-provided callbacks for each event.

    Args:
        on_eval: Called on each action evaluation.
        on_iter: Called on each iteration completion.
        on_done: Called on convergence.

    """

    def __init__(
        self,
        on_eval: Any = None,
        on_iter: Any = None,
        on_done: Any = None,
    ) -> None:
        """Initialize the callback observer.

        Args:
            on_eval: Called on each action evaluation.
            on_iter: Called on each iteration completion.
            on_done: Called on convergence.

        """
        self._on_eval = on_eval
        self._on_iter = on_iter
        self._on_done = on_done

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
        if self._on_eval is not None:
            self._on_eval(action, delta, elapsed_ms)

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
        if self._on_iter is not None:
            self._on_iter(iteration, best_action, objective)

    def on_convergence(self, iterations: int, final_objective: float) -> None:
        """Called when the solver terminates.

        Args:
            iterations: Total iterations completed.
            final_objective: Best objective value found.

        """
        if self._on_done is not None:
            self._on_done(iterations, final_objective)
