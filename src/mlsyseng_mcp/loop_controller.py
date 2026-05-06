"""State convergence loop for iterative expert-driven optimization.

Implements the convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state vector at a given iteration."""
    iteration: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    actions_taken: List[str] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Convert metrics dict to a sorted, deterministic vector."""
        return [self.metrics[k] for k in sorted(self.metrics.keys())]


@dataclass
class ConvergenceResult:
    """Result of a convergence loop run."""
    converged: bool
    iterations: int
    final_state: Optional[LoopState]
    history: List[LoopState]
    delta_history: List[float]
    reason: str


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class LoopController:
    """Controls the state convergence loop for expert-driven optimization."""

    def __init__(self, epsilon: float = 0.001, max_iterations: int = 10,
                 patience: int = 3):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.history: List[LoopState] = []
        self.delta_history: List[float] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoopController":
        return cls(
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
        )

    def check_convergence(self, current: LoopState,
                          previous: Optional[LoopState]) -> bool:
        """Check if ||state[n] - state[n-1]||_2 < epsilon."""
        if previous is None:
            return False
        try:
            curr_vec = current.to_vector()
            prev_vec = previous.to_vector()
            if not curr_vec or not prev_vec:
                return False
            delta = l2_norm(curr_vec, prev_vec)
            self.delta_history.append(delta)
            logger.info("Iteration %d: delta=%.6f (epsilon=%.6f)",
                        current.iteration, delta, self.epsilon)
            return delta < self.epsilon
        except ValueError as e:
            logger.warning("Convergence check failed: %s", e)
            return False

    def run(self, step_fn: Callable[[int, Optional[LoopState]], LoopState],
            progress_fn: Optional[Callable[[int, LoopState, float], None]] = None
            ) -> ConvergenceResult:
        """Run the convergence loop.

        Args:
            step_fn: Called each iteration with (iteration_num, previous_state).
                     Must return the new LoopState.
            progress_fn: Optional callback (iteration, state, delta) for progress.

        Returns:
            ConvergenceResult with convergence status and history.
        """
        self.history = []
        self.delta_history = []
        consecutive_converging = 0
        previous: Optional[LoopState] = None

        for i in range(self.max_iterations):
            current = step_fn(i, previous)
            current.iteration = i
            self.history.append(current)

            if self.check_convergence(current, previous):
                consecutive_converging += 1
            else:
                consecutive_converging = 0

            delta = self.delta_history[-1] if self.delta_history else float("inf")
            if progress_fn:
                progress_fn(i, current, delta)

            if consecutive_converging >= self.patience:
                logger.info("Converged after %d iterations (patience=%d)",
                            i + 1, self.patience)
                return ConvergenceResult(
                    converged=True,
                    iterations=i + 1,
                    final_state=current,
                    history=self.history,
                    delta_history=self.delta_history,
                    reason=f"Converged: {self.patience} consecutive deltas < {self.epsilon}",
                )

            previous = current

        final = self.history[-1] if self.history else None
        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_state=final,
            history=self.history,
            delta_history=self.delta_history,
            reason=f"Max iterations ({self.max_iterations}) reached without convergence",
        )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the loop run."""
        return {
            "total_iterations": len(self.history),
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "delta_history": self.delta_history,
            "final_metrics": (
                self.history[-1].metrics if self.history else {}),
            "converged": (
                len(self.delta_history) >= self.patience and
                all(d < self.epsilon
                    for d in self.delta_history[-self.patience:])),
        }
