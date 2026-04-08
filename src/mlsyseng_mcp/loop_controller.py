"""State convergence loop controller for MLSysEng MoE system.

Implements iterative refinement with L2-norm convergence checking.
Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state vector at an iteration."""

    iteration: int
    metrics: dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> list[float]:
        """Convert metrics dict to a sorted vector for comparison."""
        return [v for _, v in sorted(self.metrics.items())]

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class ConvergenceResult:
    """Result of the convergence loop."""

    converged: bool
    iterations_run: int
    final_state: LoopState | None
    history: list[dict]
    l2_norms: list[float]
    reason: str


def l2_norm(v1: list[float], v2: list[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        raise ValueError(
            f"Vector dimension mismatch: {len(v1)} vs {len(v2)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class LoopController:
    """Controls the state convergence loop for expert-driven competition building.

    The loop runs an iteration function repeatedly, tracking a state vector
    of metrics. It exits when the L2 norm of the state difference falls
    below epsilon for `patience` consecutive iterations, or when
    max_iterations is reached.
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        objective: str = "minimize_validation_loss",
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.objective = objective
        self.history: list[LoopState] = []
        self.l2_norms: list[float] = []

    @classmethod
    def from_config(cls, config: dict) -> "LoopController":
        """Create a LoopController from an expert's loop_config."""
        return cls(
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
            objective=config.get("objective", "minimize_validation_loss"),
        )

    def check_convergence(
        self, current: LoopState, previous: LoopState
    ) -> tuple[float, bool]:
        """Check if states have converged.

        Returns (l2_norm_value, is_converging).
        """
        v_curr = current.to_vector()
        v_prev = previous.to_vector()

        if not v_curr or not v_prev:
            return float("inf"), False

        try:
            norm = l2_norm(v_curr, v_prev)
        except ValueError:
            return float("inf"), False

        return norm, norm < self.epsilon

    def run(
        self,
        iteration_fn: Callable[[int, LoopState | None], dict[str, float]],
        initial_metrics: dict[str, float] | None = None,
    ) -> ConvergenceResult:
        """Run the convergence loop.

        Args:
            iteration_fn: Called each iteration with (iteration_number, previous_state).
                          Must return a dict of metric_name -> metric_value.
            initial_metrics: Optional initial metrics to seed the loop.

        Returns:
            ConvergenceResult with convergence status and history.
        """
        self.history = []
        self.l2_norms = []
        converging_count = 0

        if initial_metrics:
            initial = LoopState(iteration=0, metrics=initial_metrics)
            self.history.append(initial)

        for i in range(1, self.max_iterations + 1):
            prev = self.history[-1] if self.history else None

            try:
                metrics = iteration_fn(i, prev)
            except Exception as e:
                logger.error("Iteration %d failed: %s", i, e)
                return ConvergenceResult(
                    converged=False,
                    iterations_run=i,
                    final_state=prev,
                    history=[s.to_dict() for s in self.history],
                    l2_norms=self.l2_norms,
                    reason=f"Iteration {i} failed: {e}",
                )

            current = LoopState(iteration=i, metrics=metrics)
            self.history.append(current)

            if prev is not None:
                norm, is_converging = self.check_convergence(current, prev)
                self.l2_norms.append(norm)
                logger.info(
                    "Iteration %d: L2 norm = %.6f (epsilon = %.6f, converging = %s)",
                    i,
                    norm,
                    self.epsilon,
                    is_converging,
                )

                if is_converging:
                    converging_count += 1
                    if converging_count >= self.patience:
                        return ConvergenceResult(
                            converged=True,
                            iterations_run=i,
                            final_state=current,
                            history=[s.to_dict() for s in self.history],
                            l2_norms=self.l2_norms,
                            reason=(
                                f"Converged after {i} iterations "
                                f"({converging_count} consecutive under epsilon)"
                            ),
                        )
                else:
                    converging_count = 0

        final = self.history[-1] if self.history else None
        return ConvergenceResult(
            converged=False,
            iterations_run=self.max_iterations,
            final_state=final,
            history=[s.to_dict() for s in self.history],
            l2_norms=self.l2_norms,
            reason=f"Max iterations ({self.max_iterations}) reached without convergence",
        )

    def get_status(self) -> dict:
        """Return current loop status."""
        return {
            "iterations_completed": len(self.history),
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "objective": self.objective,
            "l2_norms": self.l2_norms,
            "converging": (
                len(self.l2_norms) > 0 and self.l2_norms[-1] < self.epsilon
            ),
        }
