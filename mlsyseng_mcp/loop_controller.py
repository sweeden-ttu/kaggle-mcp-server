"""State convergence loop for the MoE system.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents a single iteration's state vector."""

    iteration: int
    metrics: Dict[str, float]
    expert_outputs: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> List[float]:
        return sorted(self.metrics.values())


def l2_norm_diff(v1: List[float], v2: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0.0] * (max_len - len(v1))
        v2 = v2 + [0.0] * (max_len - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@dataclass
class ConvergenceConfig:
    """Configuration for the convergence loop."""

    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConvergenceConfig":
        return cls(
            objective=d.get("objective", "minimize_validation_loss"),
            epsilon=d.get("epsilon", 0.001),
            max_iterations=d.get("max_iterations", 10),
            patience=d.get("patience", 3),
        )


class LoopController:
    """Manages the state convergence loop for expert-driven competition building."""

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        self.config = config or ConvergenceConfig()
        self.history: List[LoopState] = []
        self._converging_count = 0

    def reset(self) -> None:
        self.history.clear()
        self._converging_count = 0

    def check_convergence(self, current: LoopState) -> Dict[str, Any]:
        """
        Check if the loop has converged.

        Returns dict with:
          - converged: bool
          - l2_diff: float (nan for first iteration)
          - converging_count: int
          - reason: str
        """
        if not self.history:
            self.history.append(current)
            return {
                "converged": False,
                "l2_diff": float("nan"),
                "converging_count": 0,
                "reason": "first_iteration",
            }

        prev = self.history[-1]
        v_curr = current.to_vector()
        v_prev = prev.to_vector()
        diff = l2_norm_diff(v_curr, v_prev)

        self.history.append(current)

        if diff < self.config.epsilon:
            self._converging_count += 1
        else:
            self._converging_count = 0

        converged = self._converging_count >= self.config.patience

        if converged:
            reason = f"converged: {self._converging_count} consecutive iterations below epsilon={self.config.epsilon}"
        elif current.iteration >= self.config.max_iterations:
            reason = f"max_iterations_reached: {self.config.max_iterations}"
            converged = True
        else:
            reason = "continuing"

        return {
            "converged": converged,
            "l2_diff": diff,
            "converging_count": self._converging_count,
            "reason": reason,
        }

    def run(
        self,
        step_fn: Callable[[int, Optional[LoopState]], LoopState],
        on_step: Optional[Callable[[int, LoopState, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the convergence loop.

        Args:
            step_fn: Called each iteration with (iteration, previous_state) -> new_state.
                      Must return a LoopState with updated metrics.
            on_step: Optional callback after each step for logging/reporting.

        Returns:
            Summary dict with final state, convergence info, and history.
        """
        self.reset()
        prev_state: Optional[LoopState] = None

        for iteration in range(1, self.config.max_iterations + 1):
            state = step_fn(iteration, prev_state)
            state.iteration = iteration

            convergence = self.check_convergence(state)

            if on_step:
                on_step(iteration, state, convergence)

            logger.info(
                "Iteration %d: l2_diff=%.6f converging=%d/%d",
                iteration,
                convergence["l2_diff"] if not math.isnan(convergence["l2_diff"]) else -1,
                convergence["converging_count"],
                self.config.patience,
            )

            if convergence["converged"]:
                break

            prev_state = state

        final_state = self.history[-1] if self.history else None
        return {
            "total_iterations": len(self.history),
            "converged": convergence["converged"] if self.history else False,
            "final_metrics": final_state.metrics if final_state else {},
            "convergence_reason": convergence.get("reason", "unknown"),
            "history": [
                {
                    "iteration": s.iteration,
                    "metrics": s.metrics,
                    "l2_diff": l2_norm_diff(
                        s.to_vector(),
                        self.history[i - 1].to_vector() if i > 0 else s.to_vector(),
                    ),
                }
                for i, s in enumerate(self.history)
            ],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loop execution."""
        if not self.history:
            return {"status": "not_started", "iterations": 0}

        return {
            "status": "completed",
            "iterations": len(self.history),
            "final_metrics": self.history[-1].metrics,
            "converging_count": self._converging_count,
            "config": {
                "objective": self.config.objective,
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "patience": self.config.patience,
            },
        }
