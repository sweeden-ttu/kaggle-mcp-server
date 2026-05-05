"""State convergence loop controller for MLSysEng MoE system.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state of a convergence iteration."""

    iteration: int
    metrics: Dict[str, float]
    state_vector: List[float]
    converged: bool = False
    delta_norm: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "metrics": self.metrics,
            "state_vector": self.state_vector,
            "converged": self.converged,
            "delta_norm": self.delta_norm,
        }


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""

    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopConfig":
        return cls(
            objective=d.get("objective", "minimize_validation_loss"),
            epsilon=d.get("epsilon", 0.001),
            max_iterations=d.get("max_iterations", 10),
            patience=d.get("patience", 3),
        )


def l2_norm(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


class LoopController:
    """Controls the state convergence loop for expert-driven competition building."""

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: List[LoopState] = []
        self._consecutive_converging = 0

    def reset(self):
        self.history.clear()
        self._consecutive_converging = 0

    def step(
        self,
        state_vector: List[float],
        metrics: Dict[str, float],
    ) -> LoopState:
        """Execute one iteration step and check convergence."""
        iteration = len(self.history)

        if iteration == 0:
            state = LoopState(
                iteration=iteration,
                metrics=metrics,
                state_vector=state_vector,
                converged=False,
                delta_norm=float("inf"),
            )
            self.history.append(state)
            return state

        prev = self.history[-1]
        delta = l2_norm(state_vector, prev.state_vector)

        if delta < self.config.epsilon:
            self._consecutive_converging += 1
        else:
            self._consecutive_converging = 0

        converged = self._consecutive_converging >= self.config.patience

        state = LoopState(
            iteration=iteration,
            metrics=metrics,
            state_vector=state_vector,
            converged=converged,
            delta_norm=delta,
        )
        self.history.append(state)

        logger.info(
            "Iteration %d: delta=%.6f, converged=%s, patience=%d/%d",
            iteration,
            delta,
            converged,
            self._consecutive_converging,
            self.config.patience,
        )
        return state

    def should_stop(self) -> bool:
        """Check if the loop should stop (converged or max iterations reached)."""
        if not self.history:
            return False
        latest = self.history[-1]
        if latest.converged:
            return True
        if latest.iteration >= self.config.max_iterations - 1:
            return True
        return False

    def run(
        self,
        step_fn: Callable[[int, Optional[LoopState]], tuple],
        initial_state: Optional[List[float]] = None,
    ) -> List[LoopState]:
        """Run the full convergence loop.

        Args:
            step_fn: Callable that takes (iteration, prev_state) and returns
                     (state_vector, metrics) tuple.
            initial_state: Optional initial state vector for iteration 0.
        """
        self.reset()

        for i in range(self.config.max_iterations):
            prev = self.history[-1] if self.history else None
            state_vector, metrics = step_fn(i, prev)
            state = self.step(state_vector, metrics)

            if self.should_stop():
                logger.info(
                    "Loop stopped at iteration %d (converged=%s)",
                    state.iteration,
                    state.converged,
                )
                break

        return self.history

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loop execution."""
        if not self.history:
            return {"status": "not_started"}

        final = self.history[-1]
        return {
            "total_iterations": len(self.history),
            "converged": final.converged,
            "final_delta_norm": final.delta_norm,
            "final_metrics": final.metrics,
            "epsilon": self.config.epsilon,
            "patience": self.config.patience,
            "objective": self.config.objective,
            "history": [s.to_dict() for s in self.history],
        }
