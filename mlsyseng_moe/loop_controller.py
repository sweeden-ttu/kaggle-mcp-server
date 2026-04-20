"""State convergence loop controller for MLSysEng MoE.

Implements the iterative convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

Used to iteratively refine competition entries until convergence.
"""

import json
import logging
import math
from dataclasses import dataclass, field
import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class StateVector:
    """Represents the current state of a competition entry iteration."""
    iteration: int = 0
    validation_loss: float = float("inf")
    accuracy: float = 0.0
    f1_score: float = 0.0
    features_count: int = 0
    model_score: float = 0.0
    timestamp: str = ""

    def to_vector(self) -> list[float]:
        return [
            self.validation_loss,
            self.accuracy,
            self.f1_score,
            float(self.features_count),
            self.model_score,
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "StateVector":
        return cls(
            iteration=d.get("iteration", 0),
            validation_loss=d.get("validation_loss", float("inf")),
            accuracy=d.get("accuracy", 0.0),
            f1_score=d.get("f1_score", 0.0),
            features_count=d.get("features_count", 0),
            model_score=d.get("model_score", 0.0),
            timestamp=d.get("timestamp", ""),
        )

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "validation_loss": self.validation_loss,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "features_count": self.features_count,
            "model_score": self.model_score,
            "timestamp": self.timestamp,
        }


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        return cls(
            objective=d.get("objective", "minimize_validation_loss"),
            epsilon=d.get("epsilon", 0.001),
            max_iterations=int(d.get("max_iterations", 10)),
            patience=int(d.get("patience", 3)),
        )


@dataclass
class ConvergenceResult:
    """Result of a convergence loop execution."""
    converged: bool = False
    iterations_run: int = 0
    final_state: Optional[StateVector] = None
    history: list[StateVector] = field(default_factory=list)
    convergence_deltas: list[float] = field(default_factory=list)
    exit_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "iterations_run": self.iterations_run,
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "convergence_deltas": self.convergence_deltas,
            "exit_reason": self.exit_reason,
            "history": [s.to_dict() for s in self.history],
        }


def l2_norm(v1: list[float], v2: list[float]) -> float:
    """Compute L2 norm of vector difference: ||v1 - v2||_2."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class LoopController:
    """Controls the state convergence loop for iterative refinement."""

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: list[StateVector] = []
        self.convergence_deltas: list[float] = []
        self._patience_counter = 0

    def check_convergence(self, current: StateVector, previous: StateVector) -> tuple[bool, float]:
        """Check if the state has converged.

        Returns (converged, delta) where delta is the L2 norm of the state difference.
        """
        current_vec = current.to_vector()
        previous_vec = previous.to_vector()
        delta = l2_norm(current_vec, previous_vec)
        converged = delta < self.config.epsilon
        return converged, delta

    def step(self, state: StateVector) -> tuple[bool, str]:
        """Process one iteration step.

        Returns (should_stop, reason).
        """
        state.iteration = len(self.history)
        state.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.history.append(state)

        if len(self.history) >= self.config.max_iterations:
            return True, f"max_iterations_reached ({self.config.max_iterations})"

        if len(self.history) < 2:
            return False, "need_more_data"

        previous = self.history[-2]
        converged, delta = self.check_convergence(state, previous)
        self.convergence_deltas.append(delta)

        if converged:
            self._patience_counter += 1
            if self._patience_counter >= self.config.patience:
                return True, f"converged (delta={delta:.6f} < epsilon={self.config.epsilon}, patience={self._patience_counter})"
            return False, f"converging (delta={delta:.6f}, patience={self._patience_counter}/{self.config.patience})"
        else:
            self._patience_counter = 0
            return False, f"not_converged (delta={delta:.6f})"

    def run(
        self,
        step_fn: Callable[[int, Optional[StateVector]], StateVector],
    ) -> ConvergenceResult:
        """Run the full convergence loop.

        Args:
            step_fn: Function that takes (iteration, previous_state) and returns new StateVector.
        """
        result = ConvergenceResult()

        for i in range(self.config.max_iterations):
            previous = self.history[-1] if self.history else None

            try:
                new_state = step_fn(i, previous)
            except Exception as e:
                logger.error(f"Step {i} failed: {e}")
                result.exit_reason = f"step_failed: {e}"
                break

            should_stop, reason = self.step(new_state)
            logger.info(f"Iteration {i}: {reason}")

            if should_stop:
                result.exit_reason = reason
                result.converged = "converged" in reason
                break
        else:
            result.exit_reason = "max_iterations_exhausted"

        result.iterations_run = len(self.history)
        result.final_state = self.history[-1] if self.history else None
        result.history = list(self.history)
        result.convergence_deltas = list(self.convergence_deltas)

        return result

    def reset(self):
        """Reset the controller state."""
        self.history.clear()
        self.convergence_deltas.clear()
        self._patience_counter = 0
