"""State convergence loop controller for MLSysEng MoE system.

Implements the iterative convergence loop with exit condition:
  ||state[n] - state[n-1]||_2 < epsilon

Each iteration refines the competition entry by consulting experts
and applying their recommended skills.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StateVector:
    """Represents a state in the convergence loop."""

    iteration: int
    scores: Dict[str, float] = field(default_factory=dict)
    features_count: int = 0
    models_tried: int = 0
    best_score: float = 0.0
    validation_loss: float = float("inf")
    expert_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> List[float]:
        """Convert state to a numeric vector for norm computation."""
        components = [
            self.features_count / 100.0,
            self.models_tried / 10.0,
            self.best_score,
            min(self.validation_loss, 10.0) / 10.0,
        ]
        for _, contrib in sorted(self.expert_contributions.items()):
            components.append(contrib)
        return components

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "scores": self.scores,
            "features_count": self.features_count,
            "models_tried": self.models_tried,
            "best_score": self.best_score,
            "validation_loss": self.validation_loss,
            "expert_contributions": self.expert_contributions,
            "timestamp": self.timestamp,
        }


def l2_norm_diff(v1: List[float], v2: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    max_len = max(len(v1), len(v2))
    padded_v1 = v1 + [0.0] * (max_len - len(v1))
    padded_v2 = v2 + [0.0] * (max_len - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(padded_v1, padded_v2)))


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""

    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3
    exit_condition: str = "||state[n] - state[n-1]||_2 < epsilon"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopConfig":
        return cls(
            objective=d.get("objective", "minimize_validation_loss"),
            epsilon=d.get("epsilon", 0.001),
            max_iterations=d.get("max_iterations", 10),
            patience=d.get("patience", 3),
            exit_condition=d.get("exit_condition", "||state[n] - state[n-1]||_2 < epsilon"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "exit_condition": self.exit_condition,
        }


class LoopController:
    """Controls the state convergence loop for competition entry building.

    The loop iterates through expert consultations, applying skills and
    tracking state changes. It exits when the L2 norm of the state
    difference falls below epsilon for `patience` consecutive iterations.
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.states: List[StateVector] = []
        self.convergence_history: List[float] = []
        self._converging_count = 0

    @property
    def current_iteration(self) -> int:
        return len(self.states)

    @property
    def has_converged(self) -> bool:
        return self._converging_count >= self.config.patience

    @property
    def should_stop(self) -> bool:
        if self.current_iteration >= self.config.max_iterations:
            return True
        return self.has_converged

    def record_state(self, state: StateVector) -> Dict[str, Any]:
        """Record a new state and check convergence."""
        self.states.append(state)

        if len(self.states) < 2:
            return {
                "iteration": state.iteration,
                "converged": False,
                "norm_diff": None,
                "message": "First iteration, no convergence check",
            }

        prev = self.states[-2].to_vector()
        curr = self.states[-1].to_vector()
        norm_diff = l2_norm_diff(prev, curr)
        self.convergence_history.append(norm_diff)

        if norm_diff < self.config.epsilon:
            self._converging_count += 1
        else:
            self._converging_count = 0

        return {
            "iteration": state.iteration,
            "norm_diff": norm_diff,
            "epsilon": self.config.epsilon,
            "converging_count": self._converging_count,
            "patience": self.config.patience,
            "converged": self.has_converged,
            "should_stop": self.should_stop,
            "message": self._convergence_message(norm_diff),
        }

    def _convergence_message(self, norm_diff: float) -> str:
        if self.has_converged:
            return (
                f"Converged: ||state[n] - state[n-1]||_2 = {norm_diff:.6f} < "
                f"epsilon = {self.config.epsilon} for {self._converging_count} "
                f"consecutive iterations (patience = {self.config.patience})"
            )
        if norm_diff < self.config.epsilon:
            return (
                f"Below epsilon ({norm_diff:.6f} < {self.config.epsilon}), "
                f"converging count: {self._converging_count}/{self.config.patience}"
            )
        if self.current_iteration >= self.config.max_iterations:
            return f"Max iterations reached ({self.config.max_iterations})"
        return (
            f"Not converged: ||state[n] - state[n-1]||_2 = {norm_diff:.6f} >= "
            f"epsilon = {self.config.epsilon}"
        )

    def run(
        self,
        step_fn: Callable[[int, Optional[StateVector]], StateVector],
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop.

        Args:
            step_fn: Function that takes (iteration_number, previous_state)
                     and returns a new StateVector.
            on_step: Optional callback called after each step with convergence info.

        Returns:
            Summary of the loop execution.
        """
        while not self.should_stop:
            iteration = self.current_iteration
            prev_state = self.states[-1] if self.states else None

            try:
                new_state = step_fn(iteration, prev_state)
                new_state.iteration = iteration
            except Exception as e:
                logger.error(f"Step {iteration} failed: {e}")
                break

            result = self.record_state(new_state)

            if on_step:
                on_step(result)

            if result["should_stop"]:
                break

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loop execution."""
        return {
            "total_iterations": len(self.states),
            "converged": self.has_converged,
            "max_iterations": self.config.max_iterations,
            "epsilon": self.config.epsilon,
            "patience": self.config.patience,
            "convergence_history": self.convergence_history,
            "final_state": self.states[-1].to_dict() if self.states else None,
            "objective": self.config.objective,
        }

    def reset(self):
        """Reset the controller for a new run."""
        self.states.clear()
        self.convergence_history.clear()
        self._converging_count = 0
