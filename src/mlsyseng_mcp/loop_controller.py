"""State convergence loop controller for the MLSysEng MoE system.

Implements iterative refinement with L2 norm convergence detection:
  exit_condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class LoopState:
    """Tracks the state vector and metadata for one iteration."""
    iteration: int = 0
    state_vector: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    converged: bool = False
    notes: str = ""


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3


def l2_norm(a: list[float], b: list[float]) -> float:
    """Compute the L2 norm (Euclidean distance) between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if not a:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class LoopController:
    """Manages the state convergence loop for competition entries.

    The loop runs an iteration function repeatedly until:
    - The L2 norm between consecutive state vectors drops below epsilon
    - The max_iterations limit is reached
    - Patience is exhausted (consecutive converging iterations)
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: list[LoopState] = []
        self._converge_count = 0

    @classmethod
    def from_expert_config(cls, loop_config_json: str) -> "LoopController":
        """Create a LoopController from an expert's loop_config JSON."""
        data = json.loads(loop_config_json)
        config = LoopConfig(
            objective=data.get("objective", "minimize_validation_loss"),
            epsilon=data.get("epsilon", 0.001),
            max_iterations=data.get("max_iterations", 10),
            patience=data.get("patience", 3),
        )
        return cls(config)

    def should_continue(self) -> bool:
        """Check whether the loop should continue iterating."""
        if len(self.history) >= self.config.max_iterations:
            return False
        if self._converge_count >= self.config.patience:
            return False
        return True

    def record_state(self, state_vector: list[float], metrics: Optional[dict] = None, notes: str = "") -> LoopState:
        """Record a new state and check convergence."""
        state = LoopState(
            iteration=len(self.history),
            state_vector=list(state_vector),
            metrics=metrics or {},
            timestamp=time.time(),
            notes=notes,
        )

        if self.history:
            prev = self.history[-1]
            delta = l2_norm(state.state_vector, prev.state_vector)
            state.metrics["l2_delta"] = delta

            if delta < self.config.epsilon:
                self._converge_count += 1
                state.converged = True
            else:
                self._converge_count = 0
                state.converged = False
        else:
            state.metrics["l2_delta"] = float("inf")

        self.history.append(state)
        return state

    def run(
        self,
        iterate_fn: Callable[[int, Optional[LoopState]], tuple[list[float], dict]],
    ) -> dict:
        """Run the convergence loop with the given iteration function.

        iterate_fn(iteration_number, previous_state) -> (state_vector, metrics)
        """
        while self.should_continue():
            prev = self.history[-1] if self.history else None
            iteration = len(self.history)

            state_vector, metrics = iterate_fn(iteration, prev)
            state = self.record_state(state_vector, metrics)

            if not self.should_continue():
                break

        return self.get_summary()

    def get_summary(self) -> dict:
        """Return a summary of the loop execution."""
        total_iterations = len(self.history)
        final_state = self.history[-1] if self.history else None

        deltas = [
            s.metrics.get("l2_delta", float("inf"))
            for s in self.history
            if "l2_delta" in s.metrics and s.metrics["l2_delta"] != float("inf")
        ]

        return {
            "objective": self.config.objective,
            "total_iterations": total_iterations,
            "converged": self._converge_count >= self.config.patience,
            "final_l2_delta": deltas[-1] if deltas else None,
            "min_l2_delta": min(deltas) if deltas else None,
            "epsilon": self.config.epsilon,
            "patience": self.config.patience,
            "patience_count": self._converge_count,
            "final_metrics": final_state.metrics if final_state else {},
            "history": [
                {
                    "iteration": s.iteration,
                    "l2_delta": s.metrics.get("l2_delta"),
                    "converged": s.converged,
                    "metrics": {k: v for k, v in s.metrics.items() if k != "l2_delta"},
                }
                for s in self.history
            ],
        }

    def reset(self):
        """Reset the controller for a fresh run."""
        self.history.clear()
        self._converge_count = 0


def build_competition_state_vector(metrics: dict) -> list[float]:
    """Convert a metrics dict into a numeric state vector for convergence tracking.

    Standard ordering: [accuracy, f1, auc, precision, recall, loss]
    Missing values default to 0.0.
    """
    keys = ["accuracy", "f1_score", "auc_roc", "precision", "recall", "loss"]
    return [float(metrics.get(k, 0.0)) for k in keys]
