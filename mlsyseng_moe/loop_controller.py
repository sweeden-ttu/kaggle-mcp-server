"""State convergence loop controller for the MLSysEng MoE system.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents a single state in the convergence loop."""
    iteration: int
    metrics: dict
    state_vector: np.ndarray
    timestamp: float = field(default_factory=time.time)
    expert_outputs: list = field(default_factory=list)


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    exit_condition: str = "||state[n] - state[n-1]||_2 < epsilon"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3
    metric_weights: dict = field(default_factory=lambda: {"accuracy": 1.0, "f1_score": 1.0})


class ConvergenceLoop:
    """
    Manages the state convergence loop for iterative expert-driven optimization.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: list[LoopState] = []
        self.converging_count: int = 0
        self.converged: bool = False
        self.best_state: Optional[LoopState] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ConvergenceLoop":
        """Create a ConvergenceLoop from a dictionary configuration."""
        config = LoopConfig(
            objective=config_dict.get("objective", "minimize_validation_loss"),
            exit_condition=config_dict.get("exit_condition", "||state[n] - state[n-1]||_2 < epsilon"),
            epsilon=config_dict.get("epsilon", 0.001),
            max_iterations=config_dict.get("max_iterations", 10),
            patience=config_dict.get("patience", 3),
        )
        return cls(config)

    def metrics_to_vector(self, metrics: dict) -> np.ndarray:
        """Convert a metrics dictionary to a weighted state vector."""
        values = []
        for key in sorted(metrics.keys()):
            weight = self.config.metric_weights.get(key, 1.0)
            values.append(float(metrics[key]) * weight)
        return np.array(values, dtype=np.float64)

    def compute_l2_distance(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """Compute L2 norm of state vector difference."""
        if state_a.shape != state_b.shape:
            min_len = min(len(state_a), len(state_b))
            state_a = state_a[:min_len]
            state_b = state_b[:min_len]
        return float(np.linalg.norm(state_a - state_b, ord=2))

    def check_convergence(self, current_state: LoopState) -> tuple[bool, float]:
        """
        Check if the loop has converged.

        Returns (converged, distance) tuple.
        """
        if len(self.history) < 1:
            return False, float("inf")

        prev_state = self.history[-1]
        distance = self.compute_l2_distance(
            current_state.state_vector,
            prev_state.state_vector,
        )

        if distance < self.config.epsilon:
            self.converging_count += 1
        else:
            self.converging_count = 0

        converged = self.converging_count >= self.config.patience
        return converged, distance

    def update(self, metrics: dict, expert_outputs: Optional[list] = None) -> dict:
        """
        Update the loop with new metrics and check convergence.

        Returns a status dict with convergence info.
        """
        state_vector = self.metrics_to_vector(metrics)
        iteration = len(self.history)

        current_state = LoopState(
            iteration=iteration,
            metrics=metrics,
            state_vector=state_vector,
            expert_outputs=expert_outputs or [],
        )

        converged, distance = self.check_convergence(current_state)
        self.history.append(current_state)

        if self._is_better(current_state):
            self.best_state = current_state

        if converged:
            self.converged = True

        max_reached = iteration >= self.config.max_iterations - 1

        return {
            "iteration": iteration,
            "distance": distance,
            "epsilon": self.config.epsilon,
            "converging_count": self.converging_count,
            "patience": self.config.patience,
            "converged": converged,
            "max_iterations_reached": max_reached,
            "should_stop": converged or max_reached,
            "metrics": metrics,
        }

    def _is_better(self, state: LoopState) -> bool:
        """Check if a state is better than the current best."""
        if self.best_state is None:
            return True

        objective = self.config.objective
        if "minimize" in objective:
            primary_metric = list(state.metrics.values())[0] if state.metrics else float("inf")
            best_metric = list(self.best_state.metrics.values())[0] if self.best_state.metrics else float("inf")
            return primary_metric < best_metric
        else:
            primary_metric = list(state.metrics.values())[0] if state.metrics else 0
            best_metric = list(self.best_state.metrics.values())[0] if self.best_state.metrics else 0
            return primary_metric > best_metric

    def get_summary(self) -> dict:
        """Get a summary of the convergence loop state."""
        distances = []
        for i in range(1, len(self.history)):
            d = self.compute_l2_distance(
                self.history[i].state_vector,
                self.history[i - 1].state_vector,
            )
            distances.append(d)

        return {
            "total_iterations": len(self.history),
            "converged": self.converged,
            "converging_count": self.converging_count,
            "config": {
                "objective": self.config.objective,
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "patience": self.config.patience,
            },
            "distances": distances,
            "best_iteration": self.best_state.iteration if self.best_state else None,
            "best_metrics": self.best_state.metrics if self.best_state else None,
        }

    def reset(self) -> None:
        """Reset the convergence loop."""
        self.history = []
        self.converging_count = 0
        self.converged = False
        self.best_state = None


def run_convergence_loop(
    step_fn: Callable[[int, Optional[dict]], dict],
    config: Optional[LoopConfig] = None,
    initial_metrics: Optional[dict] = None,
) -> dict:
    """
    Run the full convergence loop with a step function.

    Args:
        step_fn: Function that takes (iteration, previous_metrics) and returns new metrics.
        config: Loop configuration.
        initial_metrics: Optional initial metrics to seed the loop.

    Returns:
        Summary of the convergence loop execution.
    """
    loop = ConvergenceLoop(config)
    prev_metrics = initial_metrics

    for i in range(loop.config.max_iterations):
        try:
            metrics = step_fn(i, prev_metrics)
        except Exception as e:
            logger.error(f"Step function failed at iteration {i}: {e}")
            break

        status = loop.update(metrics)
        logger.info(
            f"Iteration {i}: distance={status['distance']:.6f}, "
            f"converging={status['converging_count']}/{loop.config.patience}"
        )

        if status["should_stop"]:
            break

        prev_metrics = metrics

    return loop.get_summary()
