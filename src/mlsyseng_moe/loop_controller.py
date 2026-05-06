"""State convergence loop controller for iterative expert refinement.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
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
    """Represents a single iteration's state vector."""
    iteration: int
    metrics: dict[str, float]
    expert_outputs: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> list[float]:
        return sorted(self.metrics.values())

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "metrics": self.metrics,
            "expert_outputs": self.expert_outputs,
            "timestamp": self.timestamp,
        }


def l2_norm_diff(v1: list[float], v2: list[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0.0] * (max_len - len(v1))
        v2 = v2 + [0.0] * (max_len - len(v2))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@dataclass
class ConvergenceLoop:
    """Runs an iterative loop until state converges or limits are reached."""

    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3
    objective: str = "minimize_validation_loss"

    _history: list[LoopState] = field(default_factory=list, repr=False)
    _converging_count: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._history.clear()
        self._converging_count = 0

    @property
    def history(self) -> list[dict]:
        return [s.to_dict() for s in self._history]

    @property
    def current_iteration(self) -> int:
        return len(self._history)

    def _check_convergence(self, current: LoopState, previous: LoopState) -> tuple[bool, float]:
        v_curr = current.to_vector()
        v_prev = previous.to_vector()
        diff = l2_norm_diff(v_curr, v_prev)

        if diff < self.epsilon:
            self._converging_count += 1
        else:
            self._converging_count = 0

        converged = self._converging_count >= self.patience
        return converged, diff

    def step(self, metrics: dict[str, float], expert_outputs: list[dict] | None = None) -> dict:
        """Execute one iteration step.

        Returns dict with iteration info, convergence status, and whether to stop.
        """
        state = LoopState(
            iteration=self.current_iteration,
            metrics=metrics,
            expert_outputs=expert_outputs or [],
        )
        self._history.append(state)

        if len(self._history) < 2:
            return {
                "iteration": state.iteration,
                "metrics": metrics,
                "converged": False,
                "should_stop": False,
                "l2_diff": None,
                "converging_count": 0,
                "reason": "first_iteration",
            }

        previous = self._history[-2]
        converged, diff = self._check_convergence(state, previous)

        if state.iteration >= self.max_iterations:
            return {
                "iteration": state.iteration,
                "metrics": metrics,
                "converged": False,
                "should_stop": True,
                "l2_diff": diff,
                "converging_count": self._converging_count,
                "reason": "max_iterations_reached",
            }

        return {
            "iteration": state.iteration,
            "metrics": metrics,
            "converged": converged,
            "should_stop": converged,
            "l2_diff": diff,
            "converging_count": self._converging_count,
            "reason": "converged" if converged else "running",
        }

    def run(
        self,
        step_fn: Callable[[int, dict | None], dict[str, float]],
        initial_metrics: dict[str, float] | None = None,
    ) -> dict:
        """Run the full convergence loop.

        Args:
            step_fn: Called each iteration with (iteration_number, previous_metrics).
                     Must return a dict of metric_name -> float.
            initial_metrics: Optional starting metrics for the first iteration.

        Returns:
            Summary dict with final state, history, and convergence info.
        """
        self.reset()
        prev_metrics = initial_metrics

        for i in range(self.max_iterations + 1):
            metrics = step_fn(i, prev_metrics)
            result = self.step(metrics)

            logger.info(
                "Iteration %d: l2_diff=%s converging=%d/%d",
                i,
                result.get("l2_diff"),
                self._converging_count,
                self.patience,
            )

            if result["should_stop"]:
                return {
                    "final_iteration": i,
                    "final_metrics": metrics,
                    "converged": result["converged"],
                    "reason": result["reason"],
                    "total_iterations": i + 1,
                    "history": self.history,
                }
            prev_metrics = metrics

        return {
            "final_iteration": self.max_iterations,
            "final_metrics": prev_metrics or {},
            "converged": False,
            "reason": "max_iterations_reached",
            "total_iterations": self.max_iterations + 1,
            "history": self.history,
        }


def create_loop(config: dict | None = None) -> ConvergenceLoop:
    """Factory: create a ConvergenceLoop from a config dict."""
    cfg = config or {}
    return ConvergenceLoop(
        epsilon=cfg.get("epsilon", 0.001),
        max_iterations=cfg.get("max_iterations", 10),
        patience=cfg.get("patience", 3),
        objective=cfg.get("objective", "minimize_validation_loss"),
    )
