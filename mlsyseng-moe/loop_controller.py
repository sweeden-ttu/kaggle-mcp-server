"""State convergence loop for iterative expert refinement."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateVector:
    """Represents a measurable state in the convergence loop."""

    def __init__(self, metrics: Dict[str, float]):
        self._metrics = dict(metrics)

    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    def to_array(self) -> np.ndarray:
        return np.array(sorted(self._metrics.values()), dtype=np.float64)

    def distance(self, other: "StateVector") -> float:
        """L2 norm of state vector difference."""
        a = self.to_array()
        b = other.to_array()
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))
        return float(np.linalg.norm(a - b))

    def __repr__(self) -> str:
        return f"StateVector({self._metrics})"


class LoopController:
    """
    Manages the state convergence loop.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
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
        self.history: List[StateVector] = []
        self._converge_count = 0

    def reset(self) -> None:
        self.history = []
        self._converge_count = 0

    def record_state(self, metrics: Dict[str, float]) -> StateVector:
        state = StateVector(metrics)
        self.history.append(state)
        return state

    def check_convergence(self) -> Tuple[bool, float]:
        """
        Check if the loop has converged.

        Returns (converged: bool, distance: float).
        """
        if len(self.history) < 2:
            return False, float("inf")

        current = self.history[-1]
        previous = self.history[-2]
        dist = current.distance(previous)

        if dist < self.epsilon:
            self._converge_count += 1
        else:
            self._converge_count = 0

        converged = self._converge_count >= self.patience
        return converged, dist

    def should_continue(self) -> bool:
        """Return True if the loop should keep running."""
        if len(self.history) >= self.max_iterations:
            logger.info("Max iterations (%d) reached", self.max_iterations)
            return False
        converged, dist = self.check_convergence()
        if converged:
            logger.info(
                "Converged after %d iterations (distance=%.6f < epsilon=%.6f, patience=%d)",
                len(self.history), dist, self.epsilon, self.patience,
            )
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        dist = float("inf")
        if len(self.history) >= 2:
            dist = self.history[-1].distance(self.history[-2])

        return {
            "iteration": len(self.history),
            "max_iterations": self.max_iterations,
            "current_distance": dist,
            "epsilon": self.epsilon,
            "converge_count": self._converge_count,
            "patience": self.patience,
            "converged": self._converge_count >= self.patience,
            "objective": self.objective,
        }

    def run(
        self,
        step_fn: Callable[[int, Optional[StateVector]], Dict[str, float]],
        on_step: Optional[Callable[[int, StateVector, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the convergence loop.

        Args:
            step_fn: Called each iteration with (iteration_number, previous_state).
                     Must return a dict of metric_name -> metric_value.
            on_step: Optional callback after each step with (iteration, state, status).

        Returns:
            Final status dict with full history.
        """
        self.reset()

        for i in range(self.max_iterations):
            prev = self.history[-1] if self.history else None
            metrics = step_fn(i, prev)
            state = self.record_state(metrics)
            status = self.get_status()

            if on_step:
                on_step(i, state, status)

            logger.info(
                "Iteration %d: distance=%.6f, metrics=%s",
                i, status["current_distance"], metrics,
            )

            if not self.should_continue():
                break

        final_status = self.get_status()
        final_status["history"] = [
            {"iteration": i, "metrics": s.metrics}
            for i, s in enumerate(self.history)
        ]
        return final_status


def create_competition_loop(
    competition: str,
    loop_config: Optional[Dict[str, Any]] = None,
) -> LoopController:
    """Create a configured LoopController for a competition."""
    config = loop_config or {}
    return LoopController(
        epsilon=config.get("epsilon", 0.001),
        max_iterations=config.get("max_iterations", 10),
        patience=config.get("patience", 3),
        objective=config.get("objective", "minimize_validation_loss"),
    )
