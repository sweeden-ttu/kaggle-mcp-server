"""State convergence loop controller for iterative expert refinement."""

import math
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute the L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def initial_state_vector(expert: Dict[str, Any]) -> List[float]:
    """Generate an initial state vector for an expert's competition entry.

    The state vector tracks: [loss, primary_metric, secondary_metric, iteration_frac].
    """
    return [1.0, 0.0, 0.0, 0.0]


class LoopController:
    """Manages the state convergence loop: ||state[n] - state[n-1]||_2 < epsilon."""

    def __init__(
        self,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.history: List[List[float]] = []
        self.converge_streak: int = 0

    @classmethod
    def from_expert_config(cls, expert: Dict[str, Any]) -> "LoopController":
        config = expert.get("loop_config", {})
        return cls(
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
        )

    def step(self, state: List[float]) -> Dict[str, Any]:
        """Record a new state vector and check convergence.

        Returns dict with keys: converged, delta, iteration, should_stop, reason.
        """
        iteration = len(self.history)
        self.history.append(list(state))

        if iteration == 0:
            return {
                "converged": False,
                "delta": None,
                "iteration": iteration,
                "should_stop": False,
                "reason": "initial state",
            }

        prev = self.history[-2]
        delta = l2_norm(state, prev)
        below_eps = delta < self.epsilon

        if below_eps:
            self.converge_streak += 1
        else:
            self.converge_streak = 0

        converged = self.converge_streak >= self.patience
        at_max = iteration >= self.max_iterations - 1

        should_stop = converged or at_max
        if converged:
            reason = f"Converged: {self.converge_streak} consecutive iterations below epsilon={self.epsilon}"
        elif at_max:
            reason = f"Reached max iterations ({self.max_iterations})"
        else:
            reason = f"delta={delta:.6f}, streak={self.converge_streak}/{self.patience}"

        return {
            "converged": converged,
            "delta": delta,
            "iteration": iteration,
            "should_stop": should_stop,
            "reason": reason,
        }

    def run(
        self,
        step_fn: Callable[[int, List[float]], List[float]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run the full convergence loop.

        Args:
            step_fn: Called with (iteration, current_state) -> new_state.
            initial_state: Starting state vector.

        Returns:
            Final status dict with full history.
        """
        state = initial_state or [1.0, 0.0, 0.0, 0.0]
        result = self.step(state)

        while not result["should_stop"]:
            iteration = result["iteration"] + 1
            state = step_fn(iteration, state)
            result = self.step(state)

        return {
            **result,
            "history": self.history,
            "total_iterations": len(self.history),
        }
