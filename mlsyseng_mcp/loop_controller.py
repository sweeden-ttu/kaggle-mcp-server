"""State convergence loop controller for MoE competition entries.

Implements the iterative refinement loop with L2 norm convergence check:
    exit_condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StateVector:
    """Represents the state of a competition entry at a point in time."""

    def __init__(self, values: List[float]):
        self.values = list(values)

    @property
    def dimension(self) -> int:
        return len(self.values)

    def l2_distance(self, other: "StateVector") -> float:
        """Compute L2 norm of the difference between two state vectors."""
        if self.dimension != other.dimension:
            raise ValueError(
                f"Dimension mismatch: {self.dimension} vs {other.dimension}"
            )
        diff_sq = sum((a - b) ** 2 for a, b in zip(self.values, other.values))
        return math.sqrt(diff_sq)

    def to_list(self) -> List[float]:
        return list(self.values)

    def __repr__(self) -> str:
        return f"StateVector({self.values})"


class LoopController:
    """Controls the state convergence loop for competition building.

    The loop runs an iterative refinement process where each iteration
    produces a new state vector. The loop exits when:
    - The L2 distance between consecutive states is < epsilon, OR
    - The maximum number of iterations is reached, OR
    - Patience is exhausted (consecutive converging iterations)
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

        self.state_history: List[StateVector] = []
        self.metric_history: List[Dict[str, float]] = []
        self._converging_count = 0
        self._converged = False
        self._iteration = 0

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def current_iteration(self) -> int:
        return self._iteration

    @property
    def current_state(self) -> Optional[StateVector]:
        return self.state_history[-1] if self.state_history else None

    def should_continue(self) -> bool:
        """Check whether the loop should continue iterating."""
        if self._converged:
            return False
        if self._iteration >= self.max_iterations:
            return False
        return True

    def update_state(self, new_state: List[float], metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Update the state with a new vector and check convergence.

        Returns a status dict with convergence information.
        """
        new_sv = StateVector(new_state)
        self._iteration += 1

        if metrics:
            self.metric_history.append(metrics)

        if self.state_history:
            prev_sv = self.state_history[-1]
            distance = new_sv.l2_distance(prev_sv)
            is_converging = distance < self.epsilon

            if is_converging:
                self._converging_count += 1
            else:
                self._converging_count = 0

            if self._converging_count >= self.patience:
                self._converged = True

            self.state_history.append(new_sv)

            return {
                "iteration": self._iteration,
                "l2_distance": distance,
                "epsilon": self.epsilon,
                "is_converging": is_converging,
                "converging_streak": self._converging_count,
                "patience": self.patience,
                "converged": self._converged,
                "should_continue": self.should_continue(),
                "metrics": metrics,
            }
        else:
            self.state_history.append(new_sv)
            return {
                "iteration": self._iteration,
                "l2_distance": None,
                "epsilon": self.epsilon,
                "is_converging": False,
                "converging_streak": 0,
                "patience": self.patience,
                "converged": False,
                "should_continue": True,
                "metrics": metrics,
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the loop execution."""
        distances = []
        for i in range(1, len(self.state_history)):
            d = self.state_history[i].l2_distance(self.state_history[i - 1])
            distances.append(d)

        return {
            "total_iterations": self._iteration,
            "converged": self._converged,
            "final_distance": distances[-1] if distances else None,
            "min_distance": min(distances) if distances else None,
            "max_distance": max(distances) if distances else None,
            "objective": self.objective,
            "epsilon": self.epsilon,
            "state_history": [sv.to_list() for sv in self.state_history],
            "metric_history": self.metric_history,
            "distances": distances,
        }

    def reset(self) -> None:
        """Reset the controller for a new run."""
        self.state_history = []
        self.metric_history = []
        self._converging_count = 0
        self._converged = False
        self._iteration = 0


def run_convergence_loop(
    step_fn: Callable[[int, Optional[StateVector]], Tuple[List[float], Dict[str, float]]],
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
    objective: str = "minimize_validation_loss",
) -> Dict[str, Any]:
    """Run the full convergence loop with a step function.

    Args:
        step_fn: A function that takes (iteration, current_state) and returns
                 (new_state_vector, metrics_dict).
        epsilon: Convergence threshold for L2 distance.
        max_iterations: Maximum number of iterations.
        patience: Number of consecutive converging iterations before exit.
        objective: Optimization objective name.

    Returns:
        Loop summary dict.
    """
    controller = LoopController(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
        objective=objective,
    )

    while controller.should_continue():
        current = controller.current_state
        new_state, metrics = step_fn(controller.current_iteration, current)
        status = controller.update_state(new_state, metrics)
        logger.info(
            "Iteration %d: L2=%.6f, converged=%s",
            status["iteration"],
            status["l2_distance"] or 0.0,
            status["converged"],
        )

    return controller.get_summary()
