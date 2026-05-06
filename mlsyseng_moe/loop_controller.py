"""State convergence loop for MLSysEng MoE system.

Implements the convergence condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateVector:
    """Represents a state in the convergence loop."""

    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics
        self._keys = sorted(metrics.keys())

    def to_array(self) -> np.ndarray:
        return np.array([self.metrics[k] for k in self._keys])

    @classmethod
    def from_array(cls, keys: List[str], values: np.ndarray) -> "StateVector":
        return cls(dict(zip(sorted(keys), values.tolist())))

    def distance(self, other: "StateVector") -> float:
        """Compute L2 norm of difference between two state vectors."""
        a = self.to_array()
        b = other.to_array()
        return float(np.linalg.norm(a - b))

    def __repr__(self) -> str:
        return f"StateVector({self.metrics})"


class LoopController:
    """Controls the state convergence loop for expert-guided optimization."""

    def __init__(
        self,
        objective: str = "minimize_validation_loss",
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.objective = objective
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.state_history: List[StateVector] = []
        self.convergence_history: List[float] = []
        self._converging_count = 0

    @classmethod
    def from_config(cls, loop_config: Dict[str, Any]) -> "LoopController":
        return cls(
            objective=loop_config.get("objective", "minimize_validation_loss"),
            epsilon=loop_config.get("epsilon", 0.001),
            max_iterations=loop_config.get("max_iterations", 10),
            patience=loop_config.get("patience", 3),
        )

    def reset(self):
        """Reset the loop controller state."""
        self.state_history = []
        self.convergence_history = []
        self._converging_count = 0

    def update(self, state: StateVector) -> Tuple[bool, Dict[str, Any]]:
        """
        Update the loop with a new state.

        Returns (should_continue, info_dict).
        """
        self.state_history.append(state)
        iteration = len(self.state_history)

        if iteration < 2:
            return True, {
                "iteration": iteration,
                "converged": False,
                "distance": None,
                "message": "First iteration, no convergence check possible",
            }

        prev_state = self.state_history[-2]
        distance = state.distance(prev_state)
        self.convergence_history.append(distance)

        is_converging = distance < self.epsilon
        if is_converging:
            self._converging_count += 1
        else:
            self._converging_count = 0

        converged = self._converging_count >= self.patience
        exceeded_max = iteration >= self.max_iterations

        should_continue = not converged and not exceeded_max

        info = {
            "iteration": iteration,
            "distance": distance,
            "epsilon": self.epsilon,
            "is_converging": is_converging,
            "converging_count": self._converging_count,
            "patience": self.patience,
            "converged": converged,
            "exceeded_max": exceeded_max,
            "should_continue": should_continue,
        }

        if converged:
            info["message"] = (
                f"Converged after {iteration} iterations "
                f"(distance={distance:.6f} < epsilon={self.epsilon} "
                f"for {self.patience} consecutive iterations)"
            )
        elif exceeded_max:
            info["message"] = f"Reached max iterations ({self.max_iterations})"
        else:
            info["message"] = (
                f"Iteration {iteration}: distance={distance:.6f}, "
                f"converging={self._converging_count}/{self.patience}"
            )

        logger.info(info["message"])
        return should_continue, info

    def run(
        self,
        step_fn: Callable[[int, Optional[StateVector]], StateVector],
        initial_state: Optional[StateVector] = None,
    ) -> Dict[str, Any]:
        """
        Run the convergence loop.

        Args:
            step_fn: Function that takes (iteration, previous_state) and returns new StateVector.
            initial_state: Optional initial state.

        Returns:
            Summary of the loop execution.
        """
        self.reset()

        if initial_state:
            self.state_history.append(initial_state)
            prev_state = initial_state
            start_iter = 1
        else:
            prev_state = None
            start_iter = 0

        final_info = None
        for i in range(start_iter, self.max_iterations):
            new_state = step_fn(i, prev_state)
            should_continue, info = self.update(new_state)
            final_info = info

            if not should_continue:
                break
            prev_state = new_state

        return {
            "final_state": self.state_history[-1].metrics if self.state_history else {},
            "total_iterations": len(self.state_history),
            "convergence_history": self.convergence_history,
            "converged": final_info.get("converged", False) if final_info else False,
            "final_info": final_info,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current loop state."""
        return {
            "objective": self.objective,
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "iterations_completed": len(self.state_history),
            "convergence_history": self.convergence_history,
            "current_state": self.state_history[-1].metrics if self.state_history else None,
            "converging_count": self._converging_count,
        }


def build_competition_loop(
    experts: List[Dict[str, Any]],
    competition_name: str,
    loop_config: Optional[Dict[str, Any]] = None,
) -> LoopController:
    """Build a convergence loop controller configured for a competition."""
    if loop_config is None:
        loop_config = {
            "objective": "minimize_validation_loss",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        }

    return LoopController.from_config(loop_config)
