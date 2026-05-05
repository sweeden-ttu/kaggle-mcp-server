"""State convergence loop controller for MLSysEng MoE.

Implements the iterative improvement loop with L2 norm convergence
detection and patience-based early stopping.
"""

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StateVector:
    """Represents a state vector for convergence tracking."""

    def __init__(self, values: Optional[Dict[str, float]] = None):
        self.values = values or {}
        self.timestamp = time.time()

    def l2_distance(self, other: "StateVector") -> float:
        """Compute L2 norm of difference: ||self - other||_2."""
        all_keys = set(self.values.keys()) | set(other.values.keys())
        if not all_keys:
            return 0.0
        total = 0.0
        for key in all_keys:
            diff = self.values.get(key, 0.0) - other.values.get(key, 0.0)
            total += diff * diff
        return math.sqrt(total)

    def to_dict(self) -> Dict[str, Any]:
        return {"values": self.values, "timestamp": self.timestamp}

    @classmethod
    def from_metrics(cls, metrics: Dict[str, float]) -> "StateVector":
        return cls(values=metrics)


class LoopController:
    """Controls the iterative improvement loop with convergence detection.

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
        self.convergence_count = 0
        self.iteration = 0

    def reset(self):
        """Reset the loop state."""
        self.history = []
        self.convergence_count = 0
        self.iteration = 0

    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update loop with new metrics and check convergence.

        Returns a dict with convergence info and whether to continue.
        """
        current = StateVector.from_metrics(metrics)
        self.history.append(current)
        self.iteration += 1

        result = {
            "iteration": self.iteration,
            "metrics": metrics,
            "max_iterations": self.max_iterations,
        }

        if len(self.history) < 2:
            result.update({
                "converged": False,
                "should_continue": True,
                "l2_distance": None,
                "convergence_count": 0,
                "message": "First iteration, no convergence check yet",
            })
            return result

        previous = self.history[-2]
        distance = current.l2_distance(previous)
        is_converging = distance < self.epsilon

        if is_converging:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        converged = self.convergence_count >= self.patience
        at_max = self.iteration >= self.max_iterations
        should_continue = not converged and not at_max

        if converged:
            message = (
                f"Converged: L2 distance {distance:.6f} < epsilon {self.epsilon} "
                f"for {self.convergence_count} consecutive iterations"
            )
        elif at_max:
            message = f"Reached max iterations ({self.max_iterations})"
        elif is_converging:
            message = (
                f"Converging ({self.convergence_count}/{self.patience}): "
                f"L2 distance {distance:.6f}"
            )
        else:
            message = f"Not converging: L2 distance {distance:.6f} >= epsilon {self.epsilon}"

        result.update({
            "converged": converged,
            "should_continue": should_continue,
            "l2_distance": distance,
            "convergence_count": self.convergence_count,
            "patience": self.patience,
            "epsilon": self.epsilon,
            "message": message,
        })

        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full state history."""
        history = []
        for i, state in enumerate(self.history):
            entry = {"iteration": i + 1, **state.to_dict()}
            if i > 0:
                entry["l2_distance"] = state.l2_distance(self.history[i - 1])
            history.append(entry)
        return history

    def get_best_state(self) -> Optional[Dict[str, Any]]:
        """Get the best state based on the objective."""
        if not self.history:
            return None

        minimize = "minimize" in self.objective.lower()

        objective_key = self.objective.replace("minimize_", "").replace("maximize_", "")
        candidates = [
            (i, s) for i, s in enumerate(self.history) if objective_key in s.values
        ]

        if not candidates:
            return {"iteration": len(self.history), **self.history[-1].to_dict()}

        if minimize:
            best_idx, best_state = min(candidates, key=lambda x: x[1].values[objective_key])
        else:
            best_idx, best_state = max(candidates, key=lambda x: x[1].values[objective_key])

        return {"iteration": best_idx + 1, **best_state.to_dict()}

    def run_loop(
        self,
        step_fn: Callable[[int, Optional[Dict[str, float]]], Dict[str, float]],
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the full convergence loop.

        Args:
            step_fn: Called each iteration with (iteration_number, previous_metrics).
                     Must return a dict of metric values.
            on_step: Optional callback after each step with the update result.

        Returns:
            Summary dict with final state, convergence info, and history.
        """
        self.reset()
        prev_metrics = None

        while True:
            try:
                metrics = step_fn(self.iteration + 1, prev_metrics)
            except Exception as e:
                logger.error("Step %d failed: %s", self.iteration + 1, e)
                return {
                    "status": "error",
                    "error": str(e),
                    "iteration": self.iteration,
                    "history": self.get_history(),
                }

            result = self.update(metrics)

            if on_step:
                on_step(result)

            prev_metrics = metrics

            if not result["should_continue"]:
                break

        best = self.get_best_state()

        return {
            "status": "converged" if result.get("converged") else "max_iterations",
            "total_iterations": self.iteration,
            "final_metrics": metrics,
            "best_state": best,
            "convergence_info": {
                "epsilon": self.epsilon,
                "patience": self.patience,
                "final_l2_distance": result.get("l2_distance"),
            },
            "history": self.get_history(),
        }
