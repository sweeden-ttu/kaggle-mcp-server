"""State convergence loop controller for MLSysEng MoE system.

Implements the iterative convergence loop for competition entries:
  exit_condition: ||state[n] - state[n-1]||_2 < epsilon

Each iteration runs expert recommendations, updates the state vector,
and checks for convergence.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)


class LoopController:
    """Controls the state convergence loop for competition entries."""

    def __init__(
        self,
        db: Database,
        competition: str,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db
        self.competition = competition
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self._state_history: List[List[float]] = []
        self._metric_history: List[Dict[str, float]] = []
        self._consecutive_converging = 0

    @property
    def current_iteration(self) -> int:
        return len(self._state_history)

    @property
    def converged(self) -> bool:
        return self._consecutive_converging >= self.patience

    def initialize_state(self, dimension: int = 10) -> List[float]:
        """Initialize the state vector for a new competition run."""
        state = [0.0] * dimension
        self._state_history = [state]
        self._metric_history = [{}]
        self._consecutive_converging = 0

        self.db.save_competition_state(
            competition=self.competition,
            iteration=0,
            state_vector=state,
            metric_values={},
        )
        return state

    def step(
        self,
        new_state: List[float],
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Execute one step of the convergence loop.

        Returns dict with convergence info:
          - iteration: current iteration number
          - l2_norm: L2 distance from previous state
          - converged: whether we've reached convergence
          - should_stop: whether to stop iterating
          - reason: why we stopped (if applicable)
        """
        iteration = self.current_iteration

        if self._state_history:
            prev_state = self._state_history[-1]
            l2_norm = self._l2_distance(new_state, prev_state)
        else:
            l2_norm = float("inf")

        self._state_history.append(new_state)
        self._metric_history.append(metrics)

        is_converging = l2_norm < self.epsilon
        if is_converging:
            self._consecutive_converging += 1
        else:
            self._consecutive_converging = 0

        should_stop = False
        reason = None

        if self._consecutive_converging >= self.patience:
            should_stop = True
            reason = f"Converged: {self._consecutive_converging} consecutive iterations below epsilon={self.epsilon}"
        elif iteration >= self.max_iterations:
            should_stop = True
            reason = f"Max iterations reached: {self.max_iterations}"

        self.db.save_competition_state(
            competition=self.competition,
            iteration=iteration,
            state_vector=new_state,
            metric_values=metrics,
            converged=should_stop and self._consecutive_converging >= self.patience,
        )

        return {
            "iteration": iteration,
            "l2_norm": l2_norm,
            "epsilon": self.epsilon,
            "converging": is_converging,
            "consecutive_converging": self._consecutive_converging,
            "converged": self._consecutive_converging >= self.patience,
            "should_stop": should_stop,
            "reason": reason,
            "metrics": metrics,
        }

    def run_loop(
        self,
        step_fn: Callable[[List[float], int], Tuple[List[float], Dict[str, float]]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run the full convergence loop.

        Args:
            step_fn: Function(current_state, iteration) -> (new_state, metrics)
            initial_state: Starting state vector (default: zeros of dim 10)

        Returns summary of the loop execution.
        """
        if initial_state is None:
            current_state = self.initialize_state()
        else:
            current_state = initial_state
            self._state_history = [current_state]
            self._metric_history = [{}]

        results = []
        for i in range(self.max_iterations):
            try:
                new_state, metrics = step_fn(current_state, i)
            except Exception as e:
                logger.error("Step function failed at iteration %d: %s", i, e)
                results.append({
                    "iteration": i,
                    "error": str(e),
                    "should_stop": True,
                    "reason": f"Step function error: {e}",
                })
                break

            result = self.step(new_state, metrics)
            results.append(result)

            if result["should_stop"]:
                break

            current_state = new_state

        return {
            "competition": self.competition,
            "total_iterations": len(results),
            "converged": any(r.get("converged", False) for r in results),
            "final_l2_norm": results[-1].get("l2_norm", None) if results else None,
            "final_metrics": results[-1].get("metrics", {}) if results else {},
            "history": results,
        }

    def get_history(self) -> Dict[str, Any]:
        """Get the full state history for this competition."""
        states = self.db.get_competition_states(self.competition)
        return {
            "competition": self.competition,
            "iterations": len(states),
            "converged": any(s["converged"] for s in states),
            "states": states,
        }

    @staticmethod
    def _l2_distance(a: List[float], b: List[float]) -> float:
        """Compute L2 (Euclidean) distance between two vectors."""
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def state_from_metrics(metrics: Dict[str, float], dimension: int = 10) -> List[float]:
        """Convert a metrics dictionary to a fixed-dimension state vector.

        Useful for creating state vectors from competition metric outputs.
        """
        values = list(metrics.values())
        if len(values) >= dimension:
            return values[:dimension]

        state = values + [0.0] * (dimension - len(values))
        return state
