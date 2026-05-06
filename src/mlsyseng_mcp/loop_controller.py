"""State convergence loop controller for MLSysEng MoE.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import MLSysEngDB

logger = logging.getLogger(__name__)


def l2_norm(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(vec_a) != len(vec_b):
        max_len = max(len(vec_a), len(vec_b))
        vec_a = vec_a + [0.0] * (max_len - len(vec_a))
        vec_b = vec_b + [0.0] * (max_len - len(vec_b))

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


class StateVector:
    """Represents the state of a competition entry at a given iteration."""

    def __init__(self, values: Optional[List[float]] = None):
        self.values = values or []

    @classmethod
    def from_scores(cls, scores: Dict[str, float]) -> "StateVector":
        """Create a state vector from a dictionary of metric scores."""
        sorted_keys = sorted(scores.keys())
        return cls([scores[k] for k in sorted_keys])

    @classmethod
    def initial(cls, dimension: int = 5) -> "StateVector":
        """Create an initial zero state vector."""
        return cls([0.0] * dimension)

    def distance(self, other: "StateVector") -> float:
        return l2_norm(self.values, other.values)

    def to_list(self) -> List[float]:
        return list(self.values)

    def __repr__(self) -> str:
        return f"StateVector({self.values})"


class ConvergenceResult:
    """Result of a convergence loop execution."""

    def __init__(self):
        self.iterations: List[Dict[str, Any]] = []
        self.converged: bool = False
        self.final_state: Optional[StateVector] = None
        self.total_iterations: int = 0
        self.final_l2_norm: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "total_iterations": self.total_iterations,
            "final_l2_norm": round(self.final_l2_norm, 6),
            "final_state": self.final_state.to_list() if self.final_state else [],
            "iterations": self.iterations,
        }


class LoopController:
    """Controls the state convergence loop for competition entries.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    Patience: exits after `patience` consecutive converging iterations.
    """

    def __init__(
        self,
        db: Optional[MLSysEngDB] = None,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db or MLSysEngDB()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def run(
        self,
        competition: str,
        step_fn: Callable[[str, int, StateVector], StateVector],
        initial_state: Optional[StateVector] = None,
    ) -> ConvergenceResult:
        """Run the convergence loop.

        Args:
            competition: Competition identifier.
            step_fn: Function that takes (competition, iteration, current_state) and returns next state.
            initial_state: Starting state vector (defaults to zero vector).
        """
        result = ConvergenceResult()
        state = initial_state or StateVector.initial()
        converge_count = 0

        for iteration in range(1, self.max_iterations + 1):
            prev_state = StateVector(list(state.values))

            try:
                state = step_fn(competition, iteration, state)
            except Exception as e:
                logger.error("Step function failed at iteration %d: %s", iteration, e)
                result.iterations.append({
                    "iteration": iteration,
                    "error": str(e),
                    "state": prev_state.to_list(),
                })
                break

            norm = state.distance(prev_state)
            is_converging = norm < self.epsilon

            if is_converging:
                converge_count += 1
            else:
                converge_count = 0

            iter_info = {
                "iteration": iteration,
                "l2_norm": round(norm, 6),
                "converging": is_converging,
                "converge_streak": converge_count,
                "state": state.to_list(),
            }
            result.iterations.append(iter_info)

            self.db.save_convergence_state(
                competition=competition,
                iteration=iteration,
                state_vector=state.to_list(),
                l2_norm=norm,
                converged=is_converging,
            )

            logger.info(
                "Iteration %d: L2=%.6f, converging=%s, streak=%d/%d",
                iteration, norm, is_converging, converge_count, self.patience,
            )

            if converge_count >= self.patience:
                result.converged = True
                break

        result.final_state = state
        result.total_iterations = len(result.iterations)
        result.final_l2_norm = result.iterations[-1]["l2_norm"] if result.iterations else float("inf")

        return result

    def get_history(self, competition: str) -> List[Dict[str, Any]]:
        """Get convergence history for a competition."""
        return self.db.get_convergence_history(competition)

    def check_convergence(self, competition: str) -> Dict[str, Any]:
        """Check current convergence status for a competition."""
        history = self.get_history(competition)
        if not history:
            return {"status": "no_history", "converged": False}

        latest = history[-1]
        converge_streak = 0
        for entry in reversed(history):
            if entry.get("converged"):
                converge_streak += 1
            else:
                break

        return {
            "status": "active",
            "total_iterations": len(history),
            "latest_l2_norm": latest.get("l2_norm", float("inf")),
            "converge_streak": converge_streak,
            "converged": converge_streak >= self.patience,
        }


def default_step_fn(competition: str, iteration: int, state: StateVector) -> StateVector:
    """Default step function that simulates convergence by decaying the state.

    In a real implementation, this would:
    1. Select relevant experts
    2. Generate/update notebook cells
    3. Evaluate model performance
    4. Return the new state based on metrics
    """
    decay = 0.5 ** iteration
    new_values = [v + decay * (1.0 - v) for v in state.values] if state.values else [decay] * 5
    return StateVector(new_values)
