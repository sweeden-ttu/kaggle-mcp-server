"""State convergence loop controller for the MoE system.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 (Euclidean) norm of the difference between two vectors."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class LoopController:
    """Controls state convergence iterations for competition entry building."""

    def __init__(
        self,
        db: Database,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def initialize_state(
        self,
        competition: str,
        expert_count: int,
        initial_scores: Optional[List[float]] = None,
    ) -> List[float]:
        """Create the initial state vector for a competition."""
        if initial_scores:
            state = initial_scores[:]
        else:
            state = [0.0] * max(expert_count, 1)

        self.db.save_state(
            competition=competition,
            iteration=0,
            state_vector=state,
            l2_norm=float("inf"),
            converged=False,
        )
        return state

    def step(
        self,
        competition: str,
        iteration: int,
        new_state: List[float],
    ) -> Dict[str, Any]:
        """
        Record a new state and check convergence.

        Returns dict with:
          - converged: bool
          - l2_norm: float
          - iteration: int
          - patience_remaining: int
          - should_stop: bool
        """
        history = self.db.get_state_history(competition)

        if history:
            prev_state = history[-1]["state_vector"]
            norm = l2_norm(new_state, prev_state)
        else:
            norm = float("inf")

        converged = norm < self.epsilon

        self.db.save_state(
            competition=competition,
            iteration=iteration,
            state_vector=new_state,
            l2_norm=norm,
            converged=converged,
        )

        consecutive_converged = self._count_consecutive_converged(competition)
        should_stop = (
            consecutive_converged >= self.patience
            or iteration >= self.max_iterations
        )

        return {
            "converged": converged,
            "l2_norm": round(norm, 6),
            "iteration": iteration,
            "patience_remaining": max(0, self.patience - consecutive_converged),
            "should_stop": should_stop,
            "reason": self._stop_reason(iteration, consecutive_converged),
        }

    def run_loop(
        self,
        competition: str,
        state_generator,
        expert_count: int,
    ) -> Dict[str, Any]:
        """
        Run the full convergence loop.

        Args:
            competition: Competition identifier.
            state_generator: Callable(iteration, current_state) -> new_state list.
            expert_count: Number of experts (dimension of state vector).

        Returns:
            Summary of the loop execution.
        """
        state = self.initialize_state(competition, expert_count)
        results = []

        for iteration in range(1, self.max_iterations + 1):
            new_state = state_generator(iteration, state)
            result = self.step(competition, iteration, new_state)
            results.append(result)

            logger.info(
                "Iteration %d: L2=%.6f converged=%s stop=%s",
                iteration,
                result["l2_norm"],
                result["converged"],
                result["should_stop"],
            )

            state = new_state

            if result["should_stop"]:
                break

        return {
            "competition": competition,
            "total_iterations": len(results),
            "final_state": state,
            "final_l2_norm": results[-1]["l2_norm"] if results else float("inf"),
            "converged": results[-1]["converged"] if results else False,
            "stop_reason": results[-1].get("reason", "unknown"),
            "history": results,
        }

    def get_history(self, competition: str) -> List[Dict[str, Any]]:
        """Get full state history for a competition."""
        return self.db.get_state_history(competition)

    def _count_consecutive_converged(self, competition: str) -> int:
        """Count consecutive converged iterations from the end."""
        history = self.db.get_state_history(competition)
        count = 0
        for entry in reversed(history):
            if entry["converged"]:
                count += 1
            else:
                break
        return count

    def _stop_reason(self, iteration: int, consecutive_converged: int) -> str:
        if consecutive_converged >= self.patience:
            return f"converged (patience={self.patience} reached)"
        if iteration >= self.max_iterations:
            return f"max_iterations ({self.max_iterations}) reached"
        return "running"
