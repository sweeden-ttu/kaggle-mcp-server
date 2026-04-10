"""State convergence loop with L2 norm exit condition.

Implements an iterative optimization loop that runs expert strategies
and converges when the state difference falls below epsilon.

Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class LoopController:
    """Manages the state convergence loop for competition optimization."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def run(
        self,
        competition: str,
        initial_state: List[float],
        step_fn: Callable[[List[float], int], List[float]],
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        on_step: Optional[Callable[[int, List[float], float, bool], None]] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop.

        Args:
            competition: Competition identifier.
            initial_state: Starting state vector.
            step_fn: Function(state, iteration) -> new_state that performs one
                      optimization step.
            epsilon: Convergence threshold for L2 norm.
            max_iterations: Maximum number of iterations.
            patience: Number of consecutive converging iterations before exit.
            on_step: Optional callback(iteration, state, norm, converged).

        Returns:
            Dict with final state, convergence info, and history.
        """
        state = list(initial_state)
        prev_state = list(initial_state)
        converge_count = 0
        history: List[Dict[str, Any]] = []
        converged = False

        for iteration in range(1, max_iterations + 1):
            new_state = step_fn(state, iteration)

            norm = l2_norm(new_state, state)
            is_converging = norm < epsilon

            if is_converging:
                converge_count += 1
            else:
                converge_count = 0

            self.db.record_convergence(
                competition=competition,
                iteration=iteration,
                state_vector=new_state,
                l2_norm=norm,
                converged=is_converging,
                metadata={
                    "epsilon": epsilon,
                    "patience": patience,
                    "converge_count": converge_count,
                },
            )

            step_info = {
                "iteration": iteration,
                "l2_norm": round(norm, 6),
                "converging": is_converging,
                "converge_count": converge_count,
                "state_dim": len(new_state),
            }
            history.append(step_info)

            if on_step:
                on_step(iteration, new_state, norm, is_converging)

            logger.info(
                "Iteration %d: L2=%.6f converging=%s count=%d/%d",
                iteration, norm, is_converging, converge_count, patience,
            )

            prev_state = state
            state = new_state

            if converge_count >= patience:
                converged = True
                logger.info(
                    "Converged after %d iterations (patience=%d reached)",
                    iteration, patience,
                )
                break

        return {
            "converged": converged,
            "final_state": state,
            "total_iterations": len(history),
            "final_l2_norm": history[-1]["l2_norm"] if history else 0.0,
            "epsilon": epsilon,
            "patience": patience,
            "history": history,
        }

    def get_history(self, competition: str) -> List[Dict[str, Any]]:
        """Retrieve convergence history for a competition."""
        return self.db.get_convergence_history(competition)

    def create_step_fn(
        self,
        experts: List[Dict[str, Any]],
        competition: str,
        decay: float = 0.9,
    ) -> Callable[[List[float], int], List[float]]:
        """Create a default step function based on expert weights.

        The step function simulates optimization by applying expert-weighted
        adjustments with exponential decay toward convergence.
        """

        def step(state: List[float], iteration: int) -> List[float]:
            factor = decay ** iteration
            new_state = []
            for i, val in enumerate(state):
                expert_idx = i % max(len(experts), 1)
                expert = experts[expert_idx] if experts else {}
                formula = expert.get("formula", {})

                if "minimize" in formula.get("objective", ""):
                    adjustment = -factor * abs(val) * 0.1
                else:
                    adjustment = factor * abs(val) * 0.1

                new_state.append(val + adjustment)
            return new_state

        return step
