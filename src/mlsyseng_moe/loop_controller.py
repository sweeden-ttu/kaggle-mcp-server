"""State convergence loop controller.

Implements the MoE state convergence loop with L2 norm exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

The loop iterates experts, collects their outputs as state vectors,
and converges when state changes fall below epsilon for `patience`
consecutive iterations.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class LoopController:
    """Runs the state convergence loop for competition entry building."""

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

    def run(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        initial_state: Optional[List[float]] = None,
        step_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the convergence loop.

        Args:
            competition: Competition identifier.
            experts: List of expert definitions to consult.
            initial_state: Starting state vector. If None, uses zeros.
            step_fn: Optional function(iteration, state, experts) -> new_state.
                     Defaults to a simple averaging step if not provided.

        Returns:
            Dict with convergence results including final state, iterations,
            and whether convergence was achieved.
        """
        n_experts = len(experts)
        if n_experts == 0:
            return {
                "competition": competition,
                "converged": False,
                "iterations": 0,
                "message": "No experts available",
            }

        state_dim = max(n_experts, 5)
        if initial_state is None:
            state = [0.0] * state_dim
        else:
            state = list(initial_state)
            if len(state) < state_dim:
                state.extend([0.0] * (state_dim - len(state)))

        converge_count = 0
        history: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            prev_state = list(state)

            if step_fn is not None:
                state = step_fn(iteration, state, experts)
            else:
                state = self._default_step(iteration, state, experts)

            diff = l2_norm(state, prev_state)
            converged_this_step = diff < self.epsilon

            if converged_this_step:
                converge_count += 1
            else:
                converge_count = 0

            step_info = {
                "iteration": iteration,
                "l2_diff": round(diff, 6),
                "converged_step": converged_this_step,
                "consecutive_converged": converge_count,
            }
            history.append(step_info)
            logger.info(
                "Iteration %d: L2 diff=%.6f, converge_count=%d",
                iteration, diff, converge_count,
            )

            self.db.save_loop_state(
                competition=competition,
                iteration=iteration,
                state_vector=state,
                metadata=step_info,
            )

            if converge_count >= self.patience:
                logger.info("Converged after %d iterations", iteration)
                return {
                    "competition": competition,
                    "converged": True,
                    "iterations": iteration,
                    "final_state": state,
                    "final_l2_diff": round(diff, 6),
                    "history": history,
                }

        return {
            "competition": competition,
            "converged": False,
            "iterations": self.max_iterations,
            "final_state": state,
            "final_l2_diff": round(history[-1]["l2_diff"], 6) if history else 0.0,
            "history": history,
            "message": f"Did not converge after {self.max_iterations} iterations",
        }

    def _default_step(
        self,
        iteration: int,
        state: List[float],
        experts: List[Dict[str, Any]],
    ) -> List[float]:
        """Default step function: weighted update from each expert.

        Each expert contributes a score based on its number of concepts
        and skills, decaying with iteration count to encourage convergence.
        """
        new_state = list(state)
        decay = 1.0 / (1.0 + iteration * 0.5)

        for i, expert in enumerate(experts):
            idx = i % len(new_state)
            n_concepts = len(expert.get("concepts", []))
            n_skills = len(expert.get("skill_names", []))
            contribution = (n_concepts * 0.1 + n_skills * 0.2) * decay
            new_state[idx] += contribution

        total = sum(abs(v) for v in new_state) or 1.0
        new_state = [v / total for v in new_state]

        return new_state

    def get_loop_history(self, competition: str) -> List[Dict[str, Any]]:
        """Retrieve stored loop states for a competition."""
        return self.db.get_loop_states(competition)
