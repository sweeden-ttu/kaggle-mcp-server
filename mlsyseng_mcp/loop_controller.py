"""
State convergence loop controller for the MLSysEng MoE system.

Implements the iterative refinement loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

The loop orchestrates expert-driven refinement of Kaggle competition
entries until the state vector converges or patience is exhausted.
"""

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute the L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def initial_state_vector(num_experts: int) -> List[float]:
    """Create an initial state vector (all zeros) for the given number of experts."""
    return [0.0] * max(num_experts, 1)


class LoopController:
    """
    Controls the state convergence loop for competition entry refinement.

    Each iteration:
    1. Select relevant experts
    2. Each expert contributes to the state vector
    3. Compute L2 norm between current and previous state
    4. Check convergence: ||state[n] - state[n-1]||_2 < epsilon
    5. Track patience for early stopping
    """

    def __init__(
        self,
        db,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def create_loop(self, competition: str, num_experts: int) -> Dict[str, Any]:
        """Initialize a new convergence loop for a competition."""
        loop_id = f"loop_{competition}_{uuid.uuid4().hex[:8]}"
        state = {
            "loop_id": loop_id,
            "competition": competition,
            "iteration": 0,
            "state_vector": initial_state_vector(num_experts),
            "prev_vector": [],
            "l2_norm": float("inf"),
            "converged": False,
            "patience_cnt": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.db.upsert_loop_state(state)
        return state

    def step(
        self,
        loop_id: str,
        expert_contributions: List[float],
        step_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute one iteration of the convergence loop.

        Args:
            loop_id: ID of the loop
            expert_contributions: New state values from expert processing
            step_fn: Optional callback invoked each step with (iteration, state)

        Returns:
            Updated loop state dict with convergence info
        """
        state = self.db.get_loop_state(loop_id)
        if state is None:
            raise ValueError(f"Loop {loop_id} not found")

        if state["converged"]:
            return state

        prev_vector = state["state_vector"]
        new_vector = expert_contributions

        norm = l2_norm(new_vector, prev_vector)
        iteration = state["iteration"] + 1

        converging = norm < self.epsilon
        patience_cnt = (state["patience_cnt"] + 1) if converging else 0
        converged = patience_cnt >= self.patience or iteration >= self.max_iterations

        updated = {
            "loop_id": loop_id,
            "competition": state["competition"],
            "iteration": iteration,
            "state_vector": new_vector,
            "prev_vector": prev_vector,
            "l2_norm": norm,
            "converged": converged,
            "patience_cnt": patience_cnt,
            "started_at": state["started_at"],
        }
        self.db.upsert_loop_state(updated)

        if step_fn:
            step_fn(iteration, updated)

        logger.info(
            "Loop %s iter=%d  L2=%.6f  converging=%s  patience=%d/%d",
            loop_id,
            iteration,
            norm,
            converging,
            patience_cnt,
            self.patience,
        )

        return updated

    def run_loop(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        refine_fn: Callable[[int, List[Dict[str, Any]], List[float]], List[float]],
        step_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the full convergence loop until exit condition is met.

        Args:
            competition: Competition name
            experts: List of expert dicts
            refine_fn: Function(iteration, experts, current_state) -> new_state_vector
            step_fn: Optional callback for each step

        Returns:
            Final loop state
        """
        state = self.create_loop(competition, len(experts))
        loop_id = state["loop_id"]

        for i in range(self.max_iterations):
            current_vector = state["state_vector"]
            new_vector = refine_fn(i, experts, current_vector)
            state = self.step(loop_id, new_vector, step_fn=step_fn)

            if state["converged"]:
                logger.info(
                    "Loop %s converged at iteration %d (L2=%.6f)",
                    loop_id,
                    state["iteration"],
                    state["l2_norm"],
                )
                break

        return state

    def get_status(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a loop."""
        return self.db.get_loop_state(loop_id)
