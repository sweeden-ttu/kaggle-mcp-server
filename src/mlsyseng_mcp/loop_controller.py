"""
State convergence loop controller.

Implements the MoE convergence loop with exit condition:
  ||state[n] - state[n-1]||_2 < epsilon

The loop drives iterative improvement of a competition entry by:
1. Selecting relevant experts
2. Applying their strategies
3. Computing a state vector (e.g., scores/metrics)
4. Checking convergence via L2 norm of state delta
5. Stopping when the state stabilizes or patience is exhausted
"""

import logging
import math
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from .database import ConvergenceState, MLSysEngDatabase

logger = logging.getLogger(__name__)


def _l2_norm(v: List[float]) -> float:
    """Compute the L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _l2_distance(a: List[float], b: List[float]) -> float:
    """Compute ||a - b||_2."""
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    return _l2_norm([ai - bi for ai, bi in zip(a, b)])


class LoopController:
    """Drives the state convergence loop for a competition."""

    def __init__(
        self,
        db: MLSysEngDatabase,
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
        step_fn: Callable[[int, Optional[List[float]]], List[float]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the convergence loop.

        Args:
            competition: Competition identifier (for state storage).
            step_fn: A callable(iteration, previous_state) -> new_state_vector.
                     This function encapsulates one full improvement iteration.
            initial_state: Optional starting state vector. If None, step_fn is
                          called with iteration=0 and prev_state=None to produce it.

        Returns:
            Summary dict with convergence info and final state.
        """
        prev_state = initial_state
        converging_count = 0
        history: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            new_state = step_fn(iteration, prev_state)

            if prev_state is not None:
                delta = _l2_distance(new_state, prev_state)
            else:
                delta = float("inf")

            converged = delta < self.epsilon

            snap = ConvergenceState(
                iteration=iteration,
                state_vector=new_state,
                delta_norm=delta,
                converged=converged,
            )
            self.db.save_convergence_state(competition, snap)

            history.append(asdict(snap))

            logger.info(
                "Iteration %d: delta=%.6f, converged=%s",
                iteration,
                delta,
                converged,
            )

            if converged:
                converging_count += 1
            else:
                converging_count = 0

            if converging_count >= self.patience:
                logger.info(
                    "Convergence reached after %d consecutive stable iterations",
                    converging_count,
                )
                return {
                    "status": "converged",
                    "iterations": iteration + 1,
                    "final_delta": delta,
                    "final_state": new_state,
                    "history": history,
                    "competition": competition,
                }

            prev_state = new_state

        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "final_delta": history[-1]["delta_norm"] if history else None,
            "final_state": prev_state,
            "history": history,
            "competition": competition,
        }

    def get_history(self, competition: str) -> List[Dict[str, Any]]:
        """Retrieve the convergence history for a competition."""
        states = self.db.get_convergence_history(competition)
        return [asdict(s) for s in states]

    @staticmethod
    def default_step_fn(
        experts: List[Dict[str, Any]],
    ) -> Callable[[int, Optional[List[float]]], List[float]]:
        """
        Build a default step function that simulates metric evolution.

        In a real system, this would:
        1. Generate/modify code using expert strategies
        2. Train the model
        3. Evaluate on validation set
        4. Return the metric vector

        This default implementation produces a decaying state vector
        for testing/demonstration purposes.
        """

        def step(iteration: int, prev_state: Optional[List[float]]) -> List[float]:
            n_metrics = max(
                len(experts),
                len(experts[0].get("formula", {}).get("metrics", ["accuracy"]))
                if experts
                else 1,
            )

            if prev_state is None:
                import random

                return [random.uniform(0.3, 0.7) for _ in range(n_metrics)]

            decay = 0.5 ** (iteration + 1)
            return [v + decay * (1.0 - v) * 0.1 for v in prev_state]

        return step
