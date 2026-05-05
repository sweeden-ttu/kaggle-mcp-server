"""State convergence loop controller for MLSysEng MoE."""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two state vectors."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class ConvergenceLoop:
    """
    State convergence loop with exit condition:
        ||state[n] - state[n-1]||_2 < epsilon

    Runs a step function iteratively until the state vector converges
    (L2 norm of difference falls below epsilon) or max iterations reached.
    """

    def __init__(
        self,
        competition: str,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        db: Optional[Database] = None,
    ):
        self.competition = competition
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.db = db or Database()
        self.history: List[Dict[str, Any]] = []

    def run(
        self,
        step_fn: Callable[[int, Optional[List[float]]], Tuple[List[float], Dict[str, float]]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run the convergence loop.

        Args:
            step_fn: Function that takes (iteration, previous_state) and returns
                     (new_state_vector, metrics_dict).
            initial_state: Optional initial state vector. If None, step_fn is
                          called with None for the first iteration.

        Returns:
            Dictionary with convergence results.
        """
        prev_state = initial_state
        converging_count = 0
        converged = False
        final_iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            final_iteration = iteration
            logger.info("Iteration %d/%d for %s", iteration, self.max_iterations, self.competition)

            try:
                new_state, metrics = step_fn(iteration, prev_state)
            except Exception as e:
                logger.error("Step function failed at iteration %d: %s", iteration, e)
                self.history.append({
                    "iteration": iteration,
                    "status": "error",
                    "message": str(e),
                })
                break

            if prev_state is not None:
                delta = l2_norm(new_state, prev_state)
            else:
                delta = float("inf")

            metrics["l2_delta"] = delta
            metrics["converging"] = delta < self.epsilon

            self.db.save_state(
                competition=self.competition,
                iteration=iteration,
                state_vector=new_state,
                metrics=metrics,
            )

            step_result = {
                "iteration": iteration,
                "l2_delta": round(delta, 6),
                "converging": delta < self.epsilon,
                "metrics": metrics,
                "state_dim": len(new_state),
            }
            self.history.append(step_result)

            if delta < self.epsilon:
                converging_count += 1
                logger.info(
                    "Converging (%d/%d): delta=%.6f < epsilon=%.6f",
                    converging_count, self.patience, delta, self.epsilon,
                )
                if converging_count >= self.patience:
                    converged = True
                    break
            else:
                converging_count = 0

            prev_state = new_state

        return {
            "competition": self.competition,
            "converged": converged,
            "iterations": final_iteration,
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "final_delta": self.history[-1].get("l2_delta") if self.history else None,
            "history": self.history,
        }


def default_step_fn(
    experts: List[Dict[str, Any]],
    competition: str,
) -> Callable[[int, Optional[List[float]]], Tuple[List[float], Dict[str, float]]]:
    """
    Create a default step function for competition entry building.

    Uses expert relevance scores and metrics to construct a state vector
    that converges as the system refines its selection.
    """
    def step(iteration: int, prev_state: Optional[List[float]]) -> Tuple[List[float], Dict[str, float]]:
        state = []
        for i, expert in enumerate(experts):
            base_score = expert.get("relevance_score", 0.5)
            decay = 1.0 / (1.0 + 0.1 * iteration)
            noise = (hash(f"{competition}:{expert.get('slug', i)}:{iteration}") % 1000) / 100000.0
            score = base_score * (1 - decay * 0.1) + noise
            state.append(round(score, 6))

        metrics = {
            "mean_score": round(sum(state) / len(state), 6) if state else 0,
            "max_score": round(max(state), 6) if state else 0,
            "num_experts": len(experts),
        }

        return state, metrics

    return step
