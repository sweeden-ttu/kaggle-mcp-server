"""State convergence loop controller for the MoE system.

Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import MoEDatabase

logger = logging.getLogger(__name__)


def _l2_norm(vec: List[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def _vector_diff(a: List[float], b: List[float]) -> List[float]:
    """Compute element-wise difference between two vectors."""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + [0.0] * (max_len - len(a))
        b = b + [0.0] * (max_len - len(b))
    return [x - y for x, y in zip(a, b)]


class LoopController:
    """Controls the state convergence loop for competition entries."""

    def __init__(
        self,
        db: MoEDatabase,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def _check_convergence(
        self, current: List[float], previous: List[float]
    ) -> Tuple[bool, float]:
        """Check if state has converged.

        Returns (converged, delta) where delta = ||state[n] - state[n-1]||_2.
        """
        diff = _vector_diff(current, previous)
        delta = _l2_norm(diff)
        return delta < self.epsilon, delta

    def run_loop(
        self,
        competition: str,
        step_fn: Callable[[int, List[float], Dict[str, Any]], Tuple[List[float], Dict[str, Any]]],
        initial_state: Optional[List[float]] = None,
        initial_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop.

        Args:
            competition: Competition identifier.
            step_fn: Callable(iteration, current_state, current_metrics)
                      -> (new_state, new_metrics).
                      This function implements one iteration of the optimization.
            initial_state: Starting state vector. Defaults to [0.0].
            initial_metrics: Starting metrics dict. Defaults to empty.

        Returns:
            Summary dict with final state, metrics, convergence info.
        """
        state = initial_state or [0.0]
        metrics = initial_metrics or {}
        converge_count = 0
        history: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            prev_state = list(state)

            try:
                state, metrics = step_fn(iteration, state, metrics)
            except Exception as e:
                logger.error("Step function failed at iteration %d: %s", iteration, e)
                self.db.save_loop_state(
                    competition=competition,
                    iteration=iteration,
                    state_vector=state,
                    metrics={"error": str(e), **metrics},
                    converged=False,
                )
                return {
                    "competition": competition,
                    "status": "error",
                    "iteration": iteration,
                    "error": str(e),
                    "history": history,
                }

            converged, delta = self._check_convergence(state, prev_state)

            step_info = {
                "iteration": iteration,
                "delta": round(delta, 6),
                "converged": converged,
                "metrics": metrics,
                "state_dim": len(state),
            }
            history.append(step_info)

            self.db.save_loop_state(
                competition=competition,
                iteration=iteration,
                state_vector=state,
                metrics=metrics,
                converged=converged,
            )

            logger.info(
                "Iteration %d: delta=%.6f converged=%s",
                iteration,
                delta,
                converged,
            )

            if converged:
                converge_count += 1
                if converge_count >= self.patience:
                    return {
                        "competition": competition,
                        "status": "converged",
                        "final_iteration": iteration,
                        "final_delta": round(delta, 6),
                        "final_metrics": metrics,
                        "patience_met": True,
                        "history": history,
                    }
            else:
                converge_count = 0

        return {
            "competition": competition,
            "status": "max_iterations_reached",
            "final_iteration": self.max_iterations,
            "final_delta": round(history[-1]["delta"], 6) if history else 0.0,
            "final_metrics": metrics,
            "patience_met": False,
            "history": history,
        }

    def get_loop_summary(self, competition: str) -> Dict[str, Any]:
        """Get a summary of loop states for a competition."""
        states = self.db.get_loop_states(competition)
        if not states:
            return {
                "competition": competition,
                "status": "not_started",
                "iterations": 0,
            }

        latest = states[-1]
        deltas = []
        for i in range(1, len(states)):
            prev_vec = states[i - 1].get("state_vector", [0.0])
            curr_vec = states[i].get("state_vector", [0.0])
            _, delta = self._check_convergence(curr_vec, prev_vec)
            deltas.append(delta)

        return {
            "competition": competition,
            "iterations": len(states),
            "latest_converged": bool(latest.get("converged")),
            "latest_metrics": latest.get("metrics", {}),
            "deltas": [round(d, 6) for d in deltas],
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
        }


def default_step_fn(
    iteration: int,
    state: List[float],
    metrics: Dict[str, Any],
) -> Tuple[List[float], Dict[str, Any]]:
    """Default step function that decays state toward zero (for testing).

    In production, this would be replaced by actual model training / evaluation.
    """
    decay = 0.5
    new_state = [x * decay for x in state]

    new_metrics = {
        "iteration": iteration,
        "state_norm": round(_l2_norm(new_state), 6),
        "decay_factor": decay,
    }
    return new_state, new_metrics
