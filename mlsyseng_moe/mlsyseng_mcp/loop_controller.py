"""State convergence loop controller for MoE competition entries."""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mlsyseng_moe.mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)


def l2_norm(v1: List[float], v2: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    a1 = np.array(v1)
    a2 = np.array(v2)
    return float(np.linalg.norm(a1 - a2))


class LoopController:
    """Controls the state convergence loop for competition optimization.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    The loop iterates until the state vector converges or max_iterations is reached.
    """

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

    def initialize_state(self, n_dimensions: int = 10) -> List[float]:
        """Initialize a random state vector."""
        return np.random.uniform(0.0, 1.0, n_dimensions).tolist()

    def check_convergence(
        self, current_state: List[float], previous_state: List[float]
    ) -> Tuple[bool, float]:
        """Check if the state has converged.

        Returns:
            Tuple of (converged: bool, norm_diff: float)
        """
        norm_diff = l2_norm(current_state, previous_state)
        converged = norm_diff < self.epsilon
        return converged, norm_diff

    def run_iteration(
        self,
        competition_slug: str,
        current_state: List[float],
        experts_used: List[str],
        step_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run a single iteration of the convergence loop.

        Args:
            competition_slug: The competition identifier
            current_state: Current state vector
            experts_used: List of expert slugs used in this iteration
            step_fn: Optional function that transforms the state. If None,
                     applies a default decay toward a target.

        Returns:
            Dict with new_state, metrics, and convergence info
        """
        if step_fn:
            new_state = step_fn(current_state)
        else:
            new_state = self._default_step(current_state)

        history = self.db.get_competition_history(competition_slug)
        iteration = len(history)

        converged = False
        norm_diff = float("inf")
        if history:
            prev_state = history[-1]["state_vector"]
            converged, norm_diff = self.check_convergence(new_state, prev_state)

        metrics = {
            "l2_norm_diff": norm_diff,
            "state_mean": float(np.mean(new_state)),
            "state_std": float(np.std(new_state)),
            "iteration": iteration,
        }

        self.db.save_competition_state(
            competition_slug=competition_slug,
            iteration=iteration,
            state_vector=new_state,
            metrics=metrics,
            experts_used=experts_used,
            converged=converged,
        )

        return {
            "iteration": iteration,
            "new_state": new_state,
            "metrics": metrics,
            "converged": converged,
            "norm_diff": norm_diff,
        }

    def run_loop(
        self,
        competition_slug: str,
        experts: List[Dict[str, Any]],
        step_fn: Optional[Callable] = None,
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run the full convergence loop.

        Args:
            competition_slug: Competition identifier
            experts: List of expert dicts to use
            step_fn: Optional state transformation function
            initial_state: Optional starting state vector

        Returns:
            Summary of the loop execution
        """
        state = initial_state or self.initialize_state()
        expert_slugs = [e.get("slug", e.get("expert_name", "unknown")) for e in experts]

        consecutive_converging = 0
        results = []

        for i in range(self.max_iterations):
            result = self.run_iteration(
                competition_slug=competition_slug,
                current_state=state,
                experts_used=expert_slugs,
                step_fn=step_fn,
            )
            results.append(result)
            state = result["new_state"]

            if result["converged"]:
                consecutive_converging += 1
                if consecutive_converging >= self.patience:
                    logger.info(
                        f"Converged after {i + 1} iterations "
                        f"(patience={self.patience} met)"
                    )
                    return {
                        "status": "converged",
                        "iterations": i + 1,
                        "final_state": state,
                        "final_metrics": result["metrics"],
                        "experts_used": expert_slugs,
                        "history": results,
                    }
            else:
                consecutive_converging = 0

        logger.info(f"Max iterations ({self.max_iterations}) reached without convergence")
        return {
            "status": "max_iterations_reached",
            "iterations": self.max_iterations,
            "final_state": state,
            "final_metrics": results[-1]["metrics"] if results else {},
            "experts_used": expert_slugs,
            "history": results,
        }

    def _default_step(self, state: List[float]) -> List[float]:
        """Default step function: decay toward 0.5 with noise."""
        arr = np.array(state)
        target = np.full_like(arr, 0.5)
        decay_rate = 0.3
        noise = np.random.normal(0, 0.01, len(state))
        new_state = arr + decay_rate * (target - arr) + noise
        return np.clip(new_state, 0.0, 1.0).tolist()

    def get_competition_status(self, competition_slug: str) -> Dict[str, Any]:
        """Get the current status of a competition's convergence loop."""
        history = self.db.get_competition_history(competition_slug)
        if not history:
            return {
                "competition": competition_slug,
                "status": "not_started",
                "iterations": 0,
            }

        latest = history[-1]
        return {
            "competition": competition_slug,
            "status": "converged" if latest["converged"] else "running",
            "iterations": len(history),
            "latest_metrics": latest["metrics"],
            "latest_state": latest["state_vector"],
            "converged": latest["converged"],
        }
