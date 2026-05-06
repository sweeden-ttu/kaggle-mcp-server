"""State convergence loop controller for MLSysEng MoE system.

Implements the convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
from typing import Optional, Callable

import numpy as np

from mlsyseng_mcp.database import store_convergence_state, get_convergence_history

logger = logging.getLogger(__name__)


class LoopController:
    """Controls the state convergence loop for competition optimization."""

    def __init__(
        self,
        competition: str,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        db_path: Optional[str] = None,
    ):
        self.competition = competition
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.db_path = db_path

        self.iteration = 0
        self.state_history: list[np.ndarray] = []
        self.converging_count = 0

    def compute_l2_norm(self, state_current: np.ndarray,
                        state_previous: np.ndarray) -> float:
        """Compute L2 norm of state vector difference."""
        diff = state_current - state_previous
        return float(np.linalg.norm(diff, ord=2))

    def check_convergence(self, state: np.ndarray) -> dict:
        """Check if the system has converged.

        Returns dict with convergence status and metrics.
        """
        self.state_history.append(state.copy())
        self.iteration += 1

        if len(self.state_history) < 2:
            l2_norm = float("inf")
            converged = False
        else:
            previous = self.state_history[-2]
            l2_norm = self.compute_l2_norm(state, previous)
            converged = l2_norm < self.epsilon

        if converged:
            self.converging_count += 1
        else:
            self.converging_count = 0

        should_exit = (
            self.converging_count >= self.patience
            or self.iteration >= self.max_iterations
        )

        store_convergence_state(
            competition=self.competition,
            iteration=self.iteration,
            state_vector=state.tolist(),
            l2_norm=l2_norm,
            converged=converged,
            db_path=self.db_path,
        )

        return {
            "iteration": self.iteration,
            "l2_norm": l2_norm,
            "converged": converged,
            "converging_count": self.converging_count,
            "should_exit": should_exit,
            "exit_reason": self._exit_reason(should_exit),
        }

    def _exit_reason(self, should_exit: bool) -> Optional[str]:
        if not should_exit:
            return None
        if self.converging_count >= self.patience:
            return f"Converged: {self.converging_count} consecutive iterations below epsilon={self.epsilon}"
        if self.iteration >= self.max_iterations:
            return f"Max iterations reached: {self.max_iterations}"
        return None

    def run_loop(
        self,
        step_fn: Callable[[int, Optional[np.ndarray]], np.ndarray],
        initial_state: Optional[np.ndarray] = None,
    ) -> dict:
        """Run the convergence loop.

        Args:
            step_fn: Function that takes (iteration, previous_state) and returns new state vector.
            initial_state: Optional initial state vector.

        Returns:
            Final convergence status with history.
        """
        current_state = initial_state

        while True:
            new_state = step_fn(self.iteration, current_state)

            if not isinstance(new_state, np.ndarray):
                new_state = np.array(new_state, dtype=float)

            result = self.check_convergence(new_state)
            current_state = new_state

            logger.info(
                f"Iteration {result['iteration']}: "
                f"L2={result['l2_norm']:.6f}, "
                f"converged={result['converged']}"
            )

            if result["should_exit"]:
                return {
                    "final_state": current_state.tolist(),
                    "total_iterations": self.iteration,
                    "final_l2_norm": result["l2_norm"],
                    "exit_reason": result["exit_reason"],
                    "converged": self.converging_count >= self.patience,
                }

    def get_history(self) -> list[dict]:
        """Get convergence history from database."""
        return get_convergence_history(self.competition, self.db_path)


def build_state_vector(metrics: dict) -> np.ndarray:
    """Convert a metrics dictionary to a state vector.

    Standard metrics: accuracy, f1_score, log_loss, auc, rmse
    """
    metric_keys = sorted(metrics.keys())
    return np.array([float(metrics[k]) for k in metric_keys])


def create_competition_loop(
    competition: str,
    experts: list[dict],
    db_path: Optional[str] = None,
) -> LoopController:
    """Create a loop controller configured from expert definitions."""
    if experts:
        config = experts[0].get("loop_config", {})
    else:
        config = {}

    return LoopController(
        competition=competition,
        epsilon=config.get("epsilon", 0.001),
        max_iterations=config.get("max_iterations", 10),
        patience=config.get("patience", 3),
        db_path=db_path,
    )
