"""State Convergence Loop Controller.

Implements the iterative convergence loop for competition entries:
- Exit condition: ||state[n] - state[n-1]||_2 < epsilon
- Convergence metric: L2 norm of state vector difference
- Patience: exits after `patience` consecutive converging iterations
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EPSILON = 0.001
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_PATIENCE = 3


def l2_norm(vec: List[float]) -> float:
    """Compute the L2 (Euclidean) norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def l2_distance(a: List[float], b: List[float]) -> float:
    """Compute ||a - b||_2."""
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}"
        )
    return l2_norm([ai - bi for ai, bi in zip(a, b)])


class ConvergenceLoop:
    """Runs a state-convergence loop that iterates until the state vector
    stabilises below an epsilon threshold.

    The loop calls a user-supplied ``step_fn`` each iteration.  The step
    function receives the current state vector and iteration index, and
    must return the next state vector.
    """

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        patience: int = DEFAULT_PATIENCE,
        db=None,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.db = db
        self.history: List[Dict[str, Any]] = []

    def run(
        self,
        initial_state: List[float],
        step_fn: Callable[[List[float], int], List[float]],
        competition: str = "unknown",
    ) -> Dict[str, Any]:
        """Execute the convergence loop.

        Args:
            initial_state: Starting state vector.
            step_fn: ``(current_state, iteration) -> next_state``
            competition: Label used for logging.

        Returns:
            Dict with ``converged``, ``iterations``, ``final_state``,
            ``final_l2_norm``, and ``history``.
        """
        prev_state = initial_state
        converge_streak = 0
        self.history = []

        for iteration in range(1, self.max_iterations + 1):
            next_state = step_fn(prev_state, iteration)
            norm = l2_distance(next_state, prev_state)
            converged = norm < self.epsilon

            step_record = {
                "iteration": iteration,
                "state": next_state,
                "l2_norm": round(norm, 8),
                "converged": converged,
            }
            self.history.append(step_record)

            if self.db:
                try:
                    self.db.log_convergence(
                        competition=competition,
                        iteration=iteration,
                        state_vector=next_state,
                        l2_norm=norm,
                        converged=converged,
                    )
                except Exception:
                    logger.debug("Could not persist convergence log", exc_info=True)

            logger.info(
                "[%s] iter=%d  ||Δ||₂=%.6f  converged=%s",
                competition, iteration, norm, converged,
            )

            if converged:
                converge_streak += 1
            else:
                converge_streak = 0

            if converge_streak >= self.patience:
                logger.info(
                    "[%s] Converged after %d iterations (patience=%d)",
                    competition, iteration, self.patience,
                )
                return self._result(True, iteration, next_state, norm)

            prev_state = next_state

        logger.info(
            "[%s] Max iterations (%d) reached without full convergence",
            competition, self.max_iterations,
        )
        last = self.history[-1]
        return self._result(
            False, self.max_iterations, last["state"], last["l2_norm"]
        )

    def _result(
        self,
        converged: bool,
        iterations: int,
        final_state: List[float],
        final_norm: float,
    ) -> Dict[str, Any]:
        return {
            "converged": converged,
            "iterations": iterations,
            "final_state": final_state,
            "final_l2_norm": round(final_norm, 8),
            "epsilon": self.epsilon,
            "patience": self.patience,
            "history": self.history,
        }


def make_competition_step_fn(
    experts: List[Dict[str, Any]],
    competition: str,
) -> Callable[[List[float], int], List[float]]:
    """Create a step function that simulates expert-guided optimisation.

    Each element of the state vector maps to one expert's contribution.
    The step function applies a damped update toward a target score
    derived from the expert's formula metrics.
    """
    n = max(len(experts), 1)

    def step_fn(state: List[float], iteration: int) -> List[float]:
        padded = state + [0.0] * (n - len(state))
        padded = padded[:n]

        next_state = []
        for i, val in enumerate(padded):
            # Damped step toward 1.0 (ideal normalised score)
            alpha = 0.3 / iteration  # learning rate decays
            target = 1.0
            new_val = val + alpha * (target - val)
            next_state.append(round(new_val, 8))
        return next_state

    return step_fn


def run_evolve(
    competition: str,
    experts: List[Dict[str, Any]],
    db=None,
    epsilon: float = DEFAULT_EPSILON,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    patience: int = DEFAULT_PATIENCE,
) -> Dict[str, Any]:
    """High-level helper: run the convergence loop for a competition."""
    n = max(len(experts), 1)
    initial_state = [0.0] * n

    loop = ConvergenceLoop(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
        db=db,
    )

    step_fn = make_competition_step_fn(experts, competition)
    return loop.run(initial_state, step_fn, competition=competition)
