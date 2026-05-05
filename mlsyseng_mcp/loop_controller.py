"""State convergence loop for iterative expert refinement."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Snapshot of loop state at iteration *n*."""

    iteration: int
    vector: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "vector": self.vector.tolist(),
            "metrics": self.metrics,
        }


@dataclass
class ConvergenceResult:
    converged: bool
    iterations: int
    final_delta: float
    history: List[Dict[str, Any]]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_delta": self.final_delta,
            "history": self.history,
            "message": self.message,
        }


class LoopController:
    """Run a state convergence loop with L2 norm exit condition.

    Exit condition: ``||state[n] - state[n-1]||_2 < epsilon``
    Patience: exits after *patience* consecutive converging iterations.
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def run(
        self,
        initial_state: np.ndarray,
        step_fn: Callable[[np.ndarray, int], np.ndarray],
        metric_fn: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
    ) -> ConvergenceResult:
        """Execute the convergence loop.

        Args:
            initial_state: Starting state vector.
            step_fn: ``(state, iteration) -> new_state``.
            metric_fn: Optional function to compute metrics from state.

        Returns:
            ConvergenceResult with history and convergence status.
        """
        state = initial_state.copy()
        history: List[Dict[str, Any]] = []
        consecutive = 0

        for i in range(1, self.max_iterations + 1):
            prev = state.copy()
            state = step_fn(state, i)

            delta = float(np.linalg.norm(state - prev))
            metrics = metric_fn(state) if metric_fn else {}
            metrics["l2_delta"] = delta

            snap = LoopState(iteration=i, vector=state.copy(), metrics=metrics)
            history.append(snap.to_dict())

            logger.info("Iteration %d  delta=%.6f  eps=%.6f", i, delta, self.epsilon)

            if delta < self.epsilon:
                consecutive += 1
                if consecutive >= self.patience:
                    return ConvergenceResult(
                        converged=True,
                        iterations=i,
                        final_delta=delta,
                        history=history,
                        message=(
                            f"Converged after {i} iterations "
                            f"({consecutive} consecutive below epsilon={self.epsilon})"
                        ),
                    )
            else:
                consecutive = 0

        last_delta = history[-1]["metrics"]["l2_delta"] if history else float("inf")
        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_delta=last_delta,
            history=history,
            message=f"Did not converge within {self.max_iterations} iterations",
        )


def build_competition_step_fn(
    expert_skills: List[str],
    competition_name: str,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Return a step function that simulates skill application.

    Each skill application nudges the state toward a lower-loss region.
    """

    n_skills = max(len(expert_skills), 1)
    decay = 0.8

    def _step(state: np.ndarray, iteration: int) -> np.ndarray:
        grad = np.random.default_rng(iteration).standard_normal(state.shape)
        grad /= max(np.linalg.norm(grad), 1e-8)
        lr = 0.1 * (decay ** iteration) / math.sqrt(n_skills)
        return state - lr * grad

    return _step


def default_metric_fn(state: np.ndarray) -> Dict[str, float]:
    return {
        "state_norm": float(np.linalg.norm(state)),
        "state_mean": float(np.mean(state)),
    }
