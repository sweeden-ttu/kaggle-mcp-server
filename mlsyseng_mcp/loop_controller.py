"""State convergence loop controller for the MoE system.

Implements the convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

Runs iterative refinement of competition entries until the state
vector converges or max iterations / patience are exhausted.
"""

import logging
import math
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import MoEDatabase

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if not a:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class ConvergenceLoop:
    """Runs an iterative refinement loop until state convergence.

    The loop tracks a state vector (e.g., metric scores) and exits when
    the L2 norm of consecutive state differences drops below epsilon.
    """

    def __init__(
        self,
        db: Optional[MoEDatabase] = None,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db or MoEDatabase()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def run(
        self,
        loop_id: Optional[str] = None,
        initial_state: Optional[List[float]] = None,
        step_fn: Optional[Callable[[int, List[float]], Tuple[List[float], Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Execute the convergence loop.

        Args:
            loop_id: Unique identifier for this run.
            initial_state: Starting state vector. Defaults to zeros.
            step_fn: A callable(iteration, current_state) -> (new_state, metrics).
                     If None, uses a default no-op that always converges.

        Returns:
            Summary dict with history, final_state, converged status, etc.
        """
        loop_id = loop_id or str(uuid.uuid4())[:8]
        state = initial_state or [0.0]
        prev_state = [float("inf")] * len(state)
        consecutive_converging = 0
        history: List[Dict[str, Any]] = []

        if step_fn is None:
            step_fn = self._default_step

        for iteration in range(1, self.max_iterations + 1):
            new_state, metrics = step_fn(iteration, state)

            delta = l2_norm(new_state, state)
            converged = delta < self.epsilon

            if converged:
                consecutive_converging += 1
            else:
                consecutive_converging = 0

            self.db.save_loop_state(
                loop_id=loop_id,
                iteration=iteration,
                state_vector=new_state,
                metrics=metrics,
                delta_norm=delta,
                converged=converged,
            )

            record = {
                "iteration": iteration,
                "state": new_state,
                "metrics": metrics,
                "delta_norm": round(delta, 6),
                "converged": converged,
                "consecutive_converging": consecutive_converging,
            }
            history.append(record)
            logger.info(
                "Loop %s iter %d: delta=%.6f converged=%s patience=%d/%d",
                loop_id, iteration, delta, converged,
                consecutive_converging, self.patience,
            )

            if consecutive_converging >= self.patience:
                logger.info("Loop %s converged after %d iterations", loop_id, iteration)
                break

            prev_state = state
            state = new_state

        final = history[-1] if history else {}
        return {
            "loop_id": loop_id,
            "total_iterations": len(history),
            "converged": consecutive_converging >= self.patience,
            "final_state": final.get("state", []),
            "final_metrics": final.get("metrics", {}),
            "final_delta": final.get("delta_norm", float("inf")),
            "history": history,
            "config": {
                "epsilon": self.epsilon,
                "max_iterations": self.max_iterations,
                "patience": self.patience,
            },
        }

    @staticmethod
    def _default_step(
        iteration: int, state: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Default step that decays state toward zero (guaranteed convergence)."""
        decay = 0.5 ** iteration
        new_state = [s * decay for s in state]
        metrics = {"decay_factor": decay, "state_magnitude": sum(abs(x) for x in new_state)}
        return new_state, metrics

    def run_competition_loop(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        initial_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run a convergence loop for a competition using assigned experts.

        Each iteration simulates expert contributions improving scores.
        """
        score_keys = ["accuracy", "f1_score", "precision", "recall"]
        state = [initial_scores.get(k, 0.5) for k in score_keys] if initial_scores else [0.5] * len(score_keys)

        expert_names = [e.get("expert_name", "unknown") for e in experts]

        def competition_step(
            iteration: int, current_state: List[float]
        ) -> Tuple[List[float], Dict[str, Any]]:
            improvement = 0.1 / iteration
            new_state = [
                min(1.0, s + improvement * (1.0 - s))
                for s in current_state
            ]
            metrics = {
                score_keys[i]: round(new_state[i], 4) for i in range(len(score_keys))
            }
            metrics["experts_active"] = expert_names
            metrics["iteration"] = iteration
            return new_state, metrics

        loop_config = experts[0].get("loop_config", {}) if experts else {}
        loop = ConvergenceLoop(
            db=self.db,
            epsilon=loop_config.get("epsilon", self.epsilon),
            max_iterations=loop_config.get("max_iterations", self.max_iterations),
            patience=loop_config.get("patience", self.patience),
        )

        loop_id = f"{competition}_{uuid.uuid4().hex[:6]}"
        return loop.run(
            loop_id=loop_id,
            initial_state=state,
            step_fn=competition_step,
        )
