"""State convergence loop controller for MLSysEng MoE.

Implements the iterative refinement loop with L2-norm convergence
detection for competition entry building.

Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from .database import Database, LoopState

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3
    objective: str = "minimize_validation_loss"


@dataclass
class IterationResult:
    iteration: int
    state_vector: list[float]
    metrics: dict
    l2_norm: float
    converged: bool


def l2_norm(v1: list[float], v2: list[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0.0] * (max_len - len(v1))
        v2 = v2 + [0.0] * (max_len - len(v2))

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class LoopController:
    """Controls the state convergence loop for competition entry building.

    The loop iteratively refines a state vector (representing model
    performance metrics) until convergence is detected or max iterations
    are reached.
    """

    def __init__(self, db: Database, config: Optional[ConvergenceConfig] = None):
        self.db = db
        self.config = config or ConvergenceConfig()

    def create_loop(self, competition: str) -> str:
        """Create a new convergence loop and return its ID."""
        loop_id = f"{competition}_{uuid.uuid4().hex[:8]}"
        logger.info("Created loop: %s for competition: %s", loop_id, competition)
        return loop_id

    def record_iteration(
        self,
        loop_id: str,
        competition: str,
        iteration: int,
        state_vector: list[float],
        metrics: dict,
    ) -> IterationResult:
        """Record a single iteration and check convergence."""
        prev = self.db.get_latest_loop_state(loop_id)
        if prev and prev.state_vector:
            norm = l2_norm(state_vector, prev.state_vector)
        else:
            norm = float("inf")

        converged = norm < self.config.epsilon

        state = LoopState(
            loop_id=loop_id,
            competition=competition,
            iteration=iteration,
            state_vector=state_vector,
            metrics=metrics,
            converged=converged,
            timestamp=time.time(),
        )
        self.db.save_loop_state(state)

        result = IterationResult(
            iteration=iteration,
            state_vector=state_vector,
            metrics=metrics,
            l2_norm=norm,
            converged=converged,
        )

        logger.info(
            "Loop %s iter %d: L2=%.6f converged=%s",
            loop_id, iteration, norm, converged,
        )
        return result

    def check_should_stop(self, loop_id: str) -> tuple[bool, str]:
        """Check whether the loop should stop.

        Returns (should_stop, reason).
        """
        history = self.db.get_loop_history(loop_id)
        if not history:
            return False, "no iterations yet"

        if len(history) >= self.config.max_iterations:
            return True, f"max_iterations ({self.config.max_iterations}) reached"

        consecutive_converged = 0
        for state in reversed(history):
            if state.converged:
                consecutive_converged += 1
            else:
                break

        if consecutive_converged >= self.config.patience:
            return True, (
                f"converged for {consecutive_converged} consecutive iterations "
                f"(patience={self.config.patience})"
            )

        return False, f"iteration {len(history)}/{self.config.max_iterations}"

    def run_loop(
        self,
        competition: str,
        step_fn: Callable[[int, Optional[list[float]]], tuple[list[float], dict]],
        config: Optional[ConvergenceConfig] = None,
    ) -> dict:
        """Run the full convergence loop.

        Args:
            competition: Competition identifier.
            step_fn: A callable that takes (iteration, previous_state_vector)
                     and returns (new_state_vector, metrics).
            config: Override convergence config for this run.

        Returns:
            Summary dict with loop results.
        """
        cfg = config or self.config
        self.config = cfg

        loop_id = self.create_loop(competition)
        results: list[IterationResult] = []
        prev_state: Optional[list[float]] = None

        for i in range(cfg.max_iterations):
            try:
                state_vector, metrics = step_fn(i, prev_state)
            except Exception as e:
                logger.error("Loop %s step %d failed: %s", loop_id, i, e)
                break

            result = self.record_iteration(
                loop_id=loop_id,
                competition=competition,
                iteration=i,
                state_vector=state_vector,
                metrics=metrics,
            )
            results.append(result)
            prev_state = state_vector

            should_stop, reason = self.check_should_stop(loop_id)
            if should_stop:
                logger.info("Loop %s stopping: %s", loop_id, reason)
                break

        final_metrics = results[-1].metrics if results else {}
        final_converged = any(r.converged for r in results)

        return {
            "loop_id": loop_id,
            "competition": competition,
            "total_iterations": len(results),
            "converged": final_converged,
            "final_metrics": final_metrics,
            "final_l2_norm": results[-1].l2_norm if results else None,
            "config": {
                "epsilon": cfg.epsilon,
                "max_iterations": cfg.max_iterations,
                "patience": cfg.patience,
                "objective": cfg.objective,
            },
            "history": [
                {
                    "iteration": r.iteration,
                    "l2_norm": r.l2_norm,
                    "converged": r.converged,
                    "metrics": r.metrics,
                }
                for r in results
            ],
        }

    def get_loop_summary(self, loop_id: str) -> dict:
        """Get a summary of a completed or in-progress loop."""
        history = self.db.get_loop_history(loop_id)
        if not history:
            return {"error": f"No history found for loop {loop_id}"}

        return {
            "loop_id": loop_id,
            "competition": history[0].competition,
            "total_iterations": len(history),
            "converged": any(s.converged for s in history),
            "final_metrics": history[-1].metrics if history else {},
            "iterations": [
                {
                    "iteration": s.iteration,
                    "converged": s.converged,
                    "metrics": s.metrics,
                    "timestamp": s.timestamp,
                }
                for s in history
            ],
        }
