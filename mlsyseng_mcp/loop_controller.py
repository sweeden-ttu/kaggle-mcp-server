"""State convergence loop controller for MLSysEng MoE.

Implements the iterative convergence loop with L2 norm exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

Each iteration refines competition entries by re-querying experts
and tracking convergence of a state vector.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import database as db

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute the L2 norm (Euclidean distance) between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class ConvergenceState:
    """Tracks state across convergence iterations."""

    def __init__(self, competition: str, epsilon: float = 0.001,
                 max_iterations: int = 10, patience: int = 3):
        self.competition = competition
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

        self.iteration = 0
        self.converging_count = 0
        self.history: List[Dict[str, Any]] = []
        self.state_vectors: List[List[float]] = []
        self.converged = False
        self.exit_reason: Optional[str] = None

    def update(self, state_vector: List[float],
               metrics: Dict[str, float]) -> Dict[str, Any]:
        """Record a new state and check convergence."""
        self.iteration += 1
        self.state_vectors.append(state_vector)

        delta = None
        if len(self.state_vectors) >= 2:
            prev = self.state_vectors[-2]
            curr = self.state_vectors[-1]
            delta = l2_norm(curr, prev)

            if delta < self.epsilon:
                self.converging_count += 1
            else:
                self.converging_count = 0

        step = {
            "iteration": self.iteration,
            "state_vector": state_vector,
            "metrics": metrics,
            "delta": delta,
            "converging_count": self.converging_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(step)

        if self.converging_count >= self.patience:
            self.converged = True
            self.exit_reason = (
                f"Converged: {self.converging_count} consecutive iterations "
                f"with delta < {self.epsilon}"
            )
        elif self.iteration >= self.max_iterations:
            self.converged = True
            self.exit_reason = f"Max iterations reached ({self.max_iterations})"

        return {
            "iteration": self.iteration,
            "delta": delta,
            "converged": self.converged,
            "exit_reason": self.exit_reason,
            "converging_streak": self.converging_count,
            "metrics": metrics,
        }

    def should_continue(self) -> bool:
        return not self.converged

    def summary(self) -> Dict[str, Any]:
        deltas = [h["delta"] for h in self.history if h["delta"] is not None]
        return {
            "competition": self.competition,
            "total_iterations": self.iteration,
            "converged": self.converged,
            "exit_reason": self.exit_reason,
            "epsilon": self.epsilon,
            "final_delta": deltas[-1] if deltas else None,
            "min_delta": min(deltas) if deltas else None,
            "max_delta": max(deltas) if deltas else None,
            "history": self.history,
        }


def _default_state_vector(expert: Dict, iteration: int) -> List[float]:
    """Generate a state vector from expert metrics.

    The vector encodes the expert's iteration progress and a decay factor,
    simulating convergence over iterations.
    """
    formula = expert.get("formula", {})
    metrics = formula.get("metrics", ["accuracy"])
    loop_cfg = expert.get("loop_config", {})
    max_iter = loop_cfg.get("max_iterations", 10)

    progress = iteration / max(max_iter, 1)
    decay = 1.0 / (1.0 + iteration * 0.5)

    vector = [progress]
    for i, _ in enumerate(metrics):
        base = 0.5 + 0.4 * progress
        noise = (hash(f"{expert.get('slug', '')}_{i}_{iteration}") % 100) / 10000
        vector.append(min(base + noise, 1.0))
    vector.append(decay)

    return vector


def run_convergence_loop(
    competition: str,
    experts: List[Dict[str, Any]],
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
    step_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run the convergence loop across all experts for a competition.

    For each iteration:
    1. Each expert produces a state vector
    2. Vectors are aggregated
    3. L2 norm distance is checked against epsilon
    4. Loop exits when converged or max iterations reached

    Args:
        competition: Competition name/slug
        experts: List of expert dicts
        epsilon: Convergence threshold
        max_iterations: Maximum iterations
        patience: Required consecutive converging iterations
        step_callback: Optional callback(expert, iteration, state) per step
    """
    db.init_db()

    state = ConvergenceState(
        competition=competition,
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    if not experts:
        return {
            "status": "error",
            "message": "No experts available for convergence loop",
        }

    while state.should_continue():
        agg_vector: List[float] = []
        agg_metrics: Dict[str, float] = {}
        expert_count = 0

        for expert in experts:
            sv = _default_state_vector(expert, state.iteration + 1)

            if step_callback:
                try:
                    result = step_callback(expert, state.iteration + 1, sv)
                    if result and isinstance(result, list):
                        sv = result
                except Exception as e:
                    logger.warning("step_callback failed for %s: %s",
                                   expert.get("slug"), e)

            if not agg_vector:
                agg_vector = [0.0] * len(sv)
            for i in range(min(len(sv), len(agg_vector))):
                agg_vector[i] += sv[i]
            expert_count += 1

            formula = expert.get("formula", {})
            for metric in formula.get("metrics", []):
                key = f"{expert.get('slug', 'unknown')}/{metric}"
                idx = min(1, len(sv) - 1)
                agg_metrics[key] = sv[idx] if idx < len(sv) else 0.0

        if expert_count > 0:
            agg_vector = [v / expert_count for v in agg_vector]

        step_result = state.update(agg_vector, agg_metrics)
        logger.info(
            "Iteration %d: delta=%s, converging=%d/%d",
            step_result["iteration"],
            step_result["delta"],
            step_result["converging_streak"],
            patience,
        )

    summary = state.summary()
    summary["status"] = "converged" if state.converged else "max_iterations"
    summary["experts_used"] = [e.get("slug") for e in experts]
    return summary
