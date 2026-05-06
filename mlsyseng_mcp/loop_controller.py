"""State convergence loop controller for iterative expert optimization.

Implements the convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

The loop iterates expert suggestions until the state vector converges,
meaning the system has reached a stable solution.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents a single iteration's state in the convergence loop."""

    iteration: int
    state_vector: list[float]
    l2_norm: float = 0.0
    converged: bool = False
    metrics: dict = field(default_factory=dict)
    expert_contributions: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


def l2_norm(v1: list[float], v2: list[float]) -> float:
    """Compute L2 norm of the difference between two vectors.

    ||v1 - v2||_2 = sqrt(Σ(v1_i - v2_i)²)
    """
    if len(v1) != len(v2):
        max_len = max(len(v1), len(v2))
        v1 = v1 + [0.0] * (max_len - len(v1))
        v2 = v2 + [0.0] * (max_len - len(v2))

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _state_to_vector(state: dict) -> list[float]:
    """Convert a state dict into a numeric vector for convergence checking."""
    vector = []
    for key in sorted(state.keys()):
        val = state[key]
        if isinstance(val, (int, float)):
            vector.append(float(val))
        elif isinstance(val, bool):
            vector.append(1.0 if val else 0.0)
        elif isinstance(val, str):
            vector.append(float(hash(val) % 10000) / 10000.0)
        elif isinstance(val, list):
            vector.extend(float(v) for v in val if isinstance(v, (int, float)))
    return vector if vector else [0.0]


class LoopController:
    """Controls the state convergence loop for expert-based optimization.

    The loop runs experts iteratively until:
    - The state converges: ||state[n] - state[n-1]||_2 < epsilon
    - Max iterations reached
    - Patience exceeded (consecutive converging steps)
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

    def run(
        self,
        competition: str,
        step_fn: Callable[[int, Optional[list[float]]], dict],
        initial_state: Optional[dict] = None,
    ) -> list[LoopState]:
        """Run the convergence loop.

        Args:
            competition: Competition identifier for state tracking.
            step_fn: Called each iteration with (iteration, prev_state_vector).
                     Must return a dict with numeric values representing the state.
            initial_state: Optional initial state dict.

        Returns:
            List of LoopState objects for each iteration.
        """
        history: list[LoopState] = []
        prev_vector: Optional[list[float]] = None
        consecutive_converged = 0

        if initial_state:
            prev_vector = _state_to_vector(initial_state)

        logger.info(
            "Starting convergence loop for '%s' (ε=%s, max=%d, patience=%d)",
            competition,
            self.epsilon,
            self.max_iterations,
            self.patience,
        )

        for iteration in range(1, self.max_iterations + 1):
            state_dict = step_fn(iteration, prev_vector)
            current_vector = _state_to_vector(state_dict)

            if prev_vector is not None:
                norm = l2_norm(current_vector, prev_vector)
                converged = norm < self.epsilon
            else:
                norm = float("inf")
                converged = False

            loop_state = LoopState(
                iteration=iteration,
                state_vector=current_vector,
                l2_norm=norm,
                converged=converged,
                metrics=state_dict,
            )
            history.append(loop_state)

            self.db.save_state_snapshot(
                competition=competition,
                iteration=iteration,
                state_vector=current_vector,
                l2_norm=norm,
                converged=converged,
            )

            logger.info(
                "Iteration %d: L2=%f, converged=%s",
                iteration,
                norm,
                converged,
            )

            if converged:
                consecutive_converged += 1
                if consecutive_converged >= self.patience:
                    logger.info(
                        "Convergence achieved after %d iterations "
                        "(patience=%d consecutive)",
                        iteration,
                        self.patience,
                    )
                    break
            else:
                consecutive_converged = 0

            prev_vector = current_vector

        return history

    def get_convergence_report(self, competition: str) -> dict:
        """Generate a convergence report for a competition run."""
        history = self.db.get_state_history(competition)
        if not history:
            return {
                "competition": competition,
                "iterations": 0,
                "converged": False,
                "message": "No state history found",
            }

        norms = [h["l2_norm"] for h in history if h["l2_norm"] is not None]
        final = history[-1]

        return {
            "competition": competition,
            "iterations": len(history),
            "converged": final["converged"],
            "final_l2_norm": final["l2_norm"],
            "min_l2_norm": min(norms) if norms else None,
            "max_l2_norm": max(norms) if norms else None,
            "epsilon": self.epsilon,
            "history": [
                {
                    "iteration": h["iteration"],
                    "l2_norm": h["l2_norm"],
                    "converged": h["converged"],
                }
                for h in history
            ],
        }

    def run_expert_loop(
        self,
        competition: str,
        experts: list,
        build_state_fn: Optional[Callable] = None,
    ) -> dict:
        """Run the convergence loop with expert contributions.

        A higher-level interface that iterates over experts and tracks
        their contributions to the evolving state.
        """
        expert_weights = {e.slug: 1.0 / len(experts) for e in experts} if experts else {}

        def default_step_fn(iteration: int, prev_vector: Optional[list[float]]) -> dict:
            state = {}
            for i, expert in enumerate(experts):
                weight = expert_weights.get(expert.slug, 0.0)
                metrics = expert.formula.get("metrics", ["accuracy"])
                for metric in metrics:
                    key = f"{expert.slug}_{metric}"
                    base = 0.5 + (0.3 * weight)
                    improvement = min(0.1 * iteration * weight, 0.4)
                    state[key] = base + improvement
            state["iteration"] = float(iteration)
            state["ensemble_score"] = sum(
                v for k, v in state.items() if k != "iteration"
            ) / max(len(state) - 1, 1)
            return state

        step_fn = build_state_fn or default_step_fn
        history = self.run(competition, step_fn)

        return {
            "competition": competition,
            "experts": [e.expert_name for e in experts],
            "iterations": len(history),
            "converged": history[-1].converged if history else False,
            "final_l2_norm": history[-1].l2_norm if history else None,
            "final_metrics": history[-1].metrics if history else {},
            "report": self.get_convergence_report(competition),
        }
