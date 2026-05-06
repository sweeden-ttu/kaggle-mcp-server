"""State convergence loop controller for the MoE system.

Implements the iterative refinement loop with L2 norm convergence:
    exit_condition: ||state[n] - state[n-1]||_2 < epsilon

Each iteration evaluates expert recommendations and updates
a state vector representing the current solution quality.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    objective: str = "minimize_validation_loss"
    exit_condition: str = "||state[n] - state[n-1]||_2 < epsilon"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3


@dataclass
class LoopState:
    iteration: int = 0
    state_vector: List[float] = field(default_factory=list)
    l2_norm: float = float("inf")
    converged: bool = False
    converging_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    if not a:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _default_state_vector(
    experts: List[Dict[str, Any]],
    iteration: int,
) -> List[float]:
    """Generate a default state vector based on expert configurations.

    In a real system this would be computed from actual model metrics.
    Here we simulate convergence with decreasing deltas.
    """
    base = []
    for i, expert in enumerate(experts):
        formula = expert.get("formula", {})
        if isinstance(formula, str):
            try:
                formula = json.loads(formula)
            except (json.JSONDecodeError, TypeError):
                formula = {}
        metrics = formula.get("metrics", ["accuracy"])
        for j, metric in enumerate(metrics):
            value = 0.5 + 0.4 * (1 - math.exp(-0.3 * iteration))
            value += (i * 0.01) + (j * 0.005)
            noise = math.sin(iteration * (i + 1) * 0.7) * 0.01 * math.exp(-0.2 * iteration)
            base.append(round(value + noise, 6))
    return base if base else [0.5]


class LoopController:
    """Manages the state convergence loop for competition entry building."""

    def __init__(self, db: Database, config: Optional[LoopConfig] = None):
        self.db = db
        self.config = config or LoopConfig()

    def run_loop(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        step_fn: Optional[Callable[[int, List[Dict[str, Any]], LoopState], List[float]]] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop until exit condition is met.

        Args:
            competition: Competition identifier.
            experts: List of expert dicts to use.
            step_fn: Optional callable(iteration, experts, state) -> new_state_vector.
                     If None, uses a simulated convergence function.

        Returns:
            Summary dict with convergence results.
        """
        state = LoopState()
        step_fn = step_fn or (lambda it, exps, st: _default_state_vector(exps, it))

        logger.info(
            "Starting convergence loop for '%s' with %d experts (epsilon=%.4f, max_iter=%d)",
            competition, len(experts), self.config.epsilon, self.config.max_iterations,
        )

        while state.iteration < self.config.max_iterations:
            prev_vector = list(state.state_vector)
            new_vector = step_fn(state.iteration, experts, state)
            state.state_vector = new_vector

            if prev_vector:
                state.l2_norm = l2_norm(new_vector, prev_vector)
            else:
                state.l2_norm = float("inf")

            is_converging = state.l2_norm < self.config.epsilon
            if is_converging:
                state.converging_count += 1
            else:
                state.converging_count = 0

            state.converged = state.converging_count >= self.config.patience

            step_record = {
                "iteration": state.iteration,
                "l2_norm": state.l2_norm,
                "converging": is_converging,
                "converging_count": state.converging_count,
                "state_dim": len(new_vector),
            }
            state.history.append(step_record)

            self.db.record_convergence_state(
                competition=competition,
                iteration=state.iteration,
                state_vector=new_vector,
                l2_norm=state.l2_norm,
                converged=state.converged,
            )

            logger.info(
                "Iteration %d: L2=%.6f converging=%s count=%d/%d",
                state.iteration, state.l2_norm, is_converging,
                state.converging_count, self.config.patience,
            )

            if state.converged:
                logger.info(
                    "Converged after %d iterations (patience=%d met)",
                    state.iteration + 1, self.config.patience,
                )
                break

            state.iteration += 1

        return {
            "competition": competition,
            "converged": state.converged,
            "total_iterations": state.iteration + 1,
            "final_l2_norm": state.l2_norm,
            "epsilon": self.config.epsilon,
            "patience": self.config.patience,
            "state_dim": len(state.state_vector),
            "history": state.history,
            "exit_condition": self.config.exit_condition,
        }

    def get_convergence_summary(self, competition: str) -> Dict[str, Any]:
        """Get a summary of convergence history for a competition."""
        history = self.db.get_convergence_history(competition)
        if not history:
            return {"competition": competition, "status": "no_data"}

        last = history[-1]
        return {
            "competition": competition,
            "total_iterations": len(history),
            "final_l2_norm": last.get("l2_norm", 0.0),
            "converged": bool(last.get("converged", 0)),
            "history": [
                {
                    "iteration": h["iteration"],
                    "l2_norm": h.get("l2_norm", 0.0),
                    "converged": bool(h.get("converged", 0)),
                }
                for h in history
            ],
        }
