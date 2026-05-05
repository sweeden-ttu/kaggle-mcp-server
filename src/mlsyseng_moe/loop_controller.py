"""State convergence loop for iterative expert-guided competition building."""

import json
import logging
import math
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import CompetitionEntry, MoEDatabase

logger = logging.getLogger(__name__)


def l2_norm(v1: List[float], v2: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class StateVector:
    """Represents the current state of a competition solution iteration."""

    def __init__(self, metrics: Optional[Dict[str, float]] = None):
        self.metrics = metrics or {}
        self.timestamp = time.time()

    def to_vector(self, keys: Optional[List[str]] = None) -> List[float]:
        """Convert metrics to a fixed-order vector."""
        if keys is None:
            keys = sorted(self.metrics.keys())
        return [self.metrics.get(k, 0.0) for k in keys]

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateVector":
        sv = cls(metrics=d.get("metrics", {}))
        sv.timestamp = d.get("timestamp", time.time())
        return sv


class LoopController:
    """
    Manages the state convergence loop for iterative competition solving.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    Exits after `patience` consecutive converging iterations.
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        objective: str = "minimize_validation_loss",
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.objective = objective

        self.history: List[StateVector] = []
        self.converging_count = 0
        self.iteration = 0

    def check_convergence(self, current: StateVector, previous: StateVector) -> Tuple[bool, float]:
        """Check if the state has converged based on L2 norm."""
        if not current.metrics or not previous.metrics:
            return False, float("inf")

        keys = sorted(set(current.metrics.keys()) | set(previous.metrics.keys()))
        v_curr = current.to_vector(keys)
        v_prev = previous.to_vector(keys)

        norm = l2_norm(v_curr, v_prev)
        return norm < self.epsilon, norm

    def step(self, state: StateVector) -> Dict[str, Any]:
        """Execute one step of the convergence loop."""
        self.iteration += 1
        self.history.append(state)

        result = {
            "iteration": self.iteration,
            "state": state.to_dict(),
            "converged": False,
            "should_stop": False,
            "norm": None,
            "reason": None,
        }

        if self.iteration >= self.max_iterations:
            result["should_stop"] = True
            result["reason"] = f"max_iterations ({self.max_iterations}) reached"
            return result

        if len(self.history) < 2:
            result["reason"] = "first iteration, no previous state"
            return result

        previous = self.history[-2]
        converged, norm = self.check_convergence(state, previous)
        result["norm"] = norm

        if converged:
            self.converging_count += 1
            result["converged"] = True
            if self.converging_count >= self.patience:
                result["should_stop"] = True
                result["reason"] = f"converged for {self.patience} consecutive iterations (norm={norm:.6f} < ε={self.epsilon})"
            else:
                result["reason"] = f"converging ({self.converging_count}/{self.patience} patience, norm={norm:.6f})"
        else:
            self.converging_count = 0
            result["reason"] = f"not converged (norm={norm:.6f} >= ε={self.epsilon})"

        return result

    def get_state_history(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.history]

    def reset(self):
        self.history.clear()
        self.converging_count = 0
        self.iteration = 0


def run_convergence_loop(
    competition: str,
    experts: List[Dict[str, Any]],
    step_fn: Callable[[int, List[Dict[str, Any]]], Dict[str, float]],
    db: MoEDatabase,
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> Dict[str, Any]:
    """
    Run the full convergence loop for a competition.

    Args:
        competition: Competition name/slug
        experts: List of expert dicts selected for this competition
        step_fn: Callable(iteration, experts) -> dict of metric values
        db: Database instance
        epsilon: Convergence threshold
        max_iterations: Max loop iterations
        patience: Consecutive convergence iterations before stopping
    """
    controller = LoopController(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    entry_id = f"{competition}_{uuid.uuid4().hex[:8]}"
    experts_used = [e.get("expert_name", e.get("slug", "unknown")) for e in experts]
    skills_used = set()
    for e in experts:
        for s in e.get("skills", []):
            skills_used.add(s)

    logger.info(f"Starting convergence loop for {competition} with {len(experts)} experts")

    while True:
        metrics = step_fn(controller.iteration, experts)
        state = StateVector(metrics=metrics)
        result = controller.step(state)

        logger.info(
            f"Iteration {result['iteration']}: {result['reason']} "
            f"(metrics={metrics})"
        )

        if result["should_stop"]:
            break

    final_metrics = controller.history[-1].metrics if controller.history else {}
    final_metric = final_metrics.get("validation_loss", final_metrics.get("loss", 0.0))

    entry = CompetitionEntry(
        entry_id=entry_id,
        competition=competition,
        experts_used=json.dumps(experts_used),
        skills_used=json.dumps(sorted(skills_used)),
        state_history=json.dumps(controller.get_state_history()),
        converged=controller.converging_count >= patience,
        final_metric=final_metric,
        created_at=time.time(),
    )
    db.save_entry(entry)

    return {
        "entry_id": entry_id,
        "competition": competition,
        "iterations": controller.iteration,
        "converged": controller.converging_count >= patience,
        "final_metrics": final_metrics,
        "experts_used": experts_used,
        "skills_used": sorted(skills_used),
        "state_history": controller.get_state_history(),
    }
