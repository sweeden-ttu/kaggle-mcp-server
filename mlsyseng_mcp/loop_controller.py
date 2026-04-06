"""State convergence loop controller for MoE system.

Implements the convergence loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

Each iteration refines the competition entry using expert knowledge.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Tracks the state of a convergence loop iteration."""

    iteration: int
    state_vector: List[float]
    metrics: Dict[str, float]
    expert_contributions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LoopResult:
    """Final result of the convergence loop."""

    converged: bool
    total_iterations: int
    final_metrics: Dict[str, float]
    convergence_history: List[Dict[str, Any]]
    final_state: List[float]
    exit_reason: str


def l2_norm_diff(state_a: List[float], state_b: List[float]) -> float:
    """Compute L2 norm of difference between two state vectors."""
    if len(state_a) != len(state_b):
        raise ValueError(
            f"State vectors must have same length: {len(state_a)} vs {len(state_b)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(state_a, state_b)))


class LoopController:
    """Controls the state convergence loop for competition entry building."""

    def __init__(
        self,
        objective: str = "minimize_validation_loss",
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.objective = objective
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.history: List[LoopState] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoopController":
        return cls(
            objective=config.get("objective", "minimize_validation_loss"),
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
        )

    def _check_convergence(self, current: LoopState, previous: LoopState) -> float:
        return l2_norm_diff(current.state_vector, previous.state_vector)

    def run(
        self,
        step_fn: Callable[[int, Optional[List[float]]], LoopState],
        initial_state: Optional[List[float]] = None,
    ) -> LoopResult:
        """Run the convergence loop.

        Args:
            step_fn: Callable(iteration, previous_state) -> LoopState
                     that performs one iteration of work and returns the new state.
            initial_state: Optional initial state vector.

        Returns:
            LoopResult with convergence info and final metrics.
        """
        self.history.clear()
        consecutive_converging = 0
        convergence_history = []
        exit_reason = "max_iterations_reached"

        for iteration in range(self.max_iterations):
            prev_state = initial_state if iteration == 0 else self.history[-1].state_vector
            current = step_fn(iteration, prev_state)
            self.history.append(current)

            if iteration > 0:
                diff = self._check_convergence(current, self.history[-2])
                converging = diff < self.epsilon
                convergence_history.append({
                    "iteration": iteration,
                    "l2_diff": diff,
                    "epsilon": self.epsilon,
                    "converging": converging,
                    "metrics": dict(current.metrics),
                })

                if converging:
                    consecutive_converging += 1
                    logger.info(
                        "Iteration %d: converging (%d/%d), L2 diff=%.6f",
                        iteration,
                        consecutive_converging,
                        self.patience,
                        diff,
                    )
                    if consecutive_converging >= self.patience:
                        exit_reason = f"converged_after_{iteration + 1}_iterations"
                        break
                else:
                    consecutive_converging = 0
                    logger.info(
                        "Iteration %d: not converging, L2 diff=%.6f",
                        iteration,
                        diff,
                    )
            else:
                convergence_history.append({
                    "iteration": 0,
                    "l2_diff": None,
                    "epsilon": self.epsilon,
                    "converging": False,
                    "metrics": dict(current.metrics),
                })

        final = self.history[-1]
        return LoopResult(
            converged=consecutive_converging >= self.patience,
            total_iterations=len(self.history),
            final_metrics=dict(final.metrics),
            convergence_history=convergence_history,
            final_state=list(final.state_vector),
            exit_reason=exit_reason,
        )

    def to_dict(self, result: LoopResult) -> Dict[str, Any]:
        return {
            "converged": result.converged,
            "total_iterations": result.total_iterations,
            "final_metrics": result.final_metrics,
            "convergence_history": result.convergence_history,
            "exit_reason": result.exit_reason,
            "config": {
                "objective": self.objective,
                "epsilon": self.epsilon,
                "max_iterations": self.max_iterations,
                "patience": self.patience,
            },
        }


def build_competition_step(
    experts: List[Dict[str, Any]],
    competition: str,
    iteration: int,
    previous_state: Optional[List[float]],
) -> LoopState:
    """Default step function for competition entry building.

    In a full implementation, this would:
    1. Query each expert for recommendations
    2. Generate/refine notebook code
    3. Evaluate metrics
    4. Return updated state vector

    The state vector encodes metrics like [loss, accuracy, f1, ...].
    """
    base_loss = 1.0
    base_accuracy = 0.5
    decay = 0.3

    if previous_state:
        loss = previous_state[0] * (1 - decay / (iteration + 1))
        accuracy = min(1.0, previous_state[1] + decay / (iteration + 1) * 0.5)
    else:
        loss = base_loss
        accuracy = base_accuracy

    f1 = accuracy * 0.95

    expert_names = [e.get("expert_name", "unknown") for e in experts]

    return LoopState(
        iteration=iteration,
        state_vector=[loss, accuracy, f1],
        metrics={"loss": loss, "accuracy": accuracy, "f1_score": f1},
        expert_contributions=expert_names,
    )
