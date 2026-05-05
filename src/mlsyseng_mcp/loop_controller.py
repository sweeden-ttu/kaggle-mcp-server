"""State convergence loop controller for MLSysEng MoE.

Implements the exit condition: ||state[n] - state[n-1]||_2 < epsilon
with patience-based early stopping.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state vector at a single iteration."""

    iteration: int
    values: Dict[str, float]
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> List[float]:
        return [self.values[k] for k in sorted(self.values.keys())]


@dataclass
class ConvergenceResult:
    """Result of the convergence loop."""

    converged: bool
    iterations: int
    final_state: Optional[LoopState]
    history: List[LoopState]
    l2_norms: List[float]
    reason: str


def l2_norm(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute the L2 norm of the difference between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensions must match: {len(vec_a)} != {len(vec_b)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


class LoopController:
    """Controls the state convergence loop with configurable exit conditions."""

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

    def run(
        self,
        step_fn: Callable[[LoopState, int], LoopState],
        initial_state: Dict[str, float],
    ) -> ConvergenceResult:
        """Execute the convergence loop.

        Args:
            step_fn: Function that takes (current_state, iteration) and returns the next state.
            initial_state: Initial state values dict.

        Returns:
            ConvergenceResult with convergence info and full history.
        """
        history: List[LoopState] = []
        l2_norms: List[float] = []
        consecutive_converging = 0

        current = LoopState(iteration=0, values=initial_state.copy())
        history.append(current)

        for i in range(1, self.max_iterations + 1):
            try:
                next_state = step_fn(current, i)
            except Exception as exc:
                logger.error("Step function failed at iteration %d: %s", i, exc)
                return ConvergenceResult(
                    converged=False,
                    iterations=i,
                    final_state=current,
                    history=history,
                    l2_norms=l2_norms,
                    reason=f"Step function error at iteration {i}: {exc}",
                )

            next_state.iteration = i
            history.append(next_state)

            current_vec = current.to_vector()
            next_vec = next_state.to_vector()
            norm = l2_norm(current_vec, next_vec)
            l2_norms.append(norm)

            logger.info(
                "Iteration %d: L2 norm = %.6f (epsilon = %.6f)",
                i, norm, self.epsilon,
            )

            if norm < self.epsilon:
                consecutive_converging += 1
                if consecutive_converging >= self.patience:
                    return ConvergenceResult(
                        converged=True,
                        iterations=i,
                        final_state=next_state,
                        history=history,
                        l2_norms=l2_norms,
                        reason=(
                            f"Converged after {i} iterations "
                            f"({consecutive_converging} consecutive below epsilon={self.epsilon})"
                        ),
                    )
            else:
                consecutive_converging = 0

            current = next_state

        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_state=current,
            history=history,
            l2_norms=l2_norms,
            reason=f"Max iterations ({self.max_iterations}) reached without convergence",
        )

    def run_competition(
        self,
        competition: str,
        expert_slugs: List[str],
        step_fn: Optional[Callable] = None,
    ) -> ConvergenceResult:
        """Run convergence loop for a competition entry.

        Uses a default step function that simulates metric improvement
        if no custom step_fn is provided.
        """
        initial = {
            "validation_loss": 1.0,
            "accuracy": 0.0,
            "f1_score": 0.0,
        }

        if step_fn is None:
            def default_step(state: LoopState, iteration: int) -> LoopState:
                decay = 0.7 ** iteration
                new_values = {
                    "validation_loss": state.values["validation_loss"] * (0.5 + 0.5 * decay),
                    "accuracy": min(1.0, state.values["accuracy"] + 0.1 * decay),
                    "f1_score": min(1.0, state.values["f1_score"] + 0.08 * decay),
                }
                return LoopState(
                    iteration=iteration,
                    values=new_values,
                    metrics={
                        "competition": competition,
                        "experts_used": len(expert_slugs),
                    },
                )
            step_fn = default_step

        return self.run(step_fn, initial)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoopController":
        return cls(
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
            objective=config.get("objective", "minimize_validation_loss"),
        )
