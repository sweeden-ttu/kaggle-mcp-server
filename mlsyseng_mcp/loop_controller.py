"""State convergence loop controller for the MoE system."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents a single iteration's state vector."""
    iteration: int
    state_vector: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConvergenceResult:
    """Result of the convergence loop."""
    converged: bool
    iterations: int
    final_state: Optional[np.ndarray]
    history: List[Dict[str, Any]]
    l2_norm_history: List[float]
    exit_reason: str


class LoopController:
    """
    Controls the state convergence loop for expert-driven competition solving.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
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
        self._history: List[LoopState] = []
        self._converging_count = 0

    def reset(self):
        """Reset the loop controller state."""
        self._history = []
        self._converging_count = 0

    def compute_l2_norm(self, state_current: np.ndarray, state_previous: np.ndarray) -> float:
        """Compute L2 norm of state difference: ||state[n] - state[n-1]||_2"""
        diff = state_current - state_previous
        return float(np.linalg.norm(diff, ord=2))

    def should_exit(self, state_current: np.ndarray) -> bool:
        """Check if the loop should exit based on convergence criteria."""
        if not self._history:
            return False

        state_previous = self._history[-1].state_vector
        l2_norm = self.compute_l2_norm(state_current, state_previous)

        if l2_norm < self.epsilon:
            self._converging_count += 1
        else:
            self._converging_count = 0

        return self._converging_count >= self.patience

    def step(self, state_vector: np.ndarray, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Record a single iteration step."""
        iteration = len(self._history)
        loop_state = LoopState(
            iteration=iteration,
            state_vector=state_vector.copy(),
            metrics=metrics or {},
        )

        l2_norm = None
        converging = False
        if self._history:
            l2_norm = self.compute_l2_norm(state_vector, self._history[-1].state_vector)
            converging = l2_norm < self.epsilon
            if converging:
                self._converging_count += 1
            else:
                self._converging_count = 0

        self._history.append(loop_state)

        return {
            "iteration": iteration,
            "l2_norm": l2_norm,
            "converging": converging,
            "converging_count": self._converging_count,
            "should_exit": self._converging_count >= self.patience,
            "metrics": metrics or {},
        }

    def run(
        self,
        step_fn: Callable[[int, Optional[np.ndarray]], tuple],
        initial_state: Optional[np.ndarray] = None,
    ) -> ConvergenceResult:
        """
        Run the convergence loop.

        Args:
            step_fn: Function(iteration, previous_state) -> (new_state_vector, metrics_dict)
            initial_state: Optional initial state vector
        """
        self.reset()
        l2_norms: List[float] = []

        for i in range(self.max_iterations):
            prev_state = self._history[-1].state_vector if self._history else initial_state

            new_state, metrics = step_fn(i, prev_state)
            if not isinstance(new_state, np.ndarray):
                new_state = np.array(new_state, dtype=np.float64)

            step_result = self.step(new_state, metrics)

            if step_result["l2_norm"] is not None:
                l2_norms.append(step_result["l2_norm"])

            logger.info(
                f"Iteration {i}: L2={step_result['l2_norm']}, "
                f"converging={step_result['converging']}, "
                f"count={step_result['converging_count']}"
            )

            if step_result["should_exit"]:
                return ConvergenceResult(
                    converged=True,
                    iterations=i + 1,
                    final_state=new_state,
                    history=[
                        {"iteration": s.iteration, "metrics": s.metrics}
                        for s in self._history
                    ],
                    l2_norm_history=l2_norms,
                    exit_reason=f"Converged: L2 norm < {self.epsilon} for {self.patience} consecutive iterations",
                )

        final_state = self._history[-1].state_vector if self._history else None
        return ConvergenceResult(
            converged=False,
            iterations=self.max_iterations,
            final_state=final_state,
            history=[
                {"iteration": s.iteration, "metrics": s.metrics}
                for s in self._history
            ],
            l2_norm_history=l2_norms,
            exit_reason=f"Max iterations ({self.max_iterations}) reached without convergence",
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        return {
            "iterations_completed": len(self._history),
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "converging_count": self._converging_count,
            "objective": self.objective,
            "last_l2_norm": (
                self.compute_l2_norm(
                    self._history[-1].state_vector,
                    self._history[-2].state_vector
                )
                if len(self._history) >= 2
                else None
            ),
        }
