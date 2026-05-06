"""State convergence loop controller for MLSysEng MoE."""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state vector at a given iteration."""

    iteration: int
    values: np.ndarray
    metrics: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.values))


@dataclass
class ConvergenceResult:
    """Result of a convergence loop execution."""

    converged: bool
    iterations_run: int
    final_state: Optional[LoopState]
    history: list[LoopState]
    l2_norms: list[float]
    exit_reason: str


class LoopController:
    """
    Controls the state convergence loop.

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
        self.history: list[LoopState] = []
        self.l2_norms: list[float] = []

    def reset(self) -> None:
        """Reset the controller state."""
        self.history = []
        self.l2_norms = []

    def compute_l2_distance(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """Compute L2 norm of the difference between two state vectors."""
        return float(np.linalg.norm(state_a - state_b))

    def check_convergence(self) -> tuple[bool, str]:
        """Check if the loop has converged based on recent history."""
        if len(self.history) < 2:
            return False, "insufficient_history"

        current = self.history[-1].values
        previous = self.history[-2].values
        l2_dist = self.compute_l2_distance(current, previous)
        self.l2_norms.append(l2_dist)

        if l2_dist < self.epsilon:
            consecutive_converging = 0
            for norm in reversed(self.l2_norms):
                if norm < self.epsilon:
                    consecutive_converging += 1
                else:
                    break

            if consecutive_converging >= self.patience:
                return True, f"converged_patience_{self.patience}"
            return False, f"below_epsilon_count={consecutive_converging}/{self.patience}"

        return False, f"l2_distance={l2_dist:.6f}"

    def run(
        self,
        step_fn: Callable[[int, Optional[LoopState]], np.ndarray],
        initial_state: Optional[np.ndarray] = None,
        metrics_fn: Optional[Callable[[np.ndarray], dict]] = None,
    ) -> ConvergenceResult:
        """
        Run the convergence loop.

        Args:
            step_fn: Function that takes (iteration, previous_state) and returns new state vector
            initial_state: Optional initial state vector
            metrics_fn: Optional function to compute metrics from state
        """
        self.reset()

        if initial_state is not None:
            init_loop_state = LoopState(iteration=0, values=initial_state)
            if metrics_fn:
                init_loop_state.metrics = metrics_fn(initial_state)
            self.history.append(init_loop_state)

        for i in range(1, self.max_iterations + 1):
            prev_state = self.history[-1] if self.history else None

            try:
                new_values = step_fn(i, prev_state)
            except Exception as e:
                logger.error(f"Step function failed at iteration {i}: {e}")
                return ConvergenceResult(
                    converged=False,
                    iterations_run=i,
                    final_state=self.history[-1] if self.history else None,
                    history=self.history,
                    l2_norms=self.l2_norms,
                    exit_reason=f"step_error: {e}",
                )

            new_state = LoopState(iteration=i, values=new_values)
            if metrics_fn:
                new_state.metrics = metrics_fn(new_values)
            self.history.append(new_state)

            converged, reason = self.check_convergence()
            logger.info(f"Iteration {i}: {reason}")

            if converged:
                return ConvergenceResult(
                    converged=True,
                    iterations_run=i,
                    final_state=new_state,
                    history=self.history,
                    l2_norms=self.l2_norms,
                    exit_reason=reason,
                )

        return ConvergenceResult(
            converged=False,
            iterations_run=self.max_iterations,
            final_state=self.history[-1] if self.history else None,
            history=self.history,
            l2_norms=self.l2_norms,
            exit_reason="max_iterations_reached",
        )


def create_competition_step_fn(
    expert_skills: list[str],
    competition_name: str,
) -> Callable[[int, Optional[LoopState]], np.ndarray]:
    """
    Create a step function for competition entry building.

    Each iteration refines the state vector representing:
    [validation_score, feature_importance, model_complexity, submission_confidence]
    """

    def step_fn(iteration: int, prev_state: Optional[LoopState]) -> np.ndarray:
        if prev_state is None:
            return np.array([0.5, 0.3, 0.7, 0.2])

        prev = prev_state.values
        improvement = 0.1 / iteration
        noise = np.random.normal(0, 0.01 / iteration, size=prev.shape)

        new_state = prev + improvement * (1.0 - prev) + noise
        new_state = np.clip(new_state, 0.0, 1.0)
        return new_state

    return step_fn


def run_competition_loop(
    competition_name: str,
    expert_skills: list[str],
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> dict:
    """Run a convergence loop for a competition entry."""
    controller = LoopController(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )

    step_fn = create_competition_step_fn(expert_skills, competition_name)
    result = controller.run(step_fn)

    return {
        "converged": result.converged,
        "iterations_run": result.iterations_run,
        "exit_reason": result.exit_reason,
        "final_state": result.final_state.values.tolist() if result.final_state else None,
        "l2_norms": result.l2_norms,
        "final_metrics": result.final_state.metrics if result.final_state else {},
    }
