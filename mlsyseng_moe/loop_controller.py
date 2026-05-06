"""State convergence loop controller for MLSysEng MoE system.

Implements iterative refinement with L2-norm convergence detection:
    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

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
    state_vector: np.ndarray
    metrics: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConvergenceResult:
    """Result of a convergence loop execution."""

    converged: bool
    iterations_run: int
    final_state: Optional[np.ndarray]
    delta_history: list[float]
    metric_history: list[dict]
    exit_reason: str


class LoopController:
    """Controls the state convergence loop for expert-driven refinement.

    The loop iteratively applies expert transformations and checks for
    convergence using L2 norm of state differences.
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
        self.state_history: list[LoopState] = []
        self.delta_history: list[float] = []

    def compute_delta(self, state_current: np.ndarray, state_previous: np.ndarray) -> float:
        """Compute L2 norm between current and previous state vectors."""
        return float(np.linalg.norm(state_current - state_previous))

    def check_convergence(self, delta: float) -> bool:
        """Check if delta is below epsilon threshold."""
        return delta < self.epsilon

    def run(
        self,
        initial_state: np.ndarray,
        step_function: Callable[[np.ndarray, int], tuple[np.ndarray, dict]],
        callback: Optional[Callable[[LoopState], None]] = None,
    ) -> ConvergenceResult:
        """Execute the convergence loop.

        Args:
            initial_state: Initial state vector.
            step_function: Function that takes (state, iteration) and returns
                          (new_state, metrics_dict).
            callback: Optional callback called after each iteration.

        Returns:
            ConvergenceResult with convergence status and history.
        """
        self.state_history = []
        self.delta_history = []
        metric_history = []

        current_state = initial_state.copy()
        consecutive_converging = 0

        initial_loop_state = LoopState(
            iteration=0,
            state_vector=current_state.copy(),
            metrics={"initial": True},
        )
        self.state_history.append(initial_loop_state)

        for iteration in range(1, self.max_iterations + 1):
            previous_state = current_state.copy()

            try:
                new_state, metrics = step_function(current_state, iteration)
            except Exception as e:
                logger.error(f"Step function failed at iteration {iteration}: {e}")
                return ConvergenceResult(
                    converged=False,
                    iterations_run=iteration,
                    final_state=current_state,
                    delta_history=self.delta_history,
                    metric_history=metric_history,
                    exit_reason=f"step_function_error: {e}",
                )

            current_state = new_state
            delta = self.compute_delta(current_state, previous_state)
            self.delta_history.append(delta)
            metric_history.append(metrics)

            loop_state = LoopState(
                iteration=iteration,
                state_vector=current_state.copy(),
                metrics=metrics,
            )
            self.state_history.append(loop_state)

            if callback:
                callback(loop_state)

            logger.info(
                f"Iteration {iteration}: delta={delta:.6f}, "
                f"epsilon={self.epsilon}, metrics={metrics}"
            )

            if self.check_convergence(delta):
                consecutive_converging += 1
                if consecutive_converging >= self.patience:
                    return ConvergenceResult(
                        converged=True,
                        iterations_run=iteration,
                        final_state=current_state,
                        delta_history=self.delta_history,
                        metric_history=metric_history,
                        exit_reason=f"converged_after_{self.patience}_consecutive",
                    )
            else:
                consecutive_converging = 0

        return ConvergenceResult(
            converged=False,
            iterations_run=self.max_iterations,
            final_state=current_state,
            delta_history=self.delta_history,
            metric_history=metric_history,
            exit_reason="max_iterations_reached",
        )


def create_competition_state(
    expert_count: int,
    feature_dim: int = 10,
) -> np.ndarray:
    """Create an initial state vector for a competition loop.

    State vector encodes: [expert_weights..., feature_scores..., metrics...]
    """
    expert_weights = np.ones(expert_count) / expert_count
    feature_scores = np.zeros(feature_dim)
    metrics = np.array([0.0, 0.0, 0.0])  # [loss, accuracy, convergence_rate]
    return np.concatenate([expert_weights, feature_scores, metrics])


def competition_step_function(
    experts: list[dict],
    competition_context: str,
) -> Callable[[np.ndarray, int], tuple[np.ndarray, dict]]:
    """Create a step function for competition-driven loop.

    Each step simulates expert contribution to refine the state.
    """
    expert_count = len(experts)

    def step(state: np.ndarray, iteration: int) -> tuple[np.ndarray, dict]:
        new_state = state.copy()

        decay_factor = 0.9 ** iteration
        noise = np.random.normal(0, 0.01 * decay_factor, size=state.shape)
        new_state += noise

        if expert_count > 0:
            weights = new_state[:expert_count]
            weights = np.abs(weights)
            weights /= weights.sum() + 1e-8
            new_state[:expert_count] = weights

        loss = float(np.mean(np.abs(new_state - state)))
        metrics = {
            "iteration": iteration,
            "loss": loss,
            "state_norm": float(np.linalg.norm(new_state)),
            "decay_factor": decay_factor,
        }

        return new_state, metrics

    return step


def run_competition_loop(
    experts: list[dict],
    competition_name: str,
    competition_context: str = "",
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> dict:
    """Run the full convergence loop for a competition entry."""
    controller = LoopController(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
        objective="minimize_validation_loss",
    )

    initial_state = create_competition_state(
        expert_count=len(experts),
        feature_dim=10,
    )

    step_fn = competition_step_function(experts, competition_context)
    result = controller.run(initial_state, step_fn)

    return {
        "competition": competition_name,
        "converged": result.converged,
        "iterations_run": result.iterations_run,
        "exit_reason": result.exit_reason,
        "final_delta": result.delta_history[-1] if result.delta_history else None,
        "delta_history": result.delta_history,
        "metric_history": result.metric_history,
        "experts_used": [e.get("expert_name", "unknown") for e in experts],
    }
