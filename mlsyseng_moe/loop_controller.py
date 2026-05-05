"""State convergence loop controller for the MoE system.

Implements the iterative improvement loop with L2-norm exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

The loop runs experts in sequence, tracking state vectors that represent
the quality/progress of the competition entry. Convergence is detected
when the state stops changing significantly.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state of a convergence loop iteration."""
    iteration: int
    state_vector: np.ndarray
    metrics: dict = field(default_factory=dict)
    expert_outputs: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    exit_condition: str = "||state[n] - state[n-1]||_2 < epsilon"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3
    state_dim: int = 8


class ConvergenceLoop:
    """Manages the state convergence loop for expert-driven iteration."""

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: list[LoopState] = []
        self.converged = False
        self.convergence_count = 0

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    @property
    def current_state(self) -> Optional[LoopState]:
        return self.history[-1] if self.history else None

    def initialize(self, initial_metrics: Optional[dict] = None) -> LoopState:
        """Initialize the loop with an initial state vector."""
        state_vector = np.zeros(self.config.state_dim)
        if initial_metrics:
            for i, (_, value) in enumerate(initial_metrics.items()):
                if i < self.config.state_dim and isinstance(value, (int, float)):
                    state_vector[i] = float(value)

        initial_state = LoopState(
            iteration=0,
            state_vector=state_vector,
            metrics=initial_metrics or {},
        )
        self.history.append(initial_state)
        return initial_state

    def step(self, new_metrics: dict, expert_output: str = "") -> LoopState:
        """Execute one iteration step and check convergence."""
        if not self.history:
            return self.initialize(new_metrics)

        prev_state = self.history[-1]
        new_vector = self._metrics_to_vector(new_metrics)

        new_state = LoopState(
            iteration=self.current_iteration,
            state_vector=new_vector,
            metrics=new_metrics,
            expert_outputs=[expert_output] if expert_output else [],
        )
        self.history.append(new_state)

        delta = self._compute_delta(prev_state.state_vector, new_vector)

        if delta < self.config.epsilon:
            self.convergence_count += 1
            logger.info(
                f"Iteration {new_state.iteration}: delta={delta:.6f} < epsilon={self.config.epsilon} "
                f"(converging {self.convergence_count}/{self.config.patience})"
            )
        else:
            self.convergence_count = 0
            logger.info(
                f"Iteration {new_state.iteration}: delta={delta:.6f} (not converging)"
            )

        if self.convergence_count >= self.config.patience:
            self.converged = True
            logger.info(f"Loop converged after {new_state.iteration} iterations")

        return new_state

    def should_continue(self) -> bool:
        """Check whether the loop should continue."""
        if self.converged:
            return False
        if self.current_iteration >= self.config.max_iterations:
            logger.info(f"Max iterations ({self.config.max_iterations}) reached")
            return False
        return True

    def _compute_delta(self, prev: np.ndarray, current: np.ndarray) -> float:
        """Compute L2 norm of state difference: ||state[n] - state[n-1]||_2."""
        diff = current - prev
        return float(np.linalg.norm(diff, ord=2))

    def _metrics_to_vector(self, metrics: dict) -> np.ndarray:
        """Convert a metrics dict to a fixed-size state vector."""
        vector = np.zeros(self.config.state_dim)
        for i, (_, value) in enumerate(metrics.items()):
            if i >= self.config.state_dim:
                break
            if isinstance(value, (int, float)):
                vector[i] = float(value)
        return vector

    def get_summary(self) -> dict:
        """Get a summary of the loop execution."""
        deltas = []
        for i in range(1, len(self.history)):
            delta = self._compute_delta(
                self.history[i - 1].state_vector,
                self.history[i].state_vector,
            )
            deltas.append(delta)

        return {
            "total_iterations": self.current_iteration,
            "converged": self.converged,
            "final_metrics": self.current_state.metrics if self.current_state else {},
            "deltas": deltas,
            "epsilon": self.config.epsilon,
            "patience": self.config.patience,
            "convergence_count": self.convergence_count,
        }


def run_convergence_loop(
    experts: list[dict],
    competition: str,
    initial_metrics: Optional[dict] = None,
    config: Optional[LoopConfig] = None,
    step_callback=None,
) -> dict:
    """Run the full convergence loop with a set of experts.

    Args:
        experts: List of expert definitions to cycle through.
        competition: Competition name/slug.
        initial_metrics: Starting metrics (or zeros).
        config: Loop configuration (epsilon, patience, max_iter).
        step_callback: Optional async callback(expert, iteration) -> metrics dict.

    Returns:
        Summary of the convergence loop execution.
    """
    loop = ConvergenceLoop(config)
    loop.initialize(initial_metrics or {"score": 0.0})

    iteration = 0
    while loop.should_continue():
        expert_idx = iteration % len(experts)
        expert = experts[expert_idx]

        logger.info(
            f"Iteration {iteration + 1}: Running expert '{expert.get('expert_name', 'unknown')}'"
        )

        if step_callback:
            new_metrics = step_callback(expert, iteration)
        else:
            new_metrics = _simulate_step(expert, iteration, loop.current_state)

        loop.step(new_metrics, expert_output=expert.get("expert_name", ""))
        iteration += 1

    return loop.get_summary()


def _simulate_step(
    expert: dict, iteration: int, current_state: Optional[LoopState]
) -> dict:
    """Simulate an expert step (used when no callback is provided).

    In production, this would invoke the expert's skills and return actual metrics.
    Here we simulate diminishing improvements.
    """
    base_metrics = current_state.metrics if current_state else {}
    decay = 0.5 ** (iteration + 1)

    new_metrics = {}
    for key, value in base_metrics.items():
        if isinstance(value, (int, float)):
            improvement = decay * 0.1
            new_metrics[key] = value + improvement
        else:
            new_metrics[key] = value

    if not new_metrics:
        new_metrics = {"score": decay * 0.5}

    return new_metrics
