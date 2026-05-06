"""State convergence loop controller for iterative expert-driven optimization."""

import logging
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LoopState:
    """Represents the state vector at a given iteration."""
    iteration: int
    values: np.ndarray
    metrics: dict = field(default_factory=dict)
    expert_contributions: list = field(default_factory=list)


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "LoopConfig":
        return cls(
            objective=d.get("objective", "minimize_validation_loss"),
            epsilon=d.get("epsilon", 0.001),
            max_iterations=d.get("max_iterations", 10),
            patience=d.get("patience", 3),
        )


class ConvergenceLoop:
    """Manages the state convergence loop for expert-driven optimization.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.history: list[LoopState] = []
        self.converged = False
        self.convergence_iteration: Optional[int] = None
        self._patience_counter = 0

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    @property
    def current_state(self) -> Optional[LoopState]:
        return self.history[-1] if self.history else None

    def initialize_state(self, dimension: int = 10) -> LoopState:
        """Initialize the state vector."""
        initial_values = np.zeros(dimension)
        state = LoopState(iteration=0, values=initial_values)
        self.history.append(state)
        return state

    def step(self, new_values: np.ndarray, metrics: Optional[dict] = None,
             expert_contributions: Optional[list] = None) -> dict:
        """Execute one iteration step and check convergence."""
        iteration = self.current_iteration

        if iteration >= self.config.max_iterations:
            return {
                "status": "max_iterations_reached",
                "iteration": iteration,
                "converged": False,
            }

        state = LoopState(
            iteration=iteration,
            values=new_values.copy(),
            metrics=metrics or {},
            expert_contributions=expert_contributions or [],
        )
        self.history.append(state)

        if len(self.history) >= 2:
            delta = self._compute_delta()
            is_converging = delta < self.config.epsilon

            if is_converging:
                self._patience_counter += 1
                if self._patience_counter >= self.config.patience:
                    self.converged = True
                    self.convergence_iteration = iteration
                    return {
                        "status": "converged",
                        "iteration": iteration,
                        "delta": float(delta),
                        "converged": True,
                        "patience_count": self._patience_counter,
                    }
            else:
                self._patience_counter = 0

            return {
                "status": "running",
                "iteration": iteration,
                "delta": float(delta),
                "converged": False,
                "patience_count": self._patience_counter,
            }

        return {
            "status": "initialized",
            "iteration": iteration,
            "converged": False,
        }

    def _compute_delta(self) -> float:
        """Compute L2 norm of state difference: ||state[n] - state[n-1]||_2."""
        if len(self.history) < 2:
            return float("inf")

        current = self.history[-1].values
        previous = self.history[-2].values
        return float(np.linalg.norm(current - previous, ord=2))

    def get_state_history(self) -> list[dict]:
        """Get serializable history of all states."""
        return [
            {
                "iteration": s.iteration,
                "values": s.values.tolist(),
                "metrics": s.metrics,
                "expert_contributions": s.expert_contributions,
            }
            for s in self.history
        ]

    def get_convergence_report(self) -> dict:
        """Generate a convergence report."""
        deltas = []
        for i in range(1, len(self.history)):
            delta = float(np.linalg.norm(
                self.history[i].values - self.history[i - 1].values, ord=2
            ))
            deltas.append(delta)

        return {
            "total_iterations": self.current_iteration,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "epsilon": self.config.epsilon,
            "max_iterations": self.config.max_iterations,
            "patience": self.config.patience,
            "deltas": deltas,
            "final_delta": deltas[-1] if deltas else None,
        }


def run_convergence_loop(
    experts: list[dict],
    initial_state: Optional[np.ndarray] = None,
    config: Optional[LoopConfig] = None,
    step_fn: Optional[Callable] = None,
) -> dict:
    """Run a full convergence loop with given experts.

    Args:
        experts: List of expert definitions to use.
        initial_state: Initial state vector (defaults to zeros).
        config: Loop configuration.
        step_fn: Custom step function. If None, uses a simulated step.

    Returns:
        Convergence report with full state history.
    """
    if config is None:
        config = LoopConfig()

    loop = ConvergenceLoop(config)
    dimension = len(experts) * 3 + 1

    if initial_state is None:
        initial_state = np.random.randn(dimension) * 0.1

    loop.initialize_state(dimension)
    loop.history[0].values = initial_state.copy()

    for i in range(config.max_iterations):
        if step_fn:
            new_values, metrics, contributions = step_fn(
                loop.current_state, experts, i
            )
        else:
            new_values, metrics, contributions = _default_step(
                loop.current_state, experts, i
            )

        result = loop.step(new_values, metrics, contributions)

        if result["converged"] or result["status"] == "max_iterations_reached":
            break

    report = loop.get_convergence_report()
    report["state_history"] = loop.get_state_history()
    report["experts_used"] = [e.get("expert_name", e.get("slug", "unknown")) for e in experts]
    return report


def _default_step(current_state: LoopState, experts: list[dict],
                  iteration: int) -> tuple[np.ndarray, dict, list]:
    """Default step function that simulates expert contributions with decay."""
    values = current_state.values.copy()
    decay = 0.5 ** (iteration + 1)
    perturbation = np.random.randn(len(values)) * decay

    values += perturbation

    metrics = {
        "loss": float(np.mean(np.abs(perturbation))),
        "delta_norm": float(np.linalg.norm(perturbation)),
    }

    contributions = [
        {"expert": e.get("slug", f"expert_{i}"), "weight": 1.0 / len(experts)}
        for i, e in enumerate(experts)
    ]

    return values, metrics, contributions
