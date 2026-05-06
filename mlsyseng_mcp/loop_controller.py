"""State convergence loop controller for the MoE system.

Implements the iterative refinement loop that drives expert-based competition
entry building. Uses L2-norm convergence checking with patience.

Exit condition: ||state[n] - state[n-1]||_2 < epsilon
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""
    objective: str = "minimize_validation_loss"
    exit_condition: str = "||state[n] - state[n-1]||_2 < epsilon"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoopConfig":
        return cls(
            objective=d.get("objective", cls.objective),
            exit_condition=d.get("exit_condition", cls.exit_condition),
            epsilon=d.get("epsilon", cls.epsilon),
            max_iterations=d.get("max_iterations", cls.max_iterations),
            patience=d.get("patience", cls.patience),
        )


@dataclass
class LoopState:
    """Tracks the current state of a convergence loop."""
    competition: str
    iteration: int = 0
    state_vector: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    convergence_count: int = 0


def l2_norm_diff(v1: List[float], v2: List[float]) -> float:
    """Compute the L2 norm of the difference between two vectors.

    If vectors are different lengths, pads the shorter one with zeros.
    """
    max_len = max(len(v1), len(v2))
    a = v1 + [0.0] * (max_len - len(v1))
    b = v2 + [0.0] * (max_len - len(v2))
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def check_convergence(
    current: List[float],
    previous: List[float],
    epsilon: float = 0.001,
) -> bool:
    """Check if the state has converged (L2 diff < epsilon)."""
    if not previous:
        return False
    diff = l2_norm_diff(current, previous)
    return diff < epsilon


class LoopController:
    """Manages the state convergence loop for competition entries.

    Each iteration:
    1. Collects expert recommendations
    2. Updates the state vector (metric scores)
    3. Checks convergence via L2 norm
    4. Terminates on convergence or max iterations
    """

    def __init__(self, config: Optional[LoopConfig] = None, db=None):
        self.config = config or LoopConfig()
        self.db = db
        self._active_loops: Dict[str, LoopState] = {}

    def start_loop(self, competition: str) -> LoopState:
        """Initialize a new convergence loop for a competition."""
        state = LoopState(competition=competition)
        self._active_loops[competition] = state
        logger.info("Started convergence loop for '%s'", competition)
        return state

    def get_loop(self, competition: str) -> Optional[LoopState]:
        return self._active_loops.get(competition)

    def step(
        self,
        competition: str,
        new_state_vector: List[float],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Advance the loop by one iteration.

        Args:
            competition: Competition name.
            new_state_vector: Updated state vector (e.g., metric scores).
            metrics: Optional dict of named metric values.

        Returns:
            Dict with iteration info, convergence status, and l2_diff.
        """
        state = self._active_loops.get(competition)
        if state is None:
            state = self.start_loop(competition)

        prev_vector = state.state_vector
        state.iteration += 1
        state.state_vector = new_state_vector
        state.metrics = metrics or {}

        l2_diff = l2_norm_diff(new_state_vector, prev_vector) if prev_vector else float("inf")
        is_converging = l2_diff < self.config.epsilon

        if is_converging:
            state.convergence_count += 1
        else:
            state.convergence_count = 0

        state.converged = state.convergence_count >= self.config.patience

        iteration_record = {
            "iteration": state.iteration,
            "state_vector": new_state_vector,
            "metrics": state.metrics,
            "l2_diff": l2_diff,
            "is_converging": is_converging,
            "convergence_count": state.convergence_count,
            "converged": state.converged,
        }
        state.history.append(iteration_record)

        if self.db:
            self.db.save_state_snapshot(
                competition=competition,
                iteration=state.iteration,
                state_vector=new_state_vector,
                metrics=state.metrics,
            )

        should_stop = state.converged or state.iteration >= self.config.max_iterations

        result = {
            **iteration_record,
            "should_stop": should_stop,
            "reason": self._stop_reason(state),
            "objective": self.config.objective,
        }

        logger.info(
            "Loop [%s] iter=%d, l2_diff=%.6f, converging=%s, converged=%s",
            competition, state.iteration, l2_diff, is_converging, state.converged,
        )

        return result

    def _stop_reason(self, state: LoopState) -> str:
        if state.converged:
            return f"Converged: {self.config.patience} consecutive iterations below epsilon={self.config.epsilon}"
        if state.iteration >= self.config.max_iterations:
            return f"Max iterations reached: {self.config.max_iterations}"
        return "running"

    def run_loop(
        self,
        competition: str,
        step_fn: Callable[[int, List[float]], List[float]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run the full convergence loop with a step function.

        Args:
            competition: Competition name.
            step_fn: Function that takes (iteration, current_state) and returns new_state_vector.
            initial_state: Optional initial state vector.

        Returns:
            Final loop summary.
        """
        state = self.start_loop(competition)
        current = initial_state or [0.0]

        for _ in range(self.config.max_iterations):
            new_vector = step_fn(state.iteration + 1, current)
            result = self.step(competition, new_vector)
            current = new_vector

            if result["should_stop"]:
                break

        return self.get_loop_summary(competition)

    def get_loop_summary(self, competition: str) -> Dict[str, Any]:
        """Get a summary of the loop's execution."""
        state = self._active_loops.get(competition)
        if state is None:
            return {"error": f"No active loop for '{competition}'"}

        return {
            "competition": competition,
            "total_iterations": state.iteration,
            "converged": state.converged,
            "final_state_vector": state.state_vector,
            "final_metrics": state.metrics,
            "config": {
                "objective": self.config.objective,
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "patience": self.config.patience,
            },
            "history": state.history,
        }

    def reset_loop(self, competition: str):
        """Reset a loop for a competition."""
        if competition in self._active_loops:
            del self._active_loops[competition]
