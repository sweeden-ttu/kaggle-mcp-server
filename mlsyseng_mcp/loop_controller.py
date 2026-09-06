"""State convergence loop controller for MLSysEng MoE system.

Implements the iterative refinement loop with exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

Uses L2 norm of state vector difference as convergence metric.
"""

import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)


def l2_norm(vec: List[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def l2_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute L2 distance between two vectors."""
    if len(vec_a) != len(vec_b):
        min_len = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_len]
        vec_b = vec_b[:min_len]
    diff = [a - b for a, b in zip(vec_a, vec_b)]
    return l2_norm(diff)


class ConvergenceState:
    """Tracks the convergence state of an iterative loop."""

    def __init__(
        self,
        competition: str,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.competition = competition
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.history: List[Dict[str, Any]] = []
        self.consecutive_converging = 0

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    @property
    def last_state(self) -> Optional[List[float]]:
        if self.history:
            return self.history[-1]["state_vector"]
        return None

    @property
    def converged(self) -> bool:
        return self.consecutive_converging >= self.patience

    @property
    def should_stop(self) -> bool:
        if self.current_iteration >= self.max_iterations:
            return True
        return self.converged

    def update(self, state_vector: List[float], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Update the convergence state with a new state vector.

        Returns iteration results including L2 norm and convergence status.
        """
        prev_state = self.last_state
        norm = 0.0

        if prev_state is not None:
            norm = l2_distance(state_vector, prev_state)
            if norm < self.epsilon:
                self.consecutive_converging += 1
            else:
                self.consecutive_converging = 0
        else:
            self.consecutive_converging = 0

        iteration_data = {
            "iteration": self.current_iteration,
            "state_vector": state_vector,
            "l2_norm": norm,
            "epsilon": self.epsilon,
            "below_epsilon": norm < self.epsilon if prev_state else False,
            "consecutive_converging": self.consecutive_converging,
            "converged": self.converged,
            "should_stop": self.should_stop,
            "metadata": metadata or {},
        }

        self.history.append(iteration_data)
        return iteration_data


class LoopController:
    """Controls the MoE state convergence loop for competition entries."""

    def __init__(self, db: Database):
        self.db = db

    def create_state(
        self,
        competition: str,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ) -> ConvergenceState:
        """Create a new convergence state tracker."""
        return ConvergenceState(
            competition=competition,
            epsilon=epsilon,
            max_iterations=max_iterations,
            patience=patience,
        )

    def run_loop(
        self,
        competition: str,
        step_fn: Callable[[int, Optional[List[float]]], Tuple[List[float], Dict]],
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ) -> Dict[str, Any]:
        """Run the convergence loop with a step function.

        Args:
            competition: Competition identifier
            step_fn: Function(iteration, prev_state) -> (new_state_vector, metadata)
            epsilon: Convergence threshold
            max_iterations: Maximum iterations
            patience: Consecutive converging iterations needed

        Returns:
            Final loop status with history
        """
        state = self.create_state(competition, epsilon, max_iterations, patience)

        while not state.should_stop:
            new_vector, metadata = step_fn(state.current_iteration, state.last_state)
            result = state.update(new_vector, metadata)

            self.db.save_convergence_state(
                competition=competition,
                iteration=result["iteration"],
                state_vector=new_vector,
                l2_norm=result["l2_norm"],
                converged=result["converged"],
                metadata=metadata,
            )

            logger.info(
                f"[{competition}] Iteration {result['iteration']}: "
                f"L2={result['l2_norm']:.6f}, "
                f"converging={result['consecutive_converging']}/{patience}"
            )

            if result["converged"]:
                logger.info(f"[{competition}] Converged after {result['iteration']} iterations")
                break

        return {
            "competition": competition,
            "total_iterations": state.current_iteration,
            "converged": state.converged,
            "final_l2_norm": state.history[-1]["l2_norm"] if state.history else None,
            "reason": "converged" if state.converged else "max_iterations",
            "history": [
                {
                    "iteration": h["iteration"],
                    "l2_norm": h["l2_norm"],
                    "below_epsilon": h["below_epsilon"],
                }
                for h in state.history
            ],
        }

    def build_competition_state_vector(
        self,
        expert_scores: Dict[str, float],
        skill_activations: Dict[str, float],
    ) -> List[float]:
        """Build a state vector from expert scores and skill activations.

        The state vector combines expert relevance scores and skill activation levels
        into a single vector for convergence tracking.
        """
        expert_values = [expert_scores.get(k, 0.0) for k in sorted(expert_scores.keys())]
        skill_values = [skill_activations.get(k, 0.0) for k in sorted(skill_activations.keys())]
        return expert_values + skill_values

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        """Get the full convergence history for a competition."""
        return self.db.get_convergence_history(competition)

    def resume_loop(
        self,
        competition: str,
        step_fn: Callable[[int, Optional[List[float]]], Tuple[List[float], Dict]],
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ) -> Dict[str, Any]:
        """Resume a previously started convergence loop."""
        history = self.get_convergence_history(competition)
        state = self.create_state(competition, epsilon, max_iterations, patience)

        for entry in history:
            state.update(entry["state_vector"], entry.get("metadata", {}))

        if state.should_stop:
            return {
                "competition": competition,
                "total_iterations": state.current_iteration,
                "converged": state.converged,
                "final_l2_norm": state.history[-1]["l2_norm"] if state.history else None,
                "reason": "already_converged" if state.converged else "max_iterations_reached",
                "resumed": True,
            }

        while not state.should_stop:
            new_vector, metadata = step_fn(state.current_iteration, state.last_state)
            result = state.update(new_vector, metadata)

            self.db.save_convergence_state(
                competition=competition,
                iteration=result["iteration"],
                state_vector=new_vector,
                l2_norm=result["l2_norm"],
                converged=result["converged"],
                metadata=metadata,
            )

            if result["converged"]:
                break

        return {
            "competition": competition,
            "total_iterations": state.current_iteration,
            "converged": state.converged,
            "final_l2_norm": state.history[-1]["l2_norm"] if state.history else None,
            "reason": "converged" if state.converged else "max_iterations",
            "resumed": True,
            "history": [
                {
                    "iteration": h["iteration"],
                    "l2_norm": h["l2_norm"],
                    "below_epsilon": h["below_epsilon"],
                }
                for h in state.history
            ],
        }
