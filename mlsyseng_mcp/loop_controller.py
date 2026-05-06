"""State convergence loop for the MoE system.

Exit condition: ||state[n] - state[n-1]||_2 < epsilon
Convergence: L2 norm of state vector difference
Patience: exits after `patience` consecutive converging iterations
"""

import logging
import math
import uuid
from typing import Any, Callable, Dict, List, Optional

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)


def l2_norm(v1: List[float], v2: List[float]) -> float:
    """Compute the L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        raise ValueError(
            f"Vector length mismatch: {len(v1)} vs {len(v2)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _default_state_vector() -> List[float]:
    """Return a default initial state vector."""
    return [0.0, 0.0, 0.0, 0.0, 0.0]


class LoopController:
    """Controls the state convergence loop for expert-driven competition entries."""

    def __init__(
        self,
        db: MLSysEngDatabase,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def create_session(self) -> str:
        """Create a new convergence session."""
        return str(uuid.uuid4())

    def run_loop(
        self,
        session_id: Optional[str] = None,
        initial_state: Optional[List[float]] = None,
        step_fn: Optional[Callable[[List[float], int], List[float]]] = None,
        on_step: Optional[Callable[[int, List[float], float, bool], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the convergence loop.

        Args:
            session_id: Optional session identifier (auto-generated if None).
            initial_state: Initial state vector.
            step_fn: Function(current_state, iteration) -> next_state.
                      If None, a default decay function is used.
            on_step: Optional callback(iteration, state, l2_norm, converged).

        Returns:
            Dict with convergence results.
        """
        session_id = session_id or self.create_session()
        current_state = initial_state or _default_state_vector()

        if step_fn is None:
            step_fn = self._default_step_fn

        consecutive_converging = 0
        history = []

        self.db.save_state(
            session_id=session_id,
            iteration=0,
            state_vector=current_state,
            l2_norm=float("inf"),
            converged=False,
            metadata={"phase": "initial"},
        )

        for iteration in range(1, self.max_iterations + 1):
            previous_state = current_state
            current_state = step_fn(current_state, iteration)

            norm = l2_norm(current_state, previous_state)
            converged = norm < self.epsilon

            if converged:
                consecutive_converging += 1
            else:
                consecutive_converging = 0

            step_info = {
                "iteration": iteration,
                "l2_norm": norm,
                "converged": converged,
                "consecutive_converging": consecutive_converging,
                "state": current_state,
            }
            history.append(step_info)

            self.db.save_state(
                session_id=session_id,
                iteration=iteration,
                state_vector=current_state,
                l2_norm=norm,
                converged=converged,
                metadata={
                    "consecutive_converging": consecutive_converging,
                    "phase": "converging" if converged else "exploring",
                },
            )

            if on_step:
                on_step(iteration, current_state, norm, converged)

            logger.info(
                "Iteration %d: L2=%.6f, converged=%s, patience=%d/%d",
                iteration,
                norm,
                converged,
                consecutive_converging,
                self.patience,
            )

            if consecutive_converging >= self.patience:
                logger.info(
                    "Convergence achieved after %d iterations (patience=%d)",
                    iteration,
                    self.patience,
                )
                break

        final_converged = consecutive_converging >= self.patience

        return {
            "session_id": session_id,
            "converged": final_converged,
            "total_iterations": len(history),
            "final_l2_norm": history[-1]["l2_norm"] if history else float("inf"),
            "final_state": current_state,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "history": history,
        }

    def _default_step_fn(self, state: List[float], iteration: int) -> List[float]:
        """
        Default step function that applies exponential decay toward a target.
        Simulates convergence behavior for demonstration.
        """
        decay = 0.5 ** iteration
        target = [1.0] * len(state)
        return [
            s + (t - s) * (1.0 - decay)
            for s, t in zip(state, target)
        ]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a convergence session."""
        states = self.db.get_session_states(session_id)
        if not states:
            return {"session_id": session_id, "status": "not_found"}

        l2_norms = [s["l2_norm"] for s in states if s["l2_norm"] != float("inf")]

        return {
            "session_id": session_id,
            "total_iterations": len(states) - 1,
            "final_l2_norm": l2_norms[-1] if l2_norms else None,
            "converged": states[-1].get("converged", False),
            "l2_history": l2_norms,
            "final_state": states[-1].get("state_vector", []),
        }


class CompetitionLoopController(LoopController):
    """
    Extended loop controller that integrates expert knowledge
    for competition-specific convergence.
    """

    def __init__(
        self,
        db: MLSysEngDatabase,
        experts: List[Dict[str, Any]],
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        super().__init__(db, epsilon, max_iterations, patience)
        self.experts = experts

    def build_initial_state(self) -> List[float]:
        """Build initial state vector from expert configurations."""
        if not self.experts:
            return _default_state_vector()

        state = []
        for expert in self.experts:
            relevance = expert.get("relevance", 0.5)
            state.append(relevance)

        return state

    def expert_step_fn(self, state: List[float], iteration: int) -> List[float]:
        """
        Step function that uses expert knowledge to guide convergence.
        Each dimension corresponds to an expert's contribution weight.
        """
        new_state = []
        for i, val in enumerate(state):
            if i < len(self.experts):
                expert = self.experts[i]
                loop_config = expert.get("loop_config", {})
                eps = loop_config.get("epsilon", self.epsilon)
                decay = 0.5 ** iteration
                target = expert.get("relevance", 0.8)
                new_val = val + (target - val) * (1.0 - decay)
            else:
                new_val = val * 0.99
            new_state.append(new_val)

        return new_state

    def run_competition_loop(
        self,
        session_id: Optional[str] = None,
        on_step: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop using expert-driven step function."""
        initial_state = self.build_initial_state()
        return self.run_loop(
            session_id=session_id,
            initial_state=initial_state,
            step_fn=self.expert_step_fn,
            on_step=on_step,
        )
