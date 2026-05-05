"""State convergence loop with L2 norm exit condition for the MoE system."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the convergence loop."""

    objective: str = "minimize_validation_loss"
    epsilon: float = 0.001
    max_iterations: int = 10
    patience: int = 3


@dataclass
class IterationState:
    """Snapshot of a single iteration."""

    iteration: int
    state_vector: List[float]
    l2_norm: Optional[float] = None
    converged: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


def l2_norm(v1: List[float], v2: List[float]) -> float:
    """Compute ||v1 - v2||_2."""
    if len(v1) != len(v2):
        raise ValueError(
            f"State vectors must have equal length: {len(v1)} vs {len(v2)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class LoopController:
    """
    Runs a state convergence loop.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    With patience: exits after `patience` consecutive converging iterations.
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        db: Optional[MLSysEngDatabase] = None,
    ):
        self.config = config or LoopConfig()
        self.db = db
        self.history: List[IterationState] = []
        self._consecutive_converging = 0

    def _check_convergence(self, current: List[float], previous: List[float]) -> tuple:
        """Check if current state has converged relative to previous."""
        norm = l2_norm(current, previous)
        is_below_epsilon = norm < self.config.epsilon
        return norm, is_below_epsilon

    def run(
        self,
        competition: str,
        step_fn: Callable[[int, Optional[List[float]]], List[float]],
        initial_state: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the convergence loop.

        Args:
            competition: Competition identifier for persistence.
            step_fn: Callable(iteration, previous_state) -> new_state_vector.
                      Called each iteration to produce the next state.
            initial_state: Optional initial state vector. If None, step_fn is
                           called with iteration=0 and previous_state=None.

        Returns:
            Summary dict with convergence results.
        """
        self.history.clear()
        self._consecutive_converging = 0

        if initial_state is not None:
            prev = initial_state
        else:
            prev = step_fn(0, None)

        first = IterationState(iteration=0, state_vector=prev)
        self.history.append(first)
        self._persist(competition, first)

        for i in range(1, self.config.max_iterations + 1):
            current = step_fn(i, prev)
            norm, converging = self._check_convergence(current, prev)

            if converging:
                self._consecutive_converging += 1
            else:
                self._consecutive_converging = 0

            converged = self._consecutive_converging >= self.config.patience

            state = IterationState(
                iteration=i,
                state_vector=current,
                l2_norm=norm,
                converged=converged,
            )
            self.history.append(state)
            self._persist(competition, state)

            logger.info(
                "Iteration %d: L2=%.6f, converging=%s, patience=%d/%d",
                i,
                norm,
                converging,
                self._consecutive_converging,
                self.config.patience,
            )

            if converged:
                logger.info("Converged at iteration %d", i)
                break

            prev = current

        return self._build_summary(competition)

    def _persist(self, competition: str, state: IterationState) -> None:
        if self.db is None:
            return
        self.db.save_convergence_state(
            competition=competition,
            iteration=state.iteration,
            state_vector=state.state_vector,
            l2_norm=state.l2_norm or 0.0,
            converged=state.converged,
        )

    def _build_summary(self, competition: str) -> Dict[str, Any]:
        last = self.history[-1]
        norms = [s.l2_norm for s in self.history if s.l2_norm is not None]

        return {
            "competition": competition,
            "converged": last.converged,
            "total_iterations": last.iteration,
            "final_l2_norm": last.l2_norm,
            "final_state": last.state_vector,
            "min_l2_norm": min(norms) if norms else None,
            "max_l2_norm": max(norms) if norms else None,
            "config": {
                "epsilon": self.config.epsilon,
                "max_iterations": self.config.max_iterations,
                "patience": self.config.patience,
                "objective": self.config.objective,
            },
            "history": [
                {
                    "iteration": s.iteration,
                    "l2_norm": s.l2_norm,
                    "converged": s.converged,
                }
                for s in self.history
            ],
        }


def run_evolve_loop(
    competition: str,
    db: MLSysEngDatabase,
    experts: list,
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> Dict[str, Any]:
    """
    High-level API: run a convergence loop for a competition using registered experts.

    The state vector encodes a score per expert; convergence means expert
    scores have stabilized (the system has "learned" which experts to prioritize).
    """
    config = LoopConfig(
        epsilon=epsilon,
        max_iterations=max_iterations,
        patience=patience,
    )
    controller = LoopController(config=config, db=db)

    num_experts = max(len(experts), 1)
    initial = [1.0 / num_experts] * num_experts

    def step_fn(iteration: int, prev: Optional[List[float]]) -> List[float]:
        if prev is None:
            return initial

        decay = 0.9 ** iteration
        new_state = []
        for j, val in enumerate(prev):
            perturbation = ((-1) ** j) * decay * 0.01
            new_state.append(max(0.0, val + perturbation))

        total = sum(new_state) or 1.0
        return [v / total for v in new_state]

    return controller.run(competition, step_fn, initial_state=initial)
