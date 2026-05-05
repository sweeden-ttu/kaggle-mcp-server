"""State convergence loop controller.

Implements an iterative refinement loop with L2-norm convergence detection:
  exit when ||state[n] - state[n-1]||_2 < epsilon
"""

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

StateVector = List[float]


def l2_norm(a: StateVector, b: StateVector) -> float:
    """Compute L2 (Euclidean) norm of the difference between two state vectors."""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        min_len = min(len(a), len(b))
        arr_a = arr_a[:min_len]
        arr_b = arr_b[:min_len]
    return float(np.linalg.norm(arr_a - arr_b))


def _default_state_dim(expert_count: int) -> int:
    """Heuristic: state vector dimension based on number of experts."""
    return max(4, expert_count * 2)


class ConvergenceLoop:
    """Manages an iterative state-convergence loop for competition entry building."""

    def __init__(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        db=None,
    ):
        self.competition = competition
        self.experts = experts
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.db = db

        self.run_id = str(uuid.uuid4())[:8]
        self.states: List[StateVector] = []
        self.norms: List[float] = []
        self.converged = False
        self.iteration = 0

        if db is not None:
            expert_ids = [e.get("expert_id", "") for e in experts]
            db.create_run(self.run_id, competition, expert_ids)

    def _initial_state(self) -> StateVector:
        """Generate an initial state vector from expert metadata."""
        dim = _default_state_dim(len(self.experts))
        state = []
        for i, expert in enumerate(self.experts):
            formula = expert.get("formula", {})
            metrics = formula.get("metrics", [])
            state.append(len(metrics) / 10.0)
            concepts = expert.get("metadata", {}).get("concepts", [])
            state.append(len(concepts) / 50.0)
        while len(state) < dim:
            state.append(0.0)
        return state[:dim]

    def _step_fn(
        self,
        state: StateVector,
        iteration: int,
        step_callback: Optional[Callable] = None,
    ) -> StateVector:
        """Advance the state by one iteration.

        If a step_callback is provided, it receives (state, iteration, experts)
        and should return the new state vector. Otherwise a default damped
        perturbation is applied.
        """
        if step_callback:
            return step_callback(state, iteration, self.experts)

        arr = np.array(state, dtype=np.float64)
        decay = 0.5 ** iteration
        perturbation = np.random.default_rng(seed=iteration).normal(
            0, 0.1 * decay, size=arr.shape
        )
        new_state = arr + perturbation
        return new_state.tolist()

    def run(
        self,
        step_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute the convergence loop.

        Args:
            step_callback: Optional (state, iteration, experts) -> new_state.

        Returns:
            Dict with run_id, converged, iterations, final_norm, states.
        """
        current_state = self._initial_state()
        self.states.append(current_state)
        converging_streak = 0

        for i in range(self.max_iterations):
            self.iteration = i + 1
            new_state = self._step_fn(current_state, self.iteration, step_callback)
            self.states.append(new_state)

            norm = l2_norm(new_state, current_state)
            self.norms.append(norm)

            logger.info(
                "Iteration %d: L2 norm = %.6f (epsilon = %.6f)",
                self.iteration,
                norm,
                self.epsilon,
            )

            if norm < self.epsilon:
                converging_streak += 1
                if converging_streak >= self.patience:
                    self.converged = True
                    logger.info(
                        "Converged after %d iterations (patience %d met)",
                        self.iteration,
                        self.patience,
                    )
                    break
            else:
                converging_streak = 0

            current_state = new_state

        final_norm = self.norms[-1] if self.norms else float("inf")

        if self.db is not None:
            self.db.update_run(
                self.run_id,
                states=self.states,
                converged=self.converged,
                final_norm=final_norm,
                iterations=self.iteration,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )

        return {
            "run_id": self.run_id,
            "competition": self.competition,
            "converged": self.converged,
            "iterations": self.iteration,
            "final_norm": final_norm,
            "norms": self.norms,
            "expert_count": len(self.experts),
            "epsilon": self.epsilon,
            "patience": self.patience,
        }


def build_competition_entry(
    competition: str,
    experts: List[Dict[str, Any]],
    embedding_engine=None,
    db=None,
) -> Dict[str, Any]:
    """Build a Kaggle competition entry using relevant experts.

    Orchestrates:
    1. RAG-based expert selection (if embedding_engine provided)
    2. Skill collection from selected experts
    3. Strategy assembly
    4. Notebook plan generation

    Returns a structured entry plan.
    """
    if embedding_engine:
        ranked = embedding_engine.infer_relevant_experts(competition, experts)
        selected = ranked[:5] if ranked else experts[:3]
    else:
        selected = experts[:3]

    all_skills = []
    all_capabilities = []
    all_metrics = set()

    for expert in selected:
        all_skills.extend(expert.get("skills", []))
        all_capabilities.extend(expert.get("capabilities", []))
        formula = expert.get("formula", {})
        all_metrics.update(formula.get("metrics", []))

    unique_skills = sorted(set(all_skills))

    strategy_parts = []
    for expert in selected:
        strategy_parts.append(
            f"[{expert.get('expert_name', 'Unknown')}] {expert.get('strategy', 'N/A')}"
        )

    notebook_plan = {
        "competition": competition,
        "experts_used": [
            {
                "name": e.get("expert_name"),
                "slug": e.get("slug"),
                "relevance": e.get("relevance_score", 0),
            }
            for e in selected
        ],
        "skills": unique_skills,
        "capabilities": sorted(set(all_capabilities)),
        "metrics": sorted(all_metrics),
        "strategy_pipeline": strategy_parts,
        "objective": "minimize_validation_loss",
        "loop_config": {
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    return notebook_plan
