"""State convergence loop controller for iterative competition entry building."""

import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)


def l2_norm(a: List[float], b: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _default_state_fn(
    iteration: int,
    previous_state: Optional[List[float]],
    experts: List[Dict[str, Any]],
    competition: str,
) -> List[float]:
    """Default state function that simulates convergence.

    In practice this would be replaced by actual model training metrics.
    Each dimension corresponds to an expert's contribution score.
    """
    n = max(len(experts), 1)
    if previous_state is None:
        return [1.0 / n] * n

    decay = 0.5 ** iteration
    return [v * (1.0 - decay * 0.1) for v in previous_state]


class LoopController:
    """Manages the state convergence loop for competition entries.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    """

    def __init__(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        db: Optional[Database] = None,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        state_fn: Optional[Callable] = None,
    ):
        self.competition = competition
        self.experts = experts
        self.db = db or Database()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.state_fn = state_fn or _default_state_fn

        self.history: List[Dict[str, Any]] = []
        self.current_state: Optional[List[float]] = None
        self.converged = False
        self.convergence_count = 0

    def step(self) -> Dict[str, Any]:
        """Execute one iteration of the convergence loop."""
        iteration = len(self.history)
        previous_state = self.current_state

        new_state = self.state_fn(
            iteration, previous_state, self.experts, self.competition
        )

        if previous_state is not None:
            norm = l2_norm(new_state, previous_state)
        else:
            norm = float("inf")

        is_converging = norm < self.epsilon
        if is_converging:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        self.converged = self.convergence_count >= self.patience

        step_result = {
            "iteration": iteration,
            "state": new_state,
            "previous_state": previous_state,
            "l2_norm": norm,
            "epsilon": self.epsilon,
            "is_converging": is_converging,
            "convergence_count": self.convergence_count,
            "converged": self.converged,
        }

        self.current_state = new_state
        self.history.append(step_result)

        self.db.save_convergence_state(
            self.competition, iteration, new_state, norm, self.converged
        )

        return step_result

    def run(self) -> Dict[str, Any]:
        """Run the convergence loop until exit condition or max iterations."""
        logger.info(
            f"Starting convergence loop for '{self.competition}' "
            f"(epsilon={self.epsilon}, max_iter={self.max_iterations}, patience={self.patience})"
        )

        while len(self.history) < self.max_iterations and not self.converged:
            step_result = self.step()
            logger.info(
                f"Iteration {step_result['iteration']}: "
                f"L2={step_result['l2_norm']:.6f}, "
                f"converging={step_result['is_converging']}, "
                f"patience={step_result['convergence_count']}/{self.patience}"
            )

        return {
            "competition": self.competition,
            "total_iterations": len(self.history),
            "converged": self.converged,
            "final_l2_norm": self.history[-1]["l2_norm"] if self.history else None,
            "final_state": self.current_state,
            "history": [
                {
                    "iteration": h["iteration"],
                    "l2_norm": h["l2_norm"],
                    "converged": h["converged"],
                }
                for h in self.history
            ],
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current loop status."""
        return {
            "competition": self.competition,
            "iterations_completed": len(self.history),
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "convergence_count": self.convergence_count,
            "patience": self.patience,
            "epsilon": self.epsilon,
            "current_state": self.current_state,
            "last_l2_norm": self.history[-1]["l2_norm"] if self.history else None,
        }


def build_competition_entry(
    competition: str,
    experts: List[Dict[str, Any]],
    db: Optional[Database] = None,
) -> Dict[str, Any]:
    """Build a competition entry using expert knowledge.

    Generates a structured plan combining all relevant experts' strategies.
    """
    if not experts:
        return {
            "competition": competition,
            "status": "no_experts",
            "message": "No relevant experts found for this competition",
        }

    plan_steps = []
    all_skills = set()
    all_metrics = set()

    for expert in experts:
        skills = expert.get("skills", [])
        if isinstance(skills, str):
            try:
                skills = json.loads(skills)
            except (json.JSONDecodeError, TypeError):
                skills = []
        all_skills.update(skills)

        formula = expert.get("formula", {})
        if isinstance(formula, str):
            try:
                formula = json.loads(formula)
            except (json.JSONDecodeError, TypeError):
                formula = {}
        metrics = formula.get("metrics", [])
        all_metrics.update(metrics)

        capabilities = expert.get("capabilities", [])
        if isinstance(capabilities, str):
            try:
                capabilities = json.loads(capabilities)
            except (json.JSONDecodeError, TypeError):
                capabilities = []

        plan_steps.append({
            "expert": expert.get("expert_name", "Unknown"),
            "slug": expert.get("slug", ""),
            "relevance": expert.get("relevance_score", 1.0),
            "strategy": expert.get("strategy", ""),
            "capabilities": capabilities,
            "skills": skills,
        })

    plan_steps.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    entry = {
        "competition": competition,
        "status": "ready",
        "experts_used": len(experts),
        "total_skills": sorted(all_skills),
        "metrics": sorted(all_metrics),
        "execution_plan": plan_steps,
        "pipeline": [
            "1. Data Loading & EDA",
            "2. Feature Engineering (guided by expert knowledge)",
            "3. Model Selection (based on expert strategies)",
            "4. Training with convergence loop",
            "5. Evaluation against metrics",
            "6. Submission",
        ],
    }

    return entry
