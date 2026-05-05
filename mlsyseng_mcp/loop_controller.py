"""State convergence loop for the MoE system."""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import MLSysEngDB

logger = logging.getLogger(__name__)


def l2_norm(v1: List[float], v2: List[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Vector length mismatch: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def _initial_state(num_experts: int) -> List[float]:
    """Create an initial state vector (uniform weights)."""
    if num_experts == 0:
        return [0.0]
    weight = 1.0 / num_experts
    return [weight] * num_experts


class LoopController:
    """
    Runs the state convergence loop for competition entry building.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    """

    def __init__(
        self,
        db: Optional[MLSysEngDB] = None,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
    ):
        self.db = db or MLSysEngDB()
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def run(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        step_fn: Optional[Callable[[List[float], List[Dict[str, Any]], int], List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the convergence loop.

        Args:
            competition: Competition name/slug
            experts: List of expert definitions
            step_fn: Optional function(state, experts, iteration) -> new_state.
                      If None, uses a default decay step.

        Returns:
            Dict with final_state, iterations, converged, history
        """
        num_experts = len(experts)
        state = _initial_state(num_experts)
        step = step_fn or self._default_step

        history: List[Dict[str, Any]] = []
        converging_count = 0
        converged = False
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            prev_state = state[:]
            state = step(state, experts, iteration)

            norm = l2_norm(state, prev_state)
            is_converging = norm < self.epsilon

            if is_converging:
                converging_count += 1
            else:
                converging_count = 0

            record = {
                "iteration": iteration,
                "state": state[:],
                "l2_norm": norm,
                "converging": is_converging,
                "converging_streak": converging_count,
            }
            history.append(record)

            self.db.insert_convergence_run(
                competition=competition,
                iteration=iteration,
                state_vector=state,
                l2_norm=norm,
                converged=is_converging,
            )

            logger.info(
                "Iteration %d: L2=%.6f converging=%s streak=%d",
                iteration, norm, is_converging, converging_count,
            )

            if converging_count >= self.patience:
                converged = True
                break

        return {
            "competition": competition,
            "final_state": state,
            "iterations": iteration,
            "converged": converged,
            "final_l2_norm": history[-1]["l2_norm"] if history else 0.0,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "history": history,
            "expert_weights": self._state_to_weights(state, experts),
        }

    def _default_step(
        self,
        state: List[float],
        experts: List[Dict[str, Any]],
        iteration: int,
    ) -> List[float]:
        """
        Default convergence step: exponential decay toward uniform.
        Simulates experts refining their weight allocations.
        """
        n = len(state)
        if n == 0:
            return state

        decay = 0.5 ** iteration
        uniform = 1.0 / n

        new_state = []
        for s in state:
            new_val = s + decay * (uniform - s)
            new_state.append(new_val)

        total = sum(new_state)
        if total > 0:
            new_state = [v / total for v in new_state]

        return new_state

    @staticmethod
    def _state_to_weights(
        state: List[float], experts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map state vector to expert weight assignments."""
        weights = []
        for i, expert in enumerate(experts):
            w = state[i] if i < len(state) else 0.0
            weights.append({
                "expert_name": expert.get("expert_name", f"expert_{i}"),
                "slug": expert.get("slug", f"expert_{i}"),
                "weight": round(w, 6),
            })
        return weights

    def build_competition_entry(
        self,
        competition: str,
        experts: List[Dict[str, Any]],
        convergence_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a competition entry from expert recommendations post-convergence.
        """
        if convergence_result is None:
            convergence_result = self.run(competition, experts)

        weights = convergence_result.get("expert_weights", [])
        top_experts = sorted(weights, key=lambda w: -w["weight"])

        all_skills = []
        for expert_def in experts:
            for skill in expert_def.get("skills", []):
                if skill not in all_skills:
                    all_skills.append(skill)

        strategy_parts = []
        for w in top_experts:
            expert_def = next(
                (e for e in experts if e.get("slug") == w["slug"]), None
            )
            if expert_def:
                strategy_parts.append({
                    "expert": w["expert_name"],
                    "weight": w["weight"],
                    "strategy": expert_def.get("strategy", ""),
                    "capabilities": expert_def.get("capabilities", []),
                })

        return {
            "competition": competition,
            "convergence": {
                "converged": convergence_result["converged"],
                "iterations": convergence_result["iterations"],
                "final_l2": convergence_result["final_l2_norm"],
            },
            "selected_experts": strategy_parts,
            "combined_skills": all_skills,
            "recommended_approach": self._synthesize_approach(strategy_parts),
        }

    @staticmethod
    def _synthesize_approach(expert_strategies: List[Dict[str, Any]]) -> str:
        """Synthesize a combined approach from weighted expert strategies."""
        if not expert_strategies:
            return "No experts available. Use default: EDA → Baseline → Iterate."

        top = expert_strategies[0]
        approach_parts = [
            f"Lead expert: {top['expert']} (weight: {top['weight']:.3f})",
            f"Strategy: {top.get('strategy', 'N/A')}",
            "Steps:",
            "  1. EDA and data understanding",
            "  2. Build baseline using lead expert approach",
        ]

        for i, es in enumerate(expert_strategies[1:], start=3):
            approach_parts.append(
                f"  {i}. Apply {es['expert']} insights (weight: {es['weight']:.3f})"
            )

        approach_parts.extend([
            f"  {len(expert_strategies) + 2}. Hyperparameter optimization",
            f"  {len(expert_strategies) + 3}. Final submission",
        ])

        return "\n".join(approach_parts)
