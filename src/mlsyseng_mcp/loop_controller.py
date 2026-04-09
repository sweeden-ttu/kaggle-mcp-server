"""State convergence loop controller for MLSysEng MoE.

Implements the iterative convergence loop with L2 norm exit condition:
    ||state[n] - state[n-1]||_2 < epsilon

The loop orchestrates expert recommendations and tracks convergence
toward a stable competition strategy.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional

from .database import Database
from .expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)

DEFAULT_EPSILON = 0.001
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_PATIENCE = 3


def l2_norm(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute L2 norm (Euclidean distance) between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def _encode_state(
    experts: List[Dict[str, Any]],
    scores: Dict[str, float],
    iteration: int,
) -> List[float]:
    """Encode current loop state as a numerical vector.

    The state vector captures: relevance scores for each expert,
    iteration progress, and aggregate metrics.
    """
    vector = []
    for expert in experts:
        slug = expert.get("slug", "")
        score = scores.get(slug, 0.0)
        vector.append(score)

    n_experts = len(experts) if experts else 1
    vector.append(iteration / 100.0)
    vector.append(sum(scores.values()) / max(n_experts, 1))

    return vector


class LoopController:
    """Controls the state convergence loop for competition strategy building."""

    def __init__(
        self,
        db: Optional[Database] = None,
        registry: Optional[ExpertRegistry] = None,
        epsilon: float = DEFAULT_EPSILON,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        patience: int = DEFAULT_PATIENCE,
    ):
        self.db = db or Database()
        self.registry = registry or ExpertRegistry(self.db)
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience

    def run(
        self,
        competition: str,
        description: str = "",
        on_iteration: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the convergence loop for a competition.

        Args:
            competition: Competition name/slug.
            description: Competition description for skill inference.
            on_iteration: Optional callback invoked after each iteration.

        Returns:
            Dict with convergence results including final experts and metrics.
        """
        experts = self.registry.list_experts()
        if not experts:
            return {
                "status": "no_experts",
                "message": "No experts registered. Run extract-knowledge first.",
            }

        previous_state: Optional[List[float]] = None
        consecutive_converging = 0
        history: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            scores = self._compute_relevance_scores(
                experts, competition, description, iteration
            )

            current_state = _encode_state(experts, scores, iteration)

            if previous_state is not None and len(current_state) == len(previous_state):
                norm = l2_norm(current_state, previous_state)
                converged = norm < self.epsilon
            else:
                norm = float("inf")
                converged = False

            self.db.log_convergence(
                competition=competition,
                iteration=iteration,
                state_vector=current_state,
                l2_norm=norm,
                converged=converged,
            )

            step_info = {
                "iteration": iteration,
                "l2_norm": round(norm, 6),
                "converged": converged,
                "top_experts": sorted(
                    scores.items(), key=lambda x: -x[1]
                )[:5],
                "state_dim": len(current_state),
            }
            history.append(step_info)

            if on_iteration:
                on_iteration(iteration, step_info)

            if converged:
                consecutive_converging += 1
                if consecutive_converging >= self.patience:
                    return self._build_result(
                        competition, history, experts, scores,
                        status="converged",
                        message=f"Converged after {iteration} iterations "
                                f"(patience={self.patience})",
                    )
            else:
                consecutive_converging = 0

            previous_state = current_state

        return self._build_result(
            competition, history, experts, scores,
            status="max_iterations_reached",
            message=f"Reached max iterations ({self.max_iterations}) "
                    f"without full convergence",
        )

    def _compute_relevance_scores(
        self,
        experts: List[Dict[str, Any]],
        competition: str,
        description: str,
        iteration: int,
    ) -> Dict[str, float]:
        """Compute relevance scores for each expert in the current iteration.

        Scores are based on keyword overlap with the competition description,
        with a small decay factor per iteration to encourage convergence.
        """
        desc_lower = (competition + " " + description).lower()
        desc_words = set(desc_lower.split())
        scores: Dict[str, float] = {}

        decay = 1.0 / (1.0 + 0.01 * iteration)

        for expert in experts:
            slug = expert.get("slug", "")
            score = 0.0

            for cap in expert.get("capabilities", []):
                cap_words = set(cap.lower().split())
                overlap = len(cap_words & desc_words)
                score += overlap * 0.15

            concepts = self.db.get_concepts_for_chapter(expert.get("chapter_id", 0))
            for concept in concepts:
                if concept["concept_name"].lower() in desc_lower:
                    score += 0.25

            name_words = set(expert.get("expert_name", "").lower().split())
            score += len(name_words & desc_words) * 0.1

            scores[slug] = round(score * decay, 6)

        return scores

    def _build_result(
        self,
        competition: str,
        history: List[Dict[str, Any]],
        experts: List[Dict[str, Any]],
        scores: Dict[str, float],
        status: str,
        message: str,
    ) -> Dict[str, Any]:
        """Build the final result dict."""
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_experts = []
        for slug, score in ranked:
            if score > 0:
                expert = self.registry.get_expert(slug)
                if expert:
                    top_experts.append({
                        "expert_name": expert["expert_name"],
                        "slug": slug,
                        "relevance_score": score,
                        "skills": expert.get("skills", []),
                        "strategy": expert.get("strategy", ""),
                    })

        return {
            "status": status,
            "message": message,
            "competition": competition,
            "iterations": len(history),
            "final_l2_norm": history[-1]["l2_norm"] if history else None,
            "epsilon": self.epsilon,
            "top_experts": top_experts[:5],
            "convergence_history": [
                {"iteration": h["iteration"], "l2_norm": h["l2_norm"], "converged": h["converged"]}
                for h in history
            ],
        }

    def get_convergence_history(self, competition: str) -> List[Dict[str, Any]]:
        """Retrieve convergence history for a competition."""
        return self.db.get_convergence_history(competition)
