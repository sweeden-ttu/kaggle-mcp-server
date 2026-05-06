"""State convergence loop controller for the MLSysEng MoE system.

Implements an iterative optimization loop with L2 norm convergence:
  exit_condition: ||state[n] - state[n-1]||_2 < epsilon

The loop orchestrates expert selection, skill execution, and state
tracking for Kaggle competition entries.
"""

import logging
import math
import random
from typing import Optional

from . import database as db
from . import embeddings
from . import expert_registry

logger = logging.getLogger(__name__)


def _l2_norm(a: list[float], b: list[float]) -> float:
    """Compute L2 norm of the difference between two vectors."""
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _initialize_state(experts: list[dict]) -> list[float]:
    """Create initial state vector from expert capabilities.

    Each dimension represents an expert's contribution weight,
    initialized uniformly.
    """
    n = max(len(experts), 1)
    return [1.0 / n] * n


def _update_state(
    current_state: list[float],
    experts: list[dict],
    competition: str,
    iteration: int,
) -> list[float]:
    """Update state vector based on expert evaluations.

    Simulates expert contributions converging toward optimal weights.
    In a full system, this would incorporate actual model performance
    metrics from each expert's recommended approach.
    """
    n = len(current_state)
    new_state = list(current_state)

    decay = 0.9 ** iteration
    for i in range(n):
        relevance = experts[i].get("relevance_score", 0.5) if i < len(experts) else 0.5
        perturbation = (random.random() - 0.5) * 0.1 * decay
        new_state[i] = max(0.0, current_state[i] + (relevance - 0.5) * decay * 0.1 + perturbation)

    total = sum(new_state)
    if total > 0:
        new_state = [s / total for s in new_state]

    return new_state


def run_convergence_loop(
    competition: str,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> dict:
    """Run the state convergence loop for a competition.

    Iteratively refines expert weights until convergence:
      ||state[n] - state[n-1]||_2 < epsilon

    Args:
        competition: Competition name/slug.
        epsilon: Convergence threshold.
        max_iterations: Maximum iterations before forced stop.
        patience: Consecutive converging iterations required.

    Returns:
        Dict with convergence history and final state.
    """
    db.init_db(db_path)

    experts = embeddings.get_relevant_experts(
        competition,
        n_results=10,
        db_path=db_path,
        chroma_path=chroma_path,
    )

    if not experts:
        all_experts = db.list_experts(db_path)
        if all_experts:
            experts = all_experts[:5]
        else:
            return {
                "status": "no_experts",
                "message": "No experts available. Run extract-knowledge first.",
            }

    state = _initialize_state(experts)
    history = []
    converge_count = 0

    for iteration in range(1, max_iterations + 1):
        prev_state = list(state)
        state = _update_state(state, experts, competition, iteration)

        norm = _l2_norm(state, prev_state)
        converged = norm < epsilon

        db.save_state(
            competition=competition,
            iteration=iteration,
            state_vector=state,
            l2_norm=norm,
            converged=converged,
            db_path=db_path,
        )

        step = {
            "iteration": iteration,
            "l2_norm": round(norm, 6),
            "converged": converged,
            "state": [round(s, 4) for s in state],
        }
        history.append(step)
        logger.info(
            "Iteration %d: L2=%.6f converged=%s", iteration, norm, converged
        )

        if converged:
            converge_count += 1
            if converge_count >= patience:
                break
        else:
            converge_count = 0

    expert_weights = []
    for i, expert in enumerate(experts):
        weight = state[i] if i < len(state) else 0.0
        expert_weights.append({
            "expert": expert.get("expert_name", f"expert_{i}"),
            "slug": expert.get("slug", ""),
            "weight": round(weight, 4),
            "skills": expert.get("skills", []),
        })
    expert_weights.sort(key=lambda x: -x["weight"])

    return {
        "status": "converged" if converge_count >= patience else "max_iterations",
        "competition": competition,
        "iterations": len(history),
        "final_l2_norm": history[-1]["l2_norm"] if history else None,
        "expert_weights": expert_weights,
        "history": history,
        "config": {
            "epsilon": epsilon,
            "max_iterations": max_iterations,
            "patience": patience,
        },
    }


def build_competition_entry(
    competition: str,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> dict:
    """Build a competition entry using expert knowledge and RAG.

    Selects the most relevant experts, gathers their recommended
    skills and strategies, and produces a structured entry plan.
    """
    db.init_db(db_path)

    experts = embeddings.get_relevant_experts(
        competition,
        n_results=5,
        db_path=db_path,
        chroma_path=chroma_path,
    )

    if not experts:
        all_experts = db.list_experts(db_path)
        experts = all_experts[:3] if all_experts else []

    context = embeddings.generate_context(
        competition,
        description=f"Kaggle competition: {competition}",
        chroma_path=chroma_path,
    )

    all_skills: set[str] = set()
    strategies = []
    formulas = []

    for expert in experts:
        all_skills.update(expert.get("skills", []))
        strategies.append({
            "expert": expert.get("expert_name", ""),
            "strategy": expert.get("strategy", ""),
        })
        formulas.append({
            "expert": expert.get("expert_name", ""),
            "formula": expert.get("formula", {}),
        })

    return {
        "competition": competition,
        "experts_selected": len(experts),
        "experts": [
            {
                "name": e.get("expert_name", ""),
                "slug": e.get("slug", ""),
                "capabilities": e.get("capabilities", []),
                "relevance": e.get("relevance_score", 0),
            }
            for e in experts
        ],
        "skills": sorted(all_skills),
        "strategies": strategies,
        "formulas": formulas,
        "context": context[:3000],
        "recommended_pipeline": [
            "1. Baseline model with default parameters",
            "2. EDA and feature engineering based on expert insights",
            "3. Model selection guided by chapter expertise",
            "4. Hyperparameter optimization",
            "5. Ensemble top-performing approaches",
            "6. Final submission with validation",
        ],
    }
