"""Expert registry for the MoE system.

Each chapter expert has capabilities, skills, strategy, and mathematical formulas.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import database

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CHAPTER_SKILL_MAP = {
    "ml systems": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-submitter"],
    "deep learning": ["kaggle-neural-net", "kaggle-model-trainer"],
    "optimization": ["kaggle-hypertuner", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-eng"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "neural network": ["kaggle-neural-net", "kaggle-model-trainer"],
    "natural language": ["kaggle-nlp", "kaggle-preprocessor"],
    "computer vision": ["kaggle-vision", "kaggle-preprocessor"],
    "reinforcement": ["kaggle-rl", "kaggle-model-trainer"],
    "bayesian": ["kaggle-bayesian", "kaggle-model-trainer"],
    "time series": ["kaggle-timeseries", "kaggle-preprocessor"],
    "clustering": ["kaggle-clustering", "kaggle-preprocessor"],
    "dimensionality": ["kaggle-dimreduce", "kaggle-preprocessor"],
    "evaluation": ["kaggle-evaluation", "kaggle-submitter"],
    "data": ["kaggle-preprocessor", "kaggle-eda"],
}

CONCEPT_CAPABILITIES = {
    "gradient descent": "Optimize model parameters using gradient-based methods",
    "backpropagation": "Train deep neural networks with error backpropagation",
    "loss function": "Design and select appropriate loss functions",
    "regularization": "Prevent overfitting with L1/L2/dropout regularization",
    "cross-validation": "Robust model evaluation with cross-validation",
    "neural network": "Build and train neural network architectures",
    "deep learning": "Apply deep learning to complex problems",
    "transformer": "Implement attention-based transformer models",
    "feature engineering": "Create informative features from raw data",
    "ensemble": "Combine multiple models for better predictions",
    "hyperparameter": "Systematic hyperparameter optimization",
    "model selection": "Choose the best model architecture",
    "evaluation metric": "Select and compute evaluation metrics",
    "transfer learning": "Leverage pre-trained models for new tasks",
    "data augmentation": "Augment training data for better generalization",
}


def _slugify(name: str) -> str:
    """Convert a chapter title to a slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _infer_skills(title: str, concepts: List[str]) -> List[str]:
    """Infer applicable Kaggle skills from chapter title and concepts."""
    skills = set()
    title_lower = title.lower()

    for keyword, skill_list in CHAPTER_SKILL_MAP.items():
        if keyword in title_lower:
            skills.update(skill_list)
        for concept in concepts:
            if keyword in concept.lower():
                skills.update(skill_list)

    if not skills:
        skills.add("kaggle-preprocessor")
        skills.add("kaggle-model-trainer")

    return [os.path.join(KAGGLE_SKILLS_PATH, s) for s in sorted(skills)]


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer capabilities from extracted concepts."""
    capabilities = []
    for concept in concepts:
        for key, cap in CONCEPT_CAPABILITIES.items():
            if key in concept.lower():
                capabilities.append(cap)
                break

    if not capabilities:
        capabilities = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]

    return list(dict.fromkeys(capabilities))[:8]


def _infer_formula(title: str, concepts: List[str]) -> Dict[str, Any]:
    """Infer mathematical formula/objective from chapter content."""
    title_lower = title.lower()

    if "optimization" in title_lower or "gradient" in " ".join(concepts):
        return {
            "objective": "minimize_loss",
            "function": "L = (1/n) * Σ loss(f(x_i; θ), y_i) + λ * R(θ)",
            "metrics": ["loss", "gradient_norm"],
        }
    elif "classification" in title_lower:
        return {
            "objective": "maximize_accuracy",
            "function": "acc = (TP + TN) / (TP + TN + FP + FN)",
            "metrics": ["accuracy", "f1_score", "precision", "recall"],
        }
    elif "regression" in title_lower:
        return {
            "objective": "minimize_mse",
            "function": "MSE = (1/n) * Σ (y_i - ŷ_i)²",
            "metrics": ["mse", "rmse", "r2_score"],
        }
    elif "ensemble" in title_lower:
        return {
            "objective": "minimize_ensemble_error",
            "function": "E_ensemble = E_avg - (1/M) * Σ Var(f_m)",
            "metrics": ["ensemble_score", "diversity"],
        }

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    }


def create_expert_from_chapter(
    chapter: Dict[str, Any], db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create an expert definition from an extracted chapter."""
    chapter_number = chapter["chapter_number"]
    title = chapter["title"]
    concepts = chapter.get("concepts", [])

    expert_name = f"{chapter_number:02d}_{title}"
    slug = _slugify(f"{chapter_number:02d}_{title}")

    skills = _infer_skills(title, concepts)
    capabilities = _infer_capabilities(concepts)
    formula = _infer_formula(title, concepts)

    expert_data = {
        "expert_name": expert_name,
        "slug": slug,
        "chapter_id": chapter.get("id"),
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }

    database.store_expert(expert_data, db_path=db_path)
    return expert_data


def register_all_experts(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create expert definitions for all extracted chapters."""
    chapters = database.get_all_chapters(db_path=db_path)
    experts = []

    for chapter in chapters:
        if chapter.get("extracted_text"):
            expert = create_expert_from_chapter(chapter, db_path=db_path)
            experts.append(expert)

    return experts


def get_expert_for_query(
    query: str, db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Find the most relevant expert for a given query."""
    experts = database.get_all_experts(db_path=db_path)
    if not experts:
        return None

    query_lower = query.lower()
    best_match = None
    best_score = 0

    for expert in experts:
        score = 0
        name_lower = expert["expert_name"].lower()
        if query_lower in name_lower:
            score += 10

        capabilities = expert.get("capabilities", [])
        for cap in capabilities:
            if any(word in cap.lower() for word in query_lower.split()):
                score += 2

        skills = expert.get("skills", [])
        for skill in skills:
            skill_name = os.path.basename(skill)
            if any(word in skill_name for word in query_lower.split()):
                score += 1

        if score > best_score:
            best_score = score
            best_match = expert

    return best_match


def select_experts_for_competition(
    competition_description: str,
    chapter_relevances: List[Dict[str, Any]],
    max_experts: int = 5,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select the best experts for a competition based on RAG relevance scores."""
    experts = database.get_all_experts(db_path=db_path)
    if not experts:
        return []

    expert_by_chapter: Dict[int, Dict[str, Any]] = {}
    for expert in experts:
        ch_id = expert.get("chapter_id")
        if ch_id is not None:
            expert_by_chapter[ch_id] = expert

    selected = []
    for relevance in chapter_relevances[:max_experts]:
        ch_num = relevance.get("chapter_number")
        if ch_num is not None:
            chapters = database.get_all_chapters(db_path=db_path)
            for ch in chapters:
                if ch["chapter_number"] == ch_num and ch.get("id") in expert_by_chapter:
                    expert = expert_by_chapter[ch["id"]]
                    expert["relevance_score"] = relevance.get("relevance_score", 0.0)
                    selected.append(expert)
                    break

    return selected


def export_expert_json(slug: str, output_dir: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    """Export an expert definition to a JSON file."""
    expert = database.get_expert(slug, db_path=db_path)
    if expert is None:
        return None

    out_dir = Path(output_dir or "experts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.json"

    export_data = {k: v for k, v in expert.items() if k not in ("id", "created_at", "updated_at")}
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2)

    return str(out_path)
