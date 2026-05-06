"""Expert registry for managing chapter experts and their skills/strategies."""

import os
import re
import logging
from typing import Optional

from .database import store_expert, get_all_experts, get_expert_by_slug

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = "/Users/sweeden/skills"

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPING = {
    "data_preprocessing": "kaggle-preprocessor",
    "feature_engineering": "kaggle-feature-engineer",
    "model_training": "kaggle-model-trainer",
    "hyperparameter_tuning": "kaggle-hyperparameter-tuner",
    "ensemble_methods": "kaggle-ensemble-builder",
    "deep_learning": "kaggle-deep-learning",
    "nlp": "kaggle-nlp-toolkit",
    "computer_vision": "kaggle-cv-toolkit",
    "time_series": "kaggle-time-series",
    "tabular_data": "kaggle-tabular-toolkit",
}

CONCEPT_TO_CAPABILITY = {
    "gradient descent": "Optimize model parameters using gradient-based methods",
    "backpropagation": "Train deep neural networks effectively",
    "cross-entropy": "Design and optimize classification loss functions",
    "regularization": "Prevent overfitting with regularization techniques",
    "ensemble": "Build ensemble models for improved performance",
    "feature engineering": "Create informative features from raw data",
    "hyperparameter": "Systematic hyperparameter search and tuning",
    "neural network": "Design and train neural network architectures",
    "decision tree": "Build tree-based models and ensembles",
    "cross-validation": "Robust model evaluation with cross-validation",
    "transfer learning": "Leverage pre-trained models for new tasks",
    "data augmentation": "Augment training data for better generalization",
}


def get_kaggle_skills_path() -> str:
    return os.environ.get("KAGGLE_SKILLS_PATH", DEFAULT_KAGGLE_SKILLS_PATH)


def create_expert_from_chapter(chapter_number: int, title: str,
                               concepts: list[dict],
                               db_path: Optional[str] = None) -> dict:
    """Create an expert definition from extracted chapter data."""
    slug = _generate_slug(chapter_number, title)
    capabilities = _infer_capabilities(concepts)
    skills = _infer_skills(concepts)
    formula = _generate_formula(concepts)

    expert_data = {
        "expert_name": f"{chapter_number:02d}_{title}",
        "slug": slug,
        "chapter_id": chapter_number,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }

    store_expert(expert_data, db_path)
    return expert_data


def list_experts(db_path: Optional[str] = None) -> list[dict]:
    """List all registered experts."""
    return get_all_experts(db_path)


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a specific expert by slug."""
    return get_expert_by_slug(slug, db_path)


def get_experts_for_competition(competition_name: str, relevant_chapters: list[dict],
                                db_path: Optional[str] = None) -> list[dict]:
    """Select experts based on competition relevance scores."""
    selected = []
    for ch_info in relevant_chapters:
        chapter_number = ch_info["chapter_number"]
        experts = get_all_experts(db_path)
        for expert in experts:
            if expert.get("chapter_id") == chapter_number:
                expert["relevance_score"] = ch_info.get("relevance_score", 0.0)
                selected.append(expert)
                break
    return sorted(selected, key=lambda x: x.get("relevance_score", 0), reverse=True)


def _generate_slug(chapter_number: int, title: str) -> str:
    """Generate a URL-friendly slug for an expert."""
    clean = re.sub(r"[^a-z0-9\s]", "", title.lower())
    clean = re.sub(r"\s+", "_", clean.strip())
    return f"{chapter_number:02d}_{clean}"


def _infer_capabilities(concepts: list[dict]) -> list[str]:
    """Infer expert capabilities from extracted concepts."""
    capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]

    for concept in concepts:
        name = concept["name"].lower()
        for keyword, capability in CONCEPT_TO_CAPABILITY.items():
            if keyword in name and capability not in capabilities:
                capabilities.append(capability)

    return capabilities[:8]


def _infer_skills(concepts: list[dict]) -> list[str]:
    """Infer applicable skills from concepts."""
    skills_path = get_kaggle_skills_path()
    matched_skills = set()

    concept_names = {c["name"].lower() for c in concepts}
    all_text = " ".join(concept_names)

    for keyword, skill_name in SKILL_MAPPING.items():
        keyword_words = keyword.replace("_", " ")
        if keyword_words in all_text or any(keyword_words in cn for cn in concept_names):
            matched_skills.add(os.path.join(skills_path, skill_name))

    if not matched_skills:
        matched_skills.add(os.path.join(skills_path, "kaggle-preprocessor"))
        matched_skills.add(os.path.join(skills_path, "kaggle-model-trainer"))

    return sorted(matched_skills)


def _generate_formula(concepts: list[dict]) -> dict:
    """Generate a mathematical formula/objective for the expert."""
    metrics = ["accuracy"]
    concept_names = {c["name"].lower() for c in concepts}

    if any("classification" in cn or "cross-entropy" in cn for cn in concept_names):
        metrics = ["accuracy", "f1_score", "log_loss"]
    elif any("regression" in cn for cn in concept_names):
        metrics = ["rmse", "mae", "r2_score"]
    elif any("nlp" in cn or "language" in cn for cn in concept_names):
        metrics = ["bleu", "rouge", "perplexity"]
    else:
        metrics = ["accuracy", "f1_score"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }
