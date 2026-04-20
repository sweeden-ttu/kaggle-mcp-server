"""Expert registry for MLSysEng MoE.

Manages creation, registration, and querying of chapter experts with
skills, strategies, formulas, and convergence loop configurations.
"""

import json
import logging
import os
import re
from typing import Optional

from .database import Database, Expert

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get("KAGGLE_SKILLS_PATH", "")

STRATEGY_TEMPLATES = {
    "default": "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Submit",
    "deep_learning": "Data Prep → Architecture Design → Training → Evaluation → Ensemble → Submit",
    "nlp": "Text Preprocessing → Tokenization → Embeddings → Model Training → Evaluation → Submit",
    "computer_vision": "Image Preprocessing → Augmentation → Model Architecture → Training → Evaluation → Submit",
    "tabular": "EDA → Missing Values → Feature Engineering → Model Selection → Stacking → Submit",
    "time_series": "Stationarity Check → Feature Extraction → Window Selection → Model Training → Forecast → Submit",
    "reinforcement": "Environment Setup → Policy Design → Training Loop → Evaluation → Optimization → Submit",
}

FORMULA_TEMPLATES = {
    "default": {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    },
    "regression": {
        "objective": "minimize_mse",
        "function": "L = (1/n) Σ(y_i - ŷ_i)²",
        "metrics": ["mse", "rmse", "r2_score"],
    },
    "classification": {
        "objective": "minimize_cross_entropy",
        "function": "L = -Σ y_i log(ŷ_i)",
        "metrics": ["accuracy", "f1_score", "auc_roc"],
    },
    "deep_learning": {
        "objective": "minimize_loss",
        "function": "L = CrossEntropy(y, σ(Wx + b))",
        "metrics": ["loss", "accuracy", "val_loss"],
    },
    "optimization": {
        "objective": "minimize_objective",
        "function": "θ* = argmin_θ L(θ)",
        "metrics": ["convergence_rate", "final_loss"],
    },
}

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CONCEPT_TO_STRATEGY = {
    "neural network": "deep_learning",
    "deep learning": "deep_learning",
    "convolutional": "computer_vision",
    "computer vision": "computer_vision",
    "recurrent": "time_series",
    "lstm": "time_series",
    "time series": "time_series",
    "natural language processing": "nlp",
    "transformer": "nlp",
    "attention mechanism": "nlp",
    "embedding": "nlp",
    "reinforcement learning": "reinforcement",
    "q-learning": "reinforcement",
    "policy gradient": "reinforcement",
    "decision tree": "tabular",
    "random forest": "tabular",
    "ensemble": "tabular",
    "boosting": "tabular",
    "feature engineering": "tabular",
}

CONCEPT_TO_FORMULA = {
    "neural network": "deep_learning",
    "deep learning": "deep_learning",
    "gradient descent": "optimization",
    "optimization": "optimization",
    "loss function": "optimization",
    "decision tree": "classification",
    "random forest": "classification",
    "support vector": "classification",
    "regression": "regression",
    "mean squared error": "regression",
}

SKILL_MAPPING = {
    "tabular": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-feature-engineer"],
    "deep_learning": ["kaggle-model-trainer", "kaggle-neural-arch", "kaggle-gpu-trainer"],
    "nlp": ["kaggle-text-preprocessor", "kaggle-model-trainer", "kaggle-embeddings"],
    "computer_vision": ["kaggle-image-preprocessor", "kaggle-model-trainer", "kaggle-augmentation"],
    "time_series": ["kaggle-time-series-prep", "kaggle-model-trainer", "kaggle-forecaster"],
    "reinforcement": ["kaggle-env-setup", "kaggle-rl-trainer"],
    "default": ["kaggle-preprocessor", "kaggle-model-trainer"],
}


def _infer_strategy_type(concepts: list[str]) -> str:
    """Infer the best strategy type from a list of concepts."""
    type_votes: dict[str, int] = {}
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, stype in CONCEPT_TO_STRATEGY.items():
            if keyword in concept_lower:
                type_votes[stype] = type_votes.get(stype, 0) + 1

    if not type_votes:
        return "default"
    return max(type_votes, key=type_votes.get)


def _infer_formula_type(concepts: list[str]) -> str:
    """Infer the best formula type from a list of concepts."""
    type_votes: dict[str, int] = {}
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, ftype in CONCEPT_TO_FORMULA.items():
            if keyword in concept_lower:
                type_votes[ftype] = type_votes.get(ftype, 0) + 1

    if not type_votes:
        return "default"
    return max(type_votes, key=type_votes.get)


def _slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_")


def create_expert_from_chapter(
    chapter_id: int,
    title: str,
    concepts: list[str],
    db: Database,
    skills_path: Optional[str] = None,
) -> Expert:
    """Create an expert definition from chapter data."""
    skills_base = skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    strategy_type = _infer_strategy_type(concepts)
    formula_type = _infer_formula_type(concepts)

    skill_names = SKILL_MAPPING.get(strategy_type, SKILL_MAPPING["default"])
    if skills_base:
        skill_paths = [os.path.join(skills_base, s) for s in skill_names]
    else:
        skill_paths = skill_names

    capabilities = []
    if concepts:
        capabilities.append(f"Expert in: {', '.join(concepts[:5])}")
    capabilities.append(f"Strategy focus: {strategy_type}")
    capabilities.append("Build baseline models quickly")
    capabilities.append("Systematic hyperparameter search")

    expert = Expert(
        expert_name=title,
        slug=_slugify(title),
        chapter_id=chapter_id,
        capabilities=json.dumps(capabilities),
        skills=json.dumps(skill_paths),
        strategy=STRATEGY_TEMPLATES.get(strategy_type, STRATEGY_TEMPLATES["default"]),
        formula=json.dumps(FORMULA_TEMPLATES.get(formula_type, FORMULA_TEMPLATES["default"])),
        loop_config=json.dumps(DEFAULT_LOOP_CONFIG),
    )

    expert.id = db.upsert_expert(expert)
    logger.info(f"Registered expert: {title} (strategy={strategy_type})")
    return expert


def register_experts_from_db(db: Database, skills_path: Optional[str] = None) -> list[Expert]:
    """Create experts for all extracted chapters in the database."""
    chapters = db.list_chapters(status="extracted")
    experts = []

    for chapter in chapters:
        expert = create_expert_from_chapter(
            chapter_id=chapter.id,
            title=chapter.title,
            concepts=chapter.concept_list,
            db=db,
            skills_path=skills_path,
        )
        experts.append(expert)

    return experts


def get_experts_for_competition(
    competition_description: str,
    db: Database,
    embedding_engine=None,
) -> list[dict]:
    """Select the best experts for a given competition using RAG."""
    if embedding_engine:
        return embedding_engine.get_relevant_experts(competition_description, db)

    experts = db.list_experts()
    results = []
    desc_lower = competition_description.lower()

    for expert in experts:
        score = 0
        for cap in expert.capabilities_list:
            for word in desc_lower.split():
                if len(word) > 3 and word in cap.lower():
                    score += 1

        if score > 0:
            results.append({
                "expert": expert.to_dict(),
                "relevance_score": score,
            })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results
