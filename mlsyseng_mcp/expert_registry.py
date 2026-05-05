"""Expert registry - manages chapter experts and their definitions."""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import MoEDatabase

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

CONCEPT_TO_SKILLS = {
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "sgd": ["kaggle-model-trainer"],
    "backpropagation": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "regularization": ["kaggle-model-trainer", "kaggle-feature-engineer"],
    "cross-validation": ["kaggle-model-evaluator", "kaggle-cross-validator"],
    "neural network": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "deep learning": ["kaggle-deep-learning"],
    "cnn": ["kaggle-deep-learning", "kaggle-image-processor"],
    "rnn": ["kaggle-deep-learning", "kaggle-sequence-modeler"],
    "transformer": ["kaggle-deep-learning", "kaggle-nlp"],
    "random forest": ["kaggle-model-trainer", "kaggle-ensemble"],
    "xgboost": ["kaggle-model-trainer", "kaggle-gradient-boosting"],
    "lightgbm": ["kaggle-model-trainer", "kaggle-gradient-boosting"],
    "svm": ["kaggle-model-trainer"],
    "pca": ["kaggle-feature-engineer", "kaggle-dimensionality-reducer"],
    "clustering": ["kaggle-unsupervised", "kaggle-feature-engineer"],
    "feature engineering": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-image-processor"],
    "preprocessing": ["kaggle-preprocessor"],
    "model selection": ["kaggle-model-evaluator", "kaggle-model-trainer"],
    "reinforcement learning": ["kaggle-rl-agent"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "attention mechanism": ["kaggle-deep-learning", "kaggle-nlp"],
    "normalization": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "loss function": ["kaggle-model-trainer", "kaggle-optimizer"],
    "hyperparameter": ["kaggle-hyperparameter-tuner", "kaggle-model-trainer"],
}


def _slugify(name: str) -> str:
    """Create a URL-safe slug from a name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def _map_concepts_to_skills(
    concepts: List[str],
    skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
) -> List[str]:
    """Map extracted concepts to Kaggle skill paths."""
    skill_set: set = set()
    for concept in concepts:
        key = concept.lower().strip()
        if key in CONCEPT_TO_SKILLS:
            for skill_name in CONCEPT_TO_SKILLS[key]:
                skill_set.add(os.path.join(skills_path, skill_name))
    return sorted(skill_set)


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer expert capabilities from concepts."""
    caps = []
    concept_set = {c.lower() for c in concepts}

    if concept_set & {"neural network", "deep learning", "cnn", "rnn", "transformer"}:
        caps.append("Build and train deep learning models")
    if concept_set & {"gradient descent", "sgd", "optimization", "adam"}:
        caps.append("Optimize model training with advanced optimizers")
    if concept_set & {"feature engineering", "feature selection", "preprocessing"}:
        caps.append("Engineer features for improved model performance")
    if concept_set & {"ensemble", "random forest", "xgboost", "lightgbm", "gradient boosting"}:
        caps.append("Build ensemble models for robust predictions")
    if concept_set & {"cross-validation", "model selection", "hyperparameter"}:
        caps.append("Systematic hyperparameter search and model selection")
    if concept_set & {"regularization", "dropout", "weight decay"}:
        caps.append("Apply regularization to prevent overfitting")
    if concept_set & {"clustering", "pca", "dimensionality reduction"}:
        caps.append("Perform unsupervised learning and dimensionality reduction")
    if concept_set & {"attention mechanism", "self-attention", "transformer"}:
        caps.append("Implement attention-based architectures")
    if concept_set & {"data augmentation", "normalization"}:
        caps.append("Preprocess and augment training data")

    if not caps:
        caps.append("Build baseline models quickly")
        caps.append("Systematic hyperparameter search")

    return caps


def _build_strategy(concepts: List[str]) -> str:
    """Build a strategy string from concepts."""
    stages = ["Baseline"]
    concept_set = {c.lower() for c in concepts}

    if concept_set & {"preprocessing", "data augmentation", "normalization", "feature engineering"}:
        stages.append("EDA")
        stages.append("Feature Engineering")
    else:
        stages.append("EDA")

    if concept_set & {"neural network", "deep learning"}:
        stages.append("Deep Learning")
    elif concept_set & {"ensemble", "xgboost", "lightgbm"}:
        stages.append("Gradient Boosting")
    else:
        stages.append("Model Selection")

    if concept_set & {"hyperparameter", "cross-validation"}:
        stages.append("Hyperparameter Tuning")

    stages.append("Submit")
    return " → ".join(stages)


def _build_formula(concepts: List[str]) -> Dict[str, Any]:
    """Build an objective function formula from concepts."""
    metrics = ["accuracy"]
    concept_set = {c.lower() for c in concepts}

    if concept_set & {"loss function", "cost function"}:
        objective = "minimize_loss"
    else:
        objective = "minimize_validation_loss"

    if concept_set & {"precision", "recall", "f1"}:
        metrics.extend(["f1_score", "precision", "recall"])
    if concept_set & {"auc"}:
        metrics.append("auc_roc")
    if len(metrics) == 1:
        metrics.append("f1_score")

    return {
        "objective": objective,
        "function": "L = f(X, θ, α)",
        "metrics": sorted(set(metrics)),
    }


def _build_loop_config() -> Dict[str, Any]:
    """Build default loop configuration."""
    return {
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": 0.001,
        "max_iterations": 10,
        "patience": 3,
    }


def create_expert_from_chapter(
    chapter: Dict[str, Any],
    skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
) -> Dict[str, Any]:
    """Create an expert definition from a chapter record."""
    concepts = chapter.get("concepts", [])
    title = chapter.get("title", chapter.get("chapter_id", "unknown"))
    slug = _slugify(title)

    expert = {
        "expert_name": title,
        "slug": slug,
        "chapter_id": chapter["chapter_id"],
        "capabilities": _infer_capabilities(concepts),
        "skills": _map_concepts_to_skills(concepts, skills_path),
        "strategy": _build_strategy(concepts),
        "formula": _build_formula(concepts),
        "loop_config": _build_loop_config(),
        "metadata": {
            "source_concepts": concepts,
            "concept_count": len(concepts),
        },
    }
    return expert


def register_experts_from_chapters(
    db: MoEDatabase,
    skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
) -> List[Dict[str, Any]]:
    """Register experts for all indexed chapters."""
    chapters = db.list_chapters()
    experts = []

    for chapter in chapters:
        expert = create_expert_from_chapter(chapter, skills_path)
        db.upsert_expert(expert)
        experts.append(expert)
        logger.info("Registered expert: %s", expert["expert_name"])

    return experts


def get_expert_for_query(
    query: str,
    db: MoEDatabase,
) -> Optional[Dict[str, Any]]:
    """Find the most relevant expert for a text query using keyword matching."""
    experts = db.list_experts()
    if not experts:
        return None

    query_lower = query.lower()
    best_score = 0
    best_expert = None

    for expert in experts:
        score = 0
        concepts = expert.get("metadata", {}).get("source_concepts", [])
        for concept in concepts:
            if concept.lower() in query_lower:
                score += 1

        if expert["slug"] in query_lower or expert["expert_name"].lower() in query_lower:
            score += 5

        if score > best_score:
            best_score = score
            best_expert = expert

    return best_expert
