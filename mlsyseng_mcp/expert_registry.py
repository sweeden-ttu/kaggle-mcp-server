"""
Expert registry for the MLSysEng MoE system.

Each ML Principles chapter becomes an expert with:
- capabilities: what the expert can do
- skills: Kaggle skills it recommends
- strategy: step-by-step approach
- formula: objective function + metrics
- loop_config: convergence parameters
"""

import os
import re
import uuid
from typing import Any, Dict, List, Optional

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CONCEPT_TO_SKILLS = {
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "cnn": ["kaggle-image-classifier", "kaggle-deep-learning"],
    "rnn": ["kaggle-sequence-model", "kaggle-deep-learning"],
    "lstm": ["kaggle-sequence-model", "kaggle-deep-learning"],
    "transformer": ["kaggle-transformer", "kaggle-deep-learning"],
    "decision tree": ["kaggle-tree-model", "kaggle-model-trainer"],
    "random forest": ["kaggle-tree-model", "kaggle-ensemble"],
    "xgboost": ["kaggle-boosting", "kaggle-ensemble"],
    "lightgbm": ["kaggle-boosting", "kaggle-ensemble"],
    "svm": ["kaggle-svm", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "cross validation": ["kaggle-validator", "kaggle-evaluator"],
    "ensemble": ["kaggle-ensemble", "kaggle-stacker"],
    "hyperparameter": ["kaggle-tuner", "kaggle-optimizer"],
    "regularization": ["kaggle-regularizer", "kaggle-model-trainer"],
    "clustering": ["kaggle-clusterer", "kaggle-unsupervised"],
    "pca": ["kaggle-dim-reducer", "kaggle-preprocessor"],
    "data augmentation": ["kaggle-augmenter", "kaggle-preprocessor"],
    "transfer learning": ["kaggle-transfer-learner", "kaggle-deep-learning"],
    "attention mechanism": ["kaggle-transformer", "kaggle-deep-learning"],
}

CONCEPT_TO_CAPABILITIES = {
    "gradient descent": ["Optimize model parameters using gradient-based methods"],
    "neural network": ["Build and train neural network architectures"],
    "deep learning": ["Design deep architectures for complex pattern recognition"],
    "feature engineering": ["Create informative features from raw data"],
    "ensemble": ["Combine multiple models for improved predictions"],
    "cross validation": ["Robust model evaluation with cross-validation"],
    "hyperparameter": ["Systematic hyperparameter search and optimization"],
    "regularization": ["Apply regularization to prevent overfitting"],
    "decision tree": ["Build interpretable tree-based models"],
    "clustering": ["Identify natural groupings in data"],
    "transfer learning": ["Leverage pre-trained models for new tasks"],
}


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _skill_paths(skills: List[str]) -> List[str]:
    return [os.path.join(KAGGLE_SKILLS_PATH, s) for s in skills]


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer an objective function and metrics from concepts."""
    metrics: List[str] = ["accuracy"]
    if any(c.lower() in ("precision", "recall", "f1 score") for c in concepts):
        metrics = ["f1_score", "precision", "recall"]
    if any("auc" in c.lower() or "roc" in c.lower() for c in concepts):
        metrics.append("auc_roc")
    if any("regression" in c.lower() for c in concepts):
        metrics = ["rmse", "mae", "r2_score"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": list(dict.fromkeys(metrics)),
    }


def create_expert_from_chapter(
    chapter_id: str,
    chapter_num: int,
    title: str,
    concepts: List[str],
) -> Dict[str, Any]:
    """
    Create an expert definition from a chapter's extracted data.

    Returns a dict suitable for database storage.
    """
    expert_name = f"{chapter_num:02d}_{title}"
    slug = _slugify(expert_name)

    skills_set: List[str] = []
    capabilities: List[str] = []

    for concept in concepts:
        cl = concept.lower()
        for key, skill_list in CONCEPT_TO_SKILLS.items():
            if key in cl:
                for s in skill_list:
                    if s not in skills_set:
                        skills_set.append(s)
        for key, cap_list in CONCEPT_TO_CAPABILITIES.items():
            if key in cl:
                for c in cap_list:
                    if c not in capabilities:
                        capabilities.append(c)

    if not capabilities:
        capabilities = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]
    if not skills_set:
        skills_set = ["kaggle-preprocessor", "kaggle-model-trainer"]

    return {
        "expert_id": str(uuid.uuid4()),
        "chapter_id": chapter_id,
        "expert_name": expert_name,
        "slug": slug,
        "capabilities": capabilities,
        "skills": _skill_paths(skills_set),
        "strategy": DEFAULT_STRATEGY,
        "formula": _infer_formula(concepts),
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }


def register_experts_from_db(db) -> List[Dict[str, Any]]:
    """
    Create expert definitions from all extracted chapters in the database.

    Returns the list of registered experts.
    """
    chapters = db.list_chapters()
    registered: List[Dict[str, Any]] = []

    for ch in chapters:
        if ch.get("status") != "extracted":
            continue

        expert = create_expert_from_chapter(
            chapter_id=ch["chapter_id"],
            chapter_num=ch["chapter_num"],
            title=ch["title"],
            concepts=ch.get("concepts", []),
        )
        db.upsert_expert(expert)
        registered.append(expert)

    return registered


def get_experts_for_competition(
    db,
    embedding_engine,
    competition_description: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select the best experts for a competition using RAG-based skill inference.

    Args:
        db: MoEDatabase
        embedding_engine: EmbeddingEngine
        competition_description: What the competition is about
        top_k: How many experts to select

    Returns:
        List of expert dicts ranked by relevance
    """
    inferred = embedding_engine.infer_skills(competition_description, n_results=top_k * 3)

    selected: List[Dict[str, Any]] = []
    for inf in inferred[:top_k]:
        expert_rows = [
            e
            for e in db.list_experts()
            if e["chapter_id"] == inf["chapter_id"]
        ]
        if expert_rows:
            expert = expert_rows[0]
            expert["relevance_score"] = inf["avg_score"]
            expert["matched_concepts"] = inf["concepts"]
            selected.append(expert)

    return selected
