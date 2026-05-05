"""Expert registry: creates and manages chapter experts.

Each ML Principles chapter becomes an expert with:
  - capabilities derived from extracted concepts
  - skills mapped to Kaggle skill paths
  - a strategy pipeline
  - an objective formula
  - convergence loop configuration
"""

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = (
    "Baseline → EDA → Feature Engineering → Model Selection → "
    "Hyperparameter Tuning → Ensemble → Submit"
)

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

_CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-nn-builder"],
    "deep learning": ["kaggle-model-trainer", "kaggle-nn-builder"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "backpropagation": ["kaggle-model-trainer", "kaggle-nn-builder"],
    "convolutional": ["kaggle-model-trainer", "kaggle-image-processor"],
    "recurrent": ["kaggle-model-trainer", "kaggle-sequence-modeler"],
    "transformer": ["kaggle-model-trainer", "kaggle-nlp-processor"],
    "attention": ["kaggle-model-trainer", "kaggle-nlp-processor"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-augmenter"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "bagging": ["kaggle-ensemble-builder"],
    "boosting": ["kaggle-ensemble-builder"],
    "random forest": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "cross-validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "hyperparameter": ["kaggle-model-trainer", "kaggle-optimizer"],
    "precision": ["kaggle-evaluator"],
    "recall": ["kaggle-evaluator"],
    "f1 score": ["kaggle-evaluator"],
    "accuracy": ["kaggle-evaluator"],
    "embedding": ["kaggle-nlp-processor", "kaggle-model-trainer"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "supervised": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "unsupervised": ["kaggle-model-trainer", "kaggle-clustering"],
    "pca": ["kaggle-preprocessor", "kaggle-dim-reducer"],
    "transfer learning": ["kaggle-model-trainer", "kaggle-transfer-learner"],
    "mixture of experts": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "lstm": ["kaggle-sequence-modeler", "kaggle-model-trainer"],
    "generative": ["kaggle-model-trainer", "kaggle-generative"],
    "gan": ["kaggle-model-trainer", "kaggle-generative"],
    "decision tree": ["kaggle-model-trainer", "kaggle-tree-builder"],
    "support vector": ["kaggle-model-trainer", "kaggle-svm-builder"],
}

_CONCEPT_TO_METRICS = {
    "classification": ["accuracy", "f1_score", "precision", "recall", "auc"],
    "regression": ["rmse", "mae", "r2_score"],
    "neural network": ["loss", "accuracy", "f1_score"],
    "deep learning": ["loss", "accuracy", "val_loss"],
    "ensemble": ["accuracy", "f1_score", "auc"],
    "precision": ["precision", "recall", "f1_score"],
    "recall": ["precision", "recall", "f1_score"],
    "accuracy": ["accuracy", "f1_score"],
}


def _slugify(name: str) -> str:
    """Convert a chapter title/name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _expert_id(chapter_id: str) -> str:
    return f"expert_{chapter_id}"


def _infer_skills(concepts: List[str]) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        key = concept.lower()
        for pattern, skill_list in _CONCEPT_TO_SKILLS.items():
            if pattern in key:
                for skill in skill_list:
                    skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill))
    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))
    return sorted(skills)


def _infer_capabilities(concepts: List[str], title: str) -> List[str]:
    """Generate capability descriptions from concepts."""
    caps = [f"Expert in {title}"]
    if any("neural" in c or "deep" in c for c in concepts):
        caps.append("Build and train neural network architectures")
    if any("ensemble" in c or "bagging" in c or "boosting" in c for c in concepts):
        caps.append("Construct ensemble models for improved accuracy")
    if any("feature" in c for c in concepts):
        caps.append("Advanced feature engineering and selection")
    if any("optim" in c for c in concepts):
        caps.append("Hyperparameter optimization and tuning")
    if any("transform" in c or "attention" in c for c in concepts):
        caps.append("Transformer-based model design")
    if any("regulari" in c for c in concepts):
        caps.append("Regularization techniques to prevent overfitting")
    if any("embed" in c or "nlp" in c or "token" in c for c in concepts):
        caps.append("NLP and text embedding techniques")
    caps.append("Build baseline models quickly")
    caps.append("Systematic experiment tracking")
    return caps


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Build an objective formula from concepts."""
    metrics = set()
    for concept in concepts:
        key = concept.lower()
        for pattern, metric_list in _CONCEPT_TO_METRICS.items():
            if pattern in key:
                metrics.update(metric_list)
    if not metrics:
        metrics = {"accuracy", "f1_score"}
    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": sorted(metrics),
    }


def create_expert_from_chapter(
    chapter: Dict[str, Any],
    db=None,
) -> Dict[str, Any]:
    """Create an expert definition from a chapter record.

    Args:
        chapter: dict with chapter_id, title, concepts, folder_name.
        db: MLSysEngDB instance for persistence.

    Returns:
        Expert dict.
    """
    chapter_id = chapter["chapter_id"]
    title = chapter.get("title", chapter.get("folder_name", "Unknown"))
    concepts = chapter.get("concepts", [])
    folder = chapter.get("folder_name", "")

    expert_name = f"{folder.split('_')[0]}_{title}" if "_" in folder else title
    slug = _slugify(expert_name)

    expert_data = {
        "expert_id": _expert_id(chapter_id),
        "chapter_id": chapter_id,
        "expert_name": expert_name,
        "slug": slug,
        "capabilities": _infer_capabilities(concepts, title),
        "skills": _infer_skills(concepts),
        "strategy": DEFAULT_STRATEGY,
        "formula": _infer_formula(concepts),
        "loop_config": dict(DEFAULT_LOOP_CONFIG),
        "metadata": {
            "source_folder": folder,
            "concept_count": len(concepts),
            "concepts": concepts[:20],
        },
    }

    if db is not None:
        expert_data = db.upsert_expert(expert_data)

    return expert_data


def register_all_experts(
    chapters: List[Dict[str, Any]],
    db=None,
) -> List[Dict[str, Any]]:
    """Register experts for all chapters.

    Args:
        chapters: List of chapter dicts from the database.
        db: MLSysEngDB instance.

    Returns:
        List of expert dicts.
    """
    experts = []
    for ch in chapters:
        expert = create_expert_from_chapter(ch, db=db)
        experts.append(expert)
        logger.info("Registered expert: %s (%s)", expert["expert_name"], expert["slug"])
    return experts


def get_expert_for_query(
    query: str,
    experts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the best single expert for a text query using keyword overlap."""
    query_words = set(query.lower().split())
    best_score = 0
    best_expert = None

    for expert in experts:
        concepts = expert.get("metadata", {}).get("concepts", [])
        caps = expert.get("capabilities", [])
        expert_words = set()
        for c in concepts:
            expert_words.update(c.lower().split())
        for cap in caps:
            expert_words.update(cap.lower().split())

        overlap = len(query_words & expert_words)
        if overlap > best_score:
            best_score = overlap
            best_expert = expert

    return best_expert
