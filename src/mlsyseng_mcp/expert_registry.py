"""Expert management: registers chapter experts with skills, strategies, and formulas."""

import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .database import ChapterRecord, ExpertRecord, MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

SKILL_TEMPLATES = {
    "preprocessing": {
        "name": "kaggle-preprocessor",
        "description": "Data loading, cleaning, missing value imputation, type conversion",
    },
    "eda": {
        "name": "kaggle-eda",
        "description": "Exploratory data analysis, distribution plots, correlation matrices",
    },
    "feature_engineering": {
        "name": "kaggle-feature-engineer",
        "description": "Feature creation, selection, transformation, polynomial features",
    },
    "model_training": {
        "name": "kaggle-model-trainer",
        "description": "Model selection, training, cross-validation, grid search",
    },
    "ensemble": {
        "name": "kaggle-ensembler",
        "description": "Stacking, blending, weighted averaging of models",
    },
    "deep_learning": {
        "name": "kaggle-deep-learner",
        "description": "Neural network architecture, training loops, learning schedules",
    },
    "submission": {
        "name": "kaggle-submitter",
        "description": "Format predictions, create submission CSV, Kaggle API upload",
    },
    "evaluation": {
        "name": "kaggle-evaluator",
        "description": "Metric computation, validation strategy, leaderboard probing",
    },
}

CONCEPT_TO_SKILLS = {
    "gradient descent": ["model_training", "deep_learning"],
    "backpropagation": ["deep_learning"],
    "loss function": ["model_training", "evaluation"],
    "regularization": ["model_training", "deep_learning"],
    "overfitting": ["model_training", "evaluation"],
    "underfitting": ["model_training", "evaluation"],
    "bias-variance": ["model_training", "evaluation"],
    "cross-validation": ["model_training", "evaluation"],
    "hyperparameter": ["model_training"],
    "feature engineering": ["feature_engineering", "preprocessing"],
    "dimensionality reduction": ["feature_engineering", "preprocessing"],
    "ensemble": ["ensemble", "model_training"],
    "bagging": ["ensemble"],
    "boosting": ["ensemble"],
    "random forest": ["ensemble", "model_training"],
    "neural network": ["deep_learning"],
    "deep learning": ["deep_learning"],
    "convolutional": ["deep_learning"],
    "recurrent": ["deep_learning"],
    "transformer": ["deep_learning"],
    "attention mechanism": ["deep_learning"],
    "embedding": ["deep_learning", "feature_engineering"],
    "optimization": ["model_training"],
    "decision tree": ["model_training"],
    "clustering": ["preprocessing", "feature_engineering"],
    "k-means": ["preprocessing", "feature_engineering"],
    "pca": ["feature_engineering", "preprocessing"],
    "bayesian": ["model_training"],
    "reinforcement learning": ["model_training"],
    "transfer learning": ["deep_learning"],
    "fine-tuning": ["deep_learning", "model_training"],
    "data augmentation": ["preprocessing", "deep_learning"],
    "normalization": ["preprocessing"],
    "tokenization": ["preprocessing"],
    "f1 score": ["evaluation"],
    "precision": ["evaluation"],
    "recall": ["evaluation"],
    "accuracy": ["evaluation"],
    "auc": ["evaluation"],
    "roc": ["evaluation"],
    "confusion matrix": ["evaluation"],
    "mse": ["evaluation"],
    "mae": ["evaluation"],
    "cross entropy": ["evaluation", "deep_learning"],
    "mixture of experts": ["ensemble", "model_training"],
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def _skill_paths_for_concepts(concepts: List[str], base_path: str) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    skill_keys = set()
    for concept in concepts:
        mapped = CONCEPT_TO_SKILLS.get(concept, [])
        skill_keys.update(mapped)

    skill_keys.update(["preprocessing", "submission", "evaluation"])

    return [
        os.path.join(base_path, SKILL_TEMPLATES[k]["name"])
        for k in sorted(skill_keys)
        if k in SKILL_TEMPLATES
    ]


def _capabilities_from_concepts(concepts: List[str]) -> List[str]:
    """Generate capability descriptions from concepts."""
    caps = ["Build baseline models quickly", "Systematic hyperparameter search"]
    concept_caps = {
        "gradient descent": "Apply gradient-based optimization techniques",
        "regularization": "Apply regularization to prevent overfitting",
        "ensemble": "Build ensemble models for improved accuracy",
        "feature engineering": "Create and transform features for better signal",
        "deep learning": "Design and train deep neural networks",
        "cross-validation": "Rigorous model validation and evaluation",
        "dimensionality reduction": "Reduce feature space while preserving information",
        "transfer learning": "Leverage pre-trained models for new tasks",
        "data augmentation": "Generate additional training samples",
        "attention mechanism": "Apply attention-based architectures",
        "bayesian": "Apply Bayesian methods for uncertainty quantification",
        "clustering": "Apply unsupervised learning for pattern discovery",
    }

    for concept in concepts:
        if concept in concept_caps:
            caps.append(concept_caps[concept])

    return caps[:10]


def _formula_for_concepts(concepts: List[str]) -> Dict[str, Any]:
    """Generate an objective formula based on chapter concepts."""
    metrics = ["accuracy"]

    if any(c in concepts for c in ["f1 score", "precision", "recall"]):
        metrics = ["f1_score", "precision", "recall"]
    if any(c in concepts for c in ["mse", "mae", "rmse"]):
        metrics = ["mse", "rmse"]
    if any(c in concepts for c in ["auc", "roc"]):
        metrics.append("auc_roc")
    if any(c in concepts for c in ["cross entropy", "log loss"]):
        metrics.append("log_loss")

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": sorted(set(metrics)),
    }


def register_experts_from_chapters(
    db: MLSysEngDatabase,
    skills_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create expert definitions from all indexed chapters."""
    chapters = db.list_chapters()
    if not chapters:
        return {"status": "no_chapters", "experts_created": 0}

    base_path = skills_base_path or DEFAULT_KAGGLE_SKILLS_PATH
    created = 0
    updated = 0

    for chapter in chapters:
        expert_name = chapter.title or chapter.chapter_id
        slug = _slugify(expert_name)
        skills = _skill_paths_for_concepts(chapter.concepts, base_path)
        capabilities = _capabilities_from_concepts(chapter.concepts)
        formula = _formula_for_concepts(chapter.concepts)

        existing = db.get_expert(expert_name)

        expert = ExpertRecord(
            expert_name=expert_name,
            slug=slug,
            chapter_id=chapter.chapter_id,
            capabilities=capabilities,
            skills=skills,
            strategy="Baseline → EDA → Feature Engineering → Model Selection → Submit",
            formula=formula,
        )

        db.upsert_expert(expert)
        if existing:
            updated += 1
        else:
            created += 1

    return {
        "status": "completed",
        "experts_created": created,
        "experts_updated": updated,
        "total_experts": len(chapters),
    }


def get_expert_definition(db: MLSysEngDatabase, expert_name: str) -> Optional[Dict[str, Any]]:
    """Get the full expert definition as a dictionary."""
    expert = db.get_expert(expert_name)
    if expert is None:
        expert = db.get_expert_by_slug(expert_name)
    if expert is None:
        return None

    return asdict(expert)


def query_expert(
    db: MLSysEngDatabase,
    expert_name: str,
    question: str,
) -> Dict[str, Any]:
    """Query a specific expert for advice on a topic."""
    expert = db.get_expert(expert_name)
    if expert is None:
        expert = db.get_expert_by_slug(expert_name)
    if expert is None:
        return {"error": f"Expert '{expert_name}' not found"}

    chapter = db.get_chapter(expert.chapter_id)
    chapter_preview = ""
    if chapter:
        chapter_preview = chapter.content_md[:2000]

    return {
        "expert": expert.expert_name,
        "slug": expert.slug,
        "capabilities": expert.capabilities,
        "skills": expert.skills,
        "strategy": expert.strategy,
        "formula": expert.formula,
        "chapter_context": chapter_preview,
        "question": question,
    }


def build_competition_entry(
    db: MLSysEngDatabase,
    competition: str,
    recommended_experts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a competition entry plan using expert knowledge."""
    if recommended_experts is None:
        experts = db.list_experts()
        recommended_experts = [
            {
                "expert_name": e.expert_name,
                "slug": e.slug,
                "relevance_score": 1.0,
                "capabilities": e.capabilities,
                "skills": e.skills,
                "strategy": e.strategy,
            }
            for e in experts
        ]

    all_skills = set()
    all_capabilities = set()
    strategies = []

    for rec in recommended_experts:
        all_skills.update(rec.get("skills", []))
        all_capabilities.update(rec.get("capabilities", []))
        strategies.append(rec.get("strategy", ""))

    entry = {
        "competition": competition,
        "experts_consulted": [r["expert_name"] for r in recommended_experts],
        "combined_skills": sorted(all_skills),
        "combined_capabilities": sorted(all_capabilities),
        "strategy": strategies[0] if strategies else "Baseline → EDA → Feature Engineering → Model Selection → Submit",
        "notebooks": [
            {
                "name": f"Expert_{rec['slug']}.ipynb",
                "expert": rec["expert_name"],
                "path": f"~/{competition}/Expert_{rec['slug']}.ipynb",
            }
            for rec in recommended_experts
        ],
        "pipeline_steps": [
            {"step": 1, "name": "Baseline", "description": "Quick baseline with defaults"},
            {"step": 2, "name": "EDA", "description": "Exploratory data analysis"},
            {"step": 3, "name": "Feature Engineering", "description": "Transform and create features"},
            {"step": 4, "name": "Model Selection", "description": "Compare model architectures"},
            {"step": 5, "name": "Hyperparameter Tuning", "description": "Optimize model parameters"},
            {"step": 6, "name": "Ensemble", "description": "Combine best models"},
            {"step": 7, "name": "Submit", "description": "Generate and upload submission"},
        ],
    }

    return entry
