"""Expert registry for managing chapter-based ML experts.

Each chapter in ML Principles becomes an expert with:
- Capabilities (what they can do)
- Skills (Kaggle skill paths they recommend)
- Strategy (approach pattern)
- Formula (objective function)
- Loop config (convergence parameters)
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

from . import database

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_FORMULA = {
    "objective": "minimize_validation_loss",
    "function": "L = f(X, θ, α)",
    "metrics": ["accuracy", "f1_score"],
}

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CHAPTER_EXPERT_TEMPLATES = {
    "ml_systems": {
        "capabilities": [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
            "Pipeline orchestration",
            "Model deployment and serving",
        ],
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer"],
        "strategy": "Baseline → Systematic Tuning → Pipeline Automation → Deploy",
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = Σ(y - ŷ)² / n + λ||θ||²",
            "metrics": ["rmse", "mae", "r2"],
        },
    },
    "deep_learning": {
        "capabilities": [
            "Design neural architectures",
            "Transfer learning and fine-tuning",
            "GPU-accelerated training",
            "Regularization techniques",
        ],
        "skills": ["kaggle-deep-learning", "kaggle-gpu-trainer"],
        "strategy": "Pretrained Model → Fine-tune → Architecture Search → Ensemble",
        "formula": {
            "objective": "minimize_cross_entropy",
            "function": "L = -Σ y·log(ŷ) + λ·Σ||W||²",
            "metrics": ["accuracy", "top_k_accuracy", "auc"],
        },
    },
    "optimization": {
        "capabilities": [
            "Gradient-based optimization",
            "Learning rate scheduling",
            "Convergence analysis",
            "Hyperparameter optimization",
        ],
        "skills": ["kaggle-optimizer", "kaggle-hyperparameter-tuner"],
        "strategy": "Grid Search → Bayesian Optimization → Learning Rate Finder → Early Stopping",
        "formula": {
            "objective": "minimize_generalization_error",
            "function": "θ* = argmin_θ E[L(f(x;θ), y)]",
            "metrics": ["convergence_rate", "wall_time", "best_score"],
        },
    },
    "feature_engineering": {
        "capabilities": [
            "Feature extraction and transformation",
            "Dimensionality reduction",
            "Feature selection",
            "Data augmentation",
        ],
        "skills": ["kaggle-feature-engineer", "kaggle-preprocessor"],
        "strategy": "EDA → Feature Extraction → Selection → Validation",
        "formula": {
            "objective": "maximize_mutual_information",
            "function": "I(X;Y) = H(Y) - H(Y|X)",
            "metrics": ["feature_importance", "mutual_info", "correlation"],
        },
    },
    "ensemble_methods": {
        "capabilities": [
            "Model stacking and blending",
            "Bagging and boosting",
            "Diversity-based ensemble selection",
            "Weighted voting",
        ],
        "skills": ["kaggle-ensemble", "kaggle-model-trainer"],
        "strategy": "Train Diverse Models → Validate → Stack → Blend → Submit",
        "formula": {
            "objective": "minimize_ensemble_error",
            "function": "ŷ = Σ wᵢ·fᵢ(x), Σwᵢ = 1",
            "metrics": ["ensemble_diversity", "stacking_gain", "blend_score"],
        },
    },
    "evaluation": {
        "capabilities": [
            "Cross-validation design",
            "Metric selection and analysis",
            "Statistical significance testing",
            "Leaderboard strategy",
        ],
        "skills": ["kaggle-evaluator", "kaggle-validator"],
        "strategy": "Define Metrics → CV Strategy → Validate → Leaderboard Probing",
        "formula": {
            "objective": "minimize_cv_variance",
            "function": "CV = (1/k)·Σ L(f(X_test_i), y_test_i)",
            "metrics": ["cv_mean", "cv_std", "lb_correlation"],
        },
    },
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug


def create_expert_from_chapter(
    chapter_number: int,
    chapter_title: str,
    chapter_id: int,
    concepts: list[dict],
    db_path: Optional[str] = None,
) -> dict:
    """Create an expert definition from a chapter's extracted content."""
    slug = f"{chapter_number:02d}_{_slugify(chapter_title)}"
    expert_name = f"{chapter_number:02d}_{chapter_title}"

    template = _match_template(chapter_title, concepts)

    capabilities = template.get("capabilities", [
        f"Apply {chapter_title} concepts to ML problems",
        "Systematic approach based on chapter principles",
    ])

    skills_base = template.get("skills", ["kaggle-preprocessor", "kaggle-model-trainer"])
    skills = [os.path.join(KAGGLE_SKILLS_PATH, s) for s in skills_base]

    strategy = template.get("strategy", DEFAULT_STRATEGY)

    formula = template.get("formula", DEFAULT_FORMULA.copy())
    formula["metrics"] = _infer_metrics(concepts) or formula.get("metrics", [])

    loop_config = DEFAULT_LOOP_CONFIG.copy()

    expert_id = database.insert_expert(
        expert_name=expert_name,
        slug=slug,
        chapter_id=chapter_id,
        capabilities=capabilities,
        skills=skills,
        strategy=strategy,
        formula=formula,
        loop_config=loop_config,
        db_path=db_path,
    )

    expert_def = {
        "expert_name": expert_name,
        "slug": slug,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": strategy,
        "formula": formula,
        "loop_config": loop_config,
    }

    _save_expert_file(slug, expert_def)

    return expert_def


def _match_template(title: str, concepts: list[dict]) -> dict:
    """Match a chapter to the best expert template."""
    title_lower = title.lower()
    concept_names = [c["name"].lower() for c in concepts]
    all_text = f"{title_lower} {' '.join(concept_names)}"

    best_match = ""
    best_score = 0

    scoring_keywords = {
        "ml_systems": ["system", "pipeline", "deploy", "serve", "infrastructure"],
        "deep_learning": ["neural", "deep", "network", "cnn", "rnn", "transformer"],
        "optimization": ["optimization", "gradient", "convergence", "learning rate"],
        "feature_engineering": ["feature", "extraction", "dimensionality", "pca"],
        "ensemble_methods": ["ensemble", "bagging", "boosting", "stacking", "random forest"],
        "evaluation": ["evaluation", "metric", "cross-validation", "precision", "recall"],
    }

    for template_key, keywords in scoring_keywords.items():
        score = sum(1 for kw in keywords if kw in all_text)
        if score > best_score:
            best_score = score
            best_match = template_key

    if best_match and best_score > 0:
        return CHAPTER_EXPERT_TEMPLATES[best_match]

    return {}


def _infer_metrics(concepts: list[dict]) -> list[str]:
    """Infer relevant metrics from extracted concepts."""
    metric_map = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1_score",
        "auc": "auc_roc",
        "accuracy": "accuracy",
        "loss": "loss",
        "cross-entropy": "log_loss",
        "mse": "mse",
        "rmse": "rmse",
    }

    metrics = set()
    for concept in concepts:
        name = concept.get("name", "").lower()
        for keyword, metric in metric_map.items():
            if keyword in name:
                metrics.add(metric)

    return list(metrics) if metrics else ["accuracy", "f1_score"]


def _save_expert_file(slug: str, expert_def: dict) -> None:
    """Save expert definition to a JSON file."""
    experts_dir = Path(__file__).parent.parent / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    filepath = experts_dir / f"{slug}.json"
    with open(filepath, "w") as f:
        json.dump(expert_def, f, indent=2)
    logger.info(f"Saved expert definition: {filepath}")


def register_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Create experts for all indexed chapters."""
    chapters = database.get_all_chapters(db_path)
    experts = []

    for chapter in chapters:
        concepts = database.get_concepts_for_chapter(chapter["id"], db_path)
        expert = create_expert_from_chapter(
            chapter_number=chapter["chapter_number"],
            chapter_title=chapter["title"],
            chapter_id=chapter["id"],
            concepts=concepts,
            db_path=db_path,
        )
        experts.append(expert)

    return experts


def get_expert_for_query(
    query: str,
    db_path: Optional[str] = None,
) -> Optional[dict]:
    """Find the best expert for a given query by matching capabilities."""
    experts = database.get_all_experts(db_path)
    if not experts:
        return None

    query_lower = query.lower()
    best_expert = None
    best_score = 0

    for expert in experts:
        score = 0
        for cap in expert.get("capabilities", []):
            words = cap.lower().split()
            score += sum(1 for w in words if w in query_lower)

        name_words = expert["expert_name"].lower().split("_")
        score += sum(2 for w in name_words if w in query_lower)

        if score > best_score:
            best_score = score
            best_expert = expert

    return best_expert
