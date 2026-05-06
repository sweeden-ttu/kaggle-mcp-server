"""Expert management system for MLSysEng MoE."""

import json
import os
import re
from pathlib import Path
from typing import Optional

from mlsyseng_moe import database

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPINGS = {
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "architecture": ["kaggle-model-trainer", "kaggle-neural-arch"],
    "regularization": ["kaggle-model-trainer", "kaggle-validator"],
    "training": ["kaggle-model-trainer", "kaggle-pipeline"],
    "methods": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-ensemble"],
    "representation": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "theory": ["kaggle-eda", "kaggle-validator"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
}


def _slugify(name: str) -> str:
    """Convert a chapter title to a slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _infer_capabilities(concepts: list[dict], title: str) -> list[str]:
    """Infer expert capabilities from extracted concepts."""
    capabilities = []
    categories = set(c.get("category", "general") for c in concepts)

    capability_map = {
        "optimization": "Optimize model parameters and training procedures",
        "architecture": "Design and implement neural network architectures",
        "regularization": "Apply regularization techniques to prevent overfitting",
        "training": "Configure training pipelines with proper validation",
        "methods": "Select and apply appropriate ML algorithms",
        "representation": "Engineer features and reduce dimensionality",
        "theory": "Apply statistical and probabilistic reasoning",
    }

    for cat in categories:
        if cat in capability_map:
            capabilities.append(capability_map[cat])

    capabilities.append("Build baseline models quickly")
    capabilities.append("Systematic hyperparameter search")

    return list(set(capabilities))[:8]


def _infer_skills(concepts: list[dict]) -> list[str]:
    """Infer relevant Kaggle skills from concepts."""
    skills_path = Path(DEFAULT_KAGGLE_SKILLS_PATH)
    categories = set(c.get("category", "general") for c in concepts)

    skill_names = set()
    for cat in categories:
        for skill in SKILL_MAPPINGS.get(cat, SKILL_MAPPINGS["general"]):
            skill_names.add(skill)

    return [str(skills_path / name) for name in sorted(skill_names)]


def _infer_formula(concepts: list[dict], title: str) -> dict:
    """Infer the expert's objective function based on concepts."""
    categories = set(c.get("category", "general") for c in concepts)

    if "optimization" in categories:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": ["accuracy", "f1_score", "log_loss"],
        }
    elif "architecture" in categories:
        return {
            "objective": "minimize_architecture_complexity",
            "function": "C = complexity(model) + λ * loss(X, y)",
            "metrics": ["accuracy", "inference_time", "param_count"],
        }
    elif "representation" in categories:
        return {
            "objective": "maximize_information_gain",
            "function": "IG = H(Y) - H(Y|X_features)",
            "metrics": ["mutual_information", "feature_importance"],
        }
    else:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": ["accuracy", "f1_score"],
        }


def create_expert_from_chapter(
    chapter_id: int,
    chapter_number: int,
    title: str,
    concepts: list[dict],
    db_path: Optional[str] = None,
) -> dict:
    """Create an expert definition from a chapter's extracted data."""
    formatted_name = f"{chapter_number:02d}_{title.replace(' ', '_')}"
    slug = _slugify(f"{chapter_number:02d}_{title}")

    expert_def = {
        "expert_name": formatted_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": _infer_capabilities(concepts, title),
        "skills": _infer_skills(concepts),
        "strategy": DEFAULT_STRATEGY,
        "formula": _infer_formula(concepts, title),
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }

    database.store_expert(expert_def, db_path)
    return expert_def


def register_experts_from_db(db_path: Optional[str] = None) -> list[dict]:
    """Register experts for all indexed chapters."""
    chapters = database.get_all_chapters(db_path)
    experts = []

    for chapter in chapters:
        concepts = database.get_concepts_for_chapter(chapter["id"], db_path)
        expert = create_expert_from_chapter(
            chapter_id=chapter["id"],
            chapter_number=chapter["chapter_number"],
            title=chapter["title"],
            concepts=concepts,
            db_path=db_path,
        )
        experts.append(expert)

    return experts


def get_expert_context(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get full context for an expert including chapter content."""
    expert = database.get_expert_by_slug(slug, db_path)
    if not expert:
        return None

    chapter_content = None
    concepts = []
    if expert.get("chapter_id"):
        chapter_content = database.get_chapter_content(expert["chapter_id"], db_path)
        concepts = database.get_concepts_for_chapter(expert["chapter_id"], db_path)

    return {
        "expert": expert,
        "chapter_content_preview": chapter_content[:2000] if chapter_content else None,
        "concepts": concepts,
    }


def save_expert_json(expert: dict, output_dir: Optional[str] = None) -> str:
    """Save expert definition as a JSON file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "experts")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, f"{expert['slug']}.json")

    with open(filepath, "w") as f:
        json.dump(expert, f, indent=2)

    return filepath
