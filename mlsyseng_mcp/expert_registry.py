"""Expert registry - manages chapter experts with skills, strategies, and formulas."""

import os
import re
from typing import Optional

from mlsyseng_mcp.database import (
    get_all_chapters,
    get_all_experts,
    get_chapter_concepts,
    get_expert_by_slug,
    insert_expert,
)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-submission-builder",
]

STRATEGY_TEMPLATE = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Ensemble → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug


def create_expert_from_chapter(
    chapter_number: int,
    title: str,
    chapter_id: int,
    concepts: list[dict],
    db_path: Optional[str] = None,
) -> dict:
    """Create an expert definition from a chapter and its concepts."""
    expert_name = f"{chapter_number:02d}_{title.replace(' ', '_')}"
    slug = slugify(expert_name)

    capabilities = _infer_capabilities(concepts)
    skills = _map_skills(concepts)
    formula = _build_formula(concepts)

    expert_data = {
        "expert_name": expert_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": STRATEGY_TEMPLATE,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }

    insert_expert(expert_data, db_path)
    return expert_data


def _infer_capabilities(concepts: list[dict]) -> list[str]:
    """Infer expert capabilities from concepts."""
    capability_map = {
        "algorithm": [
            "Select appropriate ML algorithms",
            "Compare model performance systematically",
        ],
        "deep_learning": [
            "Design neural network architectures",
            "Apply deep learning to structured/unstructured data",
        ],
        "optimization": [
            "Tune learning rate and optimizer settings",
            "Monitor convergence and prevent divergence",
        ],
        "regularization": [
            "Prevent overfitting with regularization",
            "Diagnose bias-variance issues",
        ],
        "methodology": [
            "Design proper train/validation/test splits",
            "Build systematic experimentation pipelines",
        ],
        "application": [
            "Apply domain-specific preprocessing",
            "Select appropriate evaluation metrics",
        ],
    }

    capabilities = set()
    capabilities.add("Build baseline models quickly")
    capabilities.add("Systematic hyperparameter search")

    categories_seen = set()
    for concept in concepts:
        cat = concept.get("category", "general")
        categories_seen.add(cat)

    for cat in categories_seen:
        if cat in capability_map:
            for cap in capability_map[cat]:
                capabilities.add(cap)

    return sorted(capabilities)


def _map_skills(concepts: list[dict]) -> list[str]:
    """Map concepts to Kaggle skill paths."""
    skills = set()
    for skill in DEFAULT_SKILLS:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill))

    concept_skill_map = {
        "deep_learning": "kaggle-deep-learning",
        "optimization": "kaggle-optimizer",
        "application": "kaggle-domain-expert",
    }

    for concept in concepts:
        cat = concept.get("category", "general")
        if cat in concept_skill_map:
            skills.add(os.path.join(KAGGLE_SKILLS_PATH, concept_skill_map[cat]))

    return sorted(skills)


def _build_formula(concepts: list[dict]) -> dict:
    """Build the mathematical formula for an expert."""
    categories = set(c.get("category", "general") for c in concepts)

    metrics = ["accuracy"]
    if "deep_learning" in categories:
        metrics.extend(["loss", "val_loss"])
    if "algorithm" in categories:
        metrics.extend(["f1_score", "auc_roc"])
    if "application" in categories:
        metrics.extend(["precision", "recall"])

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": sorted(set(metrics)),
    }


def register_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Register experts for all indexed chapters."""
    chapters = get_all_chapters(db_path)
    experts_created = []

    for chapter in chapters:
        concepts = get_chapter_concepts(chapter["id"], db_path)
        expert = create_expert_from_chapter(
            chapter_number=chapter["chapter_number"],
            title=chapter["title"],
            chapter_id=chapter["id"],
            concepts=concepts,
            db_path=db_path,
        )
        experts_created.append(expert)

    return experts_created


def list_experts(db_path: Optional[str] = None) -> list[dict]:
    """List all registered experts."""
    return get_all_experts(db_path)


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a specific expert by slug."""
    return get_expert_by_slug(slug, db_path)


def query_expert(slug: str, question: str, db_path: Optional[str] = None) -> dict:
    """Query a specific expert with a question."""
    expert = get_expert_by_slug(slug, db_path)
    if expert is None:
        return {"error": f"Expert '{slug}' not found"}

    return {
        "expert": expert["expert_name"],
        "slug": slug,
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "response": (
            f"As {expert['expert_name']}, regarding '{question}': "
            f"My strategy is {expert['strategy']}. "
            f"Key capabilities: {', '.join(expert['capabilities'][:3])}. "
            f"Objective: {expert['formula'].get('objective', 'minimize loss')}."
        ),
    }
