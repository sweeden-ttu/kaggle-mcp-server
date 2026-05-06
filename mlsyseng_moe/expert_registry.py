"""Expert management for MLSysEng MoE system."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from . import database as db

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

CHAPTER_SKILL_MAPPING = {
    "optimization": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "architecture": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "training": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-evaluator"],
    "models": ["kaggle-model-trainer", "kaggle-ensemble"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "feature_engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "evaluation": ["kaggle-evaluator", "kaggle-submitter"],
}

CHAPTER_CAPABILITIES = {
    "optimization": [
        "Optimize model parameters efficiently",
        "Select appropriate loss functions",
        "Tune learning rates and schedulers",
    ],
    "architecture": [
        "Design neural network architectures",
        "Choose appropriate model depth and width",
        "Implement attention mechanisms",
    ],
    "training": [
        "Prevent overfitting with regularization",
        "Implement cross-validation strategies",
        "Apply data augmentation techniques",
    ],
    "models": [
        "Build ensemble models",
        "Select appropriate algorithms",
        "Implement tree-based methods",
    ],
    "general": [
        "Build baseline models quickly",
        "Systematic hyperparameter search",
        "End-to-end pipeline development",
    ],
}


def create_expert_from_chapter(
    chapter_id: int,
    chapter_number: int,
    title: str,
    concepts: Optional[list[dict]] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Create an expert definition from a chapter."""
    slug = _make_slug(chapter_number, title)
    expert_name = f"{chapter_number:02d}_{title}"

    primary_category = _determine_primary_category(concepts or [])

    skills_base = CHAPTER_SKILL_MAPPING.get(primary_category, CHAPTER_SKILL_MAPPING["general"])
    skills = [
        os.path.join(KAGGLE_SKILLS_PATH, s) for s in skills_base
    ]

    capabilities = CHAPTER_CAPABILITIES.get(
        primary_category, CHAPTER_CAPABILITIES["general"]
    )
    if concepts:
        for concept in concepts[:3]:
            capabilities.append(f"Apply {concept['concept_name']} techniques")

    formula = dict(DEFAULT_FORMULA)
    if primary_category == "optimization":
        formula["function"] = "L = Σ loss(y, ŷ) + λ·R(θ)"
        formula["metrics"] = ["loss", "convergence_rate"]
    elif primary_category == "architecture":
        formula["function"] = "y = f(W·x + b)"
        formula["metrics"] = ["accuracy", "latency"]
    elif primary_category == "models":
        formula["function"] = "ŷ = Σ αᵢ·hᵢ(x)"
        formula["metrics"] = ["accuracy", "auc"]

    expert_id = db.insert_expert(
        expert_name=expert_name,
        slug=slug,
        chapter_id=chapter_id,
        capabilities=capabilities,
        skills=skills,
        strategy=DEFAULT_STRATEGY,
        formula=formula,
        loop_config=DEFAULT_LOOP_CONFIG,
        db_path=db_path,
    )

    return {
        "id": expert_id,
        "expert_name": expert_name,
        "slug": slug,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG,
    }


def register_experts_from_extraction(
    extraction_results: list[dict],
    db_path: Optional[str] = None,
) -> list[dict]:
    """Register experts for all successfully extracted chapters."""
    experts_created = []

    for result in extraction_results:
        if result.get("status") != "completed":
            continue

        chapter_id = result["chapter_id"]
        concepts = db.get_chapter_concepts(chapter_id, db_path)

        expert = create_expert_from_chapter(
            chapter_id=chapter_id,
            chapter_number=result["chapter_number"],
            title=result["title"],
            concepts=concepts,
            db_path=db_path,
        )
        experts_created.append(expert)

    return experts_created


def list_experts(db_path: Optional[str] = None) -> list[dict]:
    """List all registered experts."""
    return db.get_all_experts(db_path)


def get_expert(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get a specific expert by slug."""
    return db.get_expert_by_slug(slug, db_path)


def query_expert(slug: str, question: str, db_path: Optional[str] = None) -> dict:
    """Query a specific expert with a question."""
    expert = db.get_expert_by_slug(slug, db_path)
    if not expert:
        return {"error": f"Expert '{slug}' not found"}

    chapter_id = expert.get("chapter_id")
    context = ""
    if chapter_id:
        blocks = db.get_chapter_content(chapter_id, db_path)
        context = "\n".join(b["content"] for b in blocks[:10])

    return {
        "expert": expert["expert_name"],
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "knowledge_context": context[:2000] if context else "No extracted content available",
        "question": question,
    }


def _make_slug(chapter_number: int, title: str) -> str:
    """Create a URL-safe slug from chapter number and title."""
    clean = re.sub(r"[^a-z0-9\s]", "", title.lower())
    clean = re.sub(r"\s+", "_", clean.strip())
    return f"{chapter_number:02d}_{clean}"


def _determine_primary_category(concepts: list[dict]) -> str:
    """Determine the primary category from concept list."""
    if not concepts:
        return "general"

    category_counts: dict[str, int] = {}
    for concept in concepts:
        cat = concept.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts:
        return max(category_counts, key=category_counts.get)
    return "general"
