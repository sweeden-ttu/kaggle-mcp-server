"""Expert management for the MLSysEng MoE system."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from mlsyseng_moe.database import (
    get_all_chapters,
    get_all_experts,
    get_concepts_for_chapter,
    get_expert_by_slug,
    init_db,
    insert_expert,
)

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-submission-builder",
    "kaggle-eda-analyzer",
]

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


def slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    slug = re.sub(r"[^\w\s]", "", name.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug


def infer_capabilities(concepts: list[dict]) -> list[str]:
    """Infer expert capabilities from extracted concepts."""
    capabilities = []
    categories = set(c.get("category", "") for c in concepts)

    capability_map = {
        "optimization": "Optimize model training and convergence",
        "architecture": "Design and select neural network architectures",
        "regularization": "Apply regularization techniques to prevent overfitting",
        "evaluation": "Evaluate model performance with appropriate metrics",
        "ensemble_methods": "Build and combine ensemble models",
        "unsupervised": "Apply unsupervised learning and dimensionality reduction",
        "general": "Apply general ML principles and best practices",
    }

    for cat in categories:
        if cat in capability_map:
            capabilities.append(capability_map[cat])

    if not capabilities:
        capabilities = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]

    return capabilities


def infer_skills(concepts: list[dict]) -> list[str]:
    """Infer Kaggle skills based on chapter concepts."""
    skills = []
    categories = set(c.get("category", "") for c in concepts)

    skill_map = {
        "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
        "architecture": ["kaggle-model-trainer", "kaggle-deep-learning"],
        "regularization": ["kaggle-model-trainer", "kaggle-preprocessor"],
        "evaluation": ["kaggle-submission-builder", "kaggle-eda-analyzer"],
        "ensemble_methods": ["kaggle-model-trainer", "kaggle-ensemble-builder"],
        "unsupervised": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    }

    for cat in categories:
        if cat in skill_map:
            skills.extend(skill_map[cat])

    if not skills:
        skills = DEFAULT_SKILLS[:3]

    unique_skills = list(dict.fromkeys(skills))
    return [os.path.join(KAGGLE_SKILLS_PATH, s) for s in unique_skills]


def infer_formula(concepts: list[dict], chapter_title: str) -> dict:
    """Infer mathematical objective function from chapter content."""
    categories = set(c.get("category", "") for c in concepts)

    if "optimization" in categories:
        return {
            "objective": "minimize_training_loss",
            "function": "L = Σ loss(y_pred, y_true) + λ·R(θ)",
            "metrics": ["loss", "gradient_norm"],
        }
    elif "evaluation" in categories:
        return {
            "objective": "maximize_evaluation_metric",
            "function": "score = f(precision, recall, specificity)",
            "metrics": ["accuracy", "f1_score", "auc"],
        }
    elif "ensemble_methods" in categories:
        return {
            "objective": "minimize_ensemble_error",
            "function": "E = Σ w_i · h_i(x), minimize bias² + variance",
            "metrics": ["accuracy", "diversity"],
        }
    elif "architecture" in categories:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = CrossEntropy(softmax(Wx + b), y)",
            "metrics": ["accuracy", "loss"],
        }

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    }


def register_experts_from_chapters(
    force: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Create expert definitions from indexed chapters."""
    init_db(db_path)
    chapters = get_all_chapters(db_path)

    results = {"created": 0, "skipped": 0, "errors": 0, "experts": []}

    for chapter in chapters:
        chapter_num = chapter["chapter_number"]
        title = chapter["title"]
        expert_name = f"{chapter_num:02d}_{title}"
        slug = slugify(expert_name)

        if not force:
            existing = get_expert_by_slug(slug, db_path)
            if existing:
                results["skipped"] += 1
                continue

        try:
            concepts = get_concepts_for_chapter(chapter["id"], db_path)

            capabilities = infer_capabilities(concepts)
            skills = infer_skills(concepts)
            formula = infer_formula(concepts, title)

            insert_expert(
                expert_name=expert_name,
                slug=slug,
                chapter_id=chapter["id"],
                capabilities=capabilities,
                skills=skills,
                strategy=DEFAULT_STRATEGY,
                formula=formula,
                loop_config=DEFAULT_LOOP_CONFIG,
                db_path=db_path,
            )

            results["created"] += 1
            results["experts"].append({
                "name": expert_name,
                "slug": slug,
                "capabilities": capabilities,
                "skills": skills,
            })

        except Exception as e:
            logger.error(f"Failed to create expert for chapter {title}: {e}")
            results["errors"] += 1

    return results


def get_expert_definition(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get full expert definition by slug."""
    expert = get_expert_by_slug(slug, db_path)
    if not expert:
        return None

    return {
        "expert_name": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert["capabilities"],
        "skills": expert["skills"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "loop_config": expert["loop_config"],
    }


def list_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """List all registered experts with their definitions."""
    experts = get_all_experts(db_path)
    return [
        {
            "expert_name": e["expert_name"],
            "slug": e["slug"],
            "capabilities": e["capabilities"],
            "skills": e["skills"],
            "strategy": e["strategy"],
            "formula": e["formula"],
            "loop_config": e["loop_config"],
        }
        for e in experts
    ]


def select_experts_for_competition(
    competition_name: str,
    relevant_chapters: list[str],
    db_path: Optional[str] = None,
) -> list[dict]:
    """Select experts relevant to a competition based on RAG results."""
    experts = get_all_experts(db_path)
    selected = []

    for expert in experts:
        chapter_num = str(expert.get("chapter_id", ""))
        if chapter_num in relevant_chapters:
            selected.append({
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert["capabilities"],
                "skills": expert["skills"],
                "strategy": expert["strategy"],
            })

    if not selected and experts:
        selected = [
            {
                "expert_name": e["expert_name"],
                "slug": e["slug"],
                "capabilities": e["capabilities"],
                "skills": e["skills"],
                "strategy": e["strategy"],
            }
            for e in experts[:3]
        ]

    return selected


def save_expert_to_file(slug: str, output_dir: Optional[str] = None, db_path: Optional[str] = None) -> Optional[str]:
    """Save an expert definition to a JSON file."""
    expert = get_expert_definition(slug, db_path)
    if not expert:
        return None

    output_path = Path(output_dir or "experts")
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{slug}.json"
    with open(filepath, "w") as f:
        json.dump(expert, f, indent=2)

    return str(filepath)
