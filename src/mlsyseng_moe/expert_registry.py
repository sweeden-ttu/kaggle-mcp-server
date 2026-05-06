"""Expert registry — creates, manages, and queries chapter experts.

Each ML Principles chapter becomes an expert with capabilities, skills,
strategy, mathematical formulas, and loop configuration.
"""

import json
import logging
import os
import re
from typing import Any, Optional

from . import database as db

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
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

SKILL_MAP = {
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "architecture": ["kaggle-model-trainer", "kaggle-neural-arch-builder"],
    "training": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "evaluation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "representation": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "models": ["kaggle-model-trainer", "kaggle-ensemble-builder"],
    "probabilistic": ["kaggle-model-trainer", "kaggle-bayesian-optimizer"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _infer_capabilities(concepts: list[dict]) -> list[str]:
    """Derive expert capabilities from extracted concepts."""
    capability_templates = {
        "optimization": "Optimize model parameters using {name} techniques",
        "architecture": "Design and implement {name} architectures",
        "training": "Apply {name} for robust model training",
        "evaluation": "Evaluate models using {name}",
        "representation": "Engineer features with {name}",
        "models": "Build and tune {name} models",
        "probabilistic": "Apply {name} reasoning to predictions",
        "general": "Leverage {name} in ML pipelines",
    }
    capabilities = set()
    for concept in concepts:
        cat = concept.get("category", "general")
        template = capability_templates.get(cat, capability_templates["general"])
        capabilities.add(template.format(name=concept["name"]))
    base = [
        "Build baseline models quickly",
        "Systematic hyperparameter search",
    ]
    return base + sorted(capabilities)


def _infer_skills(concepts: list[dict]) -> list[str]:
    """Map concepts to Kaggle skill paths."""
    skill_names: set[str] = set()
    for concept in concepts:
        cat = concept.get("category", "general")
        skill_names.update(SKILL_MAP.get(cat, SKILL_MAP["general"]))
    return sorted(
        os.path.join(KAGGLE_SKILLS_PATH, s) for s in skill_names
    )


def _build_formula(concepts: list[dict]) -> dict:
    """Build objective formula from concepts."""
    metrics = set(DEFAULT_FORMULA["metrics"])
    for concept in concepts:
        name_lower = concept["name"].lower()
        if "f1" in name_lower:
            metrics.add("f1_score")
        elif "accuracy" in name_lower:
            metrics.add("accuracy")
        elif "auc" in name_lower or "roc" in name_lower:
            metrics.add("auc_roc")
        elif "precision" in name_lower:
            metrics.add("precision")
        elif "recall" in name_lower:
            metrics.add("recall")
    return {
        "objective": DEFAULT_FORMULA["objective"],
        "function": DEFAULT_FORMULA["function"],
        "metrics": sorted(metrics),
    }


def create_expert_from_chapter(
    chapter_number: str,
    title: str,
    concepts: list[dict],
    chapter_id: int | None = None,
    conn=None,
) -> dict:
    """Create an expert definition from a chapter's extracted data."""
    slug = f"{chapter_number}_{_slugify(title)}"
    expert_name = f"{chapter_number}_{title.replace(' ', '_')}"

    expert = {
        "slug": slug,
        "expert_name": expert_name,
        "chapter_id": chapter_id,
        "capabilities": _infer_capabilities(concepts),
        "skills": _infer_skills(concepts),
        "strategy": DEFAULT_STRATEGY,
        "formula": _build_formula(concepts),
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }

    if conn:
        db.upsert_expert(conn, expert)
        logger.info("Registered expert: %s", expert_name)

    return expert


def register_experts_from_extraction(
    extraction_results: list[dict], conn=None
) -> list[dict]:
    """Create experts for all extracted chapters."""
    experts = []
    for result in extraction_results:
        if not result.get("concepts"):
            continue
        chapter_id = None
        if conn:
            row = conn.execute(
                "SELECT id FROM chapters WHERE chapter_number = ?",
                (result["chapter_number"],),
            ).fetchone()
            chapter_id = row["id"] if row else None

        expert = create_expert_from_chapter(
            result["chapter_number"],
            result["title"],
            result["concepts"],
            chapter_id=chapter_id,
            conn=conn,
        )
        experts.append(expert)
    return experts


def get_expert(conn, slug: str) -> dict | None:
    return db.get_expert(conn, slug)


def list_experts(conn) -> list[dict]:
    return db.list_experts(conn)


def query_expert(conn, slug: str, question: str) -> dict:
    """Query a specific expert with a question and get contextual answer."""
    expert = db.get_expert(conn, slug)
    if not expert:
        return {"error": f"Expert '{slug}' not found"}

    chapter = db.get_chapter(conn, expert["chapter_id"]) if expert.get("chapter_id") else None
    concepts = (
        db.list_concepts(conn, expert["chapter_id"])
        if expert.get("chapter_id")
        else []
    )

    return {
        "expert": expert["expert_name"],
        "capabilities": expert.get("capabilities", []),
        "strategy": expert.get("strategy", ""),
        "formula": expert.get("formula", {}),
        "concepts": [c["name"] for c in concepts],
        "chapter_content_preview": (
            chapter["content_md"][:2000] if chapter and chapter.get("content_md") else ""
        ),
        "question": question,
    }


def select_experts_for_competition(
    conn,
    competition_description: str,
    ranked_chapters: list[dict],
) -> list[dict]:
    """Select and rank experts relevant to a competition."""
    selected = []
    all_experts = db.list_experts(conn)
    expert_by_chapter = {
        str(e.get("chapter_id")): e for e in all_experts if e.get("chapter_id")
    }

    for ranked in ranked_chapters:
        ch_num = ranked["chapter_number"]
        row = conn.execute(
            "SELECT id FROM chapters WHERE chapter_number = ?", (ch_num,)
        ).fetchone()
        if not row:
            continue
        chapter_id = str(row["id"])
        expert = expert_by_chapter.get(chapter_id)
        if expert:
            selected.append(
                {
                    **expert,
                    "relevance_score": ranked.get("relevance_score", 0),
                }
            )
    return selected


def export_expert_json(expert: dict, output_dir: str = "experts") -> str:
    """Export an expert definition to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{expert['slug']}.json")
    with open(path, "w") as f:
        json.dump(expert, f, indent=2, default=str)
    return path
