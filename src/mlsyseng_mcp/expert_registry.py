"""Expert registry for MLSysEng MoE system.

Manages expert definitions - each ML Principles chapter becomes an expert
with capabilities, skills, strategy, formula, and loop configuration.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_STRATEGY = (
    "Baseline -> EDA -> Feature Engineering -> Model Selection -> "
    "Hyperparameter Tuning -> Ensemble -> Submit"
)

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_TEMPLATES = {
    "data_preprocessing": {
        "name": "kaggle-preprocessor",
        "capabilities": [
            "Handle missing values",
            "Feature scaling",
            "Encode categoricals",
        ],
    },
    "model_training": {
        "name": "kaggle-model-trainer",
        "capabilities": [
            "Train baseline models",
            "Cross-validation",
            "Hyperparameter search",
        ],
    },
    "feature_engineering": {
        "name": "kaggle-feature-engineer",
        "capabilities": [
            "Create interaction features",
            "Polynomial features",
            "Domain-specific transforms",
        ],
    },
    "deep_learning": {
        "name": "kaggle-deep-learner",
        "capabilities": [
            "Build neural networks",
            "Transfer learning",
            "GPU training",
        ],
    },
    "ensemble": {
        "name": "kaggle-ensembler",
        "capabilities": [
            "Stacking",
            "Blending",
            "Voting ensembles",
        ],
    },
    "eda": {
        "name": "kaggle-eda",
        "capabilities": [
            "Statistical analysis",
            "Visualization",
            "Distribution analysis",
        ],
    },
}

CHAPTER_SKILL_MAP = {
    "optimization": ["model_training", "deep_learning"],
    "regularization": ["model_training", "feature_engineering"],
    "neural network": ["deep_learning", "model_training"],
    "deep learning": ["deep_learning", "model_training"],
    "feature": ["feature_engineering", "data_preprocessing"],
    "ensemble": ["ensemble", "model_training"],
    "tree": ["model_training", "ensemble"],
    "classification": ["model_training", "data_preprocessing"],
    "regression": ["model_training", "data_preprocessing"],
    "clustering": ["model_training", "eda"],
    "dimensionality": ["feature_engineering", "eda"],
    "bayesian": ["model_training"],
    "reinforcement": ["model_training"],
    "generative": ["deep_learning"],
    "convolutional": ["deep_learning"],
    "recurrent": ["deep_learning"],
    "transformer": ["deep_learning"],
    "attention": ["deep_learning"],
    "system": ["model_training", "data_preprocessing", "feature_engineering"],
    "evaluation": ["model_training", "eda"],
    "preprocessing": ["data_preprocessing"],
    "visualization": ["eda"],
}


def _slugify(name: str) -> str:
    """Convert a title to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _infer_skills_for_chapter(
    title: str, concepts: list[dict]
) -> list[str]:
    """Infer which Kaggle skills are relevant based on chapter content."""
    skill_ids = set()
    search_text = title.lower()
    for c in concepts:
        search_text += " " + c.get("concept", "").lower()

    for keyword, skills in CHAPTER_SKILL_MAP.items():
        if keyword in search_text:
            skill_ids.update(skills)

    if not skill_ids:
        skill_ids.add("model_training")
        skill_ids.add("data_preprocessing")

    return [
        os.path.join(KAGGLE_SKILLS_PATH, SKILL_TEMPLATES[s]["name"])
        for s in skill_ids
        if s in SKILL_TEMPLATES
    ]


def _infer_capabilities(concepts: list[dict]) -> list[str]:
    """Generate capability descriptions from concepts."""
    caps = ["Build baseline models quickly", "Systematic approach to ML problems"]
    concept_names = [c["concept"] for c in concepts[:8]]
    for name in concept_names:
        caps.append(f"Apply {name} techniques")
    return caps


def _build_formula(title: str) -> dict:
    """Build a formula definition for the expert."""
    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, theta, alpha)",
        "metrics": ["accuracy", "f1_score", "rmse"],
        "description": f"Optimize model performance using {title} principles",
    }


def create_expert_from_chapter(
    chapter_number: int,
    title: str,
    chapter_id: int,
    concepts: list[dict],
    strategy: str | None = None,
    loop_config: dict | None = None,
) -> dict:
    """Create an expert definition from chapter data."""
    slug = f"{chapter_number:02d}_{_slugify(title)}"
    expert_name = f"{chapter_number:02d}_{title}"

    return {
        "expert_name": expert_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": _infer_capabilities(concepts),
        "skills": _infer_skills_for_chapter(title, concepts),
        "strategy": strategy or DEFAULT_STRATEGY,
        "formula": _build_formula(title),
        "loop_config": loop_config or DEFAULT_LOOP_CONFIG.copy(),
    }


def register_experts_from_db(db: "Database") -> list[dict]:
    """Create and register experts from all indexed chapters.

    Returns list of expert definitions that were registered.
    """
    chapters = db.list_chapters()
    registered = []

    for ch in chapters:
        concepts = db.get_concepts(ch["id"])
        expert = create_expert_from_chapter(
            chapter_number=ch["chapter_number"],
            title=ch["title"],
            chapter_id=ch["id"],
            concepts=concepts,
        )
        db.upsert_expert(expert)
        registered.append(expert)
        logger.info("Registered expert: %s", expert["expert_name"])

    return registered


def save_expert_to_file(expert: dict, output_dir: str | None = None) -> str:
    """Save expert definition to a JSON file."""
    out_dir = Path(output_dir or os.path.join(os.path.dirname(__file__), "..", "..", "experts"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{expert['slug']}.json"
    with open(path, "w") as f:
        json.dump(expert, f, indent=2)
    return str(path)


def load_expert_from_file(path: str) -> dict:
    """Load expert definition from a JSON file."""
    with open(path) as f:
        return json.load(f)


def get_expert_for_competition(
    db: "Database",
    embedding_store: "EmbeddingStore",
    competition_description: str,
) -> list[dict]:
    """Use RAG to find the best experts for a competition.

    Returns ranked list of experts with relevance scores.
    """
    recommendations = embedding_store.infer_skills(competition_description)
    experts = db.list_experts()

    expert_by_chapter = {e.get("chapter_id"): e for e in experts}
    ranked = []
    for rec in recommendations:
        ch_num = rec["chapter_number"]
        chapter_id = db.get_chapter_id(ch_num)
        if chapter_id and chapter_id in expert_by_chapter:
            expert = expert_by_chapter[chapter_id]
            expert["relevance_score"] = rec["relevance_score"]
            ranked.append(expert)

    return ranked
