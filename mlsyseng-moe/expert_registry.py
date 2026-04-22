"""Expert management - registration, querying, and skill mapping."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import database as db


def _default_skills_path() -> str:
    return os.environ.get("KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills"))


def _slugify(name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    return slug


_SKILL_MAP: Dict[str, List[str]] = {
    "preprocessing": ["kaggle-preprocessor", "kaggle-data-cleaner"],
    "feature engineering": ["kaggle-feature-engineer"],
    "model training": ["kaggle-model-trainer"],
    "model selection": ["kaggle-model-trainer", "kaggle-ensemble"],
    "evaluation": ["kaggle-evaluator"],
    "visualization": ["kaggle-eda-viz"],
    "deep learning": ["kaggle-model-trainer", "kaggle-nn-trainer"],
    "ensemble": ["kaggle-ensemble"],
    "optimization": ["kaggle-model-trainer", "kaggle-optimizer"],
    "deployment": ["kaggle-submitter"],
    "eda": ["kaggle-eda-viz", "kaggle-preprocessor"],
    "neural network": ["kaggle-nn-trainer"],
    "transformer": ["kaggle-nn-trainer"],
    "regularization": ["kaggle-model-trainer"],
    "dimensionality reduction": ["kaggle-preprocessor"],
    "clustering": ["kaggle-model-trainer"],
    "nlp": ["kaggle-nlp-tools"],
    "computer vision": ["kaggle-cv-tools"],
    "time series": ["kaggle-ts-tools"],
}

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


def infer_skills(concepts: List[str], skills_path: Optional[str] = None) -> List[str]:
    """Infer which Kaggle skills are relevant based on chapter concepts."""
    base = skills_path or _default_skills_path()
    matched: set = set()

    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skill_names in _SKILL_MAP.items():
            if keyword in concept_lower:
                for s in skill_names:
                    matched.add(os.path.join(base, s))

    if not matched:
        matched.add(os.path.join(base, "kaggle-preprocessor"))
        matched.add(os.path.join(base, "kaggle-model-trainer"))

    return sorted(matched)


def infer_capabilities(concepts: List[str]) -> List[str]:
    """Generate capability descriptions from extracted concepts."""
    caps = []
    categories = set()

    for c in concepts:
        lower = c.lower()
        if any(k in lower for k in ("model", "train", "learn")):
            categories.add("model_training")
        if any(k in lower for k in ("feature", "preprocess", "clean")):
            categories.add("data_prep")
        if any(k in lower for k in ("evaluat", "metric", "score")):
            categories.add("evaluation")
        if any(k in lower for k in ("optim", "hyperparameter", "tune")):
            categories.add("optimization")
        if any(k in lower for k in ("visual", "plot", "eda")):
            categories.add("visualization")
        if any(k in lower for k in ("deep", "neural", "cnn", "rnn", "transformer")):
            categories.add("deep_learning")
        if any(k in lower for k in ("ensemble", "boost", "bag", "stack")):
            categories.add("ensemble")

    capability_text = {
        "model_training": "Build and train ML models systematically",
        "data_prep": "Preprocess and engineer features effectively",
        "evaluation": "Evaluate model performance with rigorous metrics",
        "optimization": "Systematic hyperparameter search and optimization",
        "visualization": "Exploratory data analysis and visualization",
        "deep_learning": "Apply deep learning architectures",
        "ensemble": "Combine models using ensemble methods",
    }

    for cat in sorted(categories):
        caps.append(capability_text.get(cat, f"Apply {cat} techniques"))

    if not caps:
        caps = ["Build baseline models quickly", "Systematic hyperparameter search"]

    return caps


def register_expert_from_chapter(
    conn,
    chapter_num: int,
    title: str,
    chapter_id: int,
    concepts: List[str],
    skills_path: Optional[str] = None,
    strategy: Optional[str] = None,
    formula: Optional[Dict[str, Any]] = None,
    loop_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or update an expert from chapter extraction results."""
    slug = _slugify(f"{chapter_num:02d}_{title}")
    expert_name = f"{chapter_num:02d}_{title}"
    capabilities = infer_capabilities(concepts)
    skills = infer_skills(concepts, skills_path)

    expert_id = db.upsert_expert(
        conn=conn,
        slug=slug,
        expert_name=expert_name,
        chapter_id=chapter_id,
        capabilities=capabilities,
        skills=skills,
        strategy=strategy or DEFAULT_STRATEGY,
        formula=formula or DEFAULT_FORMULA,
        loop_config=loop_config or DEFAULT_LOOP_CONFIG,
    )

    return {
        "id": expert_id,
        "slug": slug,
        "expert_name": expert_name,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": strategy or DEFAULT_STRATEGY,
        "formula": formula or DEFAULT_FORMULA,
        "loop_config": loop_config or DEFAULT_LOOP_CONFIG,
    }


def get_expert(conn, slug: str) -> Optional[Dict[str, Any]]:
    return db.get_expert_by_slug(conn, slug)


def list_all_experts(conn) -> List[Dict[str, Any]]:
    return db.get_all_experts(conn)


def save_expert_json(expert: Dict[str, Any], output_dir: str = "experts") -> str:
    """Save expert definition to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{expert['slug']}.json")
    with open(path, "w") as f:
        json.dump(expert, f, indent=2)
    return path


def build_competition_context(
    conn,
    competition: str,
    ranked_experts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a competition entry context from ranked experts."""
    all_skills = []
    all_capabilities = []
    strategies = []

    for item in ranked_experts[:5]:
        expert = item["expert"]
        all_skills.extend(expert.get("skills", []))
        all_capabilities.extend(expert.get("capabilities", []))
        strategies.append(expert.get("strategy", DEFAULT_STRATEGY))

    unique_skills = sorted(set(all_skills))
    unique_caps = sorted(set(all_capabilities))

    return {
        "competition": competition,
        "experts_used": [item["expert"]["slug"] for item in ranked_experts[:5]],
        "skills": unique_skills,
        "capabilities": unique_caps,
        "strategy": strategies[0] if strategies else DEFAULT_STRATEGY,
        "formula": ranked_experts[0]["expert"].get("formula", DEFAULT_FORMULA) if ranked_experts else DEFAULT_FORMULA,
        "loop_config": ranked_experts[0]["expert"].get("loop_config", DEFAULT_LOOP_CONFIG) if ranked_experts else DEFAULT_LOOP_CONFIG,
    }
