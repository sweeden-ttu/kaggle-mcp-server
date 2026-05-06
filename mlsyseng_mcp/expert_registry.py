"""Expert registry for MLSysEng MoE.

Manages chapter experts with skills, strategies, formulas, and loop configs.
Automatically creates expert definitions from extracted chapter content.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from . import database as db

DEFAULT_KAGGLE_SKILLS_PATH = os.path.expanduser("~/skills")

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

SKILL_MAPPING = {
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "architecture": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "regularization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "evaluation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "unsupervised": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "generative": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "nlp": ["kaggle-nlp-processor", "kaggle-deep-learning"],
    "classical": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "preprocessing": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "training": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "transfer": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "rl": ["kaggle-model-trainer"],
    "probabilistic": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
}


def _kaggle_skills_path() -> str:
    return os.environ.get("KAGGLE_SKILLS_PATH", DEFAULT_KAGGLE_SKILLS_PATH)


def _slugify(name: str) -> str:
    """Convert a chapter folder name to a slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    slug = re.sub(r"^(\d+)_", r"\1_", slug)
    return slug


def _infer_capabilities(concepts: List[Dict]) -> List[str]:
    """Infer expert capabilities from extracted concepts."""
    capabilities = []
    categories = set(c.get("category", "general") for c in concepts)

    capability_map = {
        "optimization": "Optimize model training with advanced gradient methods",
        "architecture": "Design and implement neural network architectures",
        "regularization": "Apply regularization techniques to prevent overfitting",
        "evaluation": "Evaluate model performance with appropriate metrics",
        "ensemble": "Build ensemble models for improved predictions",
        "unsupervised": "Apply unsupervised learning and dimensionality reduction",
        "generative": "Implement generative models (GANs, VAEs)",
        "nlp": "Process and model natural language data",
        "classical": "Apply classical ML algorithms effectively",
        "preprocessing": "Engineer and select features systematically",
        "training": "Manage training pipelines and hyperparameters",
        "transfer": "Leverage transfer learning and fine-tuning",
        "rl": "Apply reinforcement learning techniques",
        "probabilistic": "Use Bayesian methods and probabilistic models",
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


def _infer_skills(concepts: List[Dict]) -> List[str]:
    """Infer skill paths from concept categories."""
    skills_base = _kaggle_skills_path()
    seen = set()
    skills = []

    categories = set(c.get("category", "general") for c in concepts)
    for cat in categories:
        for skill_name in SKILL_MAPPING.get(cat, SKILL_MAPPING["general"]):
            if skill_name not in seen:
                seen.add(skill_name)
                skills.append(os.path.join(skills_base, skill_name))

    if not skills:
        skills = [
            os.path.join(skills_base, "kaggle-preprocessor"),
            os.path.join(skills_base, "kaggle-model-trainer"),
        ]

    return skills


def _infer_formula(concepts: List[Dict]) -> Dict:
    """Infer formula based on dominant concept categories."""
    categories = [c.get("category", "general") for c in concepts]
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    dominant = max(cat_counts, key=cat_counts.get) if cat_counts else "general"

    formula_map = {
        "optimization": {
            "objective": "minimize_loss",
            "function": "L = Σ ℓ(ŷ_i, y_i) + λR(θ)",
            "metrics": ["loss", "learning_rate", "convergence_rate"],
        },
        "evaluation": {
            "objective": "maximize_metric",
            "function": "M = f(precision, recall, threshold)",
            "metrics": ["accuracy", "f1_score", "auc_roc"],
        },
        "ensemble": {
            "objective": "minimize_ensemble_error",
            "function": "ŷ = Σ w_i · h_i(x)",
            "metrics": ["accuracy", "diversity", "ensemble_gain"],
        },
        "nlp": {
            "objective": "minimize_perplexity",
            "function": "L = -Σ log P(w_t | w_{<t})",
            "metrics": ["perplexity", "bleu", "rouge"],
        },
    }

    return formula_map.get(dominant, DEFAULT_FORMULA.copy())


def register_experts_from_chapters():
    """Create expert definitions from all extracted chapters.

    Returns list of registered experts.
    """
    db.init_db()
    chapters = db.get_all_chapters()
    registered = []

    for chapter in chapters:
        if not chapter.get("markdown_content"):
            continue

        slug = _slugify(chapter["folder_name"])
        concepts = db.get_concepts_for_chapter(chapter["id"])

        capabilities = _infer_capabilities(concepts)
        skills = _infer_skills(concepts)
        formula = _infer_formula(concepts)

        expert_id = db.upsert_expert(
            slug=slug,
            expert_name=chapter["folder_name"],
            chapter_id=chapter["id"],
            capabilities=capabilities,
            skills=skills,
            strategy=DEFAULT_STRATEGY,
            formula=formula,
            loop_config=DEFAULT_LOOP_CONFIG.copy(),
        )

        registered.append({
            "expert_id": expert_id,
            "slug": slug,
            "name": chapter["folder_name"],
            "capabilities": capabilities,
            "skills": skills,
            "concepts_count": len(concepts),
        })

    return registered


def get_expert_for_query(query: str) -> Optional[Dict[str, Any]]:
    """Find the best expert for a given query using keyword matching.

    For semantic matching, use embeddings.infer_skills instead.
    """
    experts = db.get_all_experts()
    if not experts:
        return None

    query_lower = query.lower()
    best_match = None
    best_score = 0

    for expert in experts:
        score = 0
        name_words = expert["expert_name"].lower().split()
        for word in name_words:
            if word in query_lower:
                score += 2

        for cap in (expert.get("capabilities") or []):
            cap_words = cap.lower().split()
            for word in cap_words:
                if len(word) > 3 and word in query_lower:
                    score += 1

        if score > best_score:
            best_score = score
            best_match = expert

    return best_match if best_score > 0 else (experts[0] if experts else None)


def export_expert_definitions() -> List[Dict[str, Any]]:
    """Export all expert definitions in the standard JSON format."""
    experts = db.get_all_experts()
    return [{
        "expert_name": e["expert_name"],
        "slug": e["slug"],
        "capabilities": e.get("capabilities", []),
        "skills": e.get("skills", []),
        "strategy": e.get("strategy", DEFAULT_STRATEGY),
        "formula": e.get("formula", DEFAULT_FORMULA),
        "loop_config": e.get("loop_config", DEFAULT_LOOP_CONFIG),
    } for e in experts]
