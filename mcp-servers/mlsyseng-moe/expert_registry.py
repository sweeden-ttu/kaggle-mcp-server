"""Expert registry for MLSysEng MoE system.

Creates and manages chapter experts with skills, strategies, and formulas.
Each ML Principles chapter becomes an expert that can recommend Kaggle skills.
"""

import json
import logging
import os
import re
from typing import Optional

try:
    from . import database as db
except ImportError:
    import database as db

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-eda-analyst",
    "kaggle-submission-builder",
]

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Submit"

CHAPTER_SKILL_MAP = {
    "introduction": {
        "skills": ["kaggle-eda-analyst", "kaggle-preprocessor"],
        "strategy": "EDA → Baseline → Feature Engineering → Submit",
        "metrics": ["accuracy", "rmse"],
    },
    "linear": {
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer"],
        "strategy": "Baseline (Linear) → Feature Engineering → Regularization → Submit",
        "metrics": ["rmse", "r2_score", "mae"],
    },
    "tree": {
        "skills": ["kaggle-feature-engineer", "kaggle-model-trainer"],
        "strategy": "Feature Engineering → Tree Models → Ensemble → Prune → Submit",
        "metrics": ["accuracy", "f1_score", "auc"],
    },
    "neural": {
        "skills": ["kaggle-model-trainer", "kaggle-preprocessor"],
        "strategy": "Baseline NN → Architecture Search → Regularization → Train → Submit",
        "metrics": ["loss", "accuracy", "f1_score"],
    },
    "deep": {
        "skills": ["kaggle-model-trainer", "kaggle-feature-engineer"],
        "strategy": "Transfer Learning → Fine-tune → Augment → Train → Submit",
        "metrics": ["loss", "accuracy", "top_k_accuracy"],
    },
    "ensemble": {
        "skills": ["kaggle-model-trainer", "kaggle-submission-builder"],
        "strategy": "Train Diverse Models → Stack → Blend → Validate → Submit",
        "metrics": ["accuracy", "f1_score", "log_loss"],
    },
    "optimization": {
        "skills": ["kaggle-model-trainer", "kaggle-preprocessor"],
        "strategy": "Baseline → Optimizer Selection → LR Schedule → Converge → Submit",
        "metrics": ["loss", "convergence_rate"],
    },
    "regularization": {
        "skills": ["kaggle-model-trainer", "kaggle-feature-engineer"],
        "strategy": "Overfit → Regularize → Cross-Validate → Submit",
        "metrics": ["val_loss", "generalization_gap"],
    },
    "feature": {
        "skills": ["kaggle-feature-engineer", "kaggle-eda-analyst"],
        "strategy": "EDA → Feature Creation → Selection → Reduction → Submit",
        "metrics": ["feature_importance", "mutual_information"],
    },
    "evaluation": {
        "skills": ["kaggle-eda-analyst", "kaggle-submission-builder"],
        "strategy": "Define Metrics → Cross-Validate → Error Analysis → Submit",
        "metrics": ["precision", "recall", "f1_score", "auc"],
    },
    "nlp": {
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer"],
        "strategy": "Tokenize → Embed → Fine-tune LLM → Evaluate → Submit",
        "metrics": ["bleu", "perplexity", "accuracy"],
    },
    "reinforcement": {
        "skills": ["kaggle-model-trainer"],
        "strategy": "Define Reward → Train Agent → Evaluate Policy → Submit",
        "metrics": ["reward", "episode_return"],
    },
    "system": {
        "skills": DEFAULT_SKILLS,
        "strategy": DEFAULT_STRATEGY,
        "metrics": ["accuracy", "f1_score"],
    },
    "cluster": {
        "skills": ["kaggle-preprocessor", "kaggle-eda-analyst"],
        "strategy": "EDA → Cluster → Validate → Label → Submit",
        "metrics": ["silhouette_score", "inertia"],
    },
    "bayes": {
        "skills": ["kaggle-model-trainer", "kaggle-preprocessor"],
        "strategy": "Prior → Likelihood → Posterior → Predict → Submit",
        "metrics": ["log_loss", "brier_score"],
    },
    "generative": {
        "skills": ["kaggle-model-trainer"],
        "strategy": "Train Generator → Evaluate Samples → Refine → Submit",
        "metrics": ["fid_score", "is_score"],
    },
    "transformer": {
        "skills": ["kaggle-model-trainer", "kaggle-preprocessor"],
        "strategy": "Pretrain → Fine-tune → Evaluate → Submit",
        "metrics": ["perplexity", "accuracy", "f1_score"],
    },
    "svm": {
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer"],
        "strategy": "Scale Features → Kernel Selection → C Tuning → Submit",
        "metrics": ["accuracy", "f1_score", "auc"],
    },
    "probability": {
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer"],
        "strategy": "Statistical Analysis → Model Fitting → Inference → Submit",
        "metrics": ["log_loss", "calibration_error"],
    },
}


def _slugify(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _match_chapter_type(chapter_name: str) -> str:
    name_lower = chapter_name.lower()
    for keyword in CHAPTER_SKILL_MAP:
        if keyword in name_lower:
            return keyword
    return "system"


def create_expert_from_chapter(
    chapter_name: str,
    chapter_id: int,
    concepts: list[str],
    db_path: Optional[str] = None,
) -> dict:
    """Create an expert definition from a chapter."""
    slug = _slugify(chapter_name)
    chapter_type = _match_chapter_type(chapter_name)
    config = CHAPTER_SKILL_MAP.get(chapter_type, CHAPTER_SKILL_MAP["system"])

    skill_paths = [
        os.path.join(KAGGLE_SKILLS_PATH, skill_name)
        for skill_name in config["skills"]
    ]

    capabilities = []
    if concepts:
        capabilities.append(f"Expert knowledge in: {', '.join(concepts[:5])}")
    capabilities.append("Build baseline models quickly")
    capabilities.append("Systematic hyperparameter search")
    if "ensemble" in chapter_type:
        capabilities.append("Model stacking and blending")
    if "feature" in chapter_type:
        capabilities.append("Advanced feature engineering and selection")
    if "deep" in chapter_type or "neural" in chapter_type:
        capabilities.append("Neural architecture design and optimization")

    expert_def = {
        "expert_name": chapter_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": capabilities,
        "skills": skill_paths,
        "strategy": config["strategy"],
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": config["metrics"],
        },
        "loop_config": {
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    }

    db.upsert_expert(expert_def, db_path)
    return expert_def


def register_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Create experts from all extracted chapters."""
    chapters = db.get_chapters_by_status("completed", db_path)
    experts = []
    for ch in chapters:
        concepts = ch.get("concepts", [])
        if isinstance(concepts, str):
            import json as _json
            concepts = _json.loads(concepts)
        expert = create_expert_from_chapter(
            ch["chapter_name"], ch["id"], concepts, db_path
        )
        experts.append(expert)
        logger.info("Registered expert: %s", expert["expert_name"])
    return experts


def get_expert_for_query(
    query: str, db_path: Optional[str] = None
) -> Optional[dict]:
    """Find the best expert for a given query based on keyword matching."""
    experts = db.get_all_experts(db_path)
    if not experts:
        return None

    query_lower = query.lower()
    best_expert = None
    best_score = 0

    for expert in experts:
        score = 0
        name_lower = expert["expert_name"].lower()
        if any(word in query_lower for word in name_lower.split("_")):
            score += 2

        capabilities = expert.get("capabilities", [])
        for cap in capabilities:
            if isinstance(cap, str):
                cap_words = cap.lower().split()
                score += sum(1 for w in cap_words if w in query_lower)

        if score > best_score:
            best_score = score
            best_expert = expert

    return best_expert


def build_competition_entry(
    competition_slug: str,
    expert_slugs: Optional[list[str]] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Build a competition entry using expert knowledge.

    If no expert_slugs provided, uses RAG to infer the best experts.
    """
    if expert_slugs:
        experts = []
        for slug in expert_slugs:
            e = db.get_expert_by_slug(slug, db_path)
            if e:
                experts.append(e)
    else:
        all_experts = db.get_all_experts(db_path)
        experts = all_experts[:3] if len(all_experts) > 3 else all_experts

    if not experts:
        return {
            "competition": competition_slug,
            "status": "no_experts",
            "message": "No experts available. Run extract-knowledge first.",
        }

    entry = {
        "competition": competition_slug,
        "experts_used": [e["slug"] for e in experts],
        "combined_strategy": _combine_strategies(experts),
        "combined_skills": _combine_skills(experts),
        "metrics": _combine_metrics(experts),
        "loop_config": experts[0].get("loop_config", {}),
        "notebooks": [],
    }

    for expert in experts:
        notebook_path = os.path.expanduser(
            f"~/{competition_slug}/Expert_{expert['slug']}.ipynb"
        )
        entry["notebooks"].append({
            "expert": expert["slug"],
            "path": notebook_path,
            "strategy": expert["strategy"],
        })
        db.upsert_competition_entry(
            competition_slug, expert["slug"], notebook_path, db_path=db_path
        )

    return entry


def _combine_strategies(experts: list[dict]) -> str:
    stages = []
    for expert in experts:
        strategy = expert.get("strategy", "")
        for stage in strategy.split("→"):
            stage = stage.strip()
            if stage and stage not in stages:
                stages.append(stage)
    return " → ".join(stages)


def _combine_skills(experts: list[dict]) -> list[str]:
    skills = []
    seen = set()
    for expert in experts:
        for skill in expert.get("skills", []):
            if skill not in seen:
                seen.add(skill)
                skills.append(skill)
    return skills


def _combine_metrics(experts: list[dict]) -> list[str]:
    metrics = []
    seen = set()
    for expert in experts:
        formula = expert.get("formula", {})
        for m in formula.get("metrics", []):
            if m not in seen:
                seen.add(m)
                metrics.append(m)
    return metrics
