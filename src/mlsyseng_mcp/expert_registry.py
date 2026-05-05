"""Expert registry for the MLSysEng MoE system.

Each ML Principles chapter becomes an expert with capabilities, skills,
strategy, formula, and loop configuration.
"""

import json
import os
import time
from typing import Optional

from .database import Expert, MLSysEngDB

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

SKILL_MAPPING = {
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "regularization": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "cross-validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "clustering": ["kaggle-preprocessor", "kaggle-clustering"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-pca"],
    "transformer": ["kaggle-deep-learning", "kaggle-nlp"],
    "attention": ["kaggle-deep-learning", "kaggle-nlp"],
    "reinforcement learning": ["kaggle-rl", "kaggle-model-trainer"],
    "bayesian": ["kaggle-bayesian", "kaggle-model-trainer"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "decision tree": ["kaggle-model-trainer", "kaggle-tree-models"],
    "random forest": ["kaggle-model-trainer", "kaggle-tree-models"],
    "boosting": ["kaggle-model-trainer", "kaggle-boosting"],
    "support vector": ["kaggle-model-trainer", "kaggle-svm"],
}

CAPABILITY_TEMPLATES = {
    "gradient descent": "Optimize model parameters through gradient-based methods",
    "neural network": "Build and train neural network architectures",
    "deep learning": "Apply deep learning techniques for complex pattern recognition",
    "regularization": "Prevent overfitting with regularization techniques",
    "cross-validation": "Robust model evaluation with cross-validation",
    "feature engineering": "Create informative features from raw data",
    "ensemble": "Combine multiple models for improved predictions",
    "clustering": "Discover natural groupings in data",
    "dimensionality reduction": "Reduce feature space while preserving information",
    "transformer": "Apply transformer architectures for sequence modeling",
    "optimization": "Systematic hyperparameter search and optimization",
    "bayesian": "Probabilistic modeling and Bayesian inference",
    "boosting": "Gradient boosting for high-performance predictions",
}


def _infer_skills(concepts: list[str], skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH) -> list[str]:
    """Infer relevant Kaggle skills from a list of ML concepts."""
    skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for key, skill_list in SKILL_MAPPING.items():
            if key in concept_lower:
                for skill in skill_list:
                    skill_path = os.path.join(skills_path, skill)
                    skills.add(skill_path)
    if not skills:
        skills.add(os.path.join(skills_path, "kaggle-preprocessor"))
        skills.add(os.path.join(skills_path, "kaggle-model-trainer"))
    return sorted(skills)


def _infer_capabilities(concepts: list[str]) -> list[str]:
    """Infer capabilities from concepts."""
    capabilities = set()
    capabilities.add("Build baseline models quickly")
    capabilities.add("Systematic hyperparameter search")

    for concept in concepts:
        concept_lower = concept.lower()
        for key, cap in CAPABILITY_TEMPLATES.items():
            if key in concept_lower:
                capabilities.add(cap)
    return sorted(capabilities)


def _build_formula(concepts: list[str]) -> dict:
    """Build a mathematical formula definition based on chapter concepts."""
    metrics = ["accuracy"]
    if any("f1" in c.lower() for c in concepts):
        metrics.append("f1_score")
    if any("auc" in c.lower() or "roc" in c.lower() for c in concepts):
        metrics.append("auc_roc")
    if any("precision" in c.lower() for c in concepts):
        metrics.append("precision")
    if any("recall" in c.lower() for c in concepts):
        metrics.append("recall")

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


def _build_loop_config(
    epsilon: float = 0.001,
    max_iterations: int = 10,
    patience: int = 3,
) -> dict:
    """Build the convergence loop configuration."""
    return {
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": epsilon,
        "max_iterations": max_iterations,
        "patience": patience,
    }


def register_experts_from_chapters(
    db: MLSysEngDB,
    skills_path: Optional[str] = None,
) -> dict:
    """Create/update expert definitions for all extracted chapters."""
    sp = skills_path or DEFAULT_KAGGLE_SKILLS_PATH
    chapters = db.list_chapters()
    results = {"registered": 0, "skipped": 0}

    for chapter in chapters:
        if chapter.status != "extracted":
            results["skipped"] += 1
            continue

        concepts = json.loads(chapter.concepts) if chapter.concepts else []
        skills = _infer_skills(concepts, sp)
        capabilities = _infer_capabilities(concepts)
        formula = _build_formula(concepts)
        loop_config = _build_loop_config()

        expert = Expert(
            expert_name=chapter.title,
            slug=chapter.slug,
            chapter_id=chapter.id,
            capabilities=json.dumps(capabilities),
            skills=json.dumps(skills),
            strategy=DEFAULT_STRATEGY,
            formula=json.dumps(formula),
            loop_config=json.dumps(loop_config),
            created_at=time.time(),
        )

        db.upsert_expert(expert)
        results["registered"] += 1

    return results


def get_expert_for_query(
    db: MLSysEngDB,
    query: str,
    embedding_store=None,
) -> list[dict]:
    """Find the most relevant experts for a given query using RAG."""
    if embedding_store is not None:
        hits = embedding_store.search(query, n_results=3)
        chapter_slugs = set()
        for hit in hits:
            meta = hit.get("metadata", {})
            slug = meta.get("chapter_slug", "")
            if slug:
                chapter_slugs.add(slug)

        matched_experts = []
        for slug in chapter_slugs:
            expert = db.get_expert(slug)
            if expert:
                matched_experts.append({
                    "expert_name": expert.expert_name,
                    "slug": expert.slug,
                    "capabilities": json.loads(expert.capabilities),
                    "skills": json.loads(expert.skills),
                    "strategy": expert.strategy,
                    "formula": json.loads(expert.formula),
                    "loop_config": json.loads(expert.loop_config),
                })
        return matched_experts

    experts = db.list_experts()
    query_lower = query.lower()
    scored = []
    for expert in experts:
        concepts = json.loads(expert.capabilities)
        score = sum(1 for c in concepts if query_lower in c.lower())
        scored.append((score, expert))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "expert_name": e.expert_name,
            "slug": e.slug,
            "capabilities": json.loads(e.capabilities),
            "skills": json.loads(e.skills),
            "strategy": e.strategy,
            "formula": json.loads(e.formula),
            "loop_config": json.loads(e.loop_config),
        }
        for _, e in scored[:3]
    ]


def export_expert_json(db: MLSysEngDB, slug: str) -> Optional[dict]:
    """Export an expert definition as a JSON-serializable dict."""
    expert = db.get_expert(slug)
    if not expert:
        return None
    return {
        "expert_name": expert.expert_name,
        "slug": expert.slug,
        "capabilities": json.loads(expert.capabilities),
        "skills": json.loads(expert.skills),
        "strategy": expert.strategy,
        "formula": json.loads(expert.formula),
        "loop_config": json.loads(expert.loop_config),
    }
