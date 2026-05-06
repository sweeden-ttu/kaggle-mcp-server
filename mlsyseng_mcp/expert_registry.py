"""Expert registry for the MLSysEng MoE system.

Manages chapter experts — each ML Principles chapter becomes an expert
with defined capabilities, skills, strategy, and mathematical formulas.
"""

import logging
import os
import re
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

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "convolutional": ["kaggle-image-classifier", "kaggle-deep-learning"],
    "transformer": ["kaggle-nlp-transformer", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "feature selection": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "gradient boosting": ["kaggle-model-trainer", "kaggle-ensemble"],
    "xgboost": ["kaggle-model-trainer", "kaggle-xgboost"],
    "random forest": ["kaggle-model-trainer", "kaggle-ensemble"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "clustering": ["kaggle-clustering", "kaggle-preprocessor"],
    "cross-validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "hyperparameter": ["kaggle-hypertuner", "kaggle-model-trainer"],
    "data augmentation": ["kaggle-augmenter", "kaggle-preprocessor"],
    "transfer learning": ["kaggle-transfer-learning", "kaggle-model-trainer"],
    "reinforcement learning": ["kaggle-rl-agent", "kaggle-model-trainer"],
    "embedding": ["kaggle-nlp-transformer", "kaggle-embedder"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-pca"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "logistic regression": ["kaggle-model-trainer", "kaggle-baseline"],
    "linear regression": ["kaggle-model-trainer", "kaggle-baseline"],
    "distributed training": ["kaggle-distributed", "kaggle-model-trainer"],
    "inference": ["kaggle-deployer", "kaggle-model-trainer"],
}

CONCEPT_TO_CAPABILITIES = {
    "neural network": ["Build and train neural network architectures"],
    "deep learning": ["Design deep learning pipelines"],
    "feature engineering": ["Systematic feature engineering and selection"],
    "gradient boosting": ["Gradient boosting model optimization"],
    "ensemble": ["Build ensemble models for improved accuracy"],
    "clustering": ["Unsupervised learning and cluster analysis"],
    "cross-validation": ["Robust model evaluation with cross-validation"],
    "hyperparameter": ["Systematic hyperparameter search"],
    "transfer learning": ["Apply transfer learning from pretrained models"],
    "regularization": ["Apply regularization techniques to prevent overfitting"],
    "data augmentation": ["Generate augmented training data"],
    "transformer": ["Build transformer-based models"],
    "reinforcement learning": ["Design reward functions and RL agents"],
    "distributed training": ["Scale training across multiple devices"],
}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def _infer_skills(concepts: list[str]) -> list[str]:
    """Map concepts to Kaggle skill paths."""
    skills: set[str] = set()
    skills.add("kaggle-preprocessor")
    skills.add("kaggle-model-trainer")

    for concept in concepts:
        concept_lower = concept.lower()
        for key, skill_list in CONCEPT_TO_SKILLS.items():
            if key in concept_lower:
                skills.update(skill_list)

    return [
        os.path.join(KAGGLE_SKILLS_PATH, s) for s in sorted(skills)
    ]


def _infer_capabilities(concepts: list[str]) -> list[str]:
    """Map concepts to expert capabilities."""
    caps: list[str] = [
        "Build baseline models quickly",
        "Systematic hyperparameter search",
    ]
    seen = set(caps)

    for concept in concepts:
        concept_lower = concept.lower()
        for key, cap_list in CONCEPT_TO_CAPABILITIES.items():
            if key in concept_lower:
                for c in cap_list:
                    if c not in seen:
                        caps.append(c)
                        seen.add(c)
    return caps


def _infer_formula(concepts: list[str]) -> dict:
    """Generate a formula definition based on chapter concepts."""
    metrics = ["accuracy", "f1_score"]
    objective = "minimize_validation_loss"

    concept_set = {c.lower() for c in concepts}

    if any("clustering" in c for c in concept_set):
        metrics = ["silhouette_score", "inertia"]
        objective = "maximize_silhouette_score"
    elif any("reinforcement" in c for c in concept_set):
        metrics = ["cumulative_reward", "episode_length"]
        objective = "maximize_cumulative_reward"
    elif any("regression" in c for c in concept_set) and not any("logistic" in c for c in concept_set):
        metrics = ["rmse", "mae", "r2_score"]
        objective = "minimize_rmse"

    return {
        "objective": objective,
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


def register_experts_from_chapters(
    db_path: Optional[str] = None,
) -> list[dict]:
    """Create expert definitions from all extracted chapters.

    Each chapter becomes an expert with inferred capabilities, skills,
    strategy, and formula based on its concepts.
    """
    chapters = db.list_chapters(db_path)
    if not chapters:
        return []

    results = []
    for chapter in chapters:
        concepts = chapter.get("concepts", [])
        expert_name = f"{chapter['chapter_number']:02d}_{chapter['title']}"
        slug = _slugify(expert_name)

        capabilities = _infer_capabilities(concepts)
        skills = _infer_skills(concepts)
        formula = _infer_formula(concepts)

        expert_id = db.upsert_expert(
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

        results.append({
            "expert_id": expert_id,
            "expert_name": expert_name,
            "slug": slug,
            "capabilities_count": len(capabilities),
            "skills_count": len(skills),
        })

    return results


def get_expert_definition(slug: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get full expert definition by slug."""
    return db.get_expert(slug, db_path)


def query_expert(
    slug: str,
    question: str,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> dict:
    """Query a specific expert with a question.

    Combines the expert's knowledge (from its chapter) with semantic
    search results to answer the question.
    """
    from . import embeddings

    expert = db.get_expert(slug, db_path)
    if not expert:
        return {"error": f"Expert '{slug}' not found"}

    chapter = None
    if expert.get("chapter_id"):
        chapters = db.list_chapters(db_path)
        for ch in chapters:
            if ch["id"] == expert["chapter_id"]:
                chapter = ch
                break

    search_results = []
    if chapter:
        search_results = embeddings.search(
            question,
            n_results=3,
            chroma_path=chroma_path,
            chapter_filter=chapter["chapter_number"],
        )

    context_snippets = [r["text"][:500] for r in search_results]

    return {
        "expert": expert["expert_name"],
        "slug": slug,
        "question": question,
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "relevant_context": context_snippets,
        "skills": expert["skills"],
    }
