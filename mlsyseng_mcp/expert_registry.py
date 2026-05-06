"""Expert registry management for MLSysEng MoE system."""

import os
import re
import json
import logging
from typing import Optional

from mlsyseng_mcp.database import (
    store_expert, get_expert, get_all_experts, get_all_chapters
)

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.path.expanduser(
    os.environ.get("KAGGLE_SKILLS_PATH", "~/skills")
)

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-submission-builder",
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
    """Convert expert name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '_', slug)
    return slug


def generate_capabilities(concepts: list[str], title: str) -> list[str]:
    """Generate expert capabilities from chapter concepts."""
    capabilities = []

    concept_to_capability = {
        "gradient descent": "Optimize model parameters using gradient-based methods",
        "backpropagation": "Train deep neural networks with backpropagation",
        "loss function": "Design and select appropriate loss functions",
        "regularization": "Apply regularization to prevent overfitting",
        "cross-validation": "Evaluate model performance with cross-validation",
        "ensemble": "Build ensemble models for improved accuracy",
        "feature engineering": "Create informative features from raw data",
        "neural network": "Design and train neural network architectures",
        "deep learning": "Apply deep learning to complex problems",
        "transformer": "Implement transformer-based models",
        "optimization": "Systematic hyperparameter optimization",
        "clustering": "Unsupervised clustering for pattern discovery",
        "classification": "Build classification models",
        "regression": "Build regression models",
        "transfer learning": "Apply transfer learning from pretrained models",
        "data augmentation": "Augment training data for better generalization",
    }

    for concept in concepts:
        if concept in concept_to_capability:
            capabilities.append(concept_to_capability[concept])

    if not capabilities:
        capabilities = [
            f"Apply {title} principles to ML problems",
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]

    return capabilities[:8]


def generate_formula(concepts: list[str], title: str) -> dict:
    """Generate mathematical formula based on chapter concepts."""
    if "optimization" in concepts or "gradient descent" in concepts:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": ["accuracy", "f1_score", "log_loss"],
        }
    elif "ensemble" in concepts or "boosting" in concepts:
        return {
            "objective": "maximize_ensemble_diversity",
            "function": "F(x) = Σ αᵢ hᵢ(x)",
            "metrics": ["accuracy", "diversity_score"],
        }
    elif "neural network" in concepts or "deep learning" in concepts:
        return {
            "objective": "minimize_cross_entropy",
            "function": "L = -Σ yᵢ log(ŷᵢ)",
            "metrics": ["accuracy", "loss", "auc"],
        }
    elif "clustering" in concepts:
        return {
            "objective": "minimize_intra_cluster_distance",
            "function": "J = Σ ||xᵢ - μₖ||²",
            "metrics": ["silhouette_score", "inertia"],
        }
    else:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ)",
            "metrics": ["accuracy", "f1_score"],
        }


def get_skill_paths(concepts: list[str]) -> list[str]:
    """Map concepts to skill paths."""
    skills = []
    base = KAGGLE_SKILLS_PATH

    skill_mapping = {
        "feature engineering": "kaggle-feature-engineer",
        "classification": "kaggle-model-trainer",
        "regression": "kaggle-model-trainer",
        "neural network": "kaggle-deep-trainer",
        "deep learning": "kaggle-deep-trainer",
        "ensemble": "kaggle-ensemble-builder",
        "data augmentation": "kaggle-augmentor",
        "clustering": "kaggle-clusterer",
    }

    for concept in concepts:
        if concept in skill_mapping:
            skill_path = os.path.join(base, skill_mapping[concept])
            if skill_path not in skills:
                skills.append(skill_path)

    for default_skill in DEFAULT_SKILLS:
        path = os.path.join(base, default_skill)
        if path not in skills:
            skills.append(path)
            if len(skills) >= 6:
                break

    return skills


def create_expert_from_chapter(chapter: dict,
                               db_path: Optional[str] = None) -> dict:
    """Create an expert definition from a chapter record."""
    chapter_num = chapter["chapter_number"]
    title = chapter["title"]
    concepts = chapter.get("concepts", [])

    expert_name = f"{chapter_num}_{title}"
    slug = slugify(expert_name)

    capabilities = generate_capabilities(concepts, title)
    skills = get_skill_paths(concepts)
    formula = generate_formula(concepts, title)

    expert_id = store_expert(
        expert_name=expert_name,
        slug=slug,
        chapter_id=chapter.get("id"),
        capabilities=capabilities,
        skills=skills,
        strategy=DEFAULT_STRATEGY,
        formula=formula,
        loop_config=DEFAULT_LOOP_CONFIG,
        db_path=db_path,
    )

    return {
        "expert_name": expert_name,
        "slug": slug,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG,
        "expert_id": expert_id,
    }


def register_all_experts(db_path: Optional[str] = None) -> list[dict]:
    """Register experts for all indexed chapters."""
    chapters = get_all_chapters(db_path)
    results = []

    for chapter in chapters:
        try:
            expert = create_expert_from_chapter(chapter, db_path=db_path)
            results.append({"status": "registered", **expert})
        except Exception as e:
            results.append({
                "status": "failed",
                "chapter": chapter["chapter_number"],
                "error": str(e),
            })

    return results


def list_experts(db_path: Optional[str] = None) -> list[dict]:
    """List all registered experts."""
    return get_all_experts(db_path)


def query_expert(slug: str, question: str,
                 db_path: Optional[str] = None) -> dict:
    """Query a specific expert for advice."""
    expert = get_expert(slug, db_path)
    if not expert:
        return {"error": f"Expert '{slug}' not found"}

    return {
        "expert": expert["expert_name"],
        "slug": expert["slug"],
        "capabilities": expert["capabilities"],
        "strategy": expert["strategy"],
        "formula": expert["formula"],
        "recommendation": _generate_recommendation(expert, question),
    }


def _generate_recommendation(expert: dict, question: str) -> str:
    """Generate a recommendation based on expert capabilities."""
    caps = expert.get("capabilities", [])
    strategy = expert.get("strategy", DEFAULT_STRATEGY)

    relevant_caps = [
        cap for cap in caps
        if any(word in question.lower() for word in cap.lower().split()[:3])
    ]

    if relevant_caps:
        cap_text = "; ".join(relevant_caps[:3])
        return f"Based on {expert['expert_name']}: {cap_text}. Strategy: {strategy}"
    else:
        return f"Expert {expert['expert_name']} suggests following: {strategy}"
