"""Expert management and registration for MLSysEng MoE."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPING = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model training": "kaggle-model-trainer",
    "deep learning": "kaggle-deep-learning",
    "ensemble": "kaggle-ensemble",
    "optimization": "kaggle-optimizer",
    "neural network": "kaggle-deep-learning",
    "classification": "kaggle-classifier",
    "regression": "kaggle-regressor",
    "clustering": "kaggle-clustering",
    "dimensionality reduction": "kaggle-dim-reduction",
    "time series": "kaggle-time-series",
    "nlp": "kaggle-nlp",
    "computer vision": "kaggle-cv",
    "reinforcement learning": "kaggle-rl",
}


def slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    slug = slug.strip("_")
    return slug


def infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer expert capabilities from extracted concepts."""
    capability_map = {
        "gradient descent": "Optimize model parameters using gradient-based methods",
        "backpropagation": "Train deep neural networks with backpropagation",
        "loss function": "Design and select appropriate loss functions",
        "regularization": "Apply regularization to prevent overfitting",
        "convolutional": "Build convolutional neural network architectures",
        "transformer": "Implement attention-based transformer models",
        "ensemble": "Combine multiple models for improved predictions",
        "feature engineering": "Create informative features from raw data",
        "cross-validation": "Evaluate models with robust cross-validation",
        "hyperparameter": "Systematic hyperparameter search and tuning",
        "clustering": "Group data points using unsupervised methods",
        "dimensionality reduction": "Reduce feature space while preserving information",
        "reinforcement learning": "Train agents through reward-based learning",
        "bayesian": "Apply Bayesian inference for uncertainty quantification",
        "optimization": "Select and apply optimization algorithms",
        "neural network": "Design and train neural network architectures",
        "deep learning": "Build deep learning pipelines end-to-end",
        "classification": "Build classification models with proper evaluation",
        "regression": "Build regression models for continuous targets",
    }

    capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, capability in capability_map.items():
            if keyword in concept_lower and capability not in capabilities:
                capabilities.append(capability)

    return capabilities


def infer_skills(concepts: List[str]) -> List[str]:
    """Infer relevant Kaggle skills from concepts."""
    skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skill_name in SKILL_MAPPING.items():
            if keyword in concept_lower:
                skill_path = os.path.join(KAGGLE_SKILLS_PATH, skill_name)
                skills.add(skill_path)

    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))

    return sorted(skills)


def infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer mathematical formula/objective from concepts."""
    metrics = ["accuracy"]
    objective = "minimize_validation_loss"

    concept_text = " ".join(concepts).lower()

    if "classification" in concept_text:
        metrics = ["accuracy", "f1_score", "auc_roc"]
    elif "regression" in concept_text:
        metrics = ["rmse", "mae", "r2_score"]
        objective = "minimize_rmse"
    elif "clustering" in concept_text:
        metrics = ["silhouette_score", "calinski_harabasz"]
        objective = "maximize_silhouette"
    elif "ranking" in concept_text:
        metrics = ["ndcg", "map"]
        objective = "maximize_ndcg"

    if "deep learning" in concept_text or "neural network" in concept_text:
        metrics.append("training_loss")

    return {
        "objective": objective,
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


def create_expert_from_chapter(
    chapter_name: str,
    concepts: List[str],
    chapter_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Create an expert definition from a chapter's extracted data."""
    slug = slugify(chapter_name)
    capabilities = infer_capabilities(concepts)
    skills = infer_skills(concepts)
    formula = infer_formula(concepts)

    return {
        "expert_name": chapter_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": formula,
        "loop_config": DEFAULT_LOOP_CONFIG.copy(),
    }


def register_experts_from_db(db) -> List[Dict[str, Any]]:
    """Register experts for all extracted chapters in the database."""
    chapters = db.list_chapters(status="extracted")
    experts = []

    for chapter in chapters:
        concepts = chapter.get("concepts", [])
        expert_data = create_expert_from_chapter(
            chapter_name=chapter["chapter_name"],
            concepts=concepts,
            chapter_id=chapter["id"],
        )
        db.upsert_expert(expert_data)
        experts.append(expert_data)

    return experts


def save_expert_json(expert_data: Dict[str, Any], output_dir: Optional[str] = None):
    """Save expert definition to JSON file."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "experts")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{expert_data['slug']}.json"
    with open(filepath, "w") as f:
        json.dump(expert_data, f, indent=2)

    return str(filepath)


def get_experts_for_competition(
    competition_description: str, experts: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Select top-k relevant experts for a competition based on keyword overlap."""
    comp_words = set(competition_description.lower().split())

    scored_experts = []
    for expert in experts:
        all_text = " ".join(expert.get("capabilities", []))
        all_text += " " + " ".join(expert.get("skills", []))
        expert_words = set(all_text.lower().split())

        overlap = len(comp_words & expert_words)
        scored_experts.append((overlap, expert))

    scored_experts.sort(key=lambda x: x[0], reverse=True)
    return [exp for _, exp in scored_experts[:top_k]]
