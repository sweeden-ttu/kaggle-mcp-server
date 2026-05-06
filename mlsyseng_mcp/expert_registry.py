"""Expert registry for managing chapter-based ML experts."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.path.expanduser(
    os.getenv("KAGGLE_SKILLS_PATH", "~/skills")
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

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
    "convolutional": ["kaggle-model-trainer", "kaggle-image-classifier"],
    "transformer": ["kaggle-model-trainer", "kaggle-nlp-processor"],
    "attention": ["kaggle-model-trainer", "kaggle-nlp-processor"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "backpropagation": ["kaggle-model-trainer"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "dropout": ["kaggle-model-trainer"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "bagging": ["kaggle-ensemble-builder"],
    "boosting": ["kaggle-ensemble-builder"],
    "random forest": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "xgboost": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "lightgbm": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "catboost": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "feature selection": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "pca": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "cross validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "classification": ["kaggle-classifier", "kaggle-model-trainer"],
    "regression": ["kaggle-regressor", "kaggle-model-trainer"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-augmenter"],
    "embedding": ["kaggle-nlp-processor", "kaggle-model-trainer"],
    "supervised learning": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "unsupervised learning": ["kaggle-clusterer", "kaggle-preprocessor"],
    "reinforcement learning": ["kaggle-rl-agent"],
    "precision": ["kaggle-evaluator"],
    "recall": ["kaggle-evaluator"],
    "f1 score": ["kaggle-evaluator"],
    "accuracy": ["kaggle-evaluator"],
    "svm": ["kaggle-model-trainer", "kaggle-classifier"],
    "support vector machine": ["kaggle-model-trainer", "kaggle-classifier"],
    "decision tree": ["kaggle-model-trainer", "kaggle-classifier"],
}

CONCEPT_TO_FORMULA = {
    "neural network": {
        "objective": "minimize_validation_loss",
        "function": "L = (1/N) * Σ loss(f(x_i; θ), y_i) + λ * ||θ||²",
        "metrics": ["accuracy", "f1_score", "log_loss"],
    },
    "classification": {
        "objective": "minimize_cross_entropy",
        "function": "L = -Σ y_i * log(p_i)",
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
    },
    "regression": {
        "objective": "minimize_mse",
        "function": "L = (1/N) * Σ (y_i - ŷ_i)²",
        "metrics": ["rmse", "mae", "r2_score"],
    },
    "ensemble": {
        "objective": "minimize_ensemble_error",
        "function": "L = f(X, θ₁, ..., θₖ, α)",
        "metrics": ["accuracy", "f1_score", "auc"],
    },
    "optimization": {
        "objective": "minimize_objective",
        "function": "θ* = argmin_θ L(θ)",
        "metrics": ["convergence_rate", "final_loss"],
    },
}


def _slugify(name: str) -> str:
    """Convert chapter name to a slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer capabilities from extracted concepts."""
    capabilities = set()

    capability_map = {
        "neural network": "Build and train neural network models",
        "deep learning": "Apply deep learning techniques",
        "convolutional": "Build CNN architectures for image tasks",
        "transformer": "Implement transformer-based models",
        "ensemble": "Build ensemble models for improved accuracy",
        "feature engineering": "Systematic feature engineering",
        "cross validation": "Rigorous model evaluation with cross-validation",
        "optimization": "Hyperparameter optimization and tuning",
        "regularization": "Apply regularization to prevent overfitting",
        "classification": "Build classification models",
        "regression": "Build regression models",
        "data augmentation": "Augment training data for better generalization",
    }

    for concept in concepts:
        for key, cap in capability_map.items():
            if key in concept:
                capabilities.add(cap)

    capabilities.add("Build baseline models quickly")
    capabilities.add("Systematic hyperparameter search")

    return sorted(capabilities)


def _infer_skills(concepts: List[str], skills_base_path: str) -> List[str]:
    """Infer skill paths from concepts."""
    skill_names = set()
    skill_names.add("kaggle-preprocessor")
    skill_names.add("kaggle-model-trainer")

    for concept in concepts:
        for key, skills in CONCEPT_TO_SKILLS.items():
            if key in concept:
                skill_names.update(skills)

    return sorted(
        os.path.join(skills_base_path, name) for name in skill_names
    )


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer the primary formula/objective from concepts."""
    for concept in concepts:
        for key, formula in CONCEPT_TO_FORMULA.items():
            if key in concept:
                return formula

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    }


class ExpertRegistry:
    """Manages expert definitions derived from ML Principles chapters."""

    def __init__(
        self,
        db: MLSysEngDatabase,
        kaggle_skills_path: Optional[str] = None,
    ):
        self.db = db
        self.kaggle_skills_path = kaggle_skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    def create_expert_from_chapter(
        self,
        chapter_name: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from chapter data."""
        slug = _slugify(chapter_name)
        capabilities = _infer_capabilities(concepts)
        skills = _infer_skills(concepts, self.kaggle_skills_path)
        formula = _infer_formula(concepts)

        expert = {
            "expert_name": chapter_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert)
        return expert

    def create_experts_from_all_chapters(self) -> Dict[str, Any]:
        """Create experts for all extracted chapters."""
        chapters = self.db.list_chapters(status="extracted")
        results = []

        for chapter in chapters:
            expert = self.create_expert_from_chapter(
                chapter_name=chapter["chapter_name"],
                concepts=chapter["concepts"],
                chapter_id=chapter["id"],
            )
            results.append(expert)

        return {
            "status": "complete",
            "experts_created": len(results),
            "experts": results,
        }

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_experts_for_competition(
        self,
        relevant_chapters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get experts most relevant to a competition based on chapter relevance."""
        experts = []
        for ch_info in relevant_chapters:
            chapter_name = ch_info.get("chapter", "")
            slug = _slugify(chapter_name)
            expert = self.get_expert(slug)
            if expert:
                expert["relevance"] = ch_info.get("relevance", 0.0)
                experts.append(expert)
        return experts

    def save_expert_json(self, slug: str, output_dir: str = "experts") -> Optional[str]:
        """Save an expert definition to a JSON file."""
        expert = self.get_expert(slug)
        if not expert:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filepath = output_path / f"{slug}.json"
        serializable = {k: v for k, v in expert.items() if k != "id"}
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

        return str(filepath)

    def save_all_experts_json(self, output_dir: str = "experts") -> List[str]:
        """Save all expert definitions to JSON files."""
        experts = self.list_experts()
        paths = []
        for expert in experts:
            path = self.save_expert_json(expert["slug"], output_dir)
            if path:
                paths.append(path)
        return paths
