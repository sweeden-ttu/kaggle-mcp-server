"""Expert registry for managing chapter-based ML experts."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database


def _default_kaggle_skills_path() -> Path:
    return Path(os.environ.get(
        "KAGGLE_SKILLS_PATH",
        os.path.expanduser("~/skills")
    ))


def _slugify(name: str) -> str:
    """Convert a name to a slug."""
    slug = re.sub(r"[^\w\s]", "", name.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug


SKILL_MAPPING = {
    "data preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model training": "kaggle-model-trainer",
    "model selection": "kaggle-model-selector",
    "hyperparameter tuning": "kaggle-hypertuner",
    "ensemble methods": "kaggle-ensemble-builder",
    "deep learning": "kaggle-deep-learner",
    "nlp": "kaggle-nlp-pipeline",
    "computer vision": "kaggle-cv-pipeline",
    "time series": "kaggle-timeseries",
    "tabular data": "kaggle-tabular",
    "optimization": "kaggle-optimizer",
}

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Ensemble → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


class ExpertRegistry:
    """Manages chapter-based ML experts with skills, strategies, and formulas."""

    def __init__(self, db: Optional[Database] = None, skills_path: Optional[Path] = None):
        self.db = db or Database()
        self.skills_path = skills_path or _default_kaggle_skills_path()

    def create_expert_from_chapter(
        self,
        chapter_number: int,
        title: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from extracted chapter data."""
        slug = f"{chapter_number:02d}_{_slugify(title)}"
        expert_name = f"{chapter_number:02d}_{title.replace(' ', '_')}"

        capabilities = self._infer_capabilities(concepts)
        skills = self._map_skills(concepts)
        formula = self._build_formula(concepts)

        expert_data = {
            "expert_name": expert_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.insert_expert(expert_data)
        return expert_data

    def _infer_capabilities(self, concepts: List[str]) -> List[str]:
        """Infer expert capabilities from concepts."""
        capability_map = {
            "neural network": "Build and train neural network architectures",
            "deep learning": "Design deep learning pipelines",
            "gradient descent": "Optimize models with gradient-based methods",
            "regularization": "Apply regularization techniques to prevent overfitting",
            "ensemble": "Build ensemble models for improved accuracy",
            "feature engineering": "Create informative features from raw data",
            "hyperparameter": "Systematic hyperparameter optimization",
            "cross-validation": "Robust model evaluation with cross-validation",
            "transfer learning": "Leverage pre-trained models effectively",
            "data augmentation": "Augment training data for better generalization",
            "distributed training": "Scale training across multiple devices",
            "mixture of experts": "Design and route through expert sub-models",
        }

        capabilities = ["Build baseline models quickly", "Systematic evaluation pipeline"]
        for concept in concepts:
            for key, capability in capability_map.items():
                if key in concept.lower() and capability not in capabilities:
                    capabilities.append(capability)

        return capabilities[:8]

    def _map_skills(self, concepts: List[str]) -> List[str]:
        """Map concepts to Kaggle skill paths."""
        skills = set()
        for concept in concepts:
            concept_lower = concept.lower()
            for key, skill_name in SKILL_MAPPING.items():
                if key in concept_lower:
                    skill_path = str(self.skills_path / skill_name)
                    skills.add(skill_path)

        if not skills:
            skills.add(str(self.skills_path / "kaggle-preprocessor"))
            skills.add(str(self.skills_path / "kaggle-model-trainer"))

        return sorted(skills)

    def _build_formula(self, concepts: List[str]) -> Dict[str, Any]:
        """Build an objective formula based on concepts."""
        metrics = ["accuracy"]
        concepts_str = " ".join(concepts).lower()

        if "classification" in concepts_str or "f1" in concepts_str:
            metrics = ["accuracy", "f1_score", "auc_roc"]
        elif "regression" in concepts_str or "mse" in concepts_str:
            metrics = ["rmse", "mae", "r2_score"]
        elif "ranking" in concepts_str:
            metrics = ["ndcg", "map"]

        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": metrics,
        }

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get a specific expert by slug."""
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        """List all registered experts."""
        return self.db.list_experts()

    def get_experts_for_competition(
        self,
        relevant_chapters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get full expert definitions for relevant chapters."""
        experts = []
        for ch_info in relevant_chapters:
            ch_num = ch_info["chapter_number"]
            all_experts = self.db.list_experts()
            for exp in all_experts:
                if exp["slug"].startswith(f"{ch_num:02d}_"):
                    full_expert = self.db.get_expert(exp["slug"])
                    if full_expert:
                        full_expert["relevance_score"] = ch_info.get("relevance_score", 0)
                        experts.append(full_expert)
        return experts

    def export_expert_json(self, slug: str) -> Optional[Dict[str, Any]]:
        """Export expert as a JSON-serializable definition."""
        expert = self.get_expert(slug)
        if not expert:
            return None
        return {
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert["capabilities"],
            "skills": expert["skills"],
            "strategy": expert["strategy"],
            "formula": expert["formula"],
            "loop_config": expert["loop_config"],
        }
