"""Expert management system for MLSysEng MoE."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
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

CONCEPT_TO_SKILL = {
    "neural network": "kaggle-model-trainer",
    "deep learning": "kaggle-model-trainer",
    "gradient descent": "kaggle-model-trainer",
    "backpropagation": "kaggle-model-trainer",
    "regularization": "kaggle-model-trainer",
    "feature engineering": "kaggle-feature-engineer",
    "PCA": "kaggle-feature-engineer",
    "dimensionality reduction": "kaggle-feature-engineer",
    "data augmentation": "kaggle-preprocessor",
    "clustering": "kaggle-preprocessor",
    "classification": "kaggle-model-trainer",
    "regression": "kaggle-model-trainer",
    "ensemble": "kaggle-ensemble-builder",
    "random forest": "kaggle-ensemble-builder",
    "boosting": "kaggle-ensemble-builder",
    "XGBoost": "kaggle-ensemble-builder",
    "LightGBM": "kaggle-ensemble-builder",
    "bagging": "kaggle-ensemble-builder",
    "cross-validation": "kaggle-evaluator",
    "overfitting": "kaggle-evaluator",
    "underfitting": "kaggle-evaluator",
    "bias-variance": "kaggle-evaluator",
    "hyperparameter": "kaggle-tuner",
    "learning rate": "kaggle-tuner",
    "natural language processing": "kaggle-nlp-pipeline",
    "tokenization": "kaggle-nlp-pipeline",
    "computer vision": "kaggle-cv-pipeline",
    "convolutional": "kaggle-cv-pipeline",
    "time series": "kaggle-ts-pipeline",
    "sequence model": "kaggle-ts-pipeline",
    "pipeline": "kaggle-preprocessor",
    "MLOps": "kaggle-preprocessor",
    "embedding": "kaggle-feature-engineer",
    "transfer learning": "kaggle-model-trainer",
    "fine-tuning": "kaggle-model-trainer",
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug


def _concepts_to_capabilities(concepts: List[str]) -> List[str]:
    """Map extracted concepts to expert capabilities."""
    capability_map = {
        "neural network": "Build and train neural network models",
        "deep learning": "Apply deep learning architectures",
        "gradient descent": "Optimize models using gradient-based methods",
        "feature engineering": "Engineer predictive features from raw data",
        "ensemble": "Build ensemble models for improved accuracy",
        "cross-validation": "Validate model performance with cross-validation",
        "hyperparameter": "Systematic hyperparameter search and tuning",
        "transfer learning": "Apply transfer learning from pretrained models",
        "natural language processing": "Process and model text data",
        "computer vision": "Process and model image data",
        "time series": "Forecast and model temporal data",
        "clustering": "Discover patterns with unsupervised learning",
        "regularization": "Apply regularization to prevent overfitting",
        "pipeline": "Build end-to-end ML pipelines",
    }

    capabilities = ["Build baseline models quickly"]
    for concept in concepts:
        if concept in capability_map:
            cap = capability_map[concept]
            if cap not in capabilities:
                capabilities.append(cap)
    if len(capabilities) == 1:
        capabilities.append("Systematic hyperparameter search")
    return capabilities[:8]


def _concepts_to_skills(concepts: List[str], skills_path: str) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    skill_names = set()
    for concept in concepts:
        if concept in CONCEPT_TO_SKILL:
            skill_names.add(CONCEPT_TO_SKILL[concept])
    if not skill_names:
        skill_names = {"kaggle-preprocessor", "kaggle-model-trainer"}
    return sorted(os.path.join(skills_path, s) for s in skill_names)


def _concepts_to_formula(concepts: List[str]) -> Dict[str, Any]:
    """Generate a formula definition based on chapter concepts."""
    metrics = ["accuracy"]
    if any(c in concepts for c in ["regression", "time series"]):
        metrics = ["rmse", "mae"]
    if any(c in concepts for c in ["classification", "ensemble"]):
        metrics = ["accuracy", "f1_score"]
    if "natural language processing" in concepts:
        metrics = ["bleu", "f1_score"]
    if "computer vision" in concepts:
        metrics = ["accuracy", "map"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


class ExpertRegistry:
    """Manages expert definitions derived from ML Principles chapters."""

    def __init__(
        self,
        db: Optional[Database] = None,
        skills_path: Optional[str] = None,
    ):
        self.db = db or Database()
        self.skills_path = skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    def create_expert_from_chapter(
        self,
        chapter_name: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter's extracted concepts."""
        slug = _slugify(chapter_name)
        capabilities = _concepts_to_capabilities(concepts)
        skills = _concepts_to_skills(concepts, self.skills_path)
        formula = _concepts_to_formula(concepts)

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

    def create_experts_from_db(self) -> List[Dict[str, Any]]:
        """Create experts for all extracted chapters in the database."""
        chapters = self.db.list_chapters(status="extracted")
        experts = []
        for ch in chapters:
            expert = self.create_expert_from_chapter(
                chapter_name=ch["chapter_name"],
                concepts=ch.get("concepts", []),
                chapter_id=ch.get("id"),
            )
            experts.append(expert)
        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert_for_competition(
        self,
        competition_description: str,
        recommendations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select experts for a competition based on RAG recommendations."""
        matched = []
        for rec in recommendations:
            chapter = rec["chapter"]
            slug = _slugify(chapter)
            expert = self.get_expert(slug)
            if expert:
                expert["relevance_score"] = rec.get("relevance_score", 0)
                expert["relevant_concepts"] = rec.get("relevant_concepts", [])
                matched.append(expert)
        return matched

    def export_expert(self, slug: str, output_dir: Optional[str] = None) -> Optional[str]:
        """Export an expert definition to JSON."""
        expert = self.get_expert(slug)
        if not expert:
            return None

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "experts")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        path = os.path.join(output_dir, f"{slug}.json")
        serializable = {k: v for k, v in expert.items() if k != "id"}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        return path

    def export_all_experts(self, output_dir: Optional[str] = None) -> List[str]:
        """Export all expert definitions to JSON files."""
        experts = self.list_experts()
        paths = []
        for exp in experts:
            path = self.export_expert(exp["slug"], output_dir)
            if path:
                paths.append(path)
        return paths
