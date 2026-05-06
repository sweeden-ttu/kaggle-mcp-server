"""Expert registry - manages chapter experts with skills, strategies, and formulas."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlsyseng_moe.mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

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

SKILL_MAPPING = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model training": "kaggle-model-trainer",
    "deep learning": "kaggle-deep-learning",
    "ensemble": "kaggle-ensemble",
    "hyperparameter": "kaggle-hyperparameter-tuner",
    "neural network": "kaggle-deep-learning",
    "classification": "kaggle-classifier",
    "regression": "kaggle-regressor",
    "time series": "kaggle-time-series",
    "nlp": "kaggle-nlp",
    "computer vision": "kaggle-cv",
    "optimization": "kaggle-optimizer",
}


def infer_skills_from_concepts(
    concepts: List[str], skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH
) -> List[str]:
    """Infer relevant Kaggle skills based on chapter concepts."""
    matched_skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skill_name in SKILL_MAPPING.items():
            if keyword in concept_lower:
                skill_path = os.path.join(skills_path, skill_name)
                matched_skills.add(skill_path)

    base_skills = [
        os.path.join(skills_path, "kaggle-preprocessor"),
        os.path.join(skills_path, "kaggle-model-trainer"),
    ]
    for s in base_skills:
        matched_skills.add(s)

    return sorted(matched_skills)


def infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer expert capabilities from concepts."""
    capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]

    concept_capability_map = {
        "deep learning": "Design and train neural network architectures",
        "neural network": "Design and train neural network architectures",
        "feature engineering": "Engineer domain-specific features",
        "ensemble": "Build ensemble models for improved performance",
        "regularization": "Apply regularization to prevent overfitting",
        "optimization": "Optimize model convergence and training",
        "classification": "Solve classification problems",
        "regression": "Solve regression problems",
        "time series": "Handle temporal data and forecasting",
        "clustering": "Perform unsupervised clustering",
        "dimensionality reduction": "Reduce feature dimensionality",
        "bayesian": "Apply Bayesian methods and probabilistic reasoning",
        "transformer": "Leverage transformer architectures",
        "attention": "Implement attention mechanisms",
    }

    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, capability in concept_capability_map.items():
            if keyword in concept_lower and capability not in capabilities:
                capabilities.append(capability)

    return capabilities


def infer_formula(concepts: List[str], chapter_name: str) -> Dict[str, Any]:
    """Infer the expert's mathematical formula/objective."""
    metrics = ["accuracy", "f1_score"]

    concept_lower = " ".join(c.lower() for c in concepts)
    if "regression" in concept_lower:
        metrics = ["rmse", "mae", "r2_score"]
    elif "classification" in concept_lower:
        metrics = ["accuracy", "f1_score", "auc_roc"]
    elif "time series" in concept_lower:
        metrics = ["mape", "rmse", "smape"]
    elif "clustering" in concept_lower:
        metrics = ["silhouette_score", "calinski_harabasz"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


class ExpertRegistry:
    """Manages the registry of chapter experts."""

    def __init__(self, db: Database, skills_path: Optional[str] = None):
        self.db = db
        self.skills_path = skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    def register_expert_from_chapter(
        self,
        chapter_name: str,
        chapter_slug: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create and register an expert from extracted chapter data."""
        skills = infer_skills_from_concepts(concepts, self.skills_path)
        capabilities = infer_capabilities(concepts)
        formula = infer_formula(concepts, chapter_name)

        expert_data = {
            "expert_name": chapter_name,
            "slug": chapter_slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_data)
        logger.info(f"Registered expert: {chapter_name}")
        return expert_data

    def register_experts_from_extraction(
        self, extraction_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Register experts from a batch of extraction results."""
        experts = []
        for result in extraction_results:
            chapter = self.db.get_chapter(result["slug"])
            chapter_id = chapter["id"] if chapter else None

            expert = self.register_expert_from_chapter(
                chapter_name=result["name"],
                chapter_slug=result["slug"],
                concepts=result.get("concepts", []),
                chapter_id=chapter_id,
            )
            experts.append(expert)
        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get an expert by slug."""
        return self.db.get_expert(slug)

    def get_expert_by_chapter(self, chapter_slug: str) -> Optional[Dict[str, Any]]:
        """Get an expert associated with a chapter slug."""
        return self.db.get_expert(chapter_slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        """List all registered experts."""
        return self.db.list_experts()

    def get_expert_full(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get full expert details including capabilities and formula."""
        return self.db.get_expert(slug)

    def select_experts_for_competition(
        self, competition_description: str, embedding_engine=None
    ) -> List[Dict[str, Any]]:
        """Select the best experts for a competition using RAG."""
        if embedding_engine:
            return embedding_engine.get_relevant_experts(
                competition_description, self
            )

        all_experts = self.db.list_experts()
        return all_experts[:3]

    def export_expert_definitions(self, output_dir: Optional[str] = None) -> str:
        """Export all expert definitions as JSON files."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "experts",
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        experts = self.db.list_experts()
        for expert_summary in experts:
            expert = self.db.get_expert(expert_summary["slug"])
            if expert:
                filepath = os.path.join(output_dir, f"{expert['slug']}.json")
                with open(filepath, "w") as f:
                    json.dump(expert, f, indent=2, default=str)

        return output_dir
