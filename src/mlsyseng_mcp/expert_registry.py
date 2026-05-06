"""
Expert registry for managing chapter experts.

Each ML Principles chapter becomes an expert with:
- Capabilities (what the expert can do)
- Skills (Kaggle skills the expert recommends)
- Strategy (step-by-step approach)
- Formula (objective function and metrics)
- Loop config (convergence parameters)
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import ChapterRecord, ExpertRecord, MLSysEngDatabase

logger = logging.getLogger(__name__)

_SKILL_TEMPLATES = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model training": "kaggle-model-trainer",
    "evaluation": "kaggle-evaluator",
    "ensemble": "kaggle-ensemble-builder",
    "deep learning": "kaggle-deep-learner",
    "hyperparameter tuning": "kaggle-hyperparameter-tuner",
    "visualization": "kaggle-visualizer",
    "nlp": "kaggle-nlp-processor",
    "computer vision": "kaggle-cv-processor",
    "time series": "kaggle-timeseries-processor",
}

_CAPABILITY_MAP = {
    "neural network": "Build and train neural network architectures",
    "deep learning": "Apply deep learning techniques effectively",
    "gradient descent": "Optimize models using gradient-based methods",
    "regularization": "Apply regularization to prevent overfitting",
    "ensemble": "Build ensemble models for improved performance",
    "feature engineering": "Engineer informative features from raw data",
    "cross-validation": "Evaluate models using cross-validation strategies",
    "hyperparameter": "Systematic hyperparameter search and tuning",
    "transformer": "Implement transformer-based architectures",
    "clustering": "Apply unsupervised clustering algorithms",
    "bayesian": "Use Bayesian methods for inference and optimization",
    "reinforcement learning": "Apply RL techniques for sequential decision making",
    "dimensionality reduction": "Reduce feature dimensionality while preserving signal",
    "transfer learning": "Leverage pre-trained models for new tasks",
    "pipeline": "Build end-to-end ML pipelines",
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _expert_id_from_chapter(chapter_id: str) -> str:
    return f"expert_{chapter_id}"


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Map concepts to human-readable capabilities."""
    caps = []
    for concept in concepts:
        for keyword, capability in _CAPABILITY_MAP.items():
            if keyword in concept:
                caps.append(capability)
                break
    if not caps:
        caps = ["Build baseline models quickly", "Systematic hyperparameter search"]
    return list(dict.fromkeys(caps))


def _infer_skills(
    concepts: List[str], skills_base_path: Optional[str] = None
) -> List[str]:
    """Map concepts to skill paths."""
    base = skills_base_path or os.environ.get(
        "KAGGLE_SKILLS_PATH", str(Path.home() / "skills")
    )
    skills = set()

    skills.add(os.path.join(base, "kaggle-preprocessor"))
    skills.add(os.path.join(base, "kaggle-model-trainer"))

    for concept in concepts:
        for keyword, skill_name in _SKILL_TEMPLATES.items():
            if keyword in concept:
                skills.add(os.path.join(base, skill_name))

    return sorted(skills)


def _infer_strategy(concepts: List[str]) -> str:
    """Pick a strategy string based on dominant concepts."""
    if any("deep learning" in c or "neural network" in c for c in concepts):
        return "Baseline → EDA → Feature Engineering → Deep Model → Ensemble → Submit"
    if any("ensemble" in c or "boosting" in c for c in concepts):
        return "Baseline → EDA → Feature Engineering → Ensemble Methods → Submit"
    if any("nlp" in c or "tokenization" in c or "bert" in c for c in concepts):
        return "Baseline → Text Preprocessing → Embeddings → Fine-Tuning → Submit"
    return "Baseline → EDA → Feature Engineering → Model Selection → Submit"


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Generate an objective function definition based on concepts."""
    metrics = ["accuracy"]
    if any("regression" in c for c in concepts):
        metrics = ["rmse", "mae"]
    elif any("classification" in c for c in concepts):
        metrics = ["accuracy", "f1_score", "auc"]
    elif any("segmentation" in c for c in concepts):
        metrics = ["dice_score", "iou"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


class ExpertRegistry:
    """Manages the lifecycle of chapter experts."""

    def __init__(self, db: MLSysEngDatabase, skills_base_path: Optional[str] = None):
        self.db = db
        self.skills_base_path = skills_base_path

    def register_from_chapter(self, chapter: ChapterRecord) -> ExpertRecord:
        """Create or update an expert from a chapter record."""
        expert_id = _expert_id_from_chapter(chapter.chapter_id)
        slug = _slugify(chapter.chapter_name)

        expert = ExpertRecord(
            expert_id=expert_id,
            expert_name=chapter.chapter_name,
            slug=slug,
            chapter_id=chapter.chapter_id,
            capabilities=_infer_capabilities(chapter.concepts),
            skills=_infer_skills(chapter.concepts, self.skills_base_path),
            strategy=_infer_strategy(chapter.concepts),
            formula=_infer_formula(chapter.concepts),
        )
        self.db.upsert_expert(expert)
        return expert

    def register_all_chapters(self) -> List[ExpertRecord]:
        """Register experts for all chapters in the database."""
        chapters = self.db.list_chapters()
        experts = []
        for chapter in chapters:
            expert = self.register_from_chapter(chapter)
            experts.append(expert)
            logger.info("Registered expert: %s (%s)", expert.expert_name, expert.slug)
        return experts

    def list_experts(self) -> List[ExpertRecord]:
        return self.db.list_experts()

    def get_expert(self, slug: str) -> Optional[ExpertRecord]:
        return self.db.get_expert_by_slug(slug)

    def get_experts_for_competition(
        self,
        competition_description: str,
        embedding_engine=None,
    ) -> List[Dict[str, Any]]:
        """
        Select the best experts for a competition using RAG-informed skill selection.
        Falls back to returning all experts if no embedding engine is available.
        """
        if embedding_engine is None:
            experts = self.db.list_experts()
            return [
                {
                    "expert": {
                        "expert_id": e.expert_id,
                        "expert_name": e.expert_name,
                        "slug": e.slug,
                        "capabilities": e.capabilities,
                        "skills": e.skills,
                        "strategy": e.strategy,
                        "formula": e.formula,
                    },
                    "relevance_score": 1.0,
                }
                for e in experts
            ]

        relevant = embedding_engine.infer_relevant_experts(competition_description)

        matched_experts = []
        for item in relevant:
            ch_id = item["chapter_id"]
            expert_id = _expert_id_from_chapter(ch_id)
            expert = self.db.get_expert(expert_id)
            if expert:
                matched_experts.append(
                    {
                        "expert": {
                            "expert_id": expert.expert_id,
                            "expert_name": expert.expert_name,
                            "slug": expert.slug,
                            "capabilities": expert.capabilities,
                            "skills": expert.skills,
                            "strategy": expert.strategy,
                            "formula": expert.formula,
                        },
                        "relevance_score": item["relevance_score"],
                        "matched_concepts": item["concepts"],
                    }
                )

        return matched_experts

    def export_expert_json(self, slug: str) -> Optional[Dict[str, Any]]:
        """Export a single expert definition as a JSON-serializable dict."""
        expert = self.get_expert(slug)
        if not expert:
            return None
        from dataclasses import asdict

        return asdict(expert)

    def export_all_experts_json(self) -> List[Dict[str, Any]]:
        """Export all expert definitions."""
        from dataclasses import asdict

        return [asdict(e) for e in self.list_experts()]
