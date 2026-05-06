"""Expert registry for managing chapter-based ML experts.

Each ML Principles chapter becomes an expert with capabilities,
skills, strategy, and mathematical formulas.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from mlsyseng_mcp.database import Database, ExpertRecord, ChapterRecord

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

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "convolutional neural": ["kaggle-image-classifier", "kaggle-deep-learning"],
    "recurrent neural": ["kaggle-sequence-model", "kaggle-deep-learning"],
    "transformer": ["kaggle-nlp-processor", "kaggle-deep-learning"],
    "attention mechanism": ["kaggle-nlp-processor", "kaggle-deep-learning"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "regularization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "cross validation": ["kaggle-model-trainer", "kaggle-validator"],
    "ensemble methods": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "random forest": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "decision tree": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "optimization": ["kaggle-hyperparameter-tuner", "kaggle-model-trainer"],
    "clustering": ["kaggle-preprocessor", "kaggle-unsupervised"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "principal component": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "support vector": ["kaggle-model-trainer", "kaggle-kernel-svm"],
    "bayesian": ["kaggle-model-trainer", "kaggle-probabilistic"],
    "reinforcement learning": ["kaggle-rl-agent", "kaggle-model-trainer"],
    "transfer learning": ["kaggle-deep-learning", "kaggle-transfer-learner"],
    "fine tuning": ["kaggle-deep-learning", "kaggle-transfer-learner"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-augmentor"],
    "embedding": ["kaggle-nlp-processor", "kaggle-feature-engineer"],
    "normalization": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "supervised learning": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "unsupervised learning": ["kaggle-preprocessor", "kaggle-unsupervised"],
}

CONCEPT_TO_CAPABILITIES = {
    "neural network": ["Build neural network architectures", "Train deep models"],
    "deep learning": ["Design deep learning pipelines", "GPU-accelerated training"],
    "feature engineering": [
        "Transform raw features",
        "Create derived features",
    ],
    "regularization": ["Prevent overfitting", "Apply L1/L2 penalties"],
    "cross validation": ["Robust model evaluation", "K-fold stratified splits"],
    "ensemble methods": ["Combine multiple models", "Stacking and blending"],
    "optimization": [
        "Systematic hyperparameter search",
        "Learning rate scheduling",
    ],
    "gradient descent": ["Optimize loss functions", "Implement custom optimizers"],
    "clustering": ["Unsupervised pattern discovery", "Customer segmentation"],
    "transformer": ["Sequence modeling", "Self-attention architectures"],
    "transfer learning": ["Leverage pre-trained models", "Domain adaptation"],
}


def _infer_skills(concepts: list[str], skills_path: str) -> list[str]:
    """Infer relevant Kaggle skills from chapter concepts."""
    skill_set = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for key, skills in CONCEPT_TO_SKILLS.items():
            if key in concept_lower:
                for skill in skills:
                    skill_path = os.path.join(skills_path, skill)
                    skill_set.add(skill_path)
    if not skill_set:
        default_skill = os.path.join(skills_path, "kaggle-model-trainer")
        skill_set.add(default_skill)
    return sorted(skill_set)


def _infer_capabilities(concepts: list[str]) -> list[str]:
    """Infer expert capabilities from chapter concepts."""
    caps = set()
    caps.add("Build baseline models quickly")
    for concept in concepts:
        concept_lower = concept.lower()
        for key, capabilities in CONCEPT_TO_CAPABILITIES.items():
            if key in concept_lower:
                caps.update(capabilities)
    return sorted(caps)


def _make_formula(concepts: list[str]) -> dict:
    """Generate an objective formula based on chapter concepts."""
    metrics = ["accuracy"]
    if any("precision" in c or "recall" in c or "f1" in c for c in concepts):
        metrics = ["accuracy", "f1_score", "precision", "recall"]
    elif any("regression" in c.lower() for c in concepts):
        metrics = ["rmse", "mae", "r2_score"]
    elif any("clustering" in c.lower() for c in concepts):
        metrics = ["silhouette_score", "davies_bouldin"]
    elif any("neural" in c.lower() or "deep" in c.lower() for c in concepts):
        metrics = ["accuracy", "f1_score", "loss"]

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


class ExpertRegistry:
    """Manages the expert registry backed by SQLite."""

    def __init__(
        self,
        db: Database,
        kaggle_skills_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
    ):
        self.db = db
        self.kaggle_skills_path = kaggle_skills_path

    def create_expert_from_chapter(
        self,
        chapter: ChapterRecord,
        strategy: Optional[str] = None,
        loop_config: Optional[dict] = None,
    ) -> ExpertRecord:
        """Create or update an expert from a chapter record."""
        capabilities = _infer_capabilities(chapter.concepts)
        skills = _infer_skills(chapter.concepts, self.kaggle_skills_path)
        formula = _make_formula(chapter.concepts)

        expert = ExpertRecord(
            expert_name=chapter.chapter_name,
            slug=chapter.chapter_id,
            chapter_id=chapter.chapter_id,
            capabilities=capabilities,
            skills=skills,
            strategy=strategy or DEFAULT_STRATEGY,
            formula=formula,
            loop_config=loop_config or DEFAULT_LOOP_CONFIG.copy(),
            created_at=time.time(),
        )

        self.db.upsert_expert(expert)
        logger.info("Expert created: %s (%s)", expert.expert_name, expert.slug)
        return expert

    def register_all_chapters(self) -> list[ExpertRecord]:
        """Create experts for all indexed chapters."""
        chapters = self.db.list_chapters()
        experts = []
        for chapter in chapters:
            expert = self.create_expert_from_chapter(chapter)
            experts.append(expert)
        logger.info("Registered %d experts from %d chapters", len(experts), len(chapters))
        return experts

    def get_expert(self, name_or_slug: str) -> Optional[ExpertRecord]:
        """Look up an expert by name or slug."""
        expert = self.db.get_expert(name_or_slug)
        if expert:
            return expert
        return self.db.get_expert_by_slug(name_or_slug)

    def list_experts(self) -> list[ExpertRecord]:
        return self.db.list_experts()

    def get_experts_for_competition(
        self,
        competition_description: str,
        embeddings_engine=None,
    ) -> list[ExpertRecord]:
        """Select the best experts for a competition using RAG."""
        if embeddings_engine:
            relevant_chapter_ids = embeddings_engine.get_relevant_experts(
                competition_description, n_results=5
            )
            experts = []
            for ch_id in relevant_chapter_ids:
                expert = self.db.get_expert_by_slug(ch_id)
                if expert:
                    experts.append(expert)
            if experts:
                return experts

        return self.db.list_experts()

    def build_competition_entry(
        self,
        competition: str,
        experts: list[ExpertRecord],
    ) -> dict:
        """Build a competition entry specification from selected experts."""
        all_skills = set()
        all_capabilities = set()
        all_metrics = set()
        strategies = []

        for expert in experts:
            all_skills.update(expert.skills)
            all_capabilities.update(expert.capabilities)
            if expert.formula.get("metrics"):
                all_metrics.update(expert.formula["metrics"])
            strategies.append(f"{expert.expert_name}: {expert.strategy}")

        return {
            "competition": competition,
            "experts_used": [e.expert_name for e in experts],
            "combined_capabilities": sorted(all_capabilities),
            "skills_to_activate": sorted(all_skills),
            "metrics": sorted(all_metrics) or ["accuracy"],
            "strategies": strategies,
            "loop_config": experts[0].loop_config if experts else DEFAULT_LOOP_CONFIG,
            "formula": {
                "objective": "minimize_validation_loss",
                "function": "L = Σᵢ wᵢ · Lᵢ(X, θᵢ, αᵢ)",
                "description": "Weighted ensemble of expert losses",
            },
        }

    def export_expert_json(self, expert: ExpertRecord) -> dict:
        """Export an expert as a JSON-serializable dict for file storage."""
        return {
            "expert_name": expert.expert_name,
            "slug": expert.slug,
            "capabilities": expert.capabilities,
            "skills": expert.skills,
            "strategy": expert.strategy,
            "formula": expert.formula,
            "loop_config": expert.loop_config,
        }

    def save_experts_to_dir(self, experts_dir: str) -> int:
        """Save all experts as JSON files to a directory."""
        path = Path(experts_dir)
        path.mkdir(parents=True, exist_ok=True)

        experts = self.list_experts()
        for expert in experts:
            filepath = path / f"{expert.slug}.json"
            with open(filepath, "w") as f:
                json.dump(self.export_expert_json(expert), f, indent=2)

        logger.info("Saved %d experts to %s", len(experts), experts_dir)
        return len(experts)
