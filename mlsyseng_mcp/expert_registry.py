"""Expert registry for MLSysEng MoE system.

Manages chapter experts with skills, strategies, formulas, and loop configurations.
Each ML Principles chapter becomes an expert with Kaggle-relevant capabilities.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPING = {
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "neural network": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "deep learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "cross validation": ["kaggle-validator", "kaggle-preprocessor"],
    "regularization": ["kaggle-model-trainer", "kaggle-optimizer"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "transformer": ["kaggle-deep-learning", "kaggle-nlp"],
    "attention mechanism": ["kaggle-deep-learning", "kaggle-nlp"],
    "clustering": ["kaggle-unsupervised", "kaggle-preprocessor"],
    "pca": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "bayesian": ["kaggle-model-trainer", "kaggle-optimizer"],
    "reinforcement learning": ["kaggle-rl", "kaggle-model-trainer"],
    "generative model": ["kaggle-deep-learning", "kaggle-generative"],
    "distributed training": ["kaggle-scaling", "kaggle-model-trainer"],
    "mixture of experts": ["kaggle-moe", "kaggle-model-trainer"],
}

FORMULA_TEMPLATES = {
    "classification": {
        "objective": "minimize_cross_entropy_loss",
        "function": "L = -Σ y_i * log(ŷ_i)",
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
    },
    "regression": {
        "objective": "minimize_mse",
        "function": "L = (1/n) * Σ (y_i - ŷ_i)²",
        "metrics": ["rmse", "mae", "r2_score"],
    },
    "optimization": {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    },
    "unsupervised": {
        "objective": "minimize_inertia",
        "function": "J = Σ ||x_i - μ_k||²",
        "metrics": ["silhouette_score", "inertia"],
    },
}


def _slugify(name: str) -> str:
    """Convert expert name to a slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def _default_skills_path() -> str:
    return os.environ.get("KAGGLE_SKILLS_PATH", str(Path.home() / "skills"))


def _infer_skills(concepts: List[str]) -> List[str]:
    """Infer relevant Kaggle skills from chapter concepts."""
    skills_base = _default_skills_path()
    skill_set = set()

    for concept in concepts:
        concept_lower = concept.lower()
        for pattern, skill_names in SKILL_MAPPING.items():
            if pattern in concept_lower:
                for s in skill_names:
                    skill_set.add(f"{skills_base}/{s}")

    if not skill_set:
        skill_set.add(f"{skills_base}/kaggle-preprocessor")
        skill_set.add(f"{skills_base}/kaggle-model-trainer")

    return sorted(skill_set)


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer appropriate formula template from concepts."""
    concept_text = " ".join(concepts).lower()

    if any(kw in concept_text for kw in ["classification", "precision", "recall", "f1"]):
        return FORMULA_TEMPLATES["classification"]
    elif any(kw in concept_text for kw in ["regression", "mse", "rmse"]):
        return FORMULA_TEMPLATES["regression"]
    elif any(kw in concept_text for kw in ["clustering", "unsupervised", "pca"]):
        return FORMULA_TEMPLATES["unsupervised"]

    return FORMULA_TEMPLATES["optimization"]


def _infer_capabilities(title: str, concepts: List[str]) -> List[str]:
    """Infer expert capabilities from title and concepts."""
    capabilities = [f"Expert knowledge in: {title}"]

    if any("model" in c.lower() for c in concepts):
        capabilities.append("Build baseline models quickly")
    if any("optim" in c.lower() for c in concepts):
        capabilities.append("Systematic hyperparameter search")
    if any("feature" in c.lower() for c in concepts):
        capabilities.append("Advanced feature engineering")
    if any("ensemble" in c.lower() for c in concepts):
        capabilities.append("Ensemble model construction")
    if any("neural" in c.lower() or "deep" in c.lower() for c in concepts):
        capabilities.append("Deep learning architecture design")
    if any("valid" in c.lower() or "cross" in c.lower() for c in concepts):
        capabilities.append("Robust cross-validation strategies")

    if len(capabilities) < 3:
        capabilities.extend([
            "Data preprocessing and cleaning",
            "Model evaluation and selection",
        ])

    return capabilities


class ExpertRegistry:
    """Manages chapter experts with their skills, strategies, and formulas."""

    def __init__(self, db: Database):
        self.db = db

    def create_expert_from_chapter(
        self,
        chapter_number: int,
        title: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create or update an expert from chapter data."""
        expert_name = f"{chapter_number:02d}_{title}"
        slug = _slugify(expert_name)

        expert_data = {
            "expert_name": expert_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": _infer_capabilities(title, concepts),
            "skills": _infer_skills(concepts),
            "strategy": DEFAULT_STRATEGY,
            "formula": _infer_formula(concepts),
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_data)
        logger.info(f"Registered expert: {expert_name} ({len(concepts)} concepts)")
        return expert_data

    def register_all_from_chapters(self) -> List[Dict[str, Any]]:
        """Register experts for all extracted chapters."""
        chapters = self.db.get_all_chapters()
        experts = []

        for chapter in chapters:
            expert = self.create_expert_from_chapter(
                chapter_number=chapter["chapter_number"],
                title=chapter["title"],
                concepts=chapter.get("concepts", []),
                chapter_id=chapter["id"],
            )
            experts.append(expert)

        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get a specific expert by slug."""
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        """List all registered experts."""
        return self.db.get_all_experts()

    def get_experts_for_competition(
        self, competition_description: str, embedding_engine=None
    ) -> List[Dict[str, Any]]:
        """Get relevant experts for a competition using RAG."""
        if embedding_engine:
            return embedding_engine.infer_relevant_experts(
                competition_description, self.db
            )

        experts = self.list_experts()
        return experts[:3] if experts else []

    def export_expert_json(self, slug: str) -> Optional[str]:
        """Export an expert definition as JSON."""
        expert = self.get_expert(slug)
        if expert:
            export = {
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert["capabilities"],
                "skills": expert["skills"],
                "strategy": expert["strategy"],
                "formula": expert["formula"],
                "loop_config": expert["loop_config"],
            }
            return json.dumps(export, indent=2)
        return None

    def save_expert_definitions(self, output_dir: Optional[str] = None):
        """Save all expert definitions as JSON files."""
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "experts")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        experts = self.list_experts()
        for expert in experts:
            export = {
                "expert_name": expert["expert_name"],
                "slug": expert["slug"],
                "capabilities": expert["capabilities"],
                "skills": expert["skills"],
                "strategy": expert["strategy"],
                "formula": expert["formula"],
                "loop_config": expert["loop_config"],
            }
            filepath = Path(output_dir) / f"{expert['slug']}.json"
            filepath.write_text(json.dumps(export, indent=2))

        logger.info(f"Saved {len(experts)} expert definitions to {output_dir}")
