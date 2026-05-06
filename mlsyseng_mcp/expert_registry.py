"""Expert registry for MLSysEng MoE system.

Manages chapter experts with skills, strategies, formulas, and loop configs.
Experts are created from extracted chapter content and can be queried
for Kaggle competition guidance.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-submitter",
    "kaggle-eda",
]

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


class ExpertRegistry:
    """Manages expert definitions and queries."""

    def __init__(self, db: Database, skills_path: Optional[str] = None):
        self.db = db
        self.skills_path = skills_path or KAGGLE_SKILLS_PATH

    def create_expert_from_chapter(
        self,
        chapter_number: str,
        title: str,
        concepts: Optional[List[str]] = None,
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter."""
        slug = self._make_slug(chapter_number, title)
        expert_name = f"{chapter_number}_{title}"
        capabilities = self._infer_capabilities(title, concepts or [])
        skills = self._map_skills(concepts or [])
        formula = self._define_formula(title, concepts or [])

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

        self.db.upsert_expert(expert_data)
        self._save_expert_file(expert_data)
        return expert_data

    def create_experts_from_all_chapters(self) -> List[Dict[str, Any]]:
        """Create experts from all extracted chapters."""
        chapters = self.db.get_all_chapters()
        experts = []

        for ch in chapters:
            expert = self.create_expert_from_chapter(
                chapter_number=ch["chapter_number"],
                title=ch["title"],
                concepts=ch.get("concepts"),
                chapter_id=ch.get("id"),
            )
            experts.append(expert)

        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.get_all_experts()

    def query_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert for guidance on a topic."""
        expert = self.db.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter = None
        if expert.get("chapter_id"):
            chapters = self.db.get_all_chapters()
            for ch in chapters:
                if ch["id"] == expert["chapter_id"]:
                    chapter = ch
                    break

        return {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert["capabilities"],
            "strategy": expert["strategy"],
            "formula": expert["formula"],
            "skills": expert["skills"],
            "chapter_content_available": chapter is not None and bool(chapter.get("content_markdown")),
            "relevant_concepts": chapter.get("concepts", []) if chapter else [],
            "guidance": self._generate_guidance(expert, question),
        }

    def select_experts_for_competition(
        self,
        competition: str,
        inferred_chapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select the best experts for a given competition."""
        all_experts = self.list_experts()
        if not all_experts:
            return []

        if inferred_chapters:
            chapter_nums = {c["chapter_number"] for c in inferred_chapters}
            matched = []
            for expert in all_experts:
                ch_num = expert.get("slug", "").split("_")[0]
                if ch_num in chapter_nums:
                    matched.append(expert)
            if matched:
                return matched

        return all_experts

    def _infer_capabilities(self, title: str, concepts: List[str]) -> List[str]:
        """Infer expert capabilities from title and concepts."""
        capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]

        concept_capability_map = {
            "neural network": "Design and train neural network architectures",
            "deep learning": "Apply deep learning techniques",
            "feature engineering": "Advanced feature engineering and selection",
            "ensemble": "Build ensemble models for improved performance",
            "clustering": "Unsupervised clustering and segmentation",
            "dimensionality reduction": "Dimensionality reduction and feature compression",
            "natural language processing": "NLP text processing and feature extraction",
            "reinforcement learning": "RL-based optimization strategies",
            "bayesian": "Bayesian inference and probabilistic modeling",
            "optimization": "Advanced optimization techniques",
            "regularization": "Regularization strategies to prevent overfitting",
            "transfer learning": "Transfer learning from pre-trained models",
            "data augmentation": "Data augmentation for improved generalization",
            "cross-validation": "Robust cross-validation evaluation",
        }

        for concept in concepts:
            concept_lower = concept.lower()
            for key, capability in concept_capability_map.items():
                if key in concept_lower and capability not in capabilities:
                    capabilities.append(capability)

        return capabilities[:8]

    def _map_skills(self, concepts: List[str]) -> List[str]:
        """Map concepts to Kaggle skill paths."""
        skills = []
        for s in DEFAULT_SKILLS:
            skill_path = os.path.join(self.skills_path, s)
            skills.append(skill_path)

        concept_skill_map = {
            "neural network": "kaggle-neural-net",
            "deep learning": "kaggle-deep-learning",
            "natural language processing": "kaggle-nlp",
            "computer vision": "kaggle-vision",
            "time series": "kaggle-time-series",
            "ensemble": "kaggle-ensemble",
            "feature engineering": "kaggle-feature-engineer",
        }

        for concept in concepts:
            concept_lower = concept.lower()
            for key, skill_name in concept_skill_map.items():
                if key in concept_lower:
                    skill_path = os.path.join(self.skills_path, skill_name)
                    if skill_path not in skills:
                        skills.append(skill_path)

        return skills

    def _define_formula(self, title: str, concepts: List[str]) -> Dict[str, Any]:
        """Define the expert's objective function."""
        metrics = ["accuracy", "f1_score"]

        concept_lower = " ".join(concepts).lower()
        if "regression" in concept_lower:
            metrics = ["rmse", "mae", "r2_score"]
        elif "classification" in concept_lower:
            metrics = ["accuracy", "f1_score", "auc_roc"]
        elif "clustering" in concept_lower:
            metrics = ["silhouette_score", "calinski_harabasz"]
        elif "natural language" in concept_lower:
            metrics = ["bleu", "rouge", "perplexity"]

        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": metrics,
        }

    def _generate_guidance(self, expert: Dict[str, Any], question: str) -> str:
        """Generate guidance text from an expert for a given question."""
        parts = [
            f"Expert: {expert['expert_name']}",
            f"Strategy: {expert['strategy']}",
            f"Capabilities: {', '.join(expert['capabilities'])}",
            f"Objective: {expert['formula']['objective']}",
            f"Metrics: {', '.join(expert['formula']['metrics'])}",
            f"\nFor your question about '{question}':",
            f"I recommend following the strategy: {expert['strategy']}",
            f"Focus on optimizing: {', '.join(expert['formula']['metrics'])}",
        ]
        return "\n".join(parts)

    def _make_slug(self, chapter_number: str, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return f"{chapter_number}_{slug}"

    def _save_expert_file(self, expert_data: Dict[str, Any]):
        """Save expert definition to JSON file."""
        experts_dir = Path(__file__).parent.parent / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

        filepath = experts_dir / f"{expert_data['slug']}.json"
        with open(filepath, "w") as f:
            json.dump(expert_data, f, indent=2)
