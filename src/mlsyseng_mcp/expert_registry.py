"""Expert management and registration for MLSysEng MoE."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
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
    "gradient": ["kaggle-model-trainer", "kaggle-optimizer"],
    "loss": ["kaggle-model-trainer", "kaggle-evaluator"],
    "optimization": ["kaggle-model-trainer", "kaggle-optimizer"],
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "decision tree": ["kaggle-model-trainer", "kaggle-tree-models"],
    "ensemble": ["kaggle-model-trainer", "kaggle-ensemble"],
    "feature": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "cross-validation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "classification": ["kaggle-model-trainer", "kaggle-classifier"],
    "regression": ["kaggle-model-trainer", "kaggle-regressor"],
    "clustering": ["kaggle-model-trainer", "kaggle-clustering"],
    "embedding": ["kaggle-model-trainer", "kaggle-embedding"],
    "transformer": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "convolution": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "attention": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "bayesian": ["kaggle-model-trainer", "kaggle-bayesian"],
    "dimensionality": ["kaggle-preprocessor", "kaggle-dim-reduction"],
    "transfer learning": ["kaggle-model-trainer", "kaggle-transfer"],
    "fine-tuning": ["kaggle-model-trainer", "kaggle-finetuner"],
}

CONCEPT_TO_CAPABILITIES = {
    "gradient": ["Gradient-based optimization", "Learning rate scheduling"],
    "loss": ["Loss function design", "Objective function selection"],
    "optimization": ["Hyperparameter tuning", "Systematic optimization"],
    "neural network": ["Deep learning architectures", "Neural network training"],
    "decision tree": ["Tree-based model construction", "Feature importance analysis"],
    "ensemble": ["Ensemble method design", "Model stacking and blending"],
    "feature": ["Feature engineering", "Feature selection and extraction"],
    "regularization": ["Regularization techniques", "Preventing overfitting"],
    "cross-validation": ["Cross-validation strategies", "Model evaluation"],
    "classification": ["Classification model selection", "Multi-class handling"],
    "regression": ["Regression modeling", "Continuous prediction"],
    "clustering": ["Unsupervised clustering", "Cluster analysis"],
    "embedding": ["Embedding representations", "Dimensionality mapping"],
    "transformer": ["Transformer architecture design", "Self-attention mechanisms"],
    "transfer learning": ["Pre-trained model adaptation", "Domain transfer"],
}


def _slugify(name: str) -> str:
    """Convert a chapter title to a URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


class ExpertRegistry:
    """Manages creation, lookup, and querying of chapter experts."""

    def __init__(self, db: Database, skills_path: Optional[str] = None):
        self.db = db
        self.skills_path = skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    def create_expert_from_chapter(
        self,
        chapter_id: int,
        folder_name: str,
        title: str,
        concepts: List[str],
    ) -> Dict[str, Any]:
        """Create or update an expert definition from a chapter's extracted data."""
        slug = _slugify(title)

        capabilities = []
        skills = set()
        for concept in concepts:
            caps = CONCEPT_TO_CAPABILITIES.get(concept, [])
            capabilities.extend(caps)

            skill_refs = CONCEPT_TO_SKILLS.get(concept, [])
            for s in skill_refs:
                skills.add(os.path.join(self.skills_path, s))

        if not capabilities:
            capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]
        capabilities = list(dict.fromkeys(capabilities))

        if not skills:
            skills = {
                os.path.join(self.skills_path, "kaggle-preprocessor"),
                os.path.join(self.skills_path, "kaggle-model-trainer"),
            }

        metrics = ["accuracy", "f1_score"]
        if any(c in concepts for c in ["regression"]):
            metrics = ["rmse", "r2_score", "mae"]
        elif any(c in concepts for c in ["clustering"]):
            metrics = ["silhouette_score", "calinski_harabasz"]

        expert_data: Dict[str, Any] = {
            "slug": slug,
            "expert_name": title,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": sorted(skills),
            "strategy": DEFAULT_STRATEGY,
            "formula": {
                "objective": "minimize_validation_loss",
                "function": "L = f(X, θ, α)",
                "metrics": metrics,
            },
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_data)
        return expert_data

    def create_experts_from_chapters(self) -> List[Dict[str, Any]]:
        """Create experts for all indexed chapters."""
        chapters = self.db.list_chapters()
        experts: List[Dict[str, Any]] = []

        for ch in chapters:
            if not ch.get("content_md"):
                continue
            expert = self.create_expert_from_chapter(
                chapter_id=ch["id"],
                folder_name=ch["folder_name"],
                title=ch["title"],
                concepts=ch.get("concepts", []),
            )
            experts.append(expert)

        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def query_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert with a question. Returns expert context + chapter content."""
        expert = self.db.get_expert(slug)
        if expert is None:
            return {"error": f"Expert '{slug}' not found"}

        chapter = None
        if expert.get("chapter_id"):
            chapter = self.db.get_chapter_by_id(expert["chapter_id"])

        return {
            "expert": expert,
            "chapter_title": chapter["title"] if chapter else None,
            "chapter_content_preview": (
                chapter["content_md"][:2000] if chapter and chapter.get("content_md") else None
            ),
            "question": question,
            "strategy": expert.get("strategy", ""),
            "capabilities": expert.get("capabilities", []),
            "recommended_skills": expert.get("skills", []),
        }

    def build_competition_entry(
        self,
        competition: str,
        relevant_experts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a competition entry using relevant experts' knowledge.

        Args:
            competition: Competition name/slug
            relevant_experts: List of expert info dicts from skill inference

        Returns:
            Entry plan with expert recommendations.
        """
        entry: Dict[str, Any] = {
            "competition": competition,
            "experts": [],
            "combined_strategy": [],
            "all_skills": set(),
            "metrics": set(),
        }

        for expert_info in relevant_experts:
            folder = expert_info.get("folder_name", "")
            slug = _slugify(expert_info.get("title", folder))
            expert = self.db.get_expert(slug)

            if expert:
                entry["experts"].append({
                    "name": expert["expert_name"],
                    "slug": expert["slug"],
                    "relevance": expert_info.get("relevance", 0),
                    "capabilities": expert.get("capabilities", []),
                })

                for skill in expert.get("skills", []):
                    entry["all_skills"].add(skill)

                formula = expert.get("formula", {})
                if isinstance(formula, dict):
                    for m in formula.get("metrics", []):
                        entry["metrics"].add(m)

        entry["all_skills"] = sorted(entry["all_skills"])
        entry["metrics"] = sorted(entry["metrics"])

        entry["combined_strategy"] = [
            "1. Baseline: Quick model with default parameters",
            "2. EDA: Explore data distributions and correlations",
            "3. Feature Engineering: Apply expert-recommended transformations",
            "4. Model Selection: Try models suggested by relevant experts",
            "5. Hyperparameter Tuning: Optimize with cross-validation",
            "6. Ensemble: Combine top-performing models",
            "7. Submit: Generate and submit predictions",
        ]

        return entry

    def export_expert_json(self, slug: str, output_dir: Optional[str] = None) -> Optional[str]:
        """Export an expert definition as a JSON file."""
        expert = self.db.get_expert(slug)
        if expert is None:
            return None

        out_dir = Path(output_dir or "experts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.json"

        export_data = {k: v for k, v in expert.items() if k != "id"}
        with open(out_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return str(out_path)
