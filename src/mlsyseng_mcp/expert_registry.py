"""Expert management and registration for the MoE system."""

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

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

DEFAULT_FORMULA = {
    "objective": "minimize_validation_loss",
    "function": "L = f(X, theta, alpha)",
    "metrics": ["accuracy", "f1_score"],
}

_SKILL_TEMPLATES = {
    "preprocessor": "kaggle-preprocessor",
    "model_trainer": "kaggle-model-trainer",
    "feature_engineer": "kaggle-feature-engineer",
    "ensembler": "kaggle-ensembler",
    "evaluator": "kaggle-evaluator",
    "submitter": "kaggle-submitter",
}


def slugify(name: str) -> str:
    """Convert a chapter name to a URL-safe slug."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


class ExpertRegistry:
    """Manages expert definitions derived from ML Principles chapters."""

    def __init__(self, db: Database, skills_path: Optional[str] = None):
        self.db = db
        self.skills_path = skills_path or KAGGLE_SKILLS_PATH

    def create_expert_from_chapter(
        self,
        chapter_name: str,
        concepts: List[Dict[str, str]],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter and its concepts."""
        slug = slugify(chapter_name)
        capabilities = self._derive_capabilities(concepts)
        skills = self._map_skills(concepts)
        formula = self._derive_formula(concepts)

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
        self._save_expert_file(expert)
        return expert

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_experts_for_competition(
        self,
        relevant_chapters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select experts based on relevance to a competition."""
        experts = []
        for ch in relevant_chapters:
            chapter_name = ch.get("chapter", "")
            slug = slugify(chapter_name)
            expert = self.db.get_expert(slug)
            if expert:
                expert["relevance_score"] = ch.get("relevance_score", 0.0)
                experts.append(expert)
        return sorted(experts, key=lambda e: e.get("relevance_score", 0), reverse=True)

    def _derive_capabilities(self, concepts: List[Dict[str, str]]) -> List[str]:
        """Derive capabilities from extracted concepts."""
        caps = ["Build baseline models quickly", "Systematic hyperparameter search"]

        category_caps = {
            "model": "Select and configure appropriate model architectures",
            "optimization": "Apply optimization techniques for convergence",
            "regularization": "Prevent overfitting with regularization strategies",
            "evaluation": "Evaluate models with appropriate metrics",
            "ensemble": "Combine models using ensemble methods",
            "unsupervised": "Apply unsupervised learning for feature discovery",
            "probabilistic": "Use probabilistic reasoning for uncertainty estimation",
            "information_theory": "Apply information-theoretic measures",
            "training": "Engineer features and optimize training pipeline",
        }

        seen_categories = set()
        for concept in concepts:
            cat = concept.get("category", "general")
            if cat in category_caps and cat not in seen_categories:
                seen_categories.add(cat)
                caps.append(category_caps[cat])

        return caps

    def _map_skills(self, concepts: List[Dict[str, str]]) -> List[str]:
        """Map concepts to Kaggle skill paths."""
        skills = set()
        skills.add(os.path.join(self.skills_path, _SKILL_TEMPLATES["preprocessor"]))
        skills.add(os.path.join(self.skills_path, _SKILL_TEMPLATES["submitter"]))

        for concept in concepts:
            cat = concept.get("category", "general")
            if cat in ("model", "optimization", "training"):
                skills.add(
                    os.path.join(self.skills_path, _SKILL_TEMPLATES["model_trainer"])
                )
            if cat in ("training", "unsupervised"):
                skills.add(
                    os.path.join(
                        self.skills_path, _SKILL_TEMPLATES["feature_engineer"]
                    )
                )
            if cat == "ensemble":
                skills.add(
                    os.path.join(self.skills_path, _SKILL_TEMPLATES["ensembler"])
                )
            if cat == "evaluation":
                skills.add(
                    os.path.join(self.skills_path, _SKILL_TEMPLATES["evaluator"])
                )

        return sorted(skills)

    def _derive_formula(self, concepts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Derive the objective formula from concepts."""
        formula = DEFAULT_FORMULA.copy()
        metrics = set(formula["metrics"])

        for concept in concepts:
            name_lower = concept["name"].lower()
            if "precision" in name_lower:
                metrics.add("precision")
            if "recall" in name_lower:
                metrics.add("recall")
            if "auc" in name_lower or "roc" in name_lower:
                metrics.add("auc_roc")
            if "entropy" in name_lower:
                metrics.add("cross_entropy")

        formula["metrics"] = sorted(metrics)
        return formula

    def _save_expert_file(self, expert: Dict[str, Any]):
        """Persist expert definition as JSON file."""
        experts_dir = Path(__file__).parent.parent.parent / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)
        filepath = experts_dir / f"{expert['slug']}.json"
        with open(filepath, "w") as f:
            json.dump(expert, f, indent=2, default=str)

    def ask_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert with a question."""
        expert = self.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        return {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "response": (
                f"As the {expert['expert_name']} expert, I recommend applying: "
                f"{', '.join(expert.get('capabilities', [])[:3])}. "
                f"Strategy: {expert.get('strategy', DEFAULT_STRATEGY)}"
            ),
            "question": question,
        }
