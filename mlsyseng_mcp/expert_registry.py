"""Expert registry for MLSysEng MoE system."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

KAGGLE_SKILLS_PATH = os.environ.get(
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

SKILL_MAPPING = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "classification": ["kaggle-model-trainer", "kaggle-evaluator"],
    "regression": ["kaggle-model-trainer", "kaggle-evaluator"],
    "ensemble": ["kaggle-model-trainer", "kaggle-ensemble"],
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "clustering": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "cross-validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-augmenter"],
    "transfer learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "reinforcement learning": ["kaggle-model-trainer"],
}

FORMULA_TEMPLATES = {
    "classification": {
        "objective": "minimize_cross_entropy_loss",
        "function": "L = -Σ y_i log(ŷ_i)",
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
    },
    "regression": {
        "objective": "minimize_mse",
        "function": "L = (1/n) Σ (y_i - ŷ_i)²",
        "metrics": ["mse", "rmse", "mae", "r2_score"],
    },
    "default": {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": ["accuracy", "f1_score"],
    },
}


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug


def _infer_skills(concepts: List[str]) -> List[str]:
    """Infer Kaggle skill paths from extracted concepts."""
    skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skill_list in SKILL_MAPPING.items():
            if keyword in concept_lower:
                for skill in skill_list:
                    skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill))
    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))
    return sorted(skills)


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer a formula template from concepts."""
    concepts_lower = " ".join(c.lower() for c in concepts)
    if "classification" in concepts_lower:
        return FORMULA_TEMPLATES["classification"].copy()
    if "regression" in concepts_lower:
        return FORMULA_TEMPLATES["regression"].copy()
    return FORMULA_TEMPLATES["default"].copy()


class ExpertRegistry:
    """Manages chapter-based ML experts with skills, strategy, and formulas."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def create_expert_from_chapter(
        self,
        chapter_name: str,
        chapter_number: int,
        concepts: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter's extracted concepts."""
        slug = _slugify(f"{chapter_number:02d}_{chapter_name}")
        display_name = f"{chapter_number:02d}_{chapter_name.replace(' ', '_')}"

        capabilities = []
        if concepts:
            capabilities.append(f"Expert in: {', '.join(concepts[:5])}")
            capabilities.append("Build baseline models quickly")
            capabilities.append("Systematic hyperparameter search")
            if any("deep" in c.lower() or "neural" in c.lower() for c in concepts):
                capabilities.append("Deep learning architecture design")
            if any("feature" in c.lower() for c in concepts):
                capabilities.append("Advanced feature engineering")
            if any("ensemble" in c.lower() or "boosting" in c.lower() for c in concepts):
                capabilities.append("Ensemble model construction")

        expert = {
            "expert_name": display_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": _infer_skills(concepts),
            "strategy": DEFAULT_STRATEGY,
            "formula": _infer_formula(concepts),
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert)
        return expert

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert_full(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get full expert definition including formula and loop config."""
        return self.db.get_expert(slug)

    def query_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.db.get_expert(slug)
        if expert is None:
            return {"error": f"Expert '{slug}' not found"}

        chapter = None
        if expert.get("chapter_id"):
            chapters = self.db.list_chapters()
            for ch in chapters:
                if ch["id"] == expert["chapter_id"]:
                    chapter = self.db.get_chapter(ch["chapter_name"])
                    break

        return {
            "expert": expert["expert_name"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "chapter_context": chapter.get("concepts", []) if chapter else [],
            "recommendation": (
                f"Based on {expert['expert_name']}'s expertise, "
                f"consider the following strategy: {expert.get('strategy', 'N/A')}"
            ),
        }

    def save_expert_json(self, slug: str, output_dir: Optional[str] = None) -> str:
        """Save expert definition as a JSON file."""
        expert = self.db.get_expert(slug)
        if expert is None:
            raise ValueError(f"Expert '{slug}' not found")

        out_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "experts"
        out_dir.mkdir(parents=True, exist_ok=True)

        filepath = out_dir / f"{slug}.json"
        serializable = {k: v for k, v in expert.items() if k != "id"}
        filepath.write_text(json.dumps(serializable, indent=2))
        return str(filepath)
