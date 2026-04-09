"""Expert management for MLSysEng MoE.

Creates and manages chapter experts with skills, strategies, and formulas.
Each ML Principles chapter becomes an expert that can be queried.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"
DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer expert capabilities from concept names."""
    capability_map = {
        "neural network": "Build and train neural network architectures",
        "deep learning": "Apply deep learning techniques for complex patterns",
        "gradient descent": "Optimize models using gradient-based methods",
        "regularization": "Prevent overfitting with regularization strategies",
        "feature engineering": "Create and select informative features",
        "ensemble": "Combine multiple models for improved performance",
        "clustering": "Apply unsupervised clustering algorithms",
        "dimensionality reduction": "Reduce feature space while preserving information",
        "transformer": "Apply transformer architectures for sequence data",
        "bayesian": "Apply Bayesian inference and probabilistic modeling",
        "reinforcement learning": "Apply reinforcement learning strategies",
        "transfer learning": "Leverage pre-trained models for new tasks",
        "optimization": "Systematic hyperparameter optimization",
        "cross validation": "Robust model evaluation with cross-validation",
        "decision tree": "Build interpretable tree-based models",
        "random forest": "Apply random forest for robust predictions",
    }

    capabilities = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for key, cap in capability_map.items():
            if key in concept_lower:
                capabilities.add(cap)

    if not capabilities:
        capabilities.add("Build baseline models quickly")
        capabilities.add("Systematic hyperparameter search")

    return sorted(capabilities)


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer an objective formula from the concepts."""
    has_classification = any(
        c.lower() in ["logistic regression", "decision tree", "random forest", "precision", "recall"]
        for c in concepts
    )
    has_regression = any(
        c.lower() in ["linear regression"]
        for c in concepts
    )

    if has_classification:
        return {
            "objective": "maximize_f1_score",
            "function": "F1 = 2 * (P * R) / (P + R)",
            "metrics": ["accuracy", "f1_score", "precision", "recall"],
        }
    elif has_regression:
        return {
            "objective": "minimize_rmse",
            "function": "RMSE = sqrt(mean((y - y_hat)^2))",
            "metrics": ["rmse", "mae", "r2_score"],
        }
    else:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, theta, alpha)",
            "metrics": ["accuracy", "f1_score"],
        }


def _default_skills_path() -> str:
    return os.environ.get(
        "KAGGLE_SKILLS_PATH",
        os.path.expanduser("~/skills"),
    )


def _infer_skills(concepts: List[str], skills_path: Optional[str] = None) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    base = Path(skills_path or _default_skills_path())
    concept_to_skill = {
        "feature engineering": "kaggle-preprocessor",
        "model selection": "kaggle-model-trainer",
        "neural network": "kaggle-model-trainer",
        "deep learning": "kaggle-model-trainer",
        "clustering": "kaggle-preprocessor",
        "data augmentation": "kaggle-preprocessor",
        "hyperparameter": "kaggle-model-trainer",
        "ensemble": "kaggle-model-trainer",
        "pipeline": "kaggle-preprocessor",
    }

    skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for key, skill_name in concept_to_skill.items():
            if key in concept_lower:
                skills.add(str(base / skill_name))

    if not skills:
        skills.add(str(base / "kaggle-preprocessor"))
        skills.add(str(base / "kaggle-model-trainer"))

    return sorted(skills)


class ExpertRegistry:
    """Manages the registry of chapter experts."""

    def __init__(self, db: Optional[Database] = None, experts_dir: Optional[str] = None):
        self.db = db or Database()
        self.experts_dir = experts_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experts",
        )
        Path(self.experts_dir).mkdir(parents=True, exist_ok=True)

    def create_expert_from_chapter(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        """Create an expert definition from an extracted chapter."""
        chapters = self.db.list_chapters()
        chapter = None
        for ch in chapters:
            if ch["id"] == chapter_id:
                chapter = ch
                break

        if not chapter:
            logger.warning("Chapter %d not found", chapter_id)
            return None

        concepts = self.db.get_concepts_for_chapter(chapter_id)
        concept_names = [c["concept_name"] for c in concepts]

        expert = {
            "expert_name": chapter["chapter_name"],
            "slug": chapter["slug"],
            "chapter_id": chapter_id,
            "capabilities": _infer_capabilities(concept_names),
            "skills": _infer_skills(concept_names),
            "strategy": DEFAULT_STRATEGY,
            "formula": _infer_formula(concept_names),
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert)
        self._save_expert_json(expert)
        return expert

    def create_all_experts(self) -> List[Dict[str, Any]]:
        """Create experts for all extracted chapters."""
        chapters = self.db.list_chapters()
        created = []
        for ch in chapters:
            expert = self.create_expert_from_chapter(ch["id"])
            if expert:
                created.append(expert)
        return created

    def get_expert(self, name_or_slug: str) -> Optional[Dict[str, Any]]:
        expert = self.db.get_expert(name_or_slug)
        if not expert:
            expert = self.db.get_expert_by_slug(name_or_slug)
        return expert

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def query_expert(self, name_or_slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.get_expert(name_or_slug)
        if not expert:
            return {"error": f"Expert '{name_or_slug}' not found"}

        chapter = self.db.get_chapter_by_slug(expert["slug"])
        concepts = self.db.get_concepts_for_chapter(expert.get("chapter_id", 0))

        relevant_concepts = []
        question_lower = question.lower()
        for c in concepts:
            if (
                c["concept_name"].lower() in question_lower
                or any(word in c.get("description", "").lower() for word in question_lower.split())
            ):
                relevant_concepts.append(c)

        return {
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "relevant_concepts": [
                {"name": c["concept_name"], "description": c.get("description", "")}
                for c in (relevant_concepts or concepts[:5])
            ],
            "formula": expert.get("formula", {}),
            "chapter_excerpt": (chapter.get("markdown_content", "")[:2000]
                                if chapter else ""),
        }

    def recommend_experts_for_competition(
        self, competition_description: str
    ) -> List[Dict[str, Any]]:
        """Recommend experts for a given competition description."""
        experts = self.list_experts()
        scored = []

        desc_lower = competition_description.lower()
        desc_words = set(desc_lower.split())

        for expert in experts:
            score = 0.0
            capabilities = expert.get("capabilities", [])
            for cap in capabilities:
                cap_words = set(cap.lower().split())
                overlap = len(cap_words & desc_words)
                score += overlap * 0.1

            concepts = self.db.get_concepts_for_chapter(expert.get("chapter_id", 0))
            for concept in concepts:
                if concept["concept_name"].lower() in desc_lower:
                    score += 0.3

            if score > 0:
                scored.append({
                    "expert_name": expert["expert_name"],
                    "slug": expert["slug"],
                    "relevance_score": round(score, 4),
                    "capabilities": capabilities,
                    "skills": expert.get("skills", []),
                    "strategy": expert.get("strategy", ""),
                })

        scored.sort(key=lambda x: -x["relevance_score"])
        return scored

    def _save_expert_json(self, expert: Dict[str, Any]):
        """Save expert definition as JSON file."""
        path = Path(self.experts_dir) / f"{expert['slug']}.json"
        serializable = {k: v for k, v in expert.items() if k != "chapter_id"}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_experts_from_dir(self) -> List[Dict[str, Any]]:
        """Load expert definitions from JSON files in the experts directory."""
        loaded = []
        experts_path = Path(self.experts_dir)
        if not experts_path.exists():
            return loaded

        for json_file in sorted(experts_path.glob("*.json")):
            try:
                with open(json_file) as f:
                    expert = json.load(f)
                self.db.upsert_expert(expert)
                loaded.append(expert)
            except Exception as e:
                logger.error("Failed to load expert from %s: %s", json_file, e)

        return loaded
