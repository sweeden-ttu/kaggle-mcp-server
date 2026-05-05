"""Expert management – creates, registers, and queries chapter experts."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

KAGGLE_SKILLS_PATH = os.path.expanduser(
    os.environ.get("KAGGLE_SKILLS_PATH", "~/skills")
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_FORMULA = {
    "objective": "minimize_validation_loss",
    "function": "L = f(X, θ, α)",
    "metrics": ["accuracy", "f1_score"],
}

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


_SKILL_MAP = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model": "kaggle-model-trainer",
    "deep learning": "kaggle-deep-learning",
    "neural network": "kaggle-deep-learning",
    "ensemble": "kaggle-ensembler",
    "hyperparameter": "kaggle-hyperparameter-tuner",
    "visualization": "kaggle-visualizer",
    "eda": "kaggle-eda",
    "nlp": "kaggle-nlp",
    "computer vision": "kaggle-cv",
    "time series": "kaggle-time-series",
    "optimization": "kaggle-optimizer",
}


def _infer_skills(concepts: List[str]) -> List[str]:
    """Map concept terms to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        lower = concept.lower()
        for keyword, skill_name in _SKILL_MAP.items():
            if keyword in lower:
                skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill_name))
    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))
    return sorted(skills)


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Generate capability descriptions from concept terms."""
    caps: List[str] = []
    concept_set = {c.lower() for c in concepts}

    if any("model" in c for c in concept_set):
        caps.append("Build baseline models quickly")
    if any("hyperparameter" in c for c in concept_set):
        caps.append("Systematic hyperparameter search")
    if any("feature" in c for c in concept_set):
        caps.append("Advanced feature engineering")
    if any("deep learning" in c or "neural" in c for c in concept_set):
        caps.append("Design and train deep learning architectures")
    if any("ensemble" in c for c in concept_set):
        caps.append("Build ensemble models for robust predictions")
    if any("optimization" in c for c in concept_set):
        caps.append("Optimize model performance with advanced techniques")
    if any("regularization" in c for c in concept_set):
        caps.append("Apply regularization to prevent overfitting")
    if not caps:
        caps.append("Apply ML principles systematically")
        caps.append("Build baseline models quickly")
    return caps


def _infer_metrics(concepts: List[str]) -> List[str]:
    """Select metrics relevant to concept terms."""
    metrics = {"accuracy"}
    concept_lower = " ".join(concepts).lower()
    if "precision" in concept_lower or "recall" in concept_lower:
        metrics.update(["precision", "recall", "f1_score"])
    if "regression" in concept_lower or "mse" in concept_lower:
        metrics.update(["mse", "rmse", "r2_score"])
    if "auc" in concept_lower or "roc" in concept_lower:
        metrics.add("auc_roc")
    return sorted(metrics)


class ExpertRegistry:
    """Creates and manages chapter experts."""

    def __init__(self, db: Database):
        self.db = db

    def register_from_chapter(
        self,
        chapter_slug: str,
        chapter_title: str,
        concept_terms: List[str],
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter's concepts and register it."""
        capabilities = _infer_capabilities(concept_terms)
        skills = _infer_skills(concept_terms)
        metrics = _infer_metrics(concept_terms)

        formula = {**DEFAULT_FORMULA, "metrics": metrics}
        loop_config = {**DEFAULT_LOOP_CONFIG}

        expert = {
            "slug": chapter_slug,
            "expert_name": chapter_title,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": loop_config,
            "chapter_id": chapter_id,
        }

        self.db.upsert_expert(expert)
        return expert

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def get_experts_for_concepts(self, concept_terms: List[str]) -> List[Dict[str, Any]]:
        """Find experts whose capabilities overlap with given concept terms."""
        all_experts = self.list_experts()
        if not concept_terms:
            return all_experts

        concept_lower = {t.lower() for t in concept_terms}
        scored: List[tuple] = []
        for expert in all_experts:
            caps_text = " ".join(expert.get("capabilities", [])).lower()
            skills_text = " ".join(expert.get("skills", [])).lower()
            overlap = sum(1 for c in concept_lower if c in caps_text or c in skills_text)
            if overlap > 0:
                scored.append((overlap, expert))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored] if scored else all_experts[:3]

    def export_expert_json(self, slug: str, output_dir: str = "experts") -> Optional[str]:
        """Export an expert definition to a JSON file."""
        expert = self.get_expert(slug)
        if not expert:
            return None
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filepath = out_path / f"{slug}.json"
        with open(filepath, "w") as f:
            json.dump(expert, f, indent=2, default=str)
        return str(filepath)
