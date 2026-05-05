"""Expert registry for the MLSysEng MoE system.

Each ML Principles chapter becomes an expert with defined capabilities,
skills, strategy, formula, and convergence loop configuration.
"""

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MoEDatabase

KAGGLE_SKILLS_PATH = os.environ.get("KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills"))

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

CONCEPT_SKILL_MAP: Dict[str, List[str]] = {
    "preprocessing": ["kaggle-preprocessor"],
    "feature engineering": ["kaggle-feature-engineer"],
    "model selection": ["kaggle-model-trainer"],
    "deep learning": ["kaggle-model-trainer", "kaggle-nn-builder"],
    "neural network": ["kaggle-nn-builder"],
    "ensemble": ["kaggle-ensemble-builder"],
    "hyperparameter": ["kaggle-hypertuner"],
    "cross-validation": ["kaggle-cv-evaluator"],
    "eda": ["kaggle-eda-assistant"],
    "visualization": ["kaggle-eda-assistant"],
    "optimization": ["kaggle-hypertuner"],
    "regularization": ["kaggle-model-trainer"],
    "gradient descent": ["kaggle-model-trainer"],
    "boosting": ["kaggle-ensemble-builder"],
    "bagging": ["kaggle-ensemble-builder"],
    "transformer": ["kaggle-nn-builder"],
    "cnn": ["kaggle-nn-builder"],
    "rnn": ["kaggle-nn-builder"],
    "nlp": ["kaggle-nlp-pipeline"],
    "text": ["kaggle-nlp-pipeline"],
    "image": ["kaggle-cv-pipeline"],
    "time series": ["kaggle-timeseries-builder"],
    "baseline": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "submission": ["kaggle-submitter"],
}


def slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def infer_skills_from_concepts(concepts: List[str]) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        lower_concept = concept.lower()
        for keyword, skill_names in CONCEPT_SKILL_MAP.items():
            if keyword in lower_concept:
                for sn in skill_names:
                    skills.add(os.path.join(KAGGLE_SKILLS_PATH, sn))
    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))
    return sorted(skills)


def infer_capabilities_from_concepts(concepts: List[str]) -> List[str]:
    """Generate capability descriptions from concepts."""
    caps = []
    concept_set = {c.lower() for c in concepts}

    if any("baseline" in c for c in concept_set):
        caps.append("Build baseline models quickly")
    if any("hyperparameter" in c or "optimization" in c for c in concept_set):
        caps.append("Systematic hyperparameter search")
    if any("feature" in c for c in concept_set):
        caps.append("Advanced feature engineering")
    if any("deep learning" in c or "neural" in c for c in concept_set):
        caps.append("Deep learning model architecture design")
    if any("ensemble" in c or "boosting" in c or "bagging" in c for c in concept_set):
        caps.append("Ensemble and stacking strategies")
    if any("eda" in c or "visualization" in c for c in concept_set):
        caps.append("Exploratory data analysis and visualization")
    if any("cross-validation" in c for c in concept_set):
        caps.append("Robust cross-validation schemes")
    if any("nlp" in c or "text" in c for c in concept_set):
        caps.append("Natural language processing pipelines")
    if any("image" in c or "cnn" in c for c in concept_set):
        caps.append("Computer vision and image processing")
    if any("time series" in c for c in concept_set):
        caps.append("Time series forecasting methods")

    if not caps:
        caps.append("General ML model building")
        caps.append("Data preprocessing and cleaning")
    return caps


class ExpertRegistry:
    """Manages expert lifecycle: creation, lookup, and skill inference."""

    def __init__(self, db: Optional[MoEDatabase] = None):
        self.db = db or MoEDatabase()

    def register_expert_from_chapter(
        self,
        chapter_id: str,
        chapter_title: str,
        concepts: List[str],
        strategy: Optional[str] = None,
        formula: Optional[Dict[str, Any]] = None,
        loop_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or update an expert from a chapter's extracted knowledge."""
        slug = slugify(chapter_title)
        expert_id = f"expert_{slug}"
        skills = infer_skills_from_concepts(concepts)
        capabilities = infer_capabilities_from_concepts(concepts)

        expert = {
            "expert_id": expert_id,
            "chapter_id": chapter_id,
            "expert_name": chapter_title,
            "slug": slug,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": strategy or DEFAULT_STRATEGY,
            "formula": formula or DEFAULT_FORMULA,
            "loop_config": loop_config or DEFAULT_LOOP_CONFIG,
        }
        return self.db.upsert_expert(expert)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert_by_slug(slug)

    def get_experts_for_competition(
        self, competition_concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Return experts whose capabilities overlap with competition needs."""
        all_experts = self.list_experts()
        if not all_experts:
            return []

        scored = []
        comp_lower = {c.lower() for c in competition_concepts}
        for expert in all_experts:
            caps_lower = {c.lower() for c in expert.get("capabilities", [])}
            overlap = len(caps_lower & comp_lower)
            keyword_hits = 0
            for concept in comp_lower:
                for cap in caps_lower:
                    if concept in cap or cap in concept:
                        keyword_hits += 1
            score = overlap + keyword_hits * 0.5
            if score > 0:
                scored.append((score, expert))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]

    def save_expert_json(self, slug: str, output_dir: Optional[str] = None) -> str:
        """Export a single expert definition to a JSON file."""
        expert = self.get_expert(slug)
        if not expert:
            raise ValueError(f"Expert '{slug}' not found")

        out = Path(output_dir or "experts")
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(expert, f, indent=2)
        return str(path)

    def save_all_experts_json(self, output_dir: Optional[str] = None) -> List[str]:
        """Export all expert definitions."""
        paths = []
        for expert in self.list_experts():
            paths.append(self.save_expert_json(expert["slug"], output_dir))
        return paths
