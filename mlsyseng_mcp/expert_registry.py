"""Expert registry - manages chapter experts with skills, strategies, and formulas."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_STRATEGY = "Baseline \u2192 EDA \u2192 Feature Engineering \u2192 Model Selection \u2192 Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPING = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model selection": "kaggle-model-trainer",
    "neural network": "kaggle-deep-learning",
    "deep learning": "kaggle-deep-learning",
    "ensemble": "kaggle-ensemble-builder",
    "optimization": "kaggle-optimizer",
    "hyperparameter": "kaggle-hyperparameter-tuner",
    "visualization": "kaggle-visualizer",
    "data augmentation": "kaggle-augmenter",
    "cross validation": "kaggle-cv-validator",
    "reinforcement learning": "kaggle-rl-agent",
    "transformer": "kaggle-transformer-trainer",
    "embedding": "kaggle-embedding-builder",
    "regularization": "kaggle-regularizer",
}

FORMULA_TEMPLATES = {
    "classification": {
        "objective": "minimize_cross_entropy",
        "function": "L = -\u03a3[y\u00b7log(p) + (1-y)\u00b7log(1-p)]",
        "metrics": ["accuracy", "f1_score", "precision", "recall"],
    },
    "regression": {
        "objective": "minimize_mse",
        "function": "L = (1/n)\u00b7\u03a3(y - \u0177)\u00b2",
        "metrics": ["mse", "rmse", "mae", "r2"],
    },
    "optimization": {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, \u03b8, \u03b1)",
        "metrics": ["loss", "convergence_rate"],
    },
    "default": {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, \u03b8, \u03b1)",
        "metrics": ["accuracy", "f1_score"],
    },
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s]", "", name.lower())
    return re.sub(r"\s+", "_", slug).strip("_")


def _infer_formula_type(concepts: List[str]) -> str:
    text = " ".join(concepts).lower()
    if any(kw in text for kw in ("classification", "logistic", "cross entropy", "softmax")):
        return "classification"
    if any(kw in text for kw in ("regression", "mse", "linear")):
        return "regression"
    if any(kw in text for kw in ("optimization", "gradient", "convergence")):
        return "optimization"
    return "default"


def _infer_skills(concepts: List[str]) -> List[str]:
    skills = set()
    concept_text = " ".join(concepts).lower()
    for keyword, skill_name in SKILL_MAPPING.items():
        if keyword in concept_text:
            skill_path = os.path.join(KAGGLE_SKILLS_PATH, skill_name)
            skills.add(skill_path)

    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))

    return sorted(skills)


def _infer_capabilities(concepts: List[str], title: str) -> List[str]:
    caps = [f"Expert knowledge in {title}"]

    concept_lower = [c.lower() for c in concepts]
    keyword_caps = {
        "neural network": "Build and train neural network architectures",
        "optimization": "Systematic hyperparameter search and optimization",
        "feature": "Advanced feature engineering and selection",
        "ensemble": "Ensemble model construction and stacking",
        "regularization": "Regularization strategy selection",
        "preprocessing": "Data preprocessing and cleaning pipelines",
        "cross validation": "Robust cross-validation schemes",
        "deep learning": "Deep learning model design and training",
        "transformer": "Transformer architecture implementation",
        "reinforcement": "Reinforcement learning policy design",
    }
    for keyword, cap in keyword_caps.items():
        if any(keyword in c for c in concept_lower):
            caps.append(cap)

    caps.append("Build baseline models quickly")
    return caps


class ExpertRegistry:
    """Manages the lifecycle of chapter experts."""

    def __init__(self, db: Database):
        self.db = db

    def create_expert_from_chapter(
        self,
        chapter_num: int,
        title: str,
        concepts: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        formula: Optional[Dict[str, Any]] = None,
        loop_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or update an expert from a chapter definition."""
        concepts = concepts or []
        slug = _slugify(f"{chapter_num:02d}_{title}")

        formula_type = _infer_formula_type(concepts)
        expert_def = {
            "slug": slug,
            "expert_name": f"{chapter_num:02d}_{title}",
            "chapter_num": chapter_num,
            "capabilities": _infer_capabilities(concepts, title),
            "skills": _infer_skills(concepts),
            "strategy": strategy or DEFAULT_STRATEGY,
            "formula": formula or FORMULA_TEMPLATES.get(formula_type, FORMULA_TEMPLATES["default"]),
            "loop_config": loop_config or DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_def)
        return expert_def

    def create_experts_from_db(self) -> List[Dict[str, Any]]:
        """Create experts for all extracted chapters in the database."""
        chapters = self.db.list_chapters()
        experts = []
        for ch in chapters:
            concepts = json.loads(ch["concepts"]) if ch.get("concepts") else []
            expert = self.create_expert_from_chapter(
                chapter_num=ch["chapter_num"],
                title=ch["title"],
                concepts=concepts,
            )
            experts.append(expert)
        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert_for_query(
        self, query: str, experts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based expert matching (used when embeddings are unavailable)."""
        if experts is None:
            experts = self.list_experts()
        query_lower = query.lower()
        scored = []
        for e in experts:
            score = 0
            name_words = e["expert_name"].lower().split("_")
            for w in name_words:
                if w in query_lower:
                    score += 2

            caps = e.get("capabilities", [])
            if isinstance(caps, str):
                caps = json.loads(caps)
            for cap in caps:
                cap_words = cap.lower().split()
                for w in cap_words:
                    if len(w) > 3 and w in query_lower:
                        score += 1
            if score > 0:
                scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:3]]

    def export_expert_json(self, slug: str) -> Optional[str]:
        """Export an expert definition as JSON."""
        expert = self.get_expert(slug)
        if expert is None:
            return None
        export = {k: v for k, v in expert.items() if k not in ("id", "created_at", "updated_at")}
        return json.dumps(export, indent=2)
