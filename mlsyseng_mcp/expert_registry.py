"""Expert registry - manages chapter experts with skills, strategies, and formulas."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_PATH = "~/.skills"

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"

DEFAULT_FORMULA = {
    "objective": "minimize_validation_loss",
    "function": "L = f(X, theta, alpha)",
    "metrics": ["accuracy", "f1_score"],
}

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CHAPTER_SKILL_MAP = {
    "supervised": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-evaluator"],
    "unsupervised": ["kaggle-preprocessor", "kaggle-clustering", "kaggle-dimensionality"],
    "deep_learning": ["kaggle-neural-net", "kaggle-model-trainer", "kaggle-gpu-trainer"],
    "neural_network": ["kaggle-neural-net", "kaggle-model-trainer", "kaggle-gpu-trainer"],
    "optimization": ["kaggle-hyperparameter-tuner", "kaggle-model-trainer"],
    "regularization": ["kaggle-regularizer", "kaggle-model-trainer"],
    "feature": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble", "kaggle-model-trainer"],
    "evaluation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "nlp": ["kaggle-text-preprocessor", "kaggle-nlp-model"],
    "computer_vision": ["kaggle-image-preprocessor", "kaggle-cnn-trainer"],
    "time_series": ["kaggle-time-series", "kaggle-forecaster"],
    "reinforcement": ["kaggle-rl-trainer", "kaggle-environment"],
    "bayesian": ["kaggle-bayesian-optimizer", "kaggle-model-trainer"],
    "ml_systems": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-pipeline"],
}

CHAPTER_CAPABILITY_MAP = {
    "supervised": [
        "Build classification and regression models",
        "Train/test splitting and evaluation",
        "Systematic model comparison",
    ],
    "unsupervised": [
        "Clustering and segmentation",
        "Dimensionality reduction",
        "Anomaly detection",
    ],
    "deep_learning": [
        "Design neural network architectures",
        "GPU-accelerated training",
        "Transfer learning and fine-tuning",
    ],
    "optimization": [
        "Hyperparameter optimization",
        "Learning rate scheduling",
        "Convergence analysis",
    ],
    "regularization": [
        "Prevent overfitting",
        "Apply regularization techniques",
        "Cross-validation strategies",
    ],
    "feature": [
        "Feature selection and engineering",
        "Data preprocessing pipelines",
        "Missing value imputation",
    ],
    "ensemble": [
        "Build ensemble models",
        "Stacking and blending",
        "Gradient boosting tuning",
    ],
    "evaluation": [
        "Model evaluation and metrics",
        "Statistical significance testing",
        "Error analysis",
    ],
}


def _slugify(name: str) -> str:
    """Convert chapter name to a URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug


def _infer_skills(chapter_name: str, concepts: List[Dict]) -> List[str]:
    """Infer relevant skills based on chapter name and concepts."""
    skills = set()
    name_lower = chapter_name.lower()
    concept_names = [c.get("name", "").lower() for c in concepts]
    all_text = name_lower + " " + " ".join(concept_names)

    for key, skill_list in CHAPTER_SKILL_MAP.items():
        if key in all_text:
            skills.update(skill_list)

    if not skills:
        skills.update(["kaggle-preprocessor", "kaggle-model-trainer"])

    return sorted(skills)


def _infer_capabilities(chapter_name: str, concepts: List[Dict]) -> List[str]:
    """Infer capabilities based on chapter content."""
    capabilities = []
    name_lower = chapter_name.lower()
    concept_names = [c.get("name", "").lower() for c in concepts]
    all_text = name_lower + " " + " ".join(concept_names)

    for key, caps in CHAPTER_CAPABILITY_MAP.items():
        if key in all_text:
            capabilities.extend(caps)

    if not capabilities:
        capabilities = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
            "Data preprocessing and cleaning",
        ]

    return capabilities


def _build_formula(chapter_name: str) -> Dict[str, Any]:
    """Build an objective formula based on chapter type."""
    name_lower = chapter_name.lower()

    if "classification" in name_lower:
        return {
            "objective": "maximize_classification_accuracy",
            "function": "acc = TP+TN / (TP+TN+FP+FN)",
            "metrics": ["accuracy", "f1_score", "auc_roc"],
        }
    elif "regression" in name_lower:
        return {
            "objective": "minimize_regression_error",
            "function": "MSE = (1/n) * sum((y_i - y_hat_i)^2)",
            "metrics": ["mse", "rmse", "mae", "r2"],
        }
    elif "deep" in name_lower or "neural" in name_lower:
        return {
            "objective": "minimize_training_loss",
            "function": "L = -sum(y_i * log(p_i))",
            "metrics": ["loss", "accuracy", "val_loss"],
        }
    return DEFAULT_FORMULA.copy()


class ExpertRegistry:
    """Manages the registration and retrieval of chapter experts."""

    def __init__(self, db: Optional[Database] = None, skills_path: Optional[str] = None):
        self.db = db or Database()
        self.skills_path = skills_path or DEFAULT_SKILLS_PATH

    def register_expert_from_chapter(
        self,
        chapter_id: int,
        chapter_name: str,
        concepts: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Create and register an expert from a chapter."""
        if concepts is None:
            concepts = self.db.get_concepts_for_chapter(chapter_id)

        slug = _slugify(chapter_name)
        skills = _infer_skills(chapter_name, concepts)
        capabilities = _infer_capabilities(chapter_name, concepts)
        formula = _build_formula(chapter_name)

        expert_def = {
            "expert_name": chapter_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_def)
        self._save_expert_json(expert_def)

        logger.info(f"Registered expert: {chapter_name} ({slug})")
        return expert_def

    def register_all_from_db(self) -> List[Dict[str, Any]]:
        """Register experts for all chapters in the database."""
        chapters = self.db.get_chapters()
        results = []
        for ch in chapters:
            expert = self.register_expert_from_chapter(
                ch["id"], ch["chapter_name"]
            )
            results.append(expert)
        return results

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert_by_slug(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.get_experts()

    def ask_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter_content = None
        if expert.get("chapter_id"):
            chapter_content = self.db.get_chapter_content(expert["chapter_id"])

        context_snippets = []
        if chapter_content:
            q_lower = question.lower()
            paragraphs = chapter_content.split("\n\n")
            scored = []
            for p in paragraphs:
                p_stripped = p.strip()
                if not p_stripped or len(p_stripped) < 20:
                    continue
                overlap = sum(1 for w in q_lower.split() if w in p_stripped.lower())
                if overlap > 0:
                    scored.append((overlap, p_stripped))
            scored.sort(key=lambda x: x[0], reverse=True)
            context_snippets = [s[1] for s in scored[:3]]

        return {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "context": context_snippets,
            "answer_guidance": (
                f"Based on {expert['expert_name']}'s knowledge, "
                f"apply strategy: {expert.get('strategy', DEFAULT_STRATEGY)}"
            ),
        }

    def _save_expert_json(self, expert_def: Dict[str, Any]) -> None:
        """Save expert definition as a JSON file."""
        experts_dir = Path("experts")
        experts_dir.mkdir(exist_ok=True)
        path = experts_dir / f"{expert_def['slug']}.json"
        with open(path, "w") as f:
            json.dump(expert_def, f, indent=2)
