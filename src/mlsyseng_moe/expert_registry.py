"""Expert registry - manages chapter experts with skills, strategy, and formulas.

Each ML Principles chapter becomes an expert that can recommend Kaggle skills
and strategies based on its specialized knowledge.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    str(Path.home() / "skills"),
)

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "gradient descent": ["kaggle-model-trainer"],
    "convolutional": ["kaggle-image-processor", "kaggle-deep-learning"],
    "transformer": ["kaggle-nlp-processor", "kaggle-deep-learning"],
    "attention mechanism": ["kaggle-nlp-processor"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "feature selection": ["kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble-builder"],
    "random forest": ["kaggle-model-trainer"],
    "gradient boosting": ["kaggle-model-trainer", "kaggle-tabular-optimizer"],
    "xgboost": ["kaggle-model-trainer", "kaggle-tabular-optimizer"],
    "lightgbm": ["kaggle-model-trainer", "kaggle-tabular-optimizer"],
    "catboost": ["kaggle-model-trainer", "kaggle-tabular-optimizer"],
    "cross-validation": ["kaggle-model-trainer"],
    "hyperparameter": ["kaggle-hyperparameter-tuner"],
    "time series": ["kaggle-time-series"],
    "natural language processing": ["kaggle-nlp-processor"],
    "image classification": ["kaggle-image-processor"],
    "object detection": ["kaggle-image-processor"],
    "data augmentation": ["kaggle-augmentation"],
    "normalization": ["kaggle-preprocessor"],
    "embedding": ["kaggle-nlp-processor"],
    "logistic regression": ["kaggle-model-trainer"],
    "linear regression": ["kaggle-model-trainer"],
    "clustering": ["kaggle-unsupervised"],
    "dimensionality reduction": ["kaggle-feature-engineer"],
    "bayesian": ["kaggle-model-trainer"],
    "ml systems": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "model deployment": ["kaggle-submitter"],
    "distributed training": ["kaggle-deep-learning"],
}

CONCEPT_TO_METRICS = {
    "classification": ["accuracy", "f1_score", "precision", "recall", "auc"],
    "regression": ["rmse", "mae", "r2_score"],
    "deep learning": ["loss", "accuracy", "f1_score"],
    "neural network": ["loss", "accuracy"],
    "natural language processing": ["f1_score", "bleu", "perplexity"],
    "image classification": ["accuracy", "top_5_accuracy"],
    "time series": ["rmse", "mae", "mape"],
    "clustering": ["silhouette_score", "adjusted_rand_index"],
}


def _slug_from_title(title: str) -> str:
    """Convert a chapter title to a slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_")
    return slug


def _infer_skills(concepts: List[str]) -> List[str]:
    """Map concepts to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        for key, skill_list in CONCEPT_TO_SKILLS.items():
            if key in concept.lower():
                skills.update(skill_list)
    if not skills:
        skills = {"kaggle-preprocessor", "kaggle-model-trainer"}
    return sorted(skills)


def _infer_metrics(concepts: List[str]) -> List[str]:
    """Infer metrics from chapter concepts."""
    metrics = set()
    for concept in concepts:
        for key, metric_list in CONCEPT_TO_METRICS.items():
            if key in concept.lower():
                metrics.update(metric_list)
    if not metrics:
        metrics = {"accuracy", "f1_score"}
    return sorted(metrics)


def _build_capabilities(concepts: List[str], title: str) -> List[str]:
    """Generate capability descriptions from concepts."""
    caps = [f"Expert knowledge in {title}"]
    if any("feature" in c for c in concepts):
        caps.append("Feature engineering and selection")
    if any("neural" in c or "deep" in c for c in concepts):
        caps.append("Deep learning model design and training")
    if any("ensemble" in c or "boost" in c for c in concepts):
        caps.append("Ensemble methods and gradient boosting")
    if any("nlp" in c or "language" in c or "transformer" in c for c in concepts):
        caps.append("Natural language processing")
    if any("image" in c or "convolutional" in c for c in concepts):
        caps.append("Computer vision and image processing")
    if any("time series" in c for c in concepts):
        caps.append("Time series analysis and forecasting")
    if any("optim" in c or "gradient" in c for c in concepts):
        caps.append("Optimization and hyperparameter tuning")
    caps.append("Build baseline models quickly")
    caps.append("Systematic hyperparameter search")
    return caps


def create_expert_definition(
    chapter_id: str,
    title: str,
    concepts: List[str],
    skills_path: str = DEFAULT_SKILLS_PATH,
) -> Dict[str, Any]:
    """Build a full expert definition from chapter metadata."""
    slug = _slug_from_title(title)
    expert_name = f"{chapter_id}_{slug}"
    skill_names = _infer_skills(concepts)
    metrics = _infer_metrics(concepts)
    capabilities = _build_capabilities(concepts, title)

    skill_paths = [f"{skills_path}/{s}" for s in skill_names]

    return {
        "expert_name": expert_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "title": title,
        "capabilities": capabilities,
        "skills": skill_paths,
        "skill_names": skill_names,
        "strategy": DEFAULT_STRATEGY,
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, theta, alpha)",
            "metrics": metrics,
        },
        "loop_config": dict(DEFAULT_LOOP_CONFIG),
        "concepts": concepts,
    }


class ExpertRegistry:
    """Manages the collection of chapter experts."""

    def __init__(self, db: Database, skills_path: str = DEFAULT_SKILLS_PATH):
        self.db = db
        self.skills_path = skills_path

    def register_from_chapters(self) -> List[Dict[str, Any]]:
        """Create experts from all extracted chapters in the database."""
        chapters = self.db.list_chapters()
        experts = []
        for ch in chapters:
            defn = create_expert_definition(
                chapter_id=ch["chapter_id"],
                title=ch["title"],
                concepts=ch.get("concepts", []),
                skills_path=self.skills_path,
            )
            self.db.upsert_expert(defn["expert_name"], defn)
            experts.append(defn)
        return experts

    def get_expert(self, expert_name: str) -> Optional[Dict[str, Any]]:
        row = self.db.get_expert(expert_name)
        if row:
            return row["definition"]
        return None

    def list_experts(self) -> List[Dict[str, Any]]:
        rows = self.db.list_experts()
        return [r["definition"] for r in rows]

    def find_experts_for_competition(
        self,
        competition_description: str,
        relevance_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Find experts relevant to a competition, optionally using RAG results."""
        all_experts = self.list_experts()
        if not relevance_results:
            return all_experts

        relevant_chapters = {r["chapter_id"] for r in relevance_results}
        matched = [e for e in all_experts if e.get("chapter_id") in relevant_chapters]
        return matched if matched else all_experts

    def save_expert_json(self, expert_name: str, output_dir: str = "experts") -> str:
        """Save an expert definition to a JSON file."""
        defn = self.get_expert(expert_name)
        if not defn:
            raise ValueError(f"Expert not found: {expert_name}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{expert_name}.json"
        file_path.write_text(json.dumps(defn, indent=2))
        return str(file_path)
