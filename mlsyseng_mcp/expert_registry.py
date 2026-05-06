"""Expert registry for managing chapter-based ML experts.

Each ML Principles chapter becomes an expert with capabilities, skills,
strategy, mathematical formulas, and convergence loop configuration.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
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

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "convolutional": ["kaggle-image-classifier", "kaggle-deep-learning"],
    "recurrent": ["kaggle-sequence-model", "kaggle-deep-learning"],
    "transformer": ["kaggle-nlp-transformer", "kaggle-deep-learning"],
    "attention mechanism": ["kaggle-nlp-transformer"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "feature engineering": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "boosting": ["kaggle-ensemble-builder", "kaggle-xgboost"],
    "bagging": ["kaggle-ensemble-builder", "kaggle-random-forest"],
    "random forest": ["kaggle-random-forest", "kaggle-ensemble-builder"],
    "decision tree": ["kaggle-decision-tree", "kaggle-model-trainer"],
    "classification": ["kaggle-classifier", "kaggle-model-trainer"],
    "regression": ["kaggle-regressor", "kaggle-model-trainer"],
    "clustering": ["kaggle-clustering", "kaggle-unsupervised"],
    "dimensionality reduction": ["kaggle-dim-reduction", "kaggle-preprocessor"],
    "cross-validation": ["kaggle-validator", "kaggle-model-trainer"],
    "hyperparameter": ["kaggle-hypertuner", "kaggle-model-trainer"],
    "bayesian": ["kaggle-bayesian", "kaggle-model-trainer"],
    "reinforcement learning": ["kaggle-rl-agent"],
    "generative adversarial": ["kaggle-gan-builder"],
    "transfer learning": ["kaggle-transfer-learner", "kaggle-model-trainer"],
    "embedding": ["kaggle-embedding-builder", "kaggle-preprocessor"],
    "loss function": ["kaggle-model-trainer", "kaggle-optimizer"],
    "optimizer": ["kaggle-optimizer", "kaggle-model-trainer"],
}

CONCEPT_TO_CAPABILITIES = {
    "neural network": ["Build and train neural networks", "Architecture design"],
    "deep learning": ["Deep model architectures", "GPU-accelerated training"],
    "feature engineering": ["Automated feature creation", "Feature selection"],
    "ensemble": ["Combine multiple models", "Stacking and blending"],
    "gradient descent": ["Optimization algorithms", "Learning rate scheduling"],
    "regularization": ["Prevent overfitting", "L1/L2 regularization"],
    "cross-validation": ["Model validation", "K-fold strategies"],
    "hyperparameter": ["Systematic hyperparameter search", "Bayesian optimization"],
    "classification": ["Binary and multi-class classification"],
    "regression": ["Continuous value prediction"],
    "clustering": ["Unsupervised pattern discovery"],
    "transformer": ["Self-attention models", "NLP and sequence tasks"],
    "bayesian": ["Probabilistic reasoning", "Uncertainty quantification"],
    "transfer learning": ["Pretrained model adaptation", "Domain transfer"],
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _infer_skills(concepts: List[str]) -> List[str]:
    """Map chapter concepts to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        for skill_name in CONCEPT_TO_SKILLS.get(concept, []):
            skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill_name))
    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))
    return sorted(skills)


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Map chapter concepts to expert capabilities."""
    caps = set()
    for concept in concepts:
        for cap in CONCEPT_TO_CAPABILITIES.get(concept, []):
            caps.add(cap)
    caps.add("Build baseline models quickly")
    caps.add("Systematic hyperparameter search")
    return sorted(caps)


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Customize the expert formula based on concepts."""
    formula = dict(DEFAULT_FORMULA)
    metrics = set(formula["metrics"])

    if "classification" in concepts:
        metrics.update(["accuracy", "f1_score", "precision", "recall"])
    if "regression" in concepts:
        metrics.update(["rmse", "mae", "r2_score"])
        formula["objective"] = "minimize_rmse"
    if "clustering" in concepts:
        metrics.update(["silhouette_score", "inertia"])
        formula["objective"] = "maximize_silhouette"
    if "deep learning" in concepts or "neural network" in concepts:
        formula["function"] = "L = CrossEntropy(ŷ, y) + λ||θ||₂²"

    formula["metrics"] = sorted(metrics)
    return formula


class ExpertRegistry:
    """Manages the lifecycle of chapter-based ML experts."""

    def __init__(self, db: Database):
        self.db = db

    def create_expert_from_chapter(
        self,
        chapter_number: str,
        title: str,
        concepts: Optional[List[str]] = None,
        chapter_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create or update an expert from a chapter's extracted data."""
        concepts = concepts or []
        expert_name = f"{chapter_number}_{title}"
        slug = _slugify(expert_name)

        capabilities = _infer_capabilities(concepts)
        skills = _infer_skills(concepts)
        formula = _infer_formula(concepts)

        expert_id = self.db.upsert_expert(
            expert_name=expert_name,
            slug=slug,
            chapter_id=chapter_id,
            capabilities=capabilities,
            skills=skills,
            strategy=DEFAULT_STRATEGY,
            formula=formula,
            loop_config=DEFAULT_LOOP_CONFIG,
        )

        expert = self.db.get_expert(slug)
        self._save_expert_json(expert)
        return expert

    def create_experts_from_all_chapters(self) -> List[Dict[str, Any]]:
        """Create experts for all indexed chapters."""
        chapters = self.db.list_chapters()
        experts = []
        for ch in chapters:
            concepts = ch.get("concepts", "[]")
            if isinstance(concepts, str):
                try:
                    concepts = json.loads(concepts)
                except (json.JSONDecodeError, TypeError):
                    concepts = []
            expert = self.create_expert_from_chapter(
                chapter_number=ch["chapter_number"],
                title=ch["title"],
                concepts=concepts,
                chapter_id=ch["id"],
            )
            experts.append(expert)
        return experts

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_experts_for_competition(
        self, competition_description: str
    ) -> List[Dict[str, Any]]:
        """Find experts relevant to a competition (keyword-based fallback)."""
        desc_lower = competition_description.lower()
        experts = self.list_experts()
        scored = []
        for expert in experts:
            score = 0
            caps = expert.get("capabilities", [])
            if isinstance(caps, str):
                try:
                    caps = json.loads(caps)
                except (json.JSONDecodeError, TypeError):
                    caps = []
            for cap in caps:
                if any(word in desc_lower for word in cap.lower().split()):
                    score += 1
            if score > 0:
                scored.append((score, expert))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored]

    def _save_expert_json(self, expert: Dict[str, Any]):
        """Save expert definition as JSON to the experts/ directory."""
        experts_dir = Path(__file__).parent.parent / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)
        slug = expert.get("slug", "unknown")
        path = experts_dir / f"{slug}.json"
        serializable = {}
        for k, v in expert.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable[k] = v
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Saved expert definition: %s", path)

    def export_all_experts(self) -> List[str]:
        """Export all expert definitions as JSON files."""
        experts = self.list_experts()
        paths = []
        for expert in experts:
            self._save_expert_json(expert)
            paths.append(
                str(Path(__file__).parent.parent / "experts" / f"{expert['slug']}.json")
            )
        return paths
