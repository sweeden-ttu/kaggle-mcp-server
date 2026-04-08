"""Expert registry for MLSysEng MoE system.

Manages chapter experts: creation from extracted content, skill mapping,
strategy definitions, and formula/loop configuration.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Hyperparameter Tuning -> Submit"

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

CONCEPT_TO_SKILLS = {
    "neural network": ["model-trainer", "deep-learning-tuner"],
    "deep learning": ["model-trainer", "deep-learning-tuner"],
    "convolutional": ["image-preprocessor", "cnn-builder"],
    "transformer": ["nlp-preprocessor", "transformer-builder"],
    "attention": ["nlp-preprocessor", "transformer-builder"],
    "gradient descent": ["model-trainer", "hyperparameter-tuner"],
    "optimization": ["hyperparameter-tuner", "model-trainer"],
    "regularization": ["model-trainer", "hyperparameter-tuner"],
    "feature engineering": ["feature-engineer", "preprocessor"],
    "feature selection": ["feature-engineer", "preprocessor"],
    "ensemble": ["ensemble-builder", "model-trainer"],
    "boosting": ["ensemble-builder", "xgboost-tuner"],
    "random forest": ["ensemble-builder", "model-trainer"],
    "classification": ["model-trainer", "evaluator"],
    "regression": ["model-trainer", "evaluator"],
    "clustering": ["unsupervised-learner", "preprocessor"],
    "embedding": ["nlp-preprocessor", "embedding-builder"],
    "tokenization": ["nlp-preprocessor", "text-cleaner"],
    "cross-validation": ["evaluator", "hyperparameter-tuner"],
    "preprocessing": ["preprocessor", "data-cleaner"],
    "normalization": ["preprocessor", "data-cleaner"],
    "data augmentation": ["data-augmenter", "preprocessor"],
    "transfer learning": ["transfer-learner", "model-trainer"],
    "pipeline": ["pipeline-builder", "preprocessor"],
    "PCA": ["dimensionality-reducer", "feature-engineer"],
    "Bayesian": ["bayesian-optimizer", "model-trainer"],
    "reinforcement learning": ["rl-agent-builder", "environment-designer"],
    "decision tree": ["model-trainer", "interpretability-analyzer"],
    "SVM": ["model-trainer", "kernel-tuner"],
}

CONCEPT_TO_CAPABILITIES = {
    "neural network": ["Build and train neural networks", "Architecture design"],
    "deep learning": ["Deep model training", "GPU optimization"],
    "optimization": ["Systematic hyperparameter search", "Learning rate scheduling"],
    "regularization": ["Prevent overfitting", "Model generalization"],
    "feature engineering": ["Create informative features", "Domain-specific transforms"],
    "ensemble": ["Combine multiple models", "Stacking and blending"],
    "classification": ["Build classifiers", "Multi-class strategies"],
    "regression": ["Build regressors", "Target transform strategies"],
    "preprocessing": ["Data cleaning pipelines", "Handle missing values"],
    "cross-validation": ["Robust evaluation", "Stratified validation"],
    "transfer learning": ["Leverage pretrained models", "Domain adaptation"],
    "pipeline": ["End-to-end ML pipelines", "Reproducible workflows"],
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug.strip("_")


def _get_kaggle_skills_path() -> str:
    return os.environ.get(
        "KAGGLE_SKILLS_PATH",
        os.path.expanduser("~/skills"),
    )


class ExpertRegistry:
    """Manages expert creation, querying, and skill inference."""

    def __init__(self, db: Database, skills_path: Optional[str] = None):
        self.db = db
        self.skills_path = skills_path or _get_kaggle_skills_path()

    def create_expert_from_chapter(self, chapter_id: int) -> Optional[Dict[str, Any]]:
        """Create an expert from an extracted chapter."""
        chapter = None
        for ch in self.db.list_chapters():
            if ch["id"] == chapter_id:
                chapter = ch
                break

        if not chapter:
            logger.error(f"Chapter with id {chapter_id} not found")
            return None

        concepts = self.db.get_concepts_for_chapter(chapter_id)
        concept_names = [c["concept_name"] for c in concepts]

        capabilities = self._derive_capabilities(concept_names)
        skills = self._derive_skills(concept_names)
        strategy = self._derive_strategy(concept_names)
        formula = self._derive_formula(concept_names)

        expert_name = f"{chapter['chapter_num']:02d}_{chapter['title']}"
        slug = _slugify(expert_name)

        expert_data = {
            "expert_name": expert_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": strategy,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        expert_id = self.db.upsert_expert(expert_data)
        expert_data["id"] = expert_id

        self._save_expert_definition(expert_data)

        return expert_data

    def create_all_experts(self) -> List[Dict[str, Any]]:
        """Create experts for all extracted chapters."""
        chapters = self.db.list_chapters()
        results = []
        for ch in chapters:
            try:
                expert = self.create_expert_from_chapter(ch["id"])
                if expert:
                    results.append(expert)
            except Exception as e:
                logger.error(f"Failed to create expert for chapter {ch['id']}: {e}")
                results.append({
                    "chapter_id": ch["id"],
                    "status": "error",
                    "message": str(e),
                })
        return results

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def query_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.db.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter = None
        if expert.get("chapter_id"):
            for ch in self.db.list_chapters():
                if ch["id"] == expert["chapter_id"]:
                    chapter = ch
                    break

        relevant_concepts = []
        if chapter:
            concepts = self.db.get_concepts_for_chapter(chapter["id"])
            q_lower = question.lower()
            for c in concepts:
                if any(word in c["concept_name"] for word in q_lower.split()):
                    relevant_concepts.append(c)
            if not relevant_concepts:
                relevant_concepts = concepts[:5]

        return {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "relevant_concepts": [
                {"name": c["concept_name"], "description": c.get("description", "")}
                for c in relevant_concepts
            ],
            "chapter_context": chapter["title"] if chapter else "N/A",
        }

    def infer_skills_for_competition(self, competition_description: str) -> List[Dict[str, Any]]:
        """Infer which experts and skills are needed for a competition."""
        desc_lower = competition_description.lower()
        matched_experts = []

        for expert in self.db.list_experts():
            score = 0
            matched_capabilities = []

            for cap in expert.get("capabilities", []):
                if isinstance(cap, str):
                    cap_words = set(cap.lower().split())
                    desc_words = set(desc_lower.split())
                    overlap = cap_words & desc_words
                    if overlap:
                        score += len(overlap)
                        matched_capabilities.append(cap)

            concepts = []
            if expert.get("chapter_id"):
                concepts = self.db.get_concepts_for_chapter(expert["chapter_id"])
                for c in concepts:
                    if c["concept_name"].lower() in desc_lower:
                        score += 2

            if score > 0:
                matched_experts.append({
                    "expert": expert["expert_name"],
                    "slug": expert["slug"],
                    "score": score,
                    "matched_capabilities": matched_capabilities,
                    "skills": expert.get("skills", []),
                })

        matched_experts.sort(key=lambda x: x["score"], reverse=True)

        if not matched_experts:
            all_experts = self.db.list_experts()
            return [
                {
                    "expert": e["expert_name"],
                    "slug": e["slug"],
                    "score": 0,
                    "matched_capabilities": [],
                    "skills": e.get("skills", []),
                }
                for e in all_experts[:3]
            ]

        return matched_experts

    def _derive_capabilities(self, concept_names: List[str]) -> List[str]:
        capabilities = set()
        capabilities.add("Build baseline models quickly")
        for concept in concept_names:
            for key, caps in CONCEPT_TO_CAPABILITIES.items():
                if key in concept:
                    capabilities.update(caps)
        if not capabilities:
            capabilities.add("General ML modeling")
        return sorted(capabilities)

    def _derive_skills(self, concept_names: List[str]) -> List[str]:
        skills = set()
        base_path = self.skills_path
        for concept in concept_names:
            for key, skill_names in CONCEPT_TO_SKILLS.items():
                if key in concept:
                    for s in skill_names:
                        skills.add(f"{base_path}/{s}")
        if not skills:
            skills.add(f"{base_path}/preprocessor")
            skills.add(f"{base_path}/model-trainer")
        return sorted(skills)

    def _derive_strategy(self, concept_names: List[str]) -> str:
        stages = ["Baseline"]
        if any("feature" in c for c in concept_names):
            stages.append("Feature Engineering")
        if any(c in ["neural network", "deep learning", "transformer"] for c in concept_names):
            stages.append("Deep Learning")
        elif any(c in ["ensemble", "boosting", "random forest"] for c in concept_names):
            stages.append("Ensemble Methods")
        else:
            stages.append("Model Selection")
        if any("hyperparameter" in c or "optimization" in c for c in concept_names):
            stages.append("Hyperparameter Tuning")
        stages.append("Validation")
        stages.append("Submit")
        return " -> ".join(stages)

    def _derive_formula(self, concept_names: List[str]) -> Dict[str, Any]:
        formula = DEFAULT_FORMULA.copy()
        metrics = set(formula["metrics"])
        if any(c in ["classification", "precision", "recall", "F1"] for c in concept_names):
            metrics.update(["accuracy", "f1_score", "precision", "recall"])
        if any(c in ["regression"] for c in concept_names):
            metrics.update(["rmse", "mae", "r2_score"])
        if any("AUC" in c or "ROC" in c for c in concept_names):
            metrics.add("auc_roc")
        formula["metrics"] = sorted(metrics)
        return formula

    def _save_expert_definition(self, expert_data: Dict[str, Any]):
        """Save expert definition as JSON file."""
        experts_dir = Path(__file__).parent.parent.parent / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

        filepath = experts_dir / f"{expert_data['slug']}.json"
        serializable = {
            k: v for k, v in expert_data.items()
            if k not in ("id", "chapter_id", "created_at", "updated_at")
        }
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved expert definition to {filepath}")
