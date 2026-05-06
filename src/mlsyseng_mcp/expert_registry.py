"""Expert registry for MLSysEng MoE - manages chapter experts with skills, strategy, and formulas."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .database import MLSysEngDB

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Submit"

DEFAULT_SKILLS = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-submitter",
]

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "feature": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "classification": ["kaggle-model-trainer", "kaggle-evaluator"],
    "regression": ["kaggle-model-trainer", "kaggle-evaluator"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "boosting": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "transformer": ["kaggle-deep-learning", "kaggle-nlp"],
    "embedding": ["kaggle-nlp", "kaggle-feature-engineer"],
    "clustering": ["kaggle-unsupervised", "kaggle-feature-engineer"],
    "regularization": ["kaggle-model-trainer", "kaggle-optimizer"],
    "cross-validation": ["kaggle-evaluator", "kaggle-model-trainer"],
    "hyperparameter": ["kaggle-optimizer", "kaggle-model-trainer"],
    "data augmentation": ["kaggle-preprocessor", "kaggle-augmenter"],
    "transfer learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "pipeline": ["kaggle-preprocessor", "kaggle-pipeline-builder"],
    "deployment": ["kaggle-submitter", "kaggle-pipeline-builder"],
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a URL-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _infer_skills_from_concepts(concepts: List[Dict[str, Any]], skills_path: str) -> List[str]:
    """Infer Kaggle skill paths from extracted concepts."""
    skill_names = set(DEFAULT_SKILLS)

    for concept in concepts:
        cname = concept.get("concept_name", "").lower()
        for keyword, skills in CONCEPT_TO_SKILLS.items():
            if keyword.lower() in cname or cname in keyword.lower():
                skill_names.update(skills)

    return [os.path.join(skills_path, s) for s in sorted(skill_names)]


def _build_formula(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build an objective function formula from concepts."""
    metrics = ["accuracy"]
    categories = {c.get("category", "general") for c in concepts}

    if "evaluation" in categories:
        metrics.extend(["f1_score", "precision", "recall"])
    if "optimization" in categories:
        metrics.append("loss")
    if "regularization" in categories:
        metrics.append("regularization_penalty")

    metrics = sorted(set(metrics))

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": metrics,
    }


def _build_loop_config(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build convergence loop configuration."""
    complexity = len(concepts)
    max_iter = min(20, max(5, complexity))
    epsilon = 0.001 if complexity > 10 else 0.01

    return {
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": epsilon,
        "max_iterations": max_iter,
        "patience": 3,
    }


def _infer_capabilities(concepts: List[Dict[str, Any]], title: str) -> List[str]:
    """Infer capabilities from chapter concepts."""
    capabilities = ["Build baseline models quickly", "Systematic approach to problem solving"]

    categories = {c.get("category", "general") for c in concepts}

    category_capabilities = {
        "architecture": "Design and implement neural architectures",
        "optimization": "Systematic hyperparameter search and optimization",
        "regularization": "Apply regularization techniques to prevent overfitting",
        "evaluation": "Comprehensive model evaluation and comparison",
        "feature_engineering": "Advanced feature engineering and selection",
        "nlp": "Natural language processing and text feature extraction",
        "ensemble": "Build ensemble models for improved performance",
        "reinforcement_learning": "Apply reinforcement learning strategies",
        "infrastructure": "Build production-ready ML pipelines",
    }

    for cat in categories:
        if cat in category_capabilities:
            capabilities.append(category_capabilities[cat])

    return capabilities


class ExpertRegistry:
    """Registry for managing chapter experts."""

    def __init__(self, db: Optional[MLSysEngDB] = None, skills_path: Optional[str] = None):
        self.db = db or MLSysEngDB()
        self.skills_path = skills_path or DEFAULT_KAGGLE_SKILLS_PATH

    def register_expert_from_chapter(
        self,
        chapter_num: int,
        title: str,
        chapter_id: int,
        concepts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Register an expert from extracted chapter data."""
        expert_name = f"{chapter_num:02d}_{title}"
        slug = _slugify(expert_name)

        capabilities = _infer_capabilities(concepts, title)
        skills = _infer_skills_from_concepts(concepts, self.skills_path)
        strategy = DEFAULT_STRATEGY
        formula = _build_formula(concepts)
        loop_config = _build_loop_config(concepts)

        expert_id = self.db.upsert_expert(
            expert_name=expert_name,
            slug=slug,
            chapter_id=chapter_id,
            capabilities=capabilities,
            skills=skills,
            strategy=strategy,
            formula=formula,
            loop_config=loop_config,
        )

        expert = {
            "expert_id": expert_id,
            "expert_name": expert_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": strategy,
            "formula": formula,
            "loop_config": loop_config,
        }

        logger.info("Registered expert: %s (slug=%s, skills=%d)", expert_name, slug, len(skills))
        return expert

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.get_all_experts()

    def get_experts_for_competition(
        self, relevant_chapters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get experts relevant to a competition based on chapter relevance scores."""
        experts = self.list_experts()
        chapter_scores = {c["chapter_num"]: c.get("relevance_score", 0.0) for c in relevant_chapters}

        relevant = []
        for expert in experts:
            chapter_id = expert.get("chapter_id")
            if chapter_id is None:
                continue
            chapter = self.db.get_chapter_by_id(chapter_id) if hasattr(self.db, "get_chapter_by_id") else None
            ch_num = None
            if chapter:
                ch_num = chapter.get("chapter_num")
            else:
                for ch in self.db.get_all_chapters():
                    if ch["id"] == chapter_id:
                        ch_num = ch["chapter_num"]
                        break

            if ch_num and ch_num in chapter_scores:
                expert["relevance_score"] = chapter_scores[ch_num]
                relevant.append(expert)

        relevant.sort(key=lambda e: e.get("relevance_score", 0.0), reverse=True)
        return relevant

    def export_expert_definition(self, slug: str) -> Optional[Dict[str, Any]]:
        """Export a full expert definition as JSON."""
        expert = self.get_expert(slug)
        if not expert:
            return None

        return {
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "skills": expert.get("skills", []),
            "strategy": expert.get("strategy", DEFAULT_STRATEGY),
            "formula": expert.get("formula", {}),
            "loop_config": expert.get("loop_config", {}),
        }

    def export_all_definitions(self) -> List[Dict[str, Any]]:
        """Export all expert definitions."""
        experts = self.list_experts()
        return [self.export_expert_definition(e["slug"]) for e in experts if e.get("slug")]
