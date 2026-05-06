"""Expert registry for MLSysEng MoE.

Manages chapter-based experts: creation from extracted content,
capability mapping, skill recommendations, and querying.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from .database import ChapterRecord, Database, ExpertRecord

logger = logging.getLogger(__name__)

SKILL_MAPPING = {
    "preprocessing": ["kaggle-preprocessor", "kaggle-data-cleaner"],
    "feature engineering": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "feature extraction": ["kaggle-feature-engineer"],
    "feature selection": ["kaggle-feature-engineer"],
    "classification": ["kaggle-model-trainer", "kaggle-classifier"],
    "regression": ["kaggle-model-trainer", "kaggle-regressor"],
    "clustering": ["kaggle-clustering"],
    "neural network": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "deep learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "convolutional": ["kaggle-cnn-trainer", "kaggle-deep-learning"],
    "recurrent": ["kaggle-rnn-trainer", "kaggle-deep-learning"],
    "transformer": ["kaggle-transformer-trainer", "kaggle-deep-learning"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "gradient boosting": ["kaggle-gbm-trainer", "kaggle-model-trainer"],
    "xgboost": ["kaggle-xgboost-trainer", "kaggle-model-trainer"],
    "lightgbm": ["kaggle-lightgbm-trainer", "kaggle-model-trainer"],
    "random forest": ["kaggle-rf-trainer", "kaggle-model-trainer"],
    "decision tree": ["kaggle-model-trainer"],
    "support vector": ["kaggle-svm-trainer", "kaggle-model-trainer"],
    "logistic regression": ["kaggle-model-trainer"],
    "bayesian": ["kaggle-bayesian-optimizer"],
    "hyperparameter": ["kaggle-hyperparameter-tuner"],
    "cross-validation": ["kaggle-cross-validator", "kaggle-model-trainer"],
    "cross validation": ["kaggle-cross-validator", "kaggle-model-trainer"],
    "embedding": ["kaggle-embedding-trainer"],
    "transfer learning": ["kaggle-transfer-learner"],
    "fine-tuning": ["kaggle-fine-tuner"],
    "data augmentation": ["kaggle-augmentation", "kaggle-preprocessor"],
    "normalization": ["kaggle-preprocessor"],
    "standardization": ["kaggle-preprocessor"],
    "pca": ["kaggle-dim-reduction"],
    "dimensionality reduction": ["kaggle-dim-reduction"],
    "time series": ["kaggle-timeseries-trainer"],
    "optimization": ["kaggle-optimizer", "kaggle-model-trainer"],
    "pipeline": ["kaggle-pipeline-builder"],
    "model deployment": ["kaggle-submission-builder"],
    "distributed training": ["kaggle-distributed-trainer"],
    "mixture of experts": ["kaggle-moe-builder"],
}

CAPABILITY_TEMPLATES = {
    "neural network": "Build and train neural network architectures",
    "deep learning": "Design deep learning pipelines",
    "convolutional": "Apply convolutional neural networks for spatial data",
    "recurrent": "Model sequential data with recurrent networks",
    "transformer": "Leverage transformer architectures for sequence tasks",
    "gradient descent": "Optimize models using gradient-based methods",
    "regularization": "Apply regularization techniques to prevent overfitting",
    "ensemble": "Combine multiple models for improved predictions",
    "feature engineering": "Create and select informative features",
    "classification": "Build classification models",
    "regression": "Build regression models",
    "clustering": "Perform unsupervised clustering analysis",
    "bayesian": "Apply Bayesian methods for probabilistic modeling",
    "hyperparameter": "Systematic hyperparameter search and optimization",
    "cross-validation": "Evaluate models with cross-validation",
    "transfer learning": "Apply transfer learning from pre-trained models",
    "time series": "Model and forecast time series data",
    "optimization": "Optimize objective functions systematically",
    "pipeline": "Build end-to-end ML pipelines",
    "preprocessing": "Clean and preprocess raw data",
    "data augmentation": "Augment training data for better generalization",
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _infer_skills(concepts: list[str], skills_base_path: str) -> list[str]:
    """Map concepts to skill paths."""
    skill_names: set[str] = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skills in SKILL_MAPPING.items():
            if keyword in concept_lower:
                skill_names.update(skills)

    if not skill_names:
        skill_names = {"kaggle-preprocessor", "kaggle-model-trainer"}

    return sorted(os.path.join(skills_base_path, s) for s in skill_names)


def _infer_capabilities(concepts: list[str]) -> list[str]:
    """Map concepts to human-readable capabilities."""
    caps: list[str] = []
    seen = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, template in CAPABILITY_TEMPLATES.items():
            if keyword in concept_lower and template not in seen:
                caps.append(template)
                seen.add(template)

    if not caps:
        caps = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]
    return caps


def _infer_formula(concepts: list[str]) -> dict:
    """Infer a mathematical formula/objective based on concepts."""
    has_classification = any(
        c in ["classification", "logistic regression"]
        for c in [x.lower() for x in concepts]
    )
    has_regression = any(
        "regression" in c.lower() for c in concepts
        if "logistic" not in c.lower()
    )

    if has_classification:
        return {
            "objective": "minimize_cross_entropy",
            "function": "L = -Σ y_i log(ŷ_i)",
            "metrics": ["accuracy", "f1_score", "auc_roc"],
        }
    elif has_regression:
        return {
            "objective": "minimize_mse",
            "function": "L = (1/n) Σ (y_i - ŷ_i)²",
            "metrics": ["rmse", "mae", "r2_score"],
        }
    else:
        return {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": ["accuracy", "f1_score"],
        }


class ExpertRegistry:
    """Manages the creation and retrieval of chapter-based experts."""

    def __init__(self, db: Database, skills_base_path: Optional[str] = None):
        self.db = db
        self.skills_base_path = skills_base_path or os.environ.get(
            "KAGGLE_SKILLS_PATH",
            os.path.expanduser("~/skills"),
        )

    def create_expert_from_chapter(
        self, chapter: ChapterRecord
    ) -> ExpertRecord:
        """Create an expert definition from an extracted chapter."""
        skills = _infer_skills(chapter.concepts, self.skills_base_path)
        capabilities = _infer_capabilities(chapter.concepts)
        formula = _infer_formula(chapter.concepts)

        expert = ExpertRecord(
            expert_name=chapter.title,
            slug=chapter.slug,
            chapter_id=chapter.chapter_id,
            capabilities=capabilities,
            skills=skills,
            strategy="Baseline → EDA → Feature Engineering → Model Selection → Submit",
            formula=formula,
            loop_config={
                "objective": formula["objective"],
                "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
                "epsilon": 0.001,
                "max_iterations": 10,
                "patience": 3,
            },
        )
        self.db.upsert_expert(expert)
        logger.info("Created expert: %s with %d skills", expert.expert_name, len(skills))
        return expert

    def create_experts_from_all_chapters(self) -> list[ExpertRecord]:
        """Create experts for all indexed chapters."""
        chapters = self.db.list_chapters()
        experts = []
        for chapter in chapters:
            expert = self.create_expert_from_chapter(chapter)
            experts.append(expert)
        return experts

    def get_expert(self, name_or_slug: str) -> Optional[ExpertRecord]:
        """Look up an expert by name or slug."""
        expert = self.db.get_expert(name_or_slug)
        if expert:
            return expert
        return self.db.get_expert_by_slug(name_or_slug)

    def list_experts(self) -> list[ExpertRecord]:
        return self.db.list_experts()

    def query_expert(self, slug: str, question: str) -> dict:
        """Query a specific expert about a topic.

        Returns the expert's context (capabilities, strategy, relevant
        chapter content) formatted for answering the question.
        """
        expert = self.db.get_expert_by_slug(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter = self.db.get_chapter(expert.chapter_id)
        chapter_excerpt = ""
        if chapter and chapter.content_md:
            words = chapter.content_md.split()
            chapter_excerpt = " ".join(words[:500])

        return {
            "expert_name": expert.expert_name,
            "slug": expert.slug,
            "question": question,
            "capabilities": expert.capabilities,
            "strategy": expert.strategy,
            "formula": expert.formula,
            "concepts": chapter.concepts if chapter else [],
            "chapter_excerpt": chapter_excerpt,
            "skills": expert.skills,
        }

    def recommend_skills_for_competition(
        self, competition_description: str
    ) -> list[dict]:
        """Recommend skills for a competition based on expert knowledge."""
        experts = self.db.list_experts()
        all_skills: dict[str, set[str]] = {}

        desc_lower = competition_description.lower()
        for expert in experts:
            relevance = sum(
                1 for cap in expert.capabilities
                if any(w in desc_lower for w in cap.lower().split())
            )
            if relevance > 0:
                for skill in expert.skills:
                    if skill not in all_skills:
                        all_skills[skill] = set()
                    all_skills[skill].add(expert.expert_name)

        results = []
        for skill_path, recommenders in sorted(
            all_skills.items(), key=lambda x: -len(x[1])
        ):
            results.append({
                "skill": skill_path,
                "recommended_by": sorted(recommenders),
                "confidence": len(recommenders) / max(len(experts), 1),
            })
        return results

    def export_expert(self, slug: str) -> Optional[dict]:
        """Export an expert definition as a JSON-serializable dict."""
        expert = self.db.get_expert_by_slug(slug)
        if not expert:
            return None
        return {
            "expert_name": expert.expert_name,
            "slug": expert.slug,
            "capabilities": expert.capabilities,
            "skills": expert.skills,
            "strategy": expert.strategy,
            "formula": expert.formula,
            "loop_config": expert.loop_config,
        }

    def export_all_experts(self, output_dir: Optional[str] = None) -> list[str]:
        """Export all experts as JSON files. Returns list of written paths."""
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..",
                "experts",
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        written = []
        for expert in self.db.list_experts():
            data = self.export_expert(expert.slug)
            if data:
                path = os.path.join(output_dir, f"{expert.slug}.json")
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
                written.append(path)
        return written
