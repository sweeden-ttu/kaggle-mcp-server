"""Expert Registry - manages chapter experts with skills, strategies, and formulas.

Each ML Principles chapter becomes an expert with:
- Capabilities: what the expert can do
- Skills: Kaggle skills the expert recommends
- Strategy: ordered approach (Baseline -> Feature Eng -> Model -> Submit)
- Formula: objective function with metrics
- Loop config: convergence parameters
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
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

# Maps concept keywords to Kaggle skill names
_SKILL_CONCEPT_MAP = {
    "preprocessing": "kaggle-preprocessor",
    "feature engineering": "kaggle-feature-engineer",
    "model": "kaggle-model-trainer",
    "training": "kaggle-model-trainer",
    "neural network": "kaggle-deep-learning",
    "deep learning": "kaggle-deep-learning",
    "ensemble": "kaggle-ensemble-builder",
    "boosting": "kaggle-ensemble-builder",
    "hyperparameter": "kaggle-hyperparameter-tuner",
    "optimization": "kaggle-optimizer",
    "cross-validation": "kaggle-cv-validator",
    "evaluation": "kaggle-evaluator",
    "submission": "kaggle-submitter",
    "eda": "kaggle-eda-explorer",
    "visualization": "kaggle-eda-explorer",
    "nlp": "kaggle-nlp-processor",
    "text": "kaggle-nlp-processor",
    "image": "kaggle-cv-processor",
    "computer vision": "kaggle-cv-processor",
    "time series": "kaggle-timeseries",
    "tabular": "kaggle-tabular",
    "regression": "kaggle-regression",
    "classification": "kaggle-classification",
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def _infer_skills(concepts: List[str]) -> List[str]:
    """Map extracted concepts to Kaggle skill paths."""
    skills = set()
    for concept in concepts:
        concept_lower = concept.lower()
        for keyword, skill_name in _SKILL_CONCEPT_MAP.items():
            if keyword in concept_lower:
                skill_path = os.path.join(KAGGLE_SKILLS_PATH, skill_name)
                skills.add(skill_path)

    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))

    return sorted(skills)


def _infer_capabilities(concepts: List[str], title: str) -> List[str]:
    """Generate capability descriptions from chapter concepts."""
    capabilities = ["Build baseline models quickly", "Systematic hyperparameter search"]

    concept_set = {c.lower() for c in concepts}

    if any("neural" in c or "deep" in c for c in concept_set):
        capabilities.append("Design and train deep neural networks")
    if any("ensemble" in c or "boosting" in c or "bagging" in c for c in concept_set):
        capabilities.append("Build ensemble models for improved accuracy")
    if any("feature" in c for c in concept_set):
        capabilities.append("Advanced feature engineering and selection")
    if any("regularization" in c or "overfit" in c for c in concept_set):
        capabilities.append("Apply regularization to prevent overfitting")
    if any("cross-validation" in c or "validation" in c for c in concept_set):
        capabilities.append("Robust cross-validation strategies")
    if any("optim" in c for c in concept_set):
        capabilities.append("Optimization algorithm selection and tuning")
    if any("nlp" in c or "text" in c or "transformer" in c for c in concept_set):
        capabilities.append("Natural language processing and text modeling")
    if any("image" in c or "cnn" in c or "convolution" in c for c in concept_set):
        capabilities.append("Computer vision and image classification")

    return capabilities


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Infer the objective function and metrics based on concepts."""
    metrics = ["accuracy", "f1_score"]
    concept_lower = " ".join(c.lower() for c in concepts)

    if "regression" in concept_lower:
        metrics = ["rmse", "mae", "r2_score"]
        return {
            "objective": "minimize_rmse",
            "function": "L = (1/n) Σ(y_i - ŷ_i)²",
            "metrics": metrics,
        }

    if "classification" in concept_lower:
        metrics = ["accuracy", "f1_score", "auc_roc"]
        return {
            "objective": "maximize_auc",
            "function": "L = -Σ(y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i))",
            "metrics": metrics,
        }

    return {**DEFAULT_FORMULA, "metrics": metrics}


class ExpertRegistry:
    """Manages the lifecycle of chapter experts."""

    def __init__(self, db):
        self.db = db

    def create_expert_from_chapter(
        self,
        chapter_id: str,
        title: str,
        concepts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create or update an expert from a chapter's extracted data."""
        concept_terms = concepts or []
        slug = _slugify(title)
        skills = _infer_skills(concept_terms)
        capabilities = _infer_capabilities(concept_terms, title)
        formula = _infer_formula(concept_terms)

        expert = {
            "expert_name": title,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG,
            "metadata": {"concept_count": len(concept_terms)},
        }

        self.db.upsert_expert(expert)
        return expert

    def create_experts_from_extraction(self) -> List[Dict[str, Any]]:
        """Create experts for all extracted chapters."""
        chapters = self.db.list_chapters()
        experts = []

        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            title = chapter["title"]

            concept_rows = self.db.get_concepts_for_chapter(chapter_id)
            concept_terms = [c["term"] for c in concept_rows]

            expert = self.create_expert_from_chapter(chapter_id, title, concept_terms)
            experts.append(expert)

        return experts

    def get_expert(self, name_or_slug: str) -> Optional[Dict[str, Any]]:
        """Look up an expert by name or slug."""
        expert = self.db.get_expert(name_or_slug)
        if expert:
            return expert
        return self.db.get_expert_by_slug(name_or_slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def ask_expert(self, name_or_slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert with a question.

        Returns the expert's capabilities, relevant strategy, and
        recommended skills for the question.
        """
        expert = self.get_expert(name_or_slug)
        if not expert:
            return {"error": f"Expert '{name_or_slug}' not found"}

        chapter = self.db.get_chapter(expert.get("chapter_id", ""))
        context = ""
        if chapter and chapter.get("content_md"):
            context = chapter["content_md"][:2000]

        return {
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "skills": expert.get("skills", []),
            "chapter_context": context,
            "question": question,
            "recommendation": (
                f"As the {expert['expert_name']} expert, I recommend following "
                f"the strategy: {expert.get('strategy', DEFAULT_STRATEGY)}. "
                f"Key capabilities: {', '.join(expert.get('capabilities', [])[:3])}."
            ),
        }

    def build_competition_entry(
        self,
        competition: str,
        description: str = "",
        relevant_experts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a competition entry plan using expert knowledge.

        Combines recommendations from relevant experts into a cohesive
        competition strategy.
        """
        if relevant_experts is None:
            relevant_experts = self.list_experts()

        all_skills = set()
        all_capabilities = set()
        strategies = []

        for expert in relevant_experts[:5]:
            for skill in expert.get("skills", []):
                all_skills.add(skill)
            for cap in expert.get("capabilities", []):
                all_capabilities.add(cap)
            strategies.append({
                "expert": expert["expert_name"],
                "strategy": expert.get("strategy", ""),
                "formula": expert.get("formula", {}),
            })

        entry = {
            "competition": competition,
            "description": description,
            "experts_consulted": [e["expert_name"] for e in relevant_experts[:5]],
            "combined_skills": sorted(all_skills),
            "combined_capabilities": sorted(all_capabilities),
            "strategies": strategies,
            "execution_plan": {
                "phase_1": "Baseline - Quick model with minimal preprocessing",
                "phase_2": "EDA - Exploratory data analysis and understanding",
                "phase_3": "Feature Engineering - Domain-informed feature creation",
                "phase_4": "Model Selection - Try models recommended by experts",
                "phase_5": "Ensemble - Combine best models",
                "phase_6": "Optimization - Hyperparameter tuning with convergence loop",
                "phase_7": "Submit - Final submission with best configuration",
            },
            "loop_config": DEFAULT_LOOP_CONFIG,
        }

        return entry

    def save_expert_json(self, expert: Dict[str, Any], output_dir: str = "experts"):
        """Save expert definition as a JSON file."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        slug = expert.get("slug", "unknown")
        path = Path(output_dir) / f"{slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(expert, f, indent=2, ensure_ascii=False)
        return str(path)
