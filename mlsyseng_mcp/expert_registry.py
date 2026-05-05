"""Expert management and registration for the MoE system."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import MLSysEngDB

KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _infer_capabilities(concepts: List[str]) -> List[str]:
    """Infer capabilities from extracted concepts."""
    capability_map = {
        "neural network": "Build and train neural network architectures",
        "deep learning": "Apply deep learning techniques to complex problems",
        "gradient descent": "Optimize models using gradient-based methods",
        "regularization": "Apply regularization to prevent overfitting",
        "feature engineering": "Create informative features from raw data",
        "ensemble": "Build ensemble models for improved performance",
        "cross-validation": "Evaluate models using cross-validation strategies",
        "hyperparameter": "Systematic hyperparameter tuning and optimization",
        "classification": "Build classification models for categorical targets",
        "regression": "Build regression models for continuous targets",
        "clustering": "Apply clustering algorithms for unsupervised analysis",
        "dimensionality reduction": "Reduce feature dimensionality while preserving information",
        "transfer learning": "Apply transfer learning from pre-trained models",
        "transformer": "Leverage transformer architectures for sequence tasks",
        "bayesian": "Apply Bayesian methods for probabilistic modeling",
        "optimization": "Formulate and solve optimization problems",
        "convolutional": "Apply convolutional architectures for spatial data",
        "recurrent": "Apply recurrent architectures for sequential data",
    }

    capabilities = ["Build baseline models quickly"]
    concepts_lower = [c.lower() for c in concepts]

    for keyword, capability in capability_map.items():
        for concept in concepts_lower:
            if keyword in concept:
                capabilities.append(capability)
                break

    return list(dict.fromkeys(capabilities))


def _infer_skills(concepts: List[str]) -> List[str]:
    """Infer Kaggle skill paths from concepts."""
    skill_map = {
        "preprocessing": "kaggle-preprocessor",
        "feature": "kaggle-feature-engineer",
        "model": "kaggle-model-trainer",
        "neural": "kaggle-nn-trainer",
        "deep learning": "kaggle-nn-trainer",
        "ensemble": "kaggle-ensemble-builder",
        "visualization": "kaggle-eda-visualizer",
        "cross-validation": "kaggle-cv-evaluator",
        "hyperparameter": "kaggle-hp-tuner",
        "submission": "kaggle-submission-maker",
        "transformer": "kaggle-transformer-trainer",
        "embedding": "kaggle-embedding-builder",
    }

    skills = set()
    concepts_lower = [c.lower() for c in concepts]

    for keyword, skill_name in skill_map.items():
        for concept in concepts_lower:
            if keyword in concept:
                skills.add(os.path.join(KAGGLE_SKILLS_PATH, skill_name))
                break

    if not skills:
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-preprocessor"))
        skills.add(os.path.join(KAGGLE_SKILLS_PATH, "kaggle-model-trainer"))

    return sorted(skills)


def _infer_formula(concepts: List[str]) -> Dict[str, Any]:
    """Generate a mathematical formula/objective based on concepts."""
    concepts_lower = [c.lower() for c in concepts]

    metrics = ["accuracy"]
    objective = "minimize_validation_loss"
    function = "L = f(X, θ, α)"

    if any("classification" in c for c in concepts_lower):
        metrics = ["accuracy", "f1_score", "auc_roc"]
        function = "L = -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]"
    elif any("regression" in c for c in concepts_lower):
        metrics = ["rmse", "mae", "r2_score"]
        objective = "minimize_rmse"
        function = "L = (1/n)·Σ(y - ŷ)²"
    elif any("neural" in c or "deep" in c for c in concepts_lower):
        metrics = ["accuracy", "loss", "f1_score"]
        function = "L = CrossEntropy(y, softmax(Wh + b))"
    elif any("ensemble" in c for c in concepts_lower):
        metrics = ["accuracy", "f1_score"]
        function = "ŷ = Σ αᵢ·fᵢ(X), s.t. Σαᵢ=1"

    return {
        "objective": objective,
        "function": function,
        "metrics": metrics,
    }


class ExpertRegistry:
    """Manages chapter experts, their capabilities, and skill mappings."""

    def __init__(self, db: Optional[MLSysEngDB] = None):
        self.db = db or MLSysEngDB()

    def register_expert_from_chapter(
        self,
        chapter_name: str,
        concepts: List[str],
        chapter_id: Optional[int] = None,
        custom_skills: Optional[List[str]] = None,
        custom_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and register an expert from a chapter's extracted concepts."""
        slug = _slugify(chapter_name)
        capabilities = _infer_capabilities(concepts)
        skills = custom_skills or _infer_skills(concepts)
        strategy = custom_strategy or DEFAULT_STRATEGY
        formula = _infer_formula(concepts)

        expert_def = {
            "expert_name": chapter_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": strategy,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert_def)

        expert_json_path = Path("experts") / f"{slug}.json"
        expert_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(expert_json_path, "w") as f:
            json.dump(expert_def, f, indent=2)

        return expert_def

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_experts_for_chapter(self, chapter_name: str) -> List[Dict[str, Any]]:
        """Find experts associated with a chapter name."""
        all_experts = self.list_experts()
        slug = _slugify(chapter_name)
        return [
            e for e in all_experts
            if e["slug"] == slug or e.get("expert_name") == chapter_name
        ]

    def get_experts_for_competition(
        self,
        competition_description: str,
        embedding_store=None,
        max_experts: int = 5,
    ) -> List[Dict[str, Any]]:
        """Select the best experts for a competition using RAG."""
        if embedding_store is not None:
            recs = embedding_store.infer_skills_for_competition(
                competition_description, expert_registry=self
            )
            seen = set()
            selected = []
            for rec in recs:
                expert = rec.get("expert", {})
                name = expert.get("expert_name", "")
                if name not in seen:
                    seen.add(name)
                    expert["relevance_score"] = rec.get("relevance_score")
                    selected.append(expert)
                if len(selected) >= max_experts:
                    break
            return selected

        return self.list_experts()[:max_experts]

    def ask_expert(self, slug: str, question: str, embedding_store=None) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.get_expert(slug)
        if expert is None:
            return {"error": f"Expert '{slug}' not found"}

        response = {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
        }

        if embedding_store is not None:
            chapter_name = expert.get("expert_name", "")
            results = embedding_store.search(
                f"{chapter_name}: {question}", n_results=3
            )
            response["relevant_knowledge"] = results

        return response
