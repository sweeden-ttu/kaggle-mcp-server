"""Expert registry for managing chapter-based ML experts.

Each chapter becomes an expert with capabilities, skills, strategy,
mathematical formulas, and a convergence loop configuration.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

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

SKILL_MAP = {
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "model_architecture": ["kaggle-model-trainer", "kaggle-neural-net-builder"],
    "regularization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "evaluation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "data": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
}

CAPABILITY_MAP = {
    "optimization": [
        "Optimize model parameters systematically",
        "Configure learning rate schedules",
        "Apply gradient-based optimization techniques",
    ],
    "model_architecture": [
        "Design neural network architectures",
        "Select appropriate model architectures",
        "Implement attention and transformer layers",
    ],
    "regularization": [
        "Prevent overfitting with regularization",
        "Apply dropout and batch normalization",
        "Configure early stopping strategies",
    ],
    "evaluation": [
        "Evaluate models with appropriate metrics",
        "Implement cross-validation strategies",
        "Analyze confusion matrices and ROC curves",
    ],
    "data": [
        "Engineer features from raw data",
        "Handle missing values and outliers",
        "Apply dimensionality reduction",
    ],
    "ensemble": [
        "Build ensemble models",
        "Implement stacking and blending",
        "Configure boosting algorithms",
    ],
    "general": [
        "Build baseline models quickly",
        "Systematic hyperparameter search",
    ],
}


class ExpertRegistry:
    """Manages registration and querying of chapter experts."""

    def __init__(self, db: Database, skills_base_path: Optional[str] = None):
        self.db = db
        import os
        self.skills_base_path = skills_base_path or os.environ.get(
            "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills"))

    def _slugify(self, name: str) -> str:
        slug = re.sub(r"[^\w\s]", "", name.lower())
        slug = re.sub(r"\s+", "_", slug.strip())
        return slug

    def register_from_chapter(self, chapter_num: int, title: str,
                              concepts: List[Dict[str, str]],
                              chapter_id: Optional[int] = None) -> Dict[str, Any]:
        """Create an expert definition from a chapter and its extracted concepts."""
        slug = f"{chapter_num:02d}_{self._slugify(title)}"
        expert_name = f"{chapter_num:02d}_{title.replace(' ', '_')}"

        categories = set()
        for c in concepts:
            cat = c.get("category", "general")
            categories.add(cat)
        if not categories:
            categories.add("general")

        capabilities = []
        skills = set()
        for cat in categories:
            capabilities.extend(CAPABILITY_MAP.get(cat, CAPABILITY_MAP["general"]))
            for skill_name in SKILL_MAP.get(cat, SKILL_MAP["general"]):
                skills.add(f"{self.skills_base_path}/{skill_name}")

        capabilities = list(dict.fromkeys(capabilities))

        formula = dict(DEFAULT_FORMULA)
        if "evaluation" in categories:
            formula["metrics"] = ["accuracy", "f1_score", "auc_roc"]
        if "optimization" in categories:
            formula["function"] = "L = Σ loss(y, ŷ) + λ·R(θ)"

        loop_config = dict(DEFAULT_LOOP_CONFIG)

        expert_id = self.db.upsert_expert(
            slug=slug,
            expert_name=expert_name,
            chapter_id=chapter_id,
            capabilities=capabilities,
            skills=list(skills),
            strategy=DEFAULT_STRATEGY,
            formula=formula,
            loop_config=loop_config,
        )

        expert_def = {
            "id": expert_id,
            "slug": slug,
            "expert_name": expert_name,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": list(skills),
            "strategy": DEFAULT_STRATEGY,
            "formula": formula,
            "loop_config": loop_config,
        }

        logger.info("Registered expert: %s with %d capabilities, %d skills",
                     expert_name, len(capabilities), len(skills))
        return expert_def

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.get_all_experts()

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def query_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert, returning its knowledge and capabilities."""
        expert = self.db.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter_content = None
        if expert.get("chapter_id"):
            chapter_content = self.db.get_chapter_content(expert["chapter_id"])

        concepts = []
        if expert.get("chapter_id"):
            concepts = self.db.get_concepts(expert["chapter_id"])

        return {
            "expert": expert,
            "question": question,
            "chapter_content_preview": (
                chapter_content[:2000] if chapter_content else None),
            "concepts": concepts,
            "answer_context": (
                f"Expert {expert['expert_name']} specializes in: "
                f"{', '.join(expert.get('capabilities', []))}. "
                f"Strategy: {expert.get('strategy', 'N/A')}."
            ),
        }

    def build_entry(self, competition: str, description: str = "",
                    embedding_store=None) -> Dict[str, Any]:
        """Build a competition entry using expert knowledge and RAG."""
        experts = self.list_experts()

        if embedding_store and description:
            recommendations = embedding_store.infer_skills(description, experts)
        else:
            recommendations = [
                {"expert": e, "relevance_score": 0.5, "matching_context": []}
                for e in experts
            ]

        selected = recommendations[:5] if recommendations else []

        all_skills = set()
        all_capabilities = []
        strategies = []

        for rec in selected:
            expert = rec["expert"]
            for skill in expert.get("skills", []):
                all_skills.add(skill)
            all_capabilities.extend(expert.get("capabilities", []))
            strategies.append({
                "expert": expert["expert_name"],
                "strategy": expert.get("strategy", DEFAULT_STRATEGY),
                "relevance": rec["relevance_score"],
            })

        entry = {
            "competition": competition,
            "description": description,
            "selected_experts": [
                {
                    "expert_name": r["expert"]["expert_name"],
                    "slug": r["expert"]["slug"],
                    "relevance_score": r["relevance_score"],
                    "capabilities": r["expert"].get("capabilities", []),
                }
                for r in selected
            ],
            "skills": list(all_skills),
            "capabilities": list(dict.fromkeys(all_capabilities)),
            "execution_plan": strategies,
            "loop_config": DEFAULT_LOOP_CONFIG,
        }

        return entry

    def export_expert_json(self, slug: str) -> Optional[str]:
        """Export an expert definition as JSON for storage in experts/ directory."""
        expert = self.get_expert(slug)
        if not expert:
            return None
        export = {k: v for k, v in expert.items()
                  if k not in ("id", "created_at")}
        return json.dumps(export, indent=2)
