"""Expert management and registration for MLSysEng MoE.

Each ML Principles chapter becomes an expert with skills, strategy,
and mathematical formulas.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .database import MLSysEngDB

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Submit"

DEFAULT_FORMULA = {
    "objective": "minimize_validation_loss",
    "function": "L = f(X, θ, α)",
    "metrics": ["accuracy", "f1_score", "rmse"],
}

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

SKILL_MAPPING = {
    "optimization": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "regularization": ["kaggle-model-trainer", "kaggle-preprocessor"],
    "architecture": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "evaluation": ["kaggle-evaluator", "kaggle-cross-validator"],
    "ensemble": ["kaggle-ensemble-builder", "kaggle-model-trainer"],
    "classical_ml": ["kaggle-preprocessor", "kaggle-model-trainer"],
    "representation": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "advanced": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "training": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
    "general": ["kaggle-preprocessor", "kaggle-model-trainer"],
}

STRATEGY_MAPPING = {
    "optimization": "Baseline → Loss Analysis → Optimizer Tuning → LR Schedule → Validation → Submit",
    "regularization": "Baseline → Overfit Check → Regularization → Cross-Validation → Submit",
    "architecture": "Baseline → Architecture Search → Training → Validation → Submit",
    "evaluation": "Baseline → Metric Selection → Cross-Validation → Ensemble → Submit",
    "ensemble": "Baseline → Model Pool → Stacking/Blending → Validation → Submit",
    "classical_ml": "EDA → Feature Engineering → Model Selection → Tuning → Submit",
    "representation": "EDA → Feature Engineering → Dimensionality Reduction → Model → Submit",
    "advanced": "Baseline → Transfer Learning → Fine-Tuning → Validation → Submit",
    "training": "Baseline → Training Loop → Hyperparameter Search → Validation → Submit",
}

FORMULA_MAPPING = {
    "optimization": {
        "objective": "minimize_loss",
        "function": "L = Σ loss(y_pred, y_true) + λ·R(θ)",
        "metrics": ["loss", "convergence_rate"],
    },
    "regularization": {
        "objective": "minimize_regularized_loss",
        "function": "L = loss(y_pred, y_true) + α·||θ||_p",
        "metrics": ["train_loss", "val_loss", "generalization_gap"],
    },
    "architecture": {
        "objective": "maximize_model_capacity",
        "function": "L = CE(y_pred, y_true) + complexity_penalty",
        "metrics": ["accuracy", "params_count", "flops"],
    },
    "evaluation": {
        "objective": "minimize_cv_variance",
        "function": "score = (1/K) Σ metric(fold_k)",
        "metrics": ["cv_mean", "cv_std", "test_score"],
    },
    "ensemble": {
        "objective": "maximize_ensemble_diversity",
        "function": "ŷ = Σ w_i · f_i(x), Σw_i = 1",
        "metrics": ["ensemble_score", "diversity", "correlation"],
    },
}


def _slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _infer_capabilities(concepts: List[Dict[str, str]], title: str) -> List[str]:
    """Infer expert capabilities from concepts and title."""
    capabilities = [f"Expert knowledge in {title}"]

    categories = set(c.get("category", "general") for c in concepts)
    capability_map = {
        "optimization": "Systematic hyperparameter search and optimization",
        "regularization": "Prevent overfitting with regularization techniques",
        "architecture": "Design and select neural network architectures",
        "evaluation": "Rigorous model evaluation and cross-validation",
        "ensemble": "Build powerful ensemble models",
        "classical_ml": "Apply classical ML algorithms effectively",
        "representation": "Feature engineering and data representation",
        "advanced": "Apply advanced ML techniques (transfer learning, RL)",
        "training": "Efficient model training strategies",
    }

    for cat in categories:
        if cat in capability_map:
            capabilities.append(capability_map[cat])

    capabilities.append("Build baseline models quickly")

    return list(dict.fromkeys(capabilities))


def _infer_skills(concepts: List[Dict[str, str]], skills_base_path: str) -> List[str]:
    """Infer skill paths from concepts."""
    skill_names = set()
    for concept in concepts:
        cat = concept.get("category", "general")
        skill_names.update(SKILL_MAPPING.get(cat, SKILL_MAPPING["general"]))

    return [f"{skills_base_path}/{name}" for name in sorted(skill_names)]


def _infer_strategy(concepts: List[Dict[str, str]]) -> str:
    """Infer strategy from dominant concept categories."""
    category_counts: Dict[str, int] = {}
    for c in concepts:
        cat = c.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts:
        dominant = max(category_counts, key=category_counts.get)
        return STRATEGY_MAPPING.get(dominant, DEFAULT_STRATEGY)
    return DEFAULT_STRATEGY


def _infer_formula(concepts: List[Dict[str, str]]) -> Dict[str, Any]:
    """Infer objective formula from dominant concept categories."""
    category_counts: Dict[str, int] = {}
    for c in concepts:
        cat = c.get("category", "general")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts:
        dominant = max(category_counts, key=category_counts.get)
        return FORMULA_MAPPING.get(dominant, DEFAULT_FORMULA)
    return DEFAULT_FORMULA


class ExpertRegistry:
    """Manages expert creation, registration, and lookup."""

    def __init__(self, db: MLSysEngDB, skills_base_path: Optional[str] = None):
        self.db = db
        self.skills_base_path = skills_base_path or "/skills"

    def create_expert_from_chapter(
        self,
        chapter_number: int,
        title: str,
        chapter_id: int,
        concepts: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Create an expert definition from a chapter's extracted data."""
        slug = f"{chapter_number:02d}_{_slugify(title)}"
        expert_name = f"{chapter_number:02d}_{title}"

        capabilities = _infer_capabilities(concepts, title)
        skills = _infer_skills(concepts, self.skills_base_path)
        strategy = _infer_strategy(concepts)
        formula = _infer_formula(concepts)

        expert = {
            "expert_name": expert_name,
            "slug": slug,
            "chapter_id": chapter_id,
            "capabilities": capabilities,
            "skills": skills,
            "strategy": strategy,
            "formula": formula,
            "loop_config": DEFAULT_LOOP_CONFIG.copy(),
        }

        self.db.upsert_expert(expert)
        return expert

    def register_experts_from_db(self) -> List[Dict[str, Any]]:
        """Create experts for all chapters that don't have one yet."""
        chapters = self.db.get_all_chapters()
        existing = {e["slug"] for e in self.db.get_all_experts()}
        created = []

        for ch in chapters:
            slug = f"{ch['chapter_number']:02d}_{_slugify(ch['title'])}"
            if slug in existing:
                continue

            concepts = self.db.get_concepts(ch["id"])
            expert = self.create_expert_from_chapter(
                ch["chapter_number"], ch["title"], ch["id"], concepts
            )
            created.append(expert)

        return created

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.get_all_experts()

    def ask_expert(self, slug: str, question: str) -> Dict[str, Any]:
        """Query a specific expert about a topic."""
        expert = self.db.get_expert(slug)
        if not expert:
            return {"error": f"Expert '{slug}' not found"}

        chapter = None
        if expert.get("chapter_id"):
            chapters = self.db.get_all_chapters()
            for ch in chapters:
                if ch["id"] == expert["chapter_id"]:
                    chapter = ch
                    break

        concepts = []
        if expert.get("chapter_id"):
            concepts = self.db.get_concepts(expert["chapter_id"])

        return {
            "expert": expert["expert_name"],
            "slug": slug,
            "capabilities": expert.get("capabilities", []),
            "strategy": expert.get("strategy", ""),
            "formula": expert.get("formula", {}),
            "relevant_concepts": [c["concept"] for c in concepts],
            "chapter_context": (chapter or {}).get("content_md", "")[:2000],
            "question": question,
            "guidance": (
                f"As the {expert['expert_name']} expert, I recommend: "
                f"Follow the strategy '{expert.get('strategy', DEFAULT_STRATEGY)}' "
                f"and optimize using {expert.get('formula', DEFAULT_FORMULA).get('function', 'L = f(X, θ, α)')}. "
                f"Key skills: {', '.join(expert.get('skills', []))}"
            ),
        }

    def build_competition_entry(
        self,
        competition: str,
        matched_experts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build a competition entry plan using matched experts."""
        if not matched_experts:
            return {
                "competition": competition,
                "error": "No matching experts found",
            }

        primary_expert = matched_experts[0]["expert"]
        supporting = [m["expert"] for m in matched_experts[1:3]]

        all_skills = set()
        all_capabilities = set()
        for me in matched_experts[:3]:
            exp = me["expert"]
            all_skills.update(exp.get("skills", []))
            all_capabilities.update(exp.get("capabilities", []))

        return {
            "competition": competition,
            "primary_expert": {
                "name": primary_expert["expert_name"],
                "strategy": primary_expert.get("strategy", ""),
                "formula": primary_expert.get("formula", {}),
            },
            "supporting_experts": [
                {"name": e["expert_name"], "slug": e["slug"]}
                for e in supporting
            ],
            "recommended_skills": sorted(all_skills),
            "combined_capabilities": sorted(all_capabilities),
            "loop_config": primary_expert.get("loop_config", DEFAULT_LOOP_CONFIG),
            "pipeline": [
                {"step": 1, "action": "Download competition data", "tool": "kaggle-mcp"},
                {"step": 2, "action": "EDA and data understanding", "skills": ["kaggle-preprocessor"]},
                {"step": 3, "action": "Feature engineering", "skills": ["kaggle-feature-engineer"]},
                {"step": 4, "action": "Build baseline model", "skills": ["kaggle-model-trainer"]},
                {"step": 5, "action": "Iterative improvement (convergence loop)", "config": primary_expert.get("loop_config", {})},
                {"step": 6, "action": "Final submission", "tool": "kaggle-mcp"},
            ],
        }
