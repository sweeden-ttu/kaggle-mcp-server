"""Expert management with skills, strategy, and mathematical formulas.

Each ML Principles chapter becomes an expert with:
- Capabilities derived from extracted concepts
- Skills mapping to Kaggle competition tasks
- Strategy for competition workflow
- Mathematical formula (objective function)
- Loop configuration for state convergence
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CHAPTER_SKILL_MAP = {
    "introduction": {
        "capabilities": [
            "Build baseline models quickly",
            "Frame ML problems correctly",
            "Select appropriate evaluation metrics",
        ],
        "skills": ["kaggle-baseline-builder", "kaggle-eda-explorer"],
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = (1/n) * sum(loss(y_i, f(x_i)))",
            "metrics": ["accuracy", "rmse"],
        },
    },
    "supervised": {
        "capabilities": [
            "Train classification and regression models",
            "Implement cross-validation pipelines",
            "Tune hyperparameters systematically",
        ],
        "skills": ["kaggle-model-trainer", "kaggle-hyperparameter-tuner"],
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = -sum(y*log(p) + (1-y)*log(1-p))",
            "metrics": ["accuracy", "f1_score", "auc"],
        },
    },
    "unsupervised": {
        "capabilities": [
            "Perform clustering analysis",
            "Reduce dimensionality for visualization",
            "Detect anomalies and outliers",
        ],
        "skills": ["kaggle-preprocessor", "kaggle-feature-engineer"],
        "formula": {
            "objective": "minimize_reconstruction_error",
            "function": "L = sum(||x - decode(encode(x))||^2)",
            "metrics": ["silhouette_score", "inertia"],
        },
    },
    "neural": {
        "capabilities": [
            "Design neural network architectures",
            "Implement backpropagation pipelines",
            "Apply appropriate activation functions",
        ],
        "skills": ["kaggle-model-trainer", "kaggle-deep-learning"],
        "formula": {
            "objective": "minimize_cross_entropy",
            "function": "L = -sum(y*log(softmax(Wx+b)))",
            "metrics": ["accuracy", "loss"],
        },
    },
    "deep": {
        "capabilities": [
            "Build CNN/RNN/Transformer architectures",
            "Apply transfer learning and fine-tuning",
            "Implement attention mechanisms",
        ],
        "skills": ["kaggle-deep-learning", "kaggle-model-trainer"],
        "formula": {
            "objective": "minimize_task_loss",
            "function": "L = L_task + lambda * L_regularization",
            "metrics": ["accuracy", "f1_score", "perplexity"],
        },
    },
    "regularization": {
        "capabilities": [
            "Prevent overfitting with regularization",
            "Optimize training with adaptive methods",
            "Schedule learning rates effectively",
        ],
        "skills": ["kaggle-hyperparameter-tuner", "kaggle-model-trainer"],
        "formula": {
            "objective": "minimize_regularized_loss",
            "function": "L = L_data + alpha*||w||_1 + beta*||w||_2^2",
            "metrics": ["validation_loss", "generalization_gap"],
        },
    },
    "feature": {
        "capabilities": [
            "Engineer informative features",
            "Select relevant feature subsets",
            "Handle missing data and encoding",
        ],
        "skills": ["kaggle-preprocessor", "kaggle-feature-engineer"],
        "formula": {
            "objective": "maximize_feature_importance",
            "function": "I(f) = H(Y) - H(Y|f)",
            "metrics": ["mutual_information", "feature_importance"],
        },
    },
    "system": {
        "capabilities": [
            "Build end-to-end ML pipelines",
            "Systematic hyperparameter search",
            "Production model deployment patterns",
        ],
        "skills": ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-submitter"],
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, theta, alpha)",
            "metrics": ["accuracy", "f1_score", "latency"],
        },
    },
    "ensemble": {
        "capabilities": [
            "Build bagging and boosting ensembles",
            "Stack multiple model predictions",
            "Blend diverse model outputs",
        ],
        "skills": ["kaggle-model-trainer", "kaggle-ensemble-builder"],
        "formula": {
            "objective": "minimize_ensemble_loss",
            "function": "L = sum(w_i * L_i) subject to sum(w_i)=1",
            "metrics": ["accuracy", "f1_score", "auc"],
        },
    },
    "evaluation": {
        "capabilities": [
            "Design robust evaluation protocols",
            "Implement statistical significance tests",
            "Build confusion matrices and ROC curves",
        ],
        "skills": ["kaggle-evaluator", "kaggle-model-trainer"],
        "formula": {
            "objective": "maximize_evaluation_metric",
            "function": "F1 = 2*P*R/(P+R)",
            "metrics": ["precision", "recall", "f1_score", "auc"],
        },
    },
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug


def _match_chapter_type(chapter_name: str) -> str:
    """Match a chapter name to a type key in CHAPTER_SKILL_MAP."""
    name_lower = chapter_name.lower()
    priority_order = [
        "unsupervised", "supervised", "regularization", "ensemble",
        "evaluation", "feature", "deep", "neural", "introduction", "system",
    ]
    for key in priority_order:
        if key in name_lower and key in CHAPTER_SKILL_MAP:
            return key
    for key in CHAPTER_SKILL_MAP:
        if key in name_lower:
            return key
    return "system"


class ExpertRegistry:
    """Manages expert registration and retrieval."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.kaggle_skills_path = os.environ.get(
            "KAGGLE_SKILLS_PATH",
            os.path.expanduser("~/skills"),
        )

    def register_from_chapters(self) -> Dict[str, Any]:
        """Register experts from all extracted chapters."""
        chapters = self.db.list_chapters()
        registered = 0
        updated = 0

        for chapter in chapters:
            if chapter["status"] != "extracted":
                continue

            chapter_name = chapter["chapter_name"]
            slug = _slugify(chapter_name)
            chapter_type = _match_chapter_type(chapter_name)
            mapping = CHAPTER_SKILL_MAP.get(chapter_type, CHAPTER_SKILL_MAP["system"])

            skills_paths = [
                os.path.join(self.kaggle_skills_path, s)
                for s in mapping["skills"]
            ]

            expert_def = {
                "expert_name": chapter_name,
                "slug": slug,
                "chapter_id": chapter["id"],
                "capabilities": mapping["capabilities"],
                "skills": skills_paths,
                "strategy": DEFAULT_STRATEGY,
                "formula": mapping["formula"],
                "loop_config": DEFAULT_LOOP_CONFIG,
            }

            existing = self.db.get_expert(slug)
            self.db.upsert_expert(expert_def)
            if existing:
                updated += 1
            else:
                registered += 1

        return {
            "registered": registered,
            "updated": updated,
            "total_chapters": len(chapters),
        }

    def get_expert(self, slug: str) -> Optional[Dict[str, Any]]:
        return self.db.get_expert(slug)

    def list_experts(self) -> List[Dict[str, Any]]:
        return self.db.list_experts()

    def get_expert_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Find experts relevant to a query based on capabilities keyword matching."""
        experts = self.list_experts()
        query_lower = query.lower()
        scored = []

        for expert in experts:
            score = 0
            for cap in expert.get("capabilities", []):
                words = set(cap.lower().split())
                query_words = set(query_lower.split())
                overlap = words & query_words
                score += len(overlap)
            if score > 0:
                scored.append((expert, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def export_expert_json(self, slug: str) -> Optional[str]:
        """Export an expert definition as JSON."""
        expert = self.db.get_expert(slug)
        if not expert:
            return None
        export = {
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "capabilities": expert["capabilities"],
            "skills": expert["skills"],
            "strategy": expert["strategy"],
            "formula": expert["formula"],
            "loop_config": expert["loop_config"],
        }
        return json.dumps(export, indent=2)

    def save_expert_definitions(self, output_dir: Optional[str] = None):
        """Save all expert definitions to JSON files in the experts directory."""
        experts_dir = Path(output_dir or os.path.join(os.path.dirname(__file__), "..", "experts"))
        experts_dir.mkdir(parents=True, exist_ok=True)

        experts = self.list_experts()
        for expert in experts:
            json_str = self.export_expert_json(expert["slug"])
            if json_str:
                filepath = experts_dir / f"{expert['slug']}.json"
                filepath.write_text(json_str)

        return {"saved": len(experts), "directory": str(experts_dir)}
