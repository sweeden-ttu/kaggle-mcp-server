"""Expert registry: manages chapter experts with skills, strategy, and formulas."""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

EXPERTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experts")
KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

DEFAULT_STRATEGY = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Submit"

SKILL_TEMPLATES = [
    "kaggle-preprocessor",
    "kaggle-model-trainer",
    "kaggle-feature-engineer",
    "kaggle-eda",
    "kaggle-submitter",
]


def _slug_from_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def build_expert_from_chapter(
    chapter: Dict[str, Any],
    chapter_id: int,
    skills_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an expert definition from an extracted chapter."""
    skills_base = skills_path or KAGGLE_SKILLS_PATH
    name = chapter["title"]
    slug = _slug_from_name(name)
    concepts = chapter.get("concepts", [])

    capabilities = []
    if any(c for c in concepts if "gradient" in c or "optimization" in c or "sgd" in c or "adam" in c):
        capabilities.append("Optimize model training with advanced optimizers")
    if any(c for c in concepts if "neural" in c or "deep learning" in c):
        capabilities.append("Build deep learning architectures")
    if any(c for c in concepts if "tree" in c or "forest" in c or "boost" in c):
        capabilities.append("Implement tree-based ensemble methods")
    if any(c for c in concepts if "feature" in c or "pca" in c):
        capabilities.append("Perform feature engineering and selection")
    if any(c for c in concepts if "regularization" in c or "dropout" in c):
        capabilities.append("Apply regularization techniques to prevent overfitting")
    if any(c for c in concepts if "cross" in c and "validation" in c or "k-fold" in c):
        capabilities.append("Implement robust cross-validation strategies")
    if any(c for c in concepts if "nlp" in c or "tokenization" in c or "embedding" in c):
        capabilities.append("Process and model natural language data")
    if any(c for c in concepts if "clustering" in c or "k-means" in c):
        capabilities.append("Perform unsupervised clustering analysis")
    if not capabilities:
        capabilities = [
            "Build baseline models quickly",
            "Systematic hyperparameter search",
        ]

    skills = [
        os.path.join(skills_base, t)
        for t in SKILL_TEMPLATES
        if os.path.isdir(os.path.join(skills_base, t))
    ]
    if not skills:
        skills = [os.path.join(skills_base, t) for t in SKILL_TEMPLATES[:2]]

    formula_metrics = ["accuracy", "f1_score"]
    if any(c for c in concepts if "regression" in c):
        formula_metrics = ["rmse", "mae", "r2_score"]

    return {
        "expert_name": name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": DEFAULT_STRATEGY,
        "formula": {
            "objective": "minimize_validation_loss",
            "function": "L = f(X, theta, alpha)",
            "metrics": formula_metrics,
        },
        "loop_config": {**DEFAULT_LOOP_CONFIG},
    }


def save_expert_json(expert: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Save expert definition as a JSON file in the experts directory."""
    base = Path(output_dir or EXPERTS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{expert['slug']}.json"
    with open(path, "w") as f:
        json.dump(expert, f, indent=2)
    return str(path)


def load_expert_json(slug: str, experts_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    base = Path(experts_dir or EXPERTS_DIR)
    path = base / f"{slug}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_experts_from_disk(experts_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    base = Path(experts_dir or EXPERTS_DIR)
    if not base.exists():
        return []
    experts = []
    for p in sorted(base.glob("*.json")):
        with open(p) as f:
            experts.append(json.load(f))
    return experts
