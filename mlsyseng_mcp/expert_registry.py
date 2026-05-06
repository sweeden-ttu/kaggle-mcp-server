"""Expert registry for the MoE system.

Each ML Principles chapter becomes an expert with capabilities, skills,
strategy, formula, and loop configuration.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH",
    os.path.expanduser("~/skills"),
)

DEFAULT_EXPERTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experts")

CONCEPT_TO_SKILLS = {
    "neural network": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "deep learning": ["kaggle-model-trainer", "kaggle-deep-learning"],
    "gradient descent": ["kaggle-model-trainer", "kaggle-optimizer"],
    "backpropagation": ["kaggle-model-trainer"],
    "regularization": ["kaggle-model-trainer", "kaggle-regularizer"],
    "cross-validation": ["kaggle-evaluator", "kaggle-cv-splitter"],
    "ensemble": ["kaggle-ensembler", "kaggle-model-trainer"],
    "random forest": ["kaggle-model-trainer"],
    "decision tree": ["kaggle-model-trainer"],
    "logistic regression": ["kaggle-model-trainer"],
    "linear regression": ["kaggle-model-trainer"],
    "dimensionality reduction": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "pca": ["kaggle-preprocessor", "kaggle-feature-engineer"],
    "clustering": ["kaggle-model-trainer", "kaggle-unsupervised"],
    "feature engineering": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "feature selection": ["kaggle-feature-engineer"],
    "data augmentation": ["kaggle-augmenter"],
    "transfer learning": ["kaggle-model-trainer", "kaggle-transfer-learning"],
    "embedding": ["kaggle-embedder", "kaggle-nlp-preprocessor"],
    "natural language processing": ["kaggle-nlp-preprocessor", "kaggle-tokenizer"],
    "nlp": ["kaggle-nlp-preprocessor", "kaggle-tokenizer"],
    "computer vision": ["kaggle-cv-preprocessor", "kaggle-image-augmenter"],
    "classification": ["kaggle-model-trainer", "kaggle-classifier"],
    "regression": ["kaggle-model-trainer", "kaggle-regressor"],
    "anomaly detection": ["kaggle-anomaly-detector"],
    "time series": ["kaggle-timeseries-preprocessor", "kaggle-model-trainer"],
    "hyperparameter": ["kaggle-hypertuner", "kaggle-model-trainer"],
    "optimizer": ["kaggle-optimizer", "kaggle-model-trainer"],
    "transformer": ["kaggle-model-trainer", "kaggle-transformer-trainer"],
    "attention mechanism": ["kaggle-model-trainer", "kaggle-transformer-trainer"],
    "bayesian": ["kaggle-bayesian-optimizer"],
    "precision": ["kaggle-evaluator"],
    "recall": ["kaggle-evaluator"],
    "f1 score": ["kaggle-evaluator"],
    "auc": ["kaggle-evaluator"],
}

DEFAULT_STRATEGY = "Baseline → EDA → Feature Engineering → Model Selection → Hyperparameter Tuning → Ensemble → Submit"

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


def _slugify(name: str) -> str:
    """Convert a chapter name to a slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug)
    slug = re.sub(r"^[\d_]+", "", slug).strip("_")
    chap_num = re.match(r"(\d+)", name)
    prefix = chap_num.group(1).zfill(2) + "_" if chap_num else ""
    return prefix + slug if slug else prefix.rstrip("_")


def infer_skills_from_concepts(
    concepts: List[str],
    skills_base_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
) -> List[str]:
    """Map extracted concepts to Kaggle skill paths."""
    skill_set = set()
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in CONCEPT_TO_SKILLS:
            for skill_name in CONCEPT_TO_SKILLS[concept_lower]:
                skill_path = os.path.join(skills_base_path, skill_name)
                skill_set.add(skill_path)

    return sorted(skill_set)


def build_expert_definition(
    chapter_id: str,
    title: str,
    concepts: List[str],
    skills_base_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
    strategy: Optional[str] = None,
    formula: Optional[Dict[str, Any]] = None,
    loop_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete expert definition from chapter data."""
    slug = _slugify(title)
    skills = infer_skills_from_concepts(concepts, skills_base_path)

    capabilities = []
    if any(c in concepts for c in ["deep learning", "neural network", "transformer"]):
        capabilities.append("Build and train deep learning models")
    if any(c in concepts for c in ["feature engineering", "feature selection", "pca"]):
        capabilities.append("Feature engineering and dimensionality reduction")
    if any(c in concepts for c in ["ensemble", "random forest"]):
        capabilities.append("Ensemble methods and model combination")
    if any(c in concepts for c in ["cross-validation", "precision", "recall", "f1 score"]):
        capabilities.append("Model evaluation and validation")
    if any(c in concepts for c in ["hyperparameter", "optimizer", "learning rate"]):
        capabilities.append("Hyperparameter optimization")
    if any(c in concepts for c in ["regularization", "overfitting", "dropout"]):
        capabilities.append("Regularization and preventing overfitting")
    if any(c in concepts for c in ["nlp", "natural language processing", "tokenization"]):
        capabilities.append("Natural language processing pipelines")
    if any(c in concepts for c in ["computer vision", "cnn", "convolutional"]):
        capabilities.append("Computer vision and image processing")
    if any(c in concepts for c in ["bayesian", "gaussian process"]):
        capabilities.append("Bayesian methods and probabilistic modeling")
    if any(c in concepts for c in ["clustering", "k-means"]):
        capabilities.append("Unsupervised learning and clustering")

    if not capabilities:
        capabilities.append("Build baseline models quickly")
        capabilities.append("Systematic hyperparameter search")

    return {
        "expert_name": title,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": capabilities,
        "skills": skills,
        "strategy": strategy or DEFAULT_STRATEGY,
        "formula": formula or DEFAULT_FORMULA,
        "loop_config": loop_config or DEFAULT_LOOP_CONFIG,
    }


def register_experts_from_chapters(
    chapters: List[Dict[str, Any]],
    db=None,
    skills_base_path: str = DEFAULT_KAGGLE_SKILLS_PATH,
) -> List[Dict[str, Any]]:
    """Create expert definitions from extracted chapters and store them.

    Args:
        chapters: List of chapter dicts (from database or extraction).
        db: MLSysEngDatabase instance.
        skills_base_path: Base path for Kaggle skills.

    Returns:
        List of expert definition dicts.
    """
    experts = []
    for chapter in chapters:
        chapter_id = chapter.get("chapter_id", "")
        title = chapter.get("title", "Unknown")
        concepts = chapter.get("concepts", [])
        if isinstance(concepts, str):
            try:
                concepts = json.loads(concepts)
            except (json.JSONDecodeError, TypeError):
                concepts = []

        expert_def = build_expert_definition(
            chapter_id=chapter_id,
            title=title,
            concepts=concepts,
            skills_base_path=skills_base_path,
        )

        if db:
            db.upsert_expert(
                expert_name=expert_def["expert_name"],
                slug=expert_def["slug"],
                chapter_id=chapter_id,
                capabilities=expert_def["capabilities"],
                skills=expert_def["skills"],
                strategy=expert_def["strategy"],
                formula=expert_def["formula"],
                loop_config=expert_def["loop_config"],
            )

        experts.append(expert_def)

    return experts


def save_expert_json(expert: Dict[str, Any], experts_dir: Optional[str] = None):
    """Save an expert definition as a JSON file."""
    base = Path(experts_dir or DEFAULT_EXPERTS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    slug = expert.get("slug", "unknown")
    path = base / f"{slug}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expert, f, indent=2)
    return str(path)


def load_expert_json(slug: str, experts_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load an expert definition from JSON file."""
    base = Path(experts_dir or DEFAULT_EXPERTS_DIR)
    path = base / f"{slug}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_experts_for_competition(
    competition_description: str,
    all_experts: List[Dict[str, Any]],
    embedding_store=None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Select the most relevant experts for a competition using RAG.

    If an embedding store is available, uses semantic search. Otherwise
    does a simple keyword overlap.
    """
    if embedding_store:
        hits = embedding_store.search(competition_description, n_results=top_k * 2)
        relevant_chapters = set()
        for hit in hits:
            relevant_chapters.add(hit.get("chapter_id", ""))

        selected = [
            e for e in all_experts if e.get("chapter_id") in relevant_chapters
        ]
        return selected[:top_k] if selected else all_experts[:top_k]

    desc_lower = competition_description.lower()
    scored = []
    for expert in all_experts:
        capabilities = expert.get("capabilities", [])
        concepts_text = " ".join(capabilities).lower()
        overlap = sum(1 for word in desc_lower.split() if word in concepts_text)
        scored.append((overlap, expert))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
