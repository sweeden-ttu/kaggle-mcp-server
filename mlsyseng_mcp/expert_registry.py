"""Expert registry for MLSysEng MoE system.

Each ML Principles chapter becomes an expert with capabilities,
skills, strategy, formula, and loop configuration.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_KAGGLE_SKILLS_PATH = os.environ.get(
    "KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills")
)

STRATEGY_TEMPLATE = "Baseline -> EDA -> Feature Engineering -> Model Selection -> Hyperparameter Tuning -> Submit"

DEFAULT_LOOP_CONFIG = {
    "objective": "minimize_validation_loss",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 10,
    "patience": 3,
}

CONCEPT_TO_SKILLS = {
    "neural network": ["model-trainer", "deep-learning-pipeline"],
    "deep learning": ["model-trainer", "deep-learning-pipeline", "gpu-optimizer"],
    "convolutional": ["cnn-builder", "image-preprocessor"],
    "recurrent": ["rnn-builder", "sequence-preprocessor"],
    "transformer": ["transformer-builder", "attention-optimizer"],
    "gradient descent": ["optimizer-tuner", "learning-rate-scheduler"],
    "backpropagation": ["model-trainer", "gradient-analyzer"],
    "optimization": ["optimizer-tuner", "hyperparameter-search"],
    "loss function": ["loss-designer", "metric-tracker"],
    "regularization": ["regularizer", "model-trainer"],
    "dropout": ["regularizer", "model-trainer"],
    "ensemble": ["ensemble-builder", "model-stacker"],
    "bagging": ["ensemble-builder", "random-forest-trainer"],
    "boosting": ["ensemble-builder", "xgboost-trainer"],
    "random forest": ["random-forest-trainer", "feature-selector"],
    "xgboost": ["xgboost-trainer", "gradient-boosting"],
    "support vector": ["svm-trainer", "kernel-optimizer"],
    "decision tree": ["tree-builder", "rule-extractor"],
    "classification": ["classifier-trainer", "metric-tracker"],
    "regression": ["regressor-trainer", "metric-tracker"],
    "clustering": ["cluster-analyzer", "unsupervised-pipeline"],
    "feature engineering": ["feature-engineer", "preprocessor"],
    "feature selection": ["feature-selector", "importance-analyzer"],
    "dimensionality reduction": ["pca-reducer", "manifold-learner"],
    "cross validation": ["cross-validator", "model-evaluator"],
    "overfitting": ["regularizer", "early-stopper"],
    "bias variance": ["model-evaluator", "complexity-analyzer"],
    "reinforcement learning": ["rl-agent-builder", "reward-designer"],
    "natural language processing": ["nlp-pipeline", "text-preprocessor"],
    "nlp": ["nlp-pipeline", "text-preprocessor"],
    "tokenization": ["tokenizer", "text-preprocessor"],
    "embedding": ["embedding-builder", "word2vec-trainer"],
    "attention": ["attention-builder", "transformer-builder"],
    "generative adversarial": ["gan-builder", "generator-trainer"],
    "hyperparameter": ["hyperparameter-search", "optimizer-tuner"],
    "data augmentation": ["augmenter", "preprocessor"],
    "preprocessing": ["preprocessor", "data-cleaner"],
    "normalization": ["normalizer", "preprocessor"],
    "activation function": ["model-trainer", "architecture-designer"],
    "precision": ["metric-tracker", "model-evaluator"],
    "recall": ["metric-tracker", "model-evaluator"],
    "f1 score": ["metric-tracker", "model-evaluator"],
    "time series": ["time-series-pipeline", "forecaster"],
    "forecasting": ["forecaster", "time-series-pipeline"],
    "computer vision": ["cv-pipeline", "image-preprocessor"],
    "image classification": ["image-classifier", "cv-pipeline"],
    "object detection": ["object-detector", "cv-pipeline"],
    "model selection": ["model-selector", "automl-pipeline"],
    "model evaluation": ["model-evaluator", "metric-tracker"],
    "mlops": ["mlops-pipeline", "model-deployer"],
}

COMMON_SKILLS = ["kaggle-preprocessor", "kaggle-model-trainer", "kaggle-submitter"]


def _slugify(name: str) -> str:
    """Convert a name to a slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def _infer_skills_from_concepts(concepts: List[str]) -> List[str]:
    """Map concepts to skill identifiers."""
    skills = set(COMMON_SKILLS)
    for concept in concepts:
        concept_lower = concept.lower()
        for key, skill_list in CONCEPT_TO_SKILLS.items():
            if key in concept_lower:
                skills.update(skill_list)
    return sorted(skills)


def _build_formula(concepts: List[str]) -> Dict[str, Any]:
    """Build a formula specification based on chapter concepts."""
    metrics = ["accuracy"]
    for concept in concepts:
        cl = concept.lower()
        if any(t in cl for t in ("regression", "forecasting", "time series")):
            metrics = ["rmse", "mae", "r2_score"]
            break
        if any(t in cl for t in ("classification", "precision", "recall", "f1")):
            metrics = ["accuracy", "f1_score", "precision", "recall"]
            break
        if any(t in cl for t in ("clustering", "unsupervised")):
            metrics = ["silhouette_score", "calinski_harabasz"]
            break
        if any(t in cl for t in ("object detection", "segmentation")):
            metrics = ["mAP", "IoU"]
            break

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, theta, alpha)",
        "metrics": metrics,
    }


def _build_capabilities(title: str, concepts: List[str]) -> List[str]:
    """Generate capability descriptions from title and concepts."""
    capabilities = [
        f"Expert knowledge in {title}",
        "Build baseline models quickly",
        "Systematic hyperparameter search",
    ]
    if any("deep" in c.lower() for c in concepts):
        capabilities.append("Deep learning architecture design")
    if any("feature" in c.lower() for c in concepts):
        capabilities.append("Advanced feature engineering")
    if any("ensemble" in c.lower() or "boost" in c.lower() for c in concepts):
        capabilities.append("Ensemble model construction")
    if any("nlp" in c.lower() or "natural language" in c.lower() for c in concepts):
        capabilities.append("Natural language processing pipelines")
    if any("vision" in c.lower() or "image" in c.lower() or "convolutional" in c.lower() for c in concepts):
        capabilities.append("Computer vision pipelines")
    if any("time series" in c.lower() or "forecast" in c.lower() for c in concepts):
        capabilities.append("Time series analysis and forecasting")
    return capabilities


def create_expert_from_chapter(
    chapter_num: int,
    title: str,
    concepts: List[str],
    chapter_id: Optional[int] = None,
    skills_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an expert definition from chapter metadata."""
    slug = f"{chapter_num:02d}_{_slugify(title)}"
    expert_name = f"{chapter_num:02d}_{title.replace(' ', '_')}"
    skills_base = skills_base_path or DEFAULT_KAGGLE_SKILLS_PATH
    skill_names = _infer_skills_from_concepts(concepts)
    skill_paths = [os.path.join(skills_base, s) for s in skill_names]

    return {
        "expert_name": expert_name,
        "slug": slug,
        "chapter_id": chapter_id,
        "capabilities": _build_capabilities(title, concepts),
        "skills": skill_paths,
        "strategy": STRATEGY_TEMPLATE,
        "formula": _build_formula(concepts),
        "loop_config": dict(DEFAULT_LOOP_CONFIG),
    }


def register_experts_from_db(db: Database) -> List[Dict[str, Any]]:
    """Create and register experts for all indexed chapters."""
    chapters = db.list_chapters()
    results = []
    for ch in chapters:
        expert = create_expert_from_chapter(
            chapter_num=ch["chapter_num"],
            title=ch["title"],
            concepts=ch.get("concepts", []),
            chapter_id=ch["id"],
        )
        db.upsert_expert(expert)
        results.append({
            "expert_name": expert["expert_name"],
            "slug": expert["slug"],
            "skills_count": len(expert["skills"]),
            "capabilities_count": len(expert["capabilities"]),
        })
    return results


def save_expert_json(expert: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Save expert definition as JSON file."""
    out = Path(output_dir or "experts")
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{expert['slug']}.json"
    with open(filepath, "w") as f:
        json.dump(expert, f, indent=2)
    return str(filepath)


def infer_experts_for_competition(
    competition_description: str,
    db: Database,
    embedding_store=None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Use RAG to infer which experts are most relevant for a competition."""
    if embedding_store is not None:
        search_results = embedding_store.search(competition_description, n_results=top_k * 2)
        chapter_nums = set()
        for r in search_results:
            if r.get("chapter_num"):
                chapter_nums.add(r["chapter_num"])

        experts = []
        for ch_num in list(chapter_nums)[:top_k]:
            chapter = db.get_chapter(ch_num)
            if chapter:
                expert_slug = f"{ch_num:02d}_{_slugify(chapter['title'])}"
                expert = db.get_expert(expert_slug)
                if expert:
                    experts.append(expert)
        return experts

    all_experts = db.list_experts()
    desc_lower = competition_description.lower()
    scored = []
    for expert in all_experts:
        score = 0
        for cap in expert.get("capabilities", []):
            if any(word in desc_lower for word in cap.lower().split() if len(word) > 3):
                score += 1
        scored.append((score, expert))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
