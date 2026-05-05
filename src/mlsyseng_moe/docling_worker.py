"""Background PDF extraction using docling for ML Principles chapters."""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Chapter, Expert, MoEDatabase

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "regularization", "overfitting",
    "underfitting", "cross-validation", "bias-variance", "ensemble",
    "bagging", "boosting", "random forest", "support vector", "svm",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention", "embedding", "feature engineering",
    "dimensionality reduction", "pca", "clustering", "k-means",
    "decision tree", "logistic regression", "linear regression",
    "loss function", "optimization", "hyperparameter", "batch normalization",
    "dropout", "activation function", "relu", "sigmoid", "softmax",
    "learning rate", "momentum", "adam", "sgd", "generalization",
    "train/test split", "data augmentation", "transfer learning",
    "fine-tuning", "pre-training", "bayesian", "probabilistic",
    "maximum likelihood", "map estimation", "kernel", "gaussian process",
    "reinforcement learning", "policy gradient", "q-learning",
    "natural language processing", "nlp", "computer vision",
    "recommendation system", "time series", "autoencoder", "gan",
    "variational", "mixture of experts", "distillation",
    "quantization", "pruning", "model compression", "federated learning",
]

KAGGLE_SKILL_MAP = {
    "data preprocessing": ["kaggle-preprocessor", "kaggle-data-cleaner"],
    "feature engineering": ["kaggle-feature-engineer"],
    "model training": ["kaggle-model-trainer", "kaggle-hypertuner"],
    "ensemble methods": ["kaggle-ensemble-builder"],
    "deep learning": ["kaggle-deep-trainer", "kaggle-gpu-manager"],
    "nlp": ["kaggle-nlp-pipeline", "kaggle-text-preprocessor"],
    "computer vision": ["kaggle-cv-pipeline", "kaggle-augmentation"],
    "time series": ["kaggle-timeseries-pipeline"],
    "evaluation": ["kaggle-evaluator", "kaggle-cv-scorer"],
    "submission": ["kaggle-submitter"],
    "eda": ["kaggle-eda-explorer", "kaggle-visualizer"],
}


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling. Falls back to basic extraction on import failure."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, attempting fallback extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Basic PDF text extraction using PyPDF2 or pdfminer as fallback."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        pass

    logger.error(f"No PDF extraction library available for {pdf_path}")
    return ""


def extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts present in text."""
    text_lower = text.lower()
    found = []
    for concept in ML_CONCEPT_KEYWORDS:
        if concept in text_lower:
            found.append(concept)
    return sorted(set(found))


def infer_skills(concepts: List[str]) -> List[str]:
    """Map detected concepts to Kaggle skill paths."""
    skills = set()
    concept_categories = {
        "data preprocessing": ["feature engineering", "dimensionality reduction", "pca", "data augmentation"],
        "feature engineering": ["feature engineering", "embedding"],
        "model training": ["gradient descent", "backpropagation", "optimization",
                          "learning rate", "momentum", "adam", "sgd",
                          "logistic regression", "linear regression", "svm", "support vector"],
        "ensemble methods": ["ensemble", "bagging", "boosting", "random forest"],
        "deep learning": ["neural network", "deep learning", "convolutional", "recurrent",
                         "transformer", "attention", "batch normalization", "dropout",
                         "activation function", "relu", "sigmoid", "softmax",
                         "autoencoder", "gan", "variational"],
        "nlp": ["natural language processing", "nlp"],
        "computer vision": ["computer vision", "convolutional"],
        "time series": ["time series", "recurrent"],
        "evaluation": ["cross-validation", "bias-variance", "overfitting",
                       "underfitting", "generalization", "train/test split"],
        "eda": ["clustering", "k-means"],
    }
    for category, keywords in concept_categories.items():
        if any(c in concepts for c in keywords):
            for skill in KAGGLE_SKILL_MAP.get(category, []):
                skills.add(skill)
    skills.update(KAGGLE_SKILL_MAP.get("submission", []))
    return sorted(skills)


def _chapter_id_from_folder(folder_name: str) -> str:
    """Generate a stable chapter ID from folder name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", folder_name).strip("_").lower()
    return cleaned


def _slug_from_title(title: str) -> str:
    """Generate a slug from a chapter title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return slug


def _build_formula(concepts: List[str]) -> Dict[str, Any]:
    """Build a formula definition based on detected concepts."""
    metrics = ["accuracy"]
    if "cross-validation" in concepts:
        metrics.append("cv_score")
    if any(c in concepts for c in ["loss function", "gradient descent", "optimization"]):
        metrics.append("loss")
    if any(c in concepts for c in ["nlp", "natural language processing"]):
        metrics.extend(["f1_score", "precision", "recall"])

    return {
        "objective": "minimize_validation_loss",
        "function": "L = f(X, θ, α)",
        "metrics": sorted(set(metrics)),
    }


def _build_loop_config() -> Dict[str, Any]:
    return {
        "objective": "minimize_validation_loss",
        "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
        "epsilon": 0.001,
        "max_iterations": 10,
        "patience": 3,
    }


def scan_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return []

    chapters = []
    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append({
                    "folder_name": item.name,
                    "folder_path": str(item),
                    "pdf_paths": [str(p) for p in pdfs],
                })
    return chapters


def extract_chapter(
    folder_info: Dict[str, str],
    db: MoEDatabase,
    force: bool = False,
) -> Chapter:
    """Extract content from a chapter folder's PDFs and store in the database."""
    folder_name = folder_info["folder_name"]
    chapter_id = _chapter_id_from_folder(folder_name)

    existing = db.get_chapter(chapter_id)
    if existing and existing.status == "done" and not force:
        logger.info(f"Chapter {chapter_id} already extracted, skipping")
        return existing

    pdf_paths = folder_info.get("pdf_paths", [])
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    all_text = []
    for pdf_path in pdf_paths:
        logger.info(f"Extracting {pdf_path}")
        text = _extract_pdf_with_docling(pdf_path)
        if text:
            all_text.append(text)

    content_md = "\n\n---\n\n".join(all_text) if all_text else ""
    concepts = extract_concepts(content_md)

    title = re.sub(r"^\d+[_\s.-]*", "", folder_name).replace("_", " ").strip()
    if not title:
        title = folder_name

    chapter = Chapter(
        chapter_id=chapter_id,
        title=title,
        folder_path=folder_info["folder_path"],
        content_md=content_md,
        concepts=json.dumps(concepts),
        extracted_at=time.time(),
        status="done" if content_md else "error",
    )
    db.upsert_chapter(chapter)
    return chapter


def create_expert_from_chapter(chapter: Chapter, db: MoEDatabase, skills_base_path: Optional[str] = None) -> Expert:
    """Create an expert definition from an extracted chapter."""
    concepts = chapter.concept_list
    skills_raw = infer_skills(concepts)

    skills_path = skills_base_path or os.environ.get("KAGGLE_SKILLS_PATH", os.path.expanduser("~/skills"))
    skill_paths = [f"{skills_path}/{s}" for s in skills_raw]

    capabilities = []
    if any(c in concepts for c in ["gradient descent", "optimization", "learning rate"]):
        capabilities.append("Systematic hyperparameter search")
    if any(c in concepts for c in ["neural network", "deep learning"]):
        capabilities.append("Deep learning model architecture design")
    if any(c in concepts for c in ["ensemble", "bagging", "boosting"]):
        capabilities.append("Ensemble model construction")
    if any(c in concepts for c in ["feature engineering", "dimensionality reduction"]):
        capabilities.append("Feature engineering and selection")
    if any(c in concepts for c in ["cross-validation", "bias-variance"]):
        capabilities.append("Robust model evaluation")
    if any(c in concepts for c in ["regularization", "dropout"]):
        capabilities.append("Regularization strategy selection")
    if any(c in concepts for c in ["transformer", "attention"]):
        capabilities.append("Attention-based model design")
    if not capabilities:
        capabilities.append("Build baseline models quickly")

    expert = Expert(
        expert_name=chapter.title,
        slug=_slug_from_title(chapter.title),
        chapter_id=chapter.chapter_id,
        capabilities=json.dumps(capabilities),
        skills=json.dumps(skill_paths),
        strategy="Baseline → EDA → Feature Engineering → Model Selection → Submit",
        formula=json.dumps(_build_formula(concepts)),
        loop_config=json.dumps(_build_loop_config()),
        created_at=time.time(),
    )
    db.upsert_expert(expert)
    return expert


def run_full_extraction(
    db: MoEDatabase,
    base_path: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """Run full extraction pipeline: scan → extract PDFs → create experts."""
    chapters_info = scan_chapters(base_path)
    if not chapters_info:
        return {
            "status": "no_chapters_found",
            "path_searched": base_path or ML_PRINCIPLES_PATH,
            "chapters_processed": 0,
            "experts_created": 0,
        }

    chapters_processed = 0
    experts_created = 0
    errors = []

    for info in chapters_info:
        try:
            chapter = extract_chapter(info, db, force=force_reindex)
            chapters_processed += 1
            if chapter.status == "done":
                expert = create_expert_from_chapter(chapter, db)
                experts_created += 1
        except Exception as e:
            logger.error(f"Error processing {info['folder_name']}: {e}")
            errors.append({"folder": info["folder_name"], "error": str(e)})

    return {
        "status": "complete",
        "path_searched": base_path or ML_PRINCIPLES_PATH,
        "chapters_processed": chapters_processed,
        "experts_created": experts_created,
        "errors": errors,
    }
