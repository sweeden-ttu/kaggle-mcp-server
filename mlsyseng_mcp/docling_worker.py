"""Background PDF extraction using docling.

Scans chapter folders, extracts PDF text and structure, identifies
ML/AI concepts, and stores everything in the database.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import MoEDatabase

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPT_KEYWORDS = [
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "classification", "regression", "clustering", "dimensionality reduction",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention mechanism", "gradient descent",
    "backpropagation", "loss function", "activation function",
    "overfitting", "underfitting", "regularization", "dropout",
    "batch normalization", "cross-validation", "hyperparameter",
    "ensemble", "boosting", "bagging", "random forest",
    "decision tree", "support vector machine", "naive bayes",
    "logistic regression", "linear regression", "feature engineering",
    "feature selection", "principal component analysis", "pca",
    "optimization", "adam", "sgd", "learning rate",
    "bias-variance", "model selection", "evaluation metrics",
    "precision", "recall", "f1 score", "accuracy", "auc", "roc",
    "confusion matrix", "data augmentation", "transfer learning",
    "fine-tuning", "pre-training", "embedding", "tokenization",
    "natural language processing", "nlp", "computer vision",
    "generative model", "discriminative model", "autoencoder",
    "variational autoencoder", "gan", "diffusion model",
    "bayesian", "markov", "monte carlo", "kernel method",
    "time series", "forecasting", "anomaly detection",
    "recommendation system", "collaborative filtering",
    "preprocessing", "normalization", "standardization",
    "baseline", "benchmark", "kaggle", "competition",
    "eda", "exploratory data analysis", "visualization",
]


def _chapter_id(folder_name: str) -> str:
    return hashlib.md5(folder_name.encode()).hexdigest()[:12]


def _extract_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, attempting fallback extraction")
    except Exception as exc:
        logger.warning("docling extraction failed for %s: %s", pdf_path, exc)

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not available either")
    except Exception as exc:
        logger.warning("PyMuPDF extraction failed: %s", exc)

    return ""


def _extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts present in extracted text."""
    text_lower = text.lower()
    found = []
    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return sorted(set(found))


def _find_pdfs(folder: Path) -> List[Path]:
    """Find all PDF files in a folder (non-recursive)."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.pdf"))


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover chapter folders containing PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []
    if not base.exists():
        logger.info("ML Principles path does not exist: %s", base)
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = _find_pdfs(item)
            if pdfs:
                chapters.append({
                    "folder_name": item.name,
                    "folder_path": str(item),
                    "pdf_path": str(pdfs[0]),
                    "pdf_count": len(pdfs),
                })
        elif item.suffix.lower() == ".pdf":
            chapters.append({
                "folder_name": item.stem,
                "folder_path": str(item.parent),
                "pdf_path": str(item),
                "pdf_count": 1,
            })
    return chapters


def extract_chapter(
    chapter_info: Dict[str, str],
    db: MoEDatabase,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract a single chapter's content and store in the database."""
    folder_name = chapter_info["folder_name"]
    chapter_id = _chapter_id(folder_name)

    existing = db.get_chapter(chapter_id)
    if existing and existing["status"] == "extracted" and not force:
        return existing

    pdf_path = chapter_info["pdf_path"]
    logger.info("Extracting: %s", pdf_path)

    content_md = _extract_with_docling(pdf_path)
    concepts = _extract_concepts(content_md) if content_md else []

    chapter = db.upsert_chapter(
        chapter_id=chapter_id,
        title=folder_name,
        folder_path=chapter_info["folder_path"],
        pdf_path=pdf_path,
        content_md=content_md,
        concepts=concepts,
        status="extracted" if content_md else "failed",
    )
    return chapter


def extract_all_chapters(
    db: MoEDatabase,
    base_path: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Extract all discovered chapters."""
    chapters_info = discover_chapters(base_path)
    results = []
    for info in chapters_info:
        try:
            ch = extract_chapter(info, db, force=force)
            results.append(ch)
        except Exception as exc:
            logger.error("Failed to extract %s: %s", info["folder_name"], exc)
            results.append({
                "chapter_id": _chapter_id(info["folder_name"]),
                "title": info["folder_name"],
                "status": "error",
                "error": str(exc),
            })
    return results


def get_extraction_status(db: MoEDatabase) -> Dict[str, Any]:
    """Return current extraction progress."""
    stats = db.get_stats()
    chapters = db.list_chapters()
    statuses = {}
    for ch in chapters:
        s = ch.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    return {
        "total_chapters": stats["total_chapters"],
        "status_breakdown": statuses,
        "total_embeddings": stats["total_embeddings"],
        "total_experts": stats["total_experts"],
    }
