"""Docling PDF extraction worker for MLSysEng MoE.

Extracts text from ML Principles PDFs using docling (with fallback to
PyPDF2/pdfplumber) and identifies key ML/AI concepts.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .database import Chapter, ConceptEntry, Database

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "neural network", "gradient descent", "backpropagation", "loss function",
    "regularization", "overfitting", "underfitting", "bias-variance",
    "cross-validation", "ensemble", "random forest", "decision tree",
    "support vector", "kernel", "feature engineering", "dimensionality reduction",
    "principal component", "clustering", "k-means", "reinforcement learning",
    "q-learning", "policy gradient", "transformer", "attention mechanism",
    "convolutional", "recurrent", "lstm", "gru", "autoencoder",
    "generative adversarial", "variational", "bayesian", "markov",
    "optimization", "hyperparameter", "learning rate", "batch normalization",
    "dropout", "activation function", "softmax", "sigmoid", "relu",
    "embedding", "transfer learning", "fine-tuning", "data augmentation",
    "normalization", "standardization", "imputation", "feature selection",
    "model selection", "evaluation metric", "precision", "recall",
    "f1 score", "auc", "roc", "confusion matrix", "accuracy",
    "mean squared error", "cross entropy", "information gain",
    "boosting", "bagging", "stacking", "deep learning", "mlp",
    "natural language processing", "computer vision", "time series",
    "anomaly detection", "recommendation system", "collaborative filtering",
]


def _extract_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, falling back to alternative extraction")
        return _extract_fallback(pdf_path)
    except Exception as e:
        logger.warning(f"docling extraction failed for {pdf_path}: {e}")
        return _extract_fallback(pdf_path)


def _extract_fallback(pdf_path: str) -> str:
    """Fallback PDF text extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        import PyPDF2
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available. Install docling, pdfplumber, or PyPDF2."
    )
    return ""


def extract_concepts(text: str) -> list[str]:
    """Extract ML/AI concepts from text content."""
    text_lower = text.lower()
    found = []
    for keyword in CONCEPT_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return sorted(set(found))


def _parse_chapter_number(folder_name: str) -> Optional[str]:
    """Extract chapter number from folder name like '08_ML_Systems'."""
    match = re.match(r"^(\d+)", folder_name)
    if match:
        return match.group(1)
    return None


def _make_title(folder_name: str) -> str:
    """Convert folder name to a readable title."""
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return f"Chapter {int(parts[0])}: {parts[1].replace('_', ' ')}"
    return folder_name.replace("_", " ")


def scan_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return chapters

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        pdfs = list(item.glob("*.pdf"))
        if not pdfs:
            continue
        chapters.append({
            "folder_name": item.name,
            "title": _make_title(item.name),
            "pdf_path": str(pdfs[0]),
        })

    return chapters


def extract_chapter(
    pdf_path: str,
    folder_name: str,
    db: Database,
    force: bool = False,
) -> Chapter:
    """Extract content from a single chapter PDF and store in database."""
    existing = db.get_chapter_by_folder(folder_name)
    if existing and existing.status == "extracted" and not force:
        logger.info(f"Skipping already-extracted chapter: {folder_name}")
        return existing

    title = _make_title(folder_name)
    chapter = Chapter(
        folder_name=folder_name,
        title=title,
        pdf_path=pdf_path,
        status="extracting",
    )
    chapter_id = db.upsert_chapter(chapter)
    db.log_event(chapter_id, "extraction_started", f"PDF: {pdf_path}")

    content = _extract_with_docling(pdf_path)
    concepts = extract_concepts(content)

    chapter.id = chapter_id
    chapter.content_md = content
    chapter.concepts = json.dumps(concepts)
    chapter.extracted_at = datetime.utcnow().isoformat()
    chapter.status = "extracted"
    db.upsert_chapter(chapter)

    for concept in concepts:
        db.add_concept(ConceptEntry(
            chapter_id=chapter_id,
            concept=concept,
            description=f"Concept '{concept}' found in {title}",
        ))

    db.log_event(
        chapter_id, "extraction_complete",
        f"Extracted {len(content)} chars, {len(concepts)} concepts",
    )

    return chapter


def extract_all(
    base_path: Optional[str] = None,
    db: Optional[Database] = None,
    force: bool = False,
) -> dict:
    """Extract all chapters from ML Principles directory."""
    db = db or Database()
    chapters_info = scan_chapters(base_path)

    results = {"extracted": 0, "skipped": 0, "failed": 0, "chapters": []}

    for info in chapters_info:
        try:
            chapter = extract_chapter(
                pdf_path=info["pdf_path"],
                folder_name=info["folder_name"],
                db=db,
                force=force,
            )
            if chapter.status == "extracted" and not force:
                results["skipped"] += 1
            else:
                results["extracted"] += 1
            results["chapters"].append({
                "folder": info["folder_name"],
                "title": chapter.title,
                "status": chapter.status,
                "concepts": chapter.concept_list,
            })
        except Exception as e:
            logger.error(f"Failed to extract {info['folder_name']}: {e}")
            results["failed"] += 1
            results["chapters"].append({
                "folder": info["folder_name"],
                "status": "failed",
                "error": str(e),
            })

    return results
