"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "gradient", "loss", "optimization", "regularization", "neural network",
    "decision tree", "ensemble", "boosting", "bagging", "cross-validation",
    "bias", "variance", "overfitting", "underfitting", "feature",
    "hyperparameter", "kernel", "embedding", "attention", "transformer",
    "convolution", "recurrent", "reinforcement", "bayesian", "clustering",
    "dimensionality", "classification", "regression", "generalization",
    "backpropagation", "activation", "dropout", "batch normalization",
    "learning rate", "momentum", "stochastic", "mini-batch",
    "precision", "recall", "f1", "auc", "roc", "accuracy",
    "supervised", "unsupervised", "semi-supervised", "self-supervised",
    "transfer learning", "fine-tuning", "pre-training",
    "autoencoder", "gan", "diffusion", "mixture of experts",
]


def _try_docling_extract(pdf_path: str) -> str:
    """Extract text from a PDF using docling, with fallback to PyPDF2/pdfplumber."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, trying PyPDF2")
    except Exception as exc:
        logger.warning("docling extraction failed for %s: %s", pdf_path, exc)

    try:
        import PyPDF2
        text_parts: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfplumber")
    except Exception as exc:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, exc)

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n\n".join(text_parts)
    except ImportError:
        raise RuntimeError(
            "No PDF extraction library available. Install docling, PyPDF2, or pdfplumber."
        )


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text by keyword matching."""
    text_lower = text.lower()
    found = []
    for kw in CONCEPT_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return sorted(set(found))


def parse_chapter_title(folder_name: str) -> str:
    """Derive a human-readable title from a chapter folder name.

    Examples:
        '08_ML_Systems'  -> 'ML Systems'
        '03 - Neural Networks' -> 'Neural Networks'
    """
    cleaned = re.sub(r"^\d+[\s_]*[-–]?\s*", "", folder_name)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned or folder_name


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters: List[Dict[str, Any]] = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        chapters.append({
            "folder_name": entry.name,
            "title": parse_chapter_title(entry.name),
            "pdf_path": str(pdfs[0]),
            "folder_path": str(entry),
        })

    return chapters


def extract_chapter(pdf_path: str) -> Tuple[str, List[str]]:
    """Extract content and concepts from a single chapter PDF.

    Returns:
        (markdown_content, list_of_concepts)
    """
    content = _try_docling_extract(pdf_path)
    concepts = extract_concepts(content)
    return content, concepts


def extract_all_chapters(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    db: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract all chapters from the ML Principles directory.

    Args:
        base_path: Override for ML_PRINCIPLES_PATH
        force_reindex: Re-extract even if already in the database
        db: Optional Database instance for persistence

    Returns:
        List of dicts with extraction results per chapter
    """
    chapters = discover_chapters(base_path)
    results: List[Dict[str, Any]] = []

    for ch in chapters:
        folder = ch["folder_name"]
        if db and not force_reindex:
            existing = db.get_chapter(folder)
            if existing and existing.get("content_md"):
                results.append({
                    "folder_name": folder,
                    "title": ch["title"],
                    "status": "skipped",
                    "concepts": existing.get("concepts", []),
                })
                continue

        try:
            content, concepts = extract_chapter(ch["pdf_path"])
            if db:
                chapter_id = db.upsert_chapter(
                    folder_name=folder,
                    title=ch["title"],
                    content_md=content,
                    concepts=concepts,
                    pdf_path=ch["pdf_path"],
                )
                db.log_extraction(chapter_id, "completed")

            results.append({
                "folder_name": folder,
                "title": ch["title"],
                "status": "extracted",
                "concepts": concepts,
                "content_length": len(content),
            })
        except Exception as exc:
            logger.error("Failed to extract %s: %s", folder, exc)
            if db:
                chapter_id = db.upsert_chapter(
                    folder_name=folder,
                    title=ch["title"],
                    content_md="",
                    concepts=[],
                    pdf_path=ch["pdf_path"],
                )
                db.log_extraction(chapter_id, "failed", str(exc))

            results.append({
                "folder_name": folder,
                "title": ch["title"],
                "status": "failed",
                "error": str(exc),
            })

    return results
