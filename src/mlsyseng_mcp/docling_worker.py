"""Background PDF extraction worker using docling.

Scans chapter folders for PDFs, extracts text and structural data,
and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.path.expanduser(
    "~/Desktop/Machine Learning Principles - Chapters"
)

CONCEPT_CATEGORIES = {
    "optimization": [
        "gradient descent", "sgd", "adam", "learning rate", "convergence",
        "loss function", "objective", "backpropagation", "optimizer",
    ],
    "model_architecture": [
        "neural network", "layer", "activation", "transformer", "attention",
        "cnn", "rnn", "lstm", "encoder", "decoder", "embedding",
    ],
    "regularization": [
        "dropout", "batch norm", "weight decay", "l1", "l2",
        "early stopping", "data augmentation",
    ],
    "evaluation": [
        "accuracy", "precision", "recall", "f1", "auc", "roc",
        "cross-validation", "confusion matrix", "metric",
    ],
    "data": [
        "feature engineering", "preprocessing", "normalization", "scaling",
        "missing values", "imputation", "encoding", "pca", "dimensionality",
    ],
    "ensemble": [
        "random forest", "boosting", "bagging", "xgboost", "lightgbm",
        "stacking", "blending", "voting",
    ],
}


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find chapter directories and their PDFs under the base path."""
    base = Path(base_path or os.environ.get(
        "ML_PRINCIPLES_PATH", DEFAULT_ML_PRINCIPLES_PATH))
    chapters = []
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    chapter_pattern = re.compile(r"(\d+)[_\s\-]+(.+)", re.IGNORECASE)
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        match = chapter_pattern.match(entry.name)
        if not match:
            continue
        chapter_num = int(match.group(1))
        title = match.group(2).replace("_", " ").strip()
        pdfs = list(entry.glob("*.pdf"))
        if pdfs:
            chapters.append({
                "chapter_num": chapter_num,
                "title": title,
                "path": str(entry),
                "pdfs": [str(p) for p in pdfs],
            })
    return chapters


def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    """Extract markdown text from a PDF using docling, with fallback to PyPDF2.

    Returns (markdown_text, page_count).
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 0
        return md, page_count
    except ImportError:
        logger.info("docling not available, trying PyPDF2 fallback")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        import PyPDF2
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts), page_count
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfplumber fallback")
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts), page_count
    except ImportError:
        logger.warning("No PDF extraction library available (docling, PyPDF2, pdfplumber)")
        return "", 0
    except Exception as e:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, e)
        return "", 0


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """Identify ML/AI concepts from extracted text using keyword matching."""
    if not text:
        return []

    text_lower = text.lower()
    found = []
    seen = set()

    for category, keywords in CONCEPT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text_lower and keyword not in seen:
                seen.add(keyword)
                start = text_lower.find(keyword)
                ctx_start = max(0, start - 100)
                ctx_end = min(len(text), start + len(keyword) + 200)
                context = text[ctx_start:ctx_end].strip()
                context = re.sub(r"\s+", " ", context)
                found.append({
                    "concept": keyword,
                    "description": context[:300],
                    "category": category,
                })
    return found


def process_chapter(chapter_info: Dict[str, Any],
                    progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """Extract content from all PDFs in a chapter directory.

    Returns a dict with markdown_content, concepts, and page_count.
    """
    all_text = []
    total_pages = 0

    for pdf_path in chapter_info["pdfs"]:
        if progress_callback:
            progress_callback(f"Extracting {Path(pdf_path).name}...")
        text, pages = extract_pdf_text(pdf_path)
        if text:
            all_text.append(text)
            total_pages += pages

    combined = "\n\n---\n\n".join(all_text)
    concepts = extract_concepts(combined)

    return {
        "markdown_content": combined,
        "concepts": concepts,
        "page_count": total_pages,
        "word_count": len(combined.split()) if combined else 0,
    }
