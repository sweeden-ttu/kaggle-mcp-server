"""Background PDF extraction worker using docling.

Extracts text and structured data from ML Principles PDF chapters,
identifies key concepts, and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import database

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "loss function", "regularization",
    "overfitting", "underfitting", "cross-validation", "bias-variance",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention mechanism", "embedding", "feature engineering",
    "ensemble", "boosting", "bagging", "random forest", "decision tree",
    "support vector", "kernel", "dimensionality reduction", "PCA",
    "clustering", "classification", "regression", "optimization",
    "stochastic", "mini-batch", "learning rate", "momentum",
    "batch normalization", "dropout", "activation function",
    "hyperparameter", "model selection", "evaluation metric",
    "precision", "recall", "f1 score", "AUC", "ROC",
    "generalization", "transfer learning", "fine-tuning",
    "data augmentation", "normalization", "standardization",
    "reinforcement learning", "policy gradient", "value function",
    "bayesian", "probabilistic", "maximum likelihood",
    "information theory", "entropy", "mutual information",
]


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover chapter folders/PDFs in the ML Principles directory.

    Returns a list of dicts with chapter_number, title, and path.
    """
    path = Path(base_path or ML_PRINCIPLES_PATH)
    if not path.exists():
        logger.warning("ML Principles path does not exist: %s", path)
        return []

    chapters = []
    chapter_pattern = re.compile(r"(\d+)[_\s-]*(.+)", re.IGNORECASE)

    for item in sorted(path.iterdir()):
        if item.is_dir():
            match = chapter_pattern.match(item.name)
            if match:
                num = int(match.group(1))
                title = match.group(2).replace("_", " ").strip()
                pdfs = list(item.glob("*.pdf"))
                if pdfs:
                    chapters.append({
                        "chapter_number": num,
                        "title": title,
                        "path": str(pdfs[0]),
                        "folder": str(item),
                    })
        elif item.suffix.lower() == ".pdf":
            match = chapter_pattern.match(item.stem)
            if match:
                num = int(match.group(1))
                title = match.group(2).replace("_", " ").strip()
                chapters.append({
                    "chapter_number": num,
                    "title": title,
                    "path": str(item),
                    "folder": str(item.parent),
                })

    return chapters


def extract_pdf_text(pdf_path: str) -> Tuple[str, str]:
    """Extract text from a PDF using docling.

    Returns (plain_text, markdown_content).
    Falls back to basic extraction if docling is unavailable.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()
        plain_text = result.document.export_to_text() if hasattr(result.document, "export_to_text") else markdown_content
        return plain_text, markdown_content
    except ImportError:
        logger.warning("docling not available, falling back to PyPDF2/pdfplumber")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> Tuple[str, str]:
    """Fallback PDF extraction when docling is not available."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = "\n\n".join(text_parts)
        return full_text, full_text
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n\n".join(text_parts)
        return full_text, full_text
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available. Install docling, pdfplumber, or PyPDF2."
    )
    return "", ""


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text using keyword matching."""
    text_lower = text.lower()
    found = []
    for keyword in CONCEPT_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return sorted(set(found))


def process_chapter(
    chapter_info: Dict[str, Any],
    force: bool = False,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single chapter: extract PDF, identify concepts, store in DB.

    Returns a status dict.
    """
    chapter_number = chapter_info["chapter_number"]
    title = chapter_info["title"]
    pdf_path = chapter_info["path"]

    if not force:
        existing = database.get_chapter(chapter_number, db_path=db_path)
        if existing and existing.get("extracted_text"):
            return {
                "chapter_number": chapter_number,
                "title": title,
                "status": "skipped",
                "reason": "already extracted",
            }

    logger.info("Extracting chapter %d: %s", chapter_number, title)

    chapter_id = database.store_chapter(
        chapter_number=chapter_number,
        title=title,
        source_path=pdf_path,
        extracted_text="",
        markdown_content="",
        concepts=[],
        db_path=db_path,
    )
    database.record_extraction(chapter_id, "running", db_path=db_path)

    try:
        plain_text, markdown_content = extract_pdf_text(pdf_path)
        if not plain_text:
            raise ValueError(f"No text extracted from {pdf_path}")

        concepts = extract_concepts(plain_text)

        database.store_chapter(
            chapter_number=chapter_number,
            title=title,
            source_path=pdf_path,
            extracted_text=plain_text,
            markdown_content=markdown_content,
            concepts=concepts,
            db_path=db_path,
        )
        database.record_extraction(chapter_id, "completed", db_path=db_path)

        return {
            "chapter_number": chapter_number,
            "title": title,
            "status": "completed",
            "word_count": len(plain_text.split()),
            "concepts_found": len(concepts),
        }

    except Exception as e:
        error_msg = str(e)
        database.record_extraction(chapter_id, "failed", error_message=error_msg, db_path=db_path)
        logger.error("Failed to extract chapter %d: %s", chapter_number, error_msg)
        return {
            "chapter_number": chapter_number,
            "title": title,
            "status": "failed",
            "error": error_msg,
        }


def extract_all(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract all chapters from ML Principles PDFs.

    Returns a summary of the extraction process.
    """
    database.init_db(db_path)
    chapters = discover_chapters(base_path)

    if not chapters:
        return {
            "status": "no_chapters_found",
            "path_searched": base_path or ML_PRINCIPLES_PATH,
            "message": "No PDF chapters found in the specified path.",
        }

    results = []
    for chapter_info in chapters:
        result = process_chapter(chapter_info, force=force_reindex, db_path=db_path)
        results.append(result)

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "status": "done",
        "total_chapters": len(chapters),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }
