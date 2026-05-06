"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content to markdown,
and identifies key ML/AI concepts.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "convolutional", "recurrent", "transformer", "attention mechanism",
    "loss function", "optimization", "regularization", "overfitting",
    "underfitting", "cross-validation", "hyperparameter", "feature engineering",
    "ensemble", "bagging", "boosting", "random forest", "decision tree",
    "support vector", "clustering", "dimensionality reduction", "PCA",
    "embedding", "transfer learning", "fine-tuning", "data augmentation",
    "batch normalization", "dropout", "learning rate", "momentum",
    "adam optimizer", "SGD", "reinforcement learning", "reward function",
    "policy gradient", "Q-learning", "generative", "discriminative",
    "autoencoder", "GAN", "variational", "bayesian", "markov",
    "kernel method", "logistic regression", "linear regression",
    "classification", "regression", "segmentation", "object detection",
    "natural language processing", "word embedding", "tokenization",
    "beam search", "model selection", "bias-variance", "ROC", "AUC",
    "precision", "recall", "F1 score", "confusion matrix",
    "mixture of experts", "gating network", "sparse activation",
]


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover chapter folders and their PDFs.

    Returns list of dicts with chapter_number, title, source_path.
    """
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    chapter_pattern = re.compile(r"^(\d{2})[\s_-]+(.+)$")

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        match = chapter_pattern.match(entry.name)
        if not match:
            continue

        chapter_num = match.group(1)
        title = match.group(2).strip()

        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            pdfs = list(entry.glob("*.PDF"))

        if pdfs:
            chapters.append({
                "chapter_number": chapter_num,
                "title": title,
                "source_path": str(pdfs[0]),
                "folder_path": str(entry),
            })
        else:
            chapters.append({
                "chapter_number": chapter_num,
                "title": title,
                "source_path": str(entry),
                "folder_path": str(entry),
            })

    return chapters


def extract_pdf_content(pdf_path: str) -> Tuple[str, int]:
    """Extract text content from a PDF using docling.

    Returns (markdown_text, page_count).
    Falls back to PyPDF2 or basic text extraction if docling unavailable.
    """
    path = Path(pdf_path)
    if not path.exists() or not path.suffix.lower() == ".pdf":
        logger.warning("Not a valid PDF path: %s", pdf_path)
        return "", 0

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(path))
        md = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 1
        return md, page_count
    except ImportError:
        logger.info("docling not available, trying PyPDF2 fallback")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n---\n\n".join(pages), len(reader.pages)
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfplumber fallback")
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n---\n\n".join(pages), len(pdf.pages)
    except ImportError:
        logger.warning("No PDF extraction library available (docling, PyPDF2, pdfplumber)")
    except Exception as e:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, e)

    return "", 0


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text content."""
    if not text:
        return []

    text_lower = text.lower()
    found = []
    for keyword in CONCEPT_KEYWORDS:
        if keyword.lower() in text_lower:
            found.append(keyword)

    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    for match in heading_pattern.finditer(text):
        heading = match.group(1).strip()
        if len(heading) > 3 and heading not in found:
            found.append(heading)

    return sorted(set(found))


def extract_all_chapters(
    db: Database,
    base_path: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict[str, any]:
    """Extract all chapters from ML Principles PDFs.

    Returns summary dict with counts and any errors.
    """
    chapters = discover_chapters(base_path)
    results = {
        "total_discovered": len(chapters),
        "extracted": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }

    for ch in chapters:
        chapter_num = ch["chapter_number"]

        if not force_reindex:
            existing = db.get_chapter(chapter_num)
            if existing and existing.get("content_markdown"):
                results["skipped"] += 1
                continue

        db.update_extraction_status(chapter_num, "extracting")

        try:
            source = ch["source_path"]
            if source.lower().endswith(".pdf"):
                content, pages = extract_pdf_content(source)
            else:
                content, pages = _extract_folder_content(ch["folder_path"])

            if not content:
                content = f"# {ch['title']}\n\nChapter {chapter_num} - content pending extraction.\n"
                pages = 0

            concepts = extract_concepts(content)

            db.upsert_chapter(
                chapter_number=chapter_num,
                title=ch["title"],
                source_path=source,
                content_markdown=content,
                concepts=concepts,
            )
            db.update_extraction_status(
                chapter_num, "completed", pages_extracted=pages
            )
            results["extracted"] += 1

        except Exception as e:
            error_msg = f"Chapter {chapter_num}: {e}"
            logger.error("Extraction failed: %s", error_msg)
            db.update_extraction_status(chapter_num, "failed", error_message=str(e))
            results["failed"] += 1
            results["errors"].append(error_msg)

    return results


def _extract_folder_content(folder_path: str) -> Tuple[str, int]:
    """Extract content from all readable files in a folder."""
    folder = Path(folder_path)
    parts = []
    file_count = 0

    for ext in ("*.md", "*.txt", "*.rst", "*.tex"):
        for f in sorted(folder.glob(ext)):
            try:
                parts.append(f.read_text(errors="replace"))
                file_count += 1
            except Exception:
                continue

    for pdf in sorted(folder.glob("*.pdf")):
        content, pages = extract_pdf_content(str(pdf))
        if content:
            parts.append(content)
            file_count += pages

    return "\n\n---\n\n".join(parts), file_count
