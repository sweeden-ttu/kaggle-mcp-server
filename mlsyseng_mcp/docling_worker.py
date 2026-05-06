"""Background PDF extraction worker using docling (with fallback to PyPDF2/pdfplumber).

Scans ML Principles chapter folders, extracts text and structural data,
identifies key ML/AI concepts, and stores content in SQLite.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "loss function", "regularization",
    "overfitting", "underfitting", "cross-validation", "feature engineering",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention mechanism", "batch normalization",
    "dropout", "learning rate", "optimizer", "activation function",
    "hyperparameter", "ensemble", "bagging", "boosting", "random forest",
    "decision tree", "support vector", "kernel", "dimensionality reduction",
    "principal component", "clustering", "k-means", "regression",
    "classification", "reinforcement learning", "reward function",
    "policy gradient", "value function", "markov decision",
    "bayesian", "prior", "posterior", "likelihood", "evidence",
    "generative", "discriminative", "autoencoder", "variational",
    "gan", "generative adversarial", "embedding", "word2vec",
    "transfer learning", "fine-tuning", "pre-training",
    "bias-variance", "confusion matrix", "precision", "recall",
    "f1 score", "roc", "auc", "mean squared error",
    "stochastic", "mini-batch", "momentum", "adam",
    "mixture of experts", "gating network", "sparse",
]


def _try_docling_extract(pdf_path: str) -> Optional[str]:
    """Extract text from PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, falling back to PyPDF2/pdfplumber")
        return None
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)
        return None


def _try_pypdf2_extract(pdf_path: str) -> Optional[str]:
    """Fallback: extract text using PyPDF2."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n---\n\n".join(pages) if pages else None
    except ImportError:
        return None
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)
        return None


def _try_pdfplumber_extract(pdf_path: str) -> Optional[str]:
    """Fallback: extract text using pdfplumber."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n---\n\n".join(pages) if pages else None
    except ImportError:
        return None
    except Exception as e:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, e)
        return None


def _get_page_count(pdf_path: str) -> int:
    try:
        from PyPDF2 import PdfReader
        return len(PdfReader(pdf_path).pages)
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def extract_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF, trying multiple backends.

    Returns (markdown_text, page_count).
    """
    text = _try_docling_extract(pdf_path)
    if not text:
        text = _try_pypdf2_extract(pdf_path)
    if not text:
        text = _try_pdfplumber_extract(pdf_path)
    if not text:
        text = f"[Extraction failed for {Path(pdf_path).name}]"

    page_count = _get_page_count(pdf_path)
    return text, page_count


def extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts present in the extracted text."""
    text_lower = text.lower()
    found = []
    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return sorted(set(found))


def _parse_chapter_number(folder_name: str) -> Optional[str]:
    """Extract chapter number from folder name like '08_ML Systems'."""
    match = re.match(r"^(\d+)", folder_name)
    return match.group(1).zfill(2) if match else None


def _parse_chapter_title(folder_name: str) -> str:
    """Extract title from folder name like '08_ML Systems'."""
    match = re.match(r"^\d+[_\s]*(.*)", folder_name)
    return match.group(1).strip() if match else folder_name


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs.

    Returns list of dicts with keys: chapter_number, title, folder_path, pdf_files.
    """
    base = Path(base_path or ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        chapter_num = _parse_chapter_number(entry.name)
        if chapter_num is None:
            continue
        pdfs = sorted(entry.glob("*.pdf"))
        if not pdfs:
            pdfs = sorted(entry.glob("**/*.pdf"))
        chapters.append({
            "chapter_number": chapter_num,
            "title": _parse_chapter_title(entry.name),
            "folder_path": str(entry),
            "pdf_files": [str(p) for p in pdfs],
        })
    return chapters


def extract_all_chapters(
    db: Database,
    base_path: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """Extract all chapter PDFs and store in the database.

    Returns a summary dict with counts and any errors.
    """
    chapters = discover_chapters(base_path)
    results = {"total": len(chapters), "extracted": 0, "skipped": 0, "errors": []}

    for ch in chapters:
        chapter_num = ch["chapter_number"]

        if not force_reindex:
            existing = db.get_chapter(chapter_num)
            if existing and existing.get("content_md"):
                results["skipped"] += 1
                continue

        db.set_extraction_status(chapter_num, "running")

        all_text = []
        total_pages = 0
        for pdf_path in ch["pdf_files"]:
            try:
                text, pages = extract_pdf(pdf_path)
                all_text.append(text)
                total_pages += pages
            except Exception as e:
                error_msg = f"Failed to extract {pdf_path}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        if all_text:
            combined = "\n\n---\n\n".join(all_text)
            concepts = extract_concepts(combined)
            db.upsert_chapter(
                chapter_number=chapter_num,
                title=ch["title"],
                content_md=combined,
                source_path=ch["folder_path"],
                concepts=concepts,
                page_count=total_pages,
            )
            db.set_extraction_status(
                chapter_num, "completed", pages_extracted=total_pages
            )
            results["extracted"] += 1
        else:
            db.set_extraction_status(
                chapter_num, "failed", error_message="No text extracted from any PDF"
            )

    return results
