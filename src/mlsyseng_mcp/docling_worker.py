"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content, identifies
concepts, and stores everything in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import MLSysEngDB

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "regularization", "overfitting",
    "underfitting", "bias-variance", "cross-validation", "ensemble",
    "boosting", "bagging", "dropout", "batch normalization", "attention",
    "transformer", "convolution", "recurrent", "lstm", "gru",
    "embedding", "feature engineering", "hyperparameter", "optimization",
    "loss function", "activation function", "learning rate", "momentum",
    "neural network", "deep learning", "random forest", "decision tree",
    "support vector", "logistic regression", "linear regression",
    "dimensionality reduction", "pca", "clustering", "k-means",
    "reinforcement learning", "generative", "discriminative",
    "bayesian", "markov", "inference", "model selection",
    "data augmentation", "transfer learning", "fine-tuning",
    "normalization", "standardization", "preprocessing",
]


def _try_docling_extract(pdf_path: str) -> Optional[str]:
    """Extract text from PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return None
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return None


def _fallback_extract(pdf_path: str) -> str:
    """Basic PDF text extraction fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        pass

    return f"[Could not extract text from {pdf_path} - install docling, PyMuPDF, or pdfminer]"


def extract_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF file. Returns (markdown_content, page_count)."""
    content = _try_docling_extract(pdf_path)
    if content is None:
        content = _fallback_extract(pdf_path)

    page_count = 0
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        page_count = content.count("\f") + 1 if content else 0

    return content, page_count


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """Extract ML/AI concepts from chapter text."""
    text_lower = text.lower()
    found = []
    for keyword in CONCEPT_KEYWORDS:
        if keyword in text_lower:
            idx = text_lower.index(keyword)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(keyword) + 200)
            snippet = text[start:end].strip()
            snippet = re.sub(r"\s+", " ", snippet)

            category = _categorize_concept(keyword)
            found.append({
                "concept": keyword,
                "description": snippet,
                "category": category,
            })
    return found


def _categorize_concept(keyword: str) -> str:
    """Categorize a concept into a broad ML category."""
    categories = {
        "optimization": ["gradient descent", "learning rate", "momentum", "optimization", "loss function"],
        "regularization": ["regularization", "overfitting", "underfitting", "dropout", "batch normalization"],
        "architecture": ["neural network", "deep learning", "convolution", "recurrent", "lstm", "gru", "transformer", "attention"],
        "evaluation": ["cross-validation", "bias-variance", "model selection"],
        "ensemble": ["ensemble", "boosting", "bagging", "random forest"],
        "classical_ml": ["decision tree", "support vector", "logistic regression", "linear regression", "k-means", "clustering"],
        "representation": ["embedding", "feature engineering", "dimensionality reduction", "pca", "preprocessing", "normalization", "standardization"],
        "advanced": ["reinforcement learning", "generative", "discriminative", "bayesian", "markov", "inference", "transfer learning", "fine-tuning", "data augmentation"],
        "training": ["backpropagation", "activation function", "hyperparameter"],
    }
    for cat, keywords in categories.items():
        if keyword in keywords:
            return cat
    return "general"


def scan_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders/PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    chapter_pattern = re.compile(r"(\d+)[_\s-]*(.*)", re.IGNORECASE)

    for entry in sorted(base.iterdir()):
        pdfs = []
        title = entry.name

        if entry.is_dir():
            match = chapter_pattern.match(entry.name)
            if match:
                num = int(match.group(1))
                title = match.group(2).strip().replace("_", " ")
            else:
                num = hash(entry.name) % 1000
            pdfs = sorted(entry.glob("*.pdf"))
        elif entry.suffix.lower() == ".pdf":
            match = chapter_pattern.match(entry.stem)
            if match:
                num = int(match.group(1))
                title = match.group(2).strip().replace("_", " ")
            else:
                num = hash(entry.stem) % 1000
            pdfs = [entry]
        else:
            continue

        if pdfs:
            chapters.append({
                "chapter_number": num,
                "title": title or f"Chapter {num}",
                "source_path": str(entry),
                "pdf_files": [str(p) for p in pdfs],
            })

    return chapters


def run_extraction(
    db: MLSysEngDB,
    force_reindex: bool = False,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full extraction pipeline.

    1. Scan for chapter folders
    2. Extract PDF content with docling
    3. Identify concepts
    4. Store in database
    """
    chapters = scan_chapters(base_path)
    results = {"scanned": len(chapters), "extracted": 0, "skipped": 0, "errors": []}

    for ch in chapters:
        existing = db.get_chapter(ch["chapter_number"])
        if existing and not force_reindex:
            results["skipped"] += 1
            continue

        chapter_id = db.upsert_chapter(
            chapter_number=ch["chapter_number"],
            title=ch["title"],
            source_path=ch["source_path"],
            content_md="",
            page_count=0,
        )
        log_id = db.log_extraction(chapter_id, "running", f"Extracting {ch['title']}")

        try:
            all_content = []
            total_pages = 0
            for pdf_path in ch["pdf_files"]:
                content, pages = extract_pdf(pdf_path)
                all_content.append(content)
                total_pages += pages

            full_content = "\n\n---\n\n".join(all_content)
            db.upsert_chapter(
                chapter_number=ch["chapter_number"],
                title=ch["title"],
                source_path=ch["source_path"],
                content_md=full_content,
                page_count=total_pages,
            )

            concepts = extract_concepts(full_content)
            if concepts:
                db.add_concepts(chapter_id, concepts)

            db.update_extraction_log(log_id, "completed", f"Extracted {total_pages} pages, {len(concepts)} concepts")
            results["extracted"] += 1

        except Exception as e:
            db.update_extraction_log(log_id, "failed", str(e))
            results["errors"].append({"chapter": ch["title"], "error": str(e)})

    return results
