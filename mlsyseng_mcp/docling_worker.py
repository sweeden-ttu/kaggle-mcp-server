"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)


def _try_import_docling():
    """Lazy-import docling so the rest of the system works without it installed."""
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter
    except ImportError:
        return None


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Scan the ML Principles directory tree for chapter folders containing PDFs."""
    root = Path(base_path or ML_PRINCIPLES_PATH)
    if not root.exists():
        logger.warning("ML Principles path does not exist: %s", root)
        return []

    chapters: List[Dict[str, str]] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        pdfs = list(item.glob("*.pdf"))
        if pdfs:
            chapters.append({
                "chapter_name": item.name,
                "path": str(item),
                "pdf_count": len(pdfs),
                "pdf_files": [p.name for p in pdfs],
            })
    return chapters


def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text and structured data from a single PDF.

    Falls back to a lightweight text extraction when docling is unavailable.
    """
    DocumentConverter = _try_import_docling()

    if DocumentConverter is not None:
        return _extract_with_docling(pdf_path, DocumentConverter)
    return _extract_fallback(pdf_path)


def _extract_with_docling(pdf_path: str, DocumentConverter) -> Dict[str, Any]:
    """Full extraction via the docling library."""
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        markdown_text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else str(doc)

        tables = []
        if hasattr(doc, "tables"):
            for t in doc.tables:
                tables.append(str(t))

        page_count = 0
        if hasattr(result, "pages"):
            page_count = len(result.pages)

        return {
            "markdown": markdown_text,
            "tables": tables,
            "page_count": page_count,
            "method": "docling",
        }
    except Exception as exc:
        logger.error("Docling extraction failed for %s: %s", pdf_path, exc)
        return _extract_fallback(pdf_path)


def _extract_fallback(pdf_path: str) -> Dict[str, Any]:
    """Lightweight fallback using PyPDF2 or pdfplumber if available."""
    text = ""
    page_count = 0

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            text = "\n\n".join(parts)
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            text = "\n\n".join(parts)
        except ImportError:
            logger.warning("No PDF extraction library available. Install docling, pdfplumber, or PyPDF2.")
            text = f"[Unable to extract content from {pdf_path} – no PDF library installed]"

    return {
        "markdown": text,
        "tables": [],
        "page_count": page_count,
        "method": "fallback",
    }


def extract_concepts(markdown_text: str) -> List[str]:
    """
    Extract key ML/AI concepts from chapter markdown content.

    Uses heuristic pattern matching on headings, bold terms, and known ML vocabulary.
    """
    concepts: List[str] = []

    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    for m in heading_pattern.finditer(markdown_text):
        heading = m.group(1).strip()
        if len(heading) > 3 and len(heading) < 100:
            concepts.append(heading)

    bold_pattern = re.compile(r"\*\*(.+?)\*\*")
    for m in bold_pattern.finditer(markdown_text):
        term = m.group(1).strip()
        if len(term) > 3 and len(term) < 60:
            concepts.append(term)

    ml_keywords = [
        "gradient descent", "backpropagation", "neural network", "deep learning",
        "regularization", "cross-validation", "feature engineering", "hyperparameter",
        "overfitting", "underfitting", "bias-variance", "ensemble", "random forest",
        "support vector", "logistic regression", "linear regression", "decision tree",
        "convolutional", "recurrent", "transformer", "attention mechanism",
        "batch normalization", "dropout", "learning rate", "loss function",
        "optimization", "stochastic gradient", "adam optimizer", "momentum",
        "kernel method", "principal component", "dimensionality reduction",
        "clustering", "classification", "regression", "reinforcement learning",
        "supervised learning", "unsupervised learning", "semi-supervised",
        "transfer learning", "fine-tuning", "pre-training", "embedding",
        "tokenization", "sequence model", "generative model", "discriminative",
        "bayesian", "maximum likelihood", "expectation maximization",
    ]
    text_lower = markdown_text.lower()
    for kw in ml_keywords:
        if kw in text_lower:
            concepts.append(kw.title())

    seen = set()
    unique: List[str] = []
    for c in concepts:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def extract_chapter(
    chapter_path: str,
    db=None,
    status_callback=None,
) -> Tuple[str, str, List[str]]:
    """
    Extract all PDFs in a chapter folder, merge content, and return
    (chapter_name, markdown, concepts).
    """
    chapter_dir = Path(chapter_path)
    chapter_name = chapter_dir.name
    pdfs = sorted(chapter_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {chapter_path}")

    if status_callback:
        status_callback(chapter_name, "running", 0.0)

    all_markdown: List[str] = []
    total_pages = 0

    for i, pdf_path in enumerate(pdfs):
        progress = (i / len(pdfs)) * 100
        if status_callback:
            status_callback(chapter_name, "running", progress)

        result = extract_pdf_content(str(pdf_path))
        if result["markdown"]:
            all_markdown.append(f"## {pdf_path.stem}\n\n{result['markdown']}")
        total_pages += result.get("page_count", 0)

    merged_markdown = "\n\n---\n\n".join(all_markdown)
    concepts = extract_concepts(merged_markdown)

    if db is not None:
        db.upsert_chapter(
            chapter_name=chapter_name,
            source_path=str(chapter_dir),
            markdown_content=merged_markdown,
            concepts=concepts,
            page_count=total_pages,
        )

    if status_callback:
        status_callback(chapter_name, "completed", 100.0)

    return chapter_name, merged_markdown, concepts
