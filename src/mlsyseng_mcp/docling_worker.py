"""Background PDF extraction worker using docling."""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = "/Users/sweeden/Desktop/Machine Learning Principles - Chapters"


def get_ml_principles_path() -> str:
    return os.environ.get("ML_PRINCIPLES_PATH", DEFAULT_ML_PRINCIPLES_PATH)


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover chapter folders and their PDFs in the ML Principles directory."""
    path = Path(base_path or get_ml_principles_path())
    chapters = []

    if not path.exists():
        logger.warning(f"ML Principles path does not exist: {path}")
        return chapters

    for item in sorted(path.iterdir()):
        if not item.is_dir():
            continue
        match = re.match(r"(\d+)[_\s-]*(.*)", item.name)
        if not match:
            continue

        chapter_number = int(match.group(1))
        title = match.group(2).replace("_", " ").strip()

        pdfs = list(item.glob("*.pdf")) + list(item.glob("*.PDF"))
        if pdfs:
            chapters.append({
                "chapter_number": chapter_number,
                "title": title,
                "folder_path": str(item),
                "pdf_paths": [str(p) for p in pdfs],
            })

    return chapters


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF using docling or fallback methods."""
    try:
        return _extract_with_docling(pdf_path)
    except ImportError:
        logger.info("docling not available, trying PyPDF2 fallback")
        try:
            return _extract_with_pypdf2(pdf_path)
        except ImportError:
            logger.info("PyPDF2 not available, trying pdfplumber fallback")
            try:
                return _extract_with_pdfplumber(pdf_path)
            except ImportError:
                raise RuntimeError(
                    "No PDF extraction library available. Install one of: docling, PyPDF2, pdfplumber"
                )


def _extract_with_docling(pdf_path: str) -> str:
    """Extract using docling library."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


def _extract_with_pypdf2(pdf_path: str) -> str:
    """Fallback extraction using PyPDF2."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_with_pdfplumber(pdf_path: str) -> str:
    """Fallback extraction using pdfplumber."""
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_concepts(markdown_content: str) -> list[dict]:
    """Extract ML/AI concepts from chapter content using heuristic patterns."""
    concepts = []
    seen = set()

    concept_patterns = [
        (r"(?:^|\n)#+\s*(.+)", "heading"),
        (r"\*\*([A-Z][A-Za-z\s]+(?:Algorithm|Method|Model|Network|Function|Loss|Optimization|Regularization|Learning|Distribution|Theorem))\*\*", "bold_term"),
        (r"(?:^|\n)(?:Definition|Theorem|Lemma|Proposition)\s*[\d.]*[:\s]+(.+)", "formal"),
    ]

    ml_keywords = {
        "gradient descent", "backpropagation", "cross-entropy", "softmax",
        "convolution", "recurrent", "attention", "transformer", "dropout",
        "batch normalization", "adam optimizer", "learning rate", "overfitting",
        "underfitting", "bias-variance", "regularization", "ensemble",
        "random forest", "decision tree", "neural network", "deep learning",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "feature engineering", "hyperparameter", "cross-validation",
        "loss function", "activation function", "weight initialization",
        "data augmentation", "transfer learning", "fine-tuning",
    }

    for keyword in ml_keywords:
        if keyword.lower() in markdown_content.lower() and keyword not in seen:
            concepts.append({
                "name": keyword.title(),
                "description": f"ML concept: {keyword}",
                "category": "ml_concept",
                "confidence": 0.8,
            })
            seen.add(keyword)

    for pattern, category in concept_patterns:
        matches = re.findall(pattern, markdown_content)
        for match in matches[:20]:
            name = match.strip()
            if len(name) > 3 and len(name) < 100 and name.lower() not in seen:
                concepts.append({
                    "name": name,
                    "description": f"Extracted from chapter ({category})",
                    "category": category,
                    "confidence": 0.6,
                })
                seen.add(name.lower())

    return concepts


def extract_chapter(chapter_info: dict) -> dict:
    """Extract content from a single chapter and return structured data."""
    pdf_paths = chapter_info["pdf_paths"]
    all_content = []

    for pdf_path in pdf_paths:
        try:
            content = extract_pdf_content(pdf_path)
            all_content.append(content)
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path}: {e}")
            all_content.append(f"[Extraction failed: {e}]")

    combined_content = "\n\n---\n\n".join(all_content)
    concepts = extract_concepts(combined_content)

    return {
        "chapter_number": chapter_info["chapter_number"],
        "title": chapter_info["title"],
        "source_path": chapter_info["folder_path"],
        "markdown_content": combined_content,
        "concepts": concepts,
    }
