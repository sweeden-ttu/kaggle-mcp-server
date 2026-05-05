"""Docling-based PDF extraction worker for ML Principles chapters.

Scans chapter folders, extracts PDF content via docling, and stores
markdown + concepts in SQLite.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

from .database import Chapter, ConceptEntry, MLSysEngDB

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPT_PATTERNS = [
    r"\b(gradient descent|backpropagation|learning rate|loss function)\b",
    r"\b(neural network|deep learning|convolutional|recurrent)\b",
    r"\b(regularization|dropout|batch normalization|weight decay)\b",
    r"\b(cross[- ]?validation|train[- ]?test split|overfitting|underfitting)\b",
    r"\b(support vector|decision tree|random forest|boosting|bagging)\b",
    r"\b(feature engineering|feature selection|dimensionality reduction)\b",
    r"\b(hyperparameter|optimization|adam|sgd|momentum)\b",
    r"\b(attention|transformer|encoder|decoder|embedding)\b",
    r"\b(reinforcement learning|policy gradient|q-learning|reward)\b",
    r"\b(bayesian|prior|posterior|likelihood|inference)\b",
    r"\b(clustering|k-means|hierarchical|dbscan)\b",
    r"\b(principal component|pca|svd|eigenvalue)\b",
    r"\b(ensemble|stacking|blending|voting)\b",
    r"\b(precision|recall|f1[- ]?score|auc|roc)\b",
    r"\b(bias[- ]?variance|generalization|capacity)\b",
]


def _extract_concepts_from_text(text: str) -> list[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    concepts = set()
    text_lower = text.lower()
    for pattern in ML_CONCEPT_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            cleaned = m.strip().title()
            if len(cleaned) > 2:
                concepts.add(cleaned)
    return sorted(concepts)


def _parse_chapter_number(folder_name: str) -> int:
    """Extract chapter number from folder name like '08_ML_Systems'."""
    match = re.match(r"(\d+)", folder_name)
    return int(match.group(1)) if match else 0


def _make_slug(name: str) -> str:
    """Convert a chapter name to a URL-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF file using docling.

    Falls back to basic text extraction if docling is not available.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        return _fallback_extract(pdf_path)
    except Exception as e:
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
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
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    return f"[PDF extraction unavailable for {pdf_path} - install docling, pdfplumber, or PyPDF2]"


def scan_chapter_folders(base_path: Optional[str] = None) -> list[dict]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        return chapters

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue

        pdfs = list(item.glob("*.pdf"))
        if not pdfs:
            continue

        chapter_num = _parse_chapter_number(item.name)
        chapters.append({
            "folder": item.name,
            "path": str(item),
            "pdfs": [str(p) for p in pdfs],
            "chapter_number": chapter_num,
            "title": item.name,
            "slug": _make_slug(item.name),
        })

    return chapters


def extract_and_store(
    db: MLSysEngDB,
    base_path: Optional[str] = None,
    force_reindex: bool = False,
) -> dict:
    """Extract PDFs from all chapter folders and store in database.

    Returns a summary dict with extraction results.
    """
    folders = scan_chapter_folders(base_path)
    results = {"total": len(folders), "extracted": 0, "skipped": 0, "errors": []}

    for folder_info in folders:
        slug = folder_info["slug"]
        existing = db.get_chapter(slug)

        if existing and existing.status == "extracted" and not force_reindex:
            results["skipped"] += 1
            continue

        all_text = []
        for pdf_path in folder_info["pdfs"]:
            try:
                text = extract_pdf_with_docling(pdf_path)
                all_text.append(text)
            except Exception as e:
                results["errors"].append({"pdf": pdf_path, "error": str(e)})

        combined_text = "\n\n---\n\n".join(all_text)
        concepts = _extract_concepts_from_text(combined_text)

        chapter = Chapter(
            chapter_number=folder_info["chapter_number"],
            title=folder_info["title"],
            slug=slug,
            source_path=folder_info["path"],
            markdown_content=combined_text,
            concepts=json.dumps(concepts),
            extracted_at=time.time(),
            status="extracted",
        )

        chapter_id = db.upsert_chapter(chapter)

        for concept in concepts:
            db.add_concept(ConceptEntry(
                chapter_id=chapter_id,
                concept=concept,
                description=f"Concept from {folder_info['title']}",
                category="ml_principle",
            ))

        results["extracted"] += 1

    return results


def get_extraction_status(db: MLSysEngDB) -> dict:
    """Get the current extraction status across all chapters."""
    chapters = db.list_chapters()
    return {
        "total_chapters": len(chapters),
        "extracted": sum(1 for c in chapters if c.status == "extracted"),
        "pending": sum(1 for c in chapters if c.status == "pending"),
        "failed": sum(1 for c in chapters if c.status == "failed"),
        "chapters": [
            {
                "title": c.title,
                "slug": c.slug,
                "status": c.status,
                "concepts_count": len(json.loads(c.concepts)) if c.concepts else 0,
            }
            for c in chapters
        ],
    }
