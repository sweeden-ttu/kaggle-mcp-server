"""Background PDF extraction worker using docling.

Scans ML Principles chapter directories, extracts text from PDFs,
and stores content in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_CONCEPT_PATTERNS = [
    r"\b(gradient\s+descent)\b",
    r"\b(backpropagation)\b",
    r"\b(regularization)\b",
    r"\b(cross[- ]?validation)\b",
    r"\b(neural\s+network)\b",
    r"\b(deep\s+learning)\b",
    r"\b(decision\s+tree)\b",
    r"\b(random\s+forest)\b",
    r"\b(support\s+vector\s+machine)\b",
    r"\b(logistic\s+regression)\b",
    r"\b(linear\s+regression)\b",
    r"\b(clustering)\b",
    r"\b(dimensionality\s+reduction)\b",
    r"\b(feature\s+engineering)\b",
    r"\b(hyperparameter\s+tuning)\b",
    r"\b(ensemble\s+method)\b",
    r"\b(convolutional\s+neural)\b",
    r"\b(recurrent\s+neural)\b",
    r"\b(transformer)\b",
    r"\b(attention\s+mechanism)\b",
    r"\b(batch\s+normalization)\b",
    r"\b(dropout)\b",
    r"\b(transfer\s+learning)\b",
    r"\b(reinforcement\s+learning)\b",
    r"\b(generative\s+adversarial)\b",
    r"\b(autoencoder)\b",
    r"\b(variational)\b",
    r"\b(bayesian)\b",
    r"\b(markov)\b",
    r"\b(monte\s+carlo)\b",
    r"\b(optimization)\b",
    r"\b(loss\s+function)\b",
    r"\b(activation\s+function)\b",
    r"\b(learning\s+rate)\b",
    r"\b(overfitting)\b",
    r"\b(underfitting)\b",
    r"\b(bias[- ]?variance)\b",
    r"\b(precision)\b",
    r"\b(recall)\b",
    r"\b(f1[- ]?score)\b",
    r"\b(ROC|AUC)\b",
    r"\b(confusion\s+matrix)\b",
    r"\b(embedding)\b",
    r"\b(tokenization)\b",
    r"\b(data\s+augmentation)\b",
    r"\b(model\s+selection)\b",
    r"\b(pipeline)\b",
    r"\b(feature\s+selection)\b",
]


def _default_chapters_path() -> str:
    return os.environ.get(
        "ML_PRINCIPLES_PATH",
        os.path.expanduser("~/ml-principles-chapters"),
    )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug


def _extract_pdf_with_docling(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()
        page_count = getattr(result.document, "num_pages", 0) or len(text) // 3000 + 1
        return text, page_count
    except ImportError:
        logger.warning("docling not available, attempting fallback PDF extraction")
        return _extract_pdf_fallback(pdf_path)
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s, trying fallback", pdf_path, e)
        return _extract_pdf_fallback(pdf_path)


def _extract_pdf_fallback(pdf_path: str) -> Tuple[str, int]:
    """Fallback extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts), page_count
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
        return "\n\n".join(text_parts), len(reader.pages)
    except ImportError:
        pass

    return f"[Could not extract text from {pdf_path} - install docling, pdfplumber, or PyPDF2]", 0


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """Extract ML/AI concepts from text using pattern matching."""
    found = {}
    text_lower = text.lower()
    for pattern in ML_CONCEPT_PATTERNS:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            concept = match.group(1).strip()
            concept_normalized = re.sub(r"\s+", " ", concept).title()
            if concept_normalized not in found:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                category = _categorize_concept(concept_normalized)
                found[concept_normalized] = {
                    "concept_name": concept_normalized,
                    "description": context,
                    "category": category,
                    "confidence": 1.0,
                }
    return list(found.values())


def _categorize_concept(concept: str) -> str:
    categories = {
        "algorithm": [
            "gradient descent", "backpropagation", "decision tree", "random forest",
            "support vector", "logistic regression", "linear regression", "clustering",
        ],
        "architecture": [
            "neural network", "deep learning", "convolutional", "recurrent",
            "transformer", "autoencoder", "generative adversarial",
        ],
        "technique": [
            "regularization", "cross validation", "feature engineering",
            "hyperparameter", "ensemble", "batch normalization", "dropout",
            "transfer learning", "data augmentation", "feature selection",
        ],
        "metric": [
            "precision", "recall", "f1", "roc", "auc", "confusion matrix",
            "loss function",
        ],
        "theory": [
            "bayesian", "markov", "monte carlo", "bias variance",
            "overfitting", "underfitting", "variational",
        ],
    }
    concept_lower = concept.lower()
    for cat, keywords in categories.items():
        if any(kw in concept_lower for kw in keywords):
            return cat
    return "general"


def scan_chapters(chapters_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Scan directory for chapter folders or PDF files."""
    base = Path(chapters_path or _default_chapters_path())
    if not base.exists():
        logger.warning("Chapters path does not exist: %s", base)
        return []

    chapters = []
    if base.is_file() and base.suffix.lower() == ".pdf":
        chapters.append({
            "name": base.stem,
            "slug": _slugify(base.stem),
            "path": str(base),
        })
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append({
                    "name": item.name,
                    "slug": _slugify(item.name),
                    "path": str(pdfs[0]),
                })
        elif item.suffix.lower() == ".pdf":
            chapters.append({
                "name": item.stem,
                "slug": _slugify(item.stem),
                "path": str(item),
            })

    return chapters


def extract_chapter(
    chapter_info: Dict[str, str],
    db: Database,
    force: bool = False,
) -> Dict[str, str]:
    """Extract a single chapter PDF and store in database."""
    name = chapter_info["name"]
    slug = chapter_info["slug"]
    pdf_path = chapter_info["path"]

    if not force:
        existing = db.get_chapter(name)
        if existing and existing.get("markdown_content"):
            return {"status": "skipped", "chapter": name, "reason": "already extracted"}

    db.set_extraction_status(name, "extracting")
    try:
        text, page_count = _extract_pdf_with_docling(pdf_path)
        chapter_id = db.upsert_chapter(name, slug, pdf_path, text, page_count)
        concepts = extract_concepts(text)
        db.add_concepts(chapter_id, concepts)
        db.set_extraction_status(name, "completed", pages_extracted=page_count)
        return {
            "status": "completed",
            "chapter": name,
            "pages": page_count,
            "concepts_found": len(concepts),
        }
    except Exception as e:
        db.set_extraction_status(name, "failed", error_message=str(e))
        logger.error("Failed to extract %s: %s", name, e)
        return {"status": "failed", "chapter": name, "error": str(e)}


def extract_all_chapters(
    chapters_path: Optional[str] = None,
    db: Optional[Database] = None,
    force: bool = False,
) -> List[Dict[str, str]]:
    """Extract all chapters from the ML Principles directory."""
    if db is None:
        db = Database()

    chapters = scan_chapters(chapters_path)
    if not chapters:
        return [{"status": "no_chapters", "message": f"No chapters found at {chapters_path or _default_chapters_path()}"}]

    results = []
    for chapter in chapters:
        result = extract_chapter(chapter, db, force=force)
        results.append(result)

    return results
