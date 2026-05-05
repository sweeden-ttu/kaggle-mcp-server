"""Background PDF extraction worker using docling.

Extracts text and structured data from ML Principles chapter PDFs,
identifies key concepts, and stores results in the database.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

from . import database

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPT_PATTERNS = [
    r"(?:gradient\s+descent|backpropagation|forward\s+pass)",
    r"(?:regularization|dropout|batch\s+norm(?:alization)?)",
    r"(?:cross[\-\s]?entropy|mean\s+squared\s+error|loss\s+function)",
    r"(?:convolutional|recurrent|transformer|attention)",
    r"(?:overfitting|underfitting|bias[\-\s]?variance)",
    r"(?:hyperparameter|learning\s+rate|momentum|weight\s+decay)",
    r"(?:ensemble|bagging|boosting|random\s+forest)",
    r"(?:feature\s+engineering|dimensionality\s+reduction|PCA)",
    r"(?:supervised|unsupervised|reinforcement)\s+learning",
    r"(?:neural\s+network|deep\s+learning|perceptron)",
    r"(?:optimization|convergence|stochastic)",
    r"(?:activation\s+function|ReLU|sigmoid|softmax)",
    r"(?:embedding|tokenization|vocabulary)",
    r"(?:precision|recall|F1[\-\s]?score|AUC|ROC)",
]


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover chapter folders and their PDFs."""
    path = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not path.exists():
        logger.warning(f"ML Principles path does not exist: {path}")
        return chapters

    for item in sorted(path.iterdir()):
        if not item.is_dir():
            continue

        match = re.match(r"(\d+)[_\-\s]*(.*)", item.name)
        if not match:
            continue

        chapter_num = int(match.group(1))
        chapter_title = match.group(2).replace("_", " ").strip()

        pdfs = list(item.glob("*.pdf"))
        if not pdfs:
            continue

        chapters.append({
            "number": chapter_num,
            "title": chapter_title,
            "path": str(item),
            "pdf_files": [str(p) for p in pdfs],
        })

    return chapters


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF using docling.

    Falls back to a basic extraction if docling is not available.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, attempting fallback extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    logger.warning(f"No PDF extraction library available for {pdf_path}")
    return ""


def extract_concepts(text: str) -> list[dict]:
    """Extract ML/AI concepts from chapter text content."""
    concepts = []
    seen = set()

    for pattern in ML_CONCEPT_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            concept_name = match.group(0).strip().lower()
            concept_name = re.sub(r"\s+", " ", concept_name)

            if concept_name in seen:
                continue
            seen.add(concept_name)

            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()

            category = _categorize_concept(concept_name)

            concepts.append({
                "name": concept_name,
                "description": context,
                "category": category,
            })

    return concepts


def _categorize_concept(concept_name: str) -> str:
    """Categorize a concept into a broad ML category."""
    categories = {
        "optimization": ["gradient", "descent", "convergence", "stochastic", "momentum", "learning rate"],
        "architecture": ["convolutional", "recurrent", "transformer", "attention", "neural network", "perceptron"],
        "regularization": ["regularization", "dropout", "batch norm", "weight decay"],
        "evaluation": ["precision", "recall", "f1", "auc", "roc", "loss function", "cross-entropy"],
        "training": ["backpropagation", "forward pass", "overfitting", "underfitting", "bias-variance"],
        "feature_engineering": ["feature engineering", "dimensionality reduction", "pca", "embedding"],
        "ensemble": ["ensemble", "bagging", "boosting", "random forest"],
        "learning_paradigm": ["supervised", "unsupervised", "reinforcement"],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in concept_name:
                return category

    return "general"


def process_chapter(
    chapter_info: dict,
    force_reindex: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Process a single chapter: extract PDF, identify concepts, store in DB."""
    chapter_num = chapter_info["number"]
    title = chapter_info["title"]
    pdf_files = chapter_info["pdf_files"]

    all_text_parts = []
    for pdf_path in pdf_files:
        logger.info(f"Extracting: {pdf_path}")
        text = extract_pdf_content(pdf_path)
        if text:
            all_text_parts.append(text)

    full_text = "\n\n---\n\n".join(all_text_parts)

    chapter_id = database.insert_chapter(
        chapter_number=chapter_num,
        title=title,
        pdf_path=pdf_files[0],
        markdown_content=full_text,
        db_path=db_path,
    )

    database.log_extraction(chapter_id, "processing", db_path=db_path)

    concepts = extract_concepts(full_text)
    for concept in concepts:
        database.insert_concept(
            chapter_id=chapter_id,
            concept_name=concept["name"],
            description=concept["description"],
            category=concept["category"],
            db_path=db_path,
        )

    database.log_extraction(chapter_id, "completed", db_path=db_path)

    return {
        "chapter_number": chapter_num,
        "title": title,
        "chapter_id": chapter_id,
        "text_length": len(full_text),
        "concepts_found": len(concepts),
    }


def run_extraction(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Run full extraction pipeline over all discovered chapters."""
    database.init_db(db_path)

    chapters = discover_chapters(base_path)
    if not chapters:
        logger.warning("No chapters found to extract")
        return []

    results = []
    for chapter_info in chapters:
        try:
            result = process_chapter(chapter_info, force_reindex, db_path)
            results.append(result)
            logger.info(
                f"Chapter {result['chapter_number']}: {result['concepts_found']} concepts extracted"
            )
        except Exception as e:
            logger.error(f"Failed to process chapter {chapter_info['number']}: {e}")
            results.append({
                "chapter_number": chapter_info["number"],
                "title": chapter_info["title"],
                "error": str(e),
            })

    return results
