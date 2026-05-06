"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from mlsyseng_moe import database, embeddings

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CHAPTER_PATTERN = re.compile(r"^(\d+)[_\s-]+(.+)$")


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover chapter folders and their PDFs."""
    path = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not path.exists():
        logger.warning(f"ML Principles path does not exist: {path}")
        return chapters

    for item in sorted(path.iterdir()):
        if not item.is_dir():
            continue
        match = CHAPTER_PATTERN.match(item.name)
        if match:
            chapter_num = int(match.group(1))
            title = match.group(2).replace("_", " ").replace("-", " ").strip()
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append({
                    "chapter_number": chapter_num,
                    "title": title,
                    "folder_path": str(item),
                    "pdf_paths": [str(p) for p in pdfs],
                })
    return chapters


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"Docling extraction failed for {pdf_path}: {e}")
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

    logger.error(f"No PDF extraction library available for {pdf_path}")
    return ""


def extract_concepts(content: str, chapter_title: str) -> list[dict]:
    """Extract key ML/AI concepts from chapter content."""
    concept_indicators = [
        "algorithm", "model", "network", "optimization", "loss",
        "gradient", "regularization", "classifier", "regression",
        "clustering", "embedding", "transformer", "attention",
        "convolution", "recurrent", "ensemble", "boosting",
        "bagging", "cross-validation", "hyperparameter", "feature",
        "dimensionality", "kernel", "bayesian", "probability",
        "distribution", "inference", "generalization", "overfitting",
        "underfitting", "bias", "variance", "activation",
        "backpropagation", "batch", "epoch", "learning rate",
        "momentum", "dropout", "normalization", "pooling",
    ]

    content_lower = content.lower()
    found_concepts = []

    for indicator in concept_indicators:
        if indicator in content_lower:
            count = content_lower.count(indicator)
            if count >= 2:
                found_concepts.append({
                    "name": indicator.title(),
                    "description": f"Concept '{indicator}' found {count} times in {chapter_title}",
                    "category": _categorize_concept(indicator),
                })

    found_concepts.sort(key=lambda x: x["name"])
    return found_concepts[:30]


def _categorize_concept(concept: str) -> str:
    """Categorize a concept into a high-level group."""
    categories = {
        "optimization": ["optimization", "gradient", "loss", "learning rate", "momentum", "backpropagation"],
        "architecture": ["network", "transformer", "attention", "convolution", "recurrent", "pooling", "activation"],
        "regularization": ["regularization", "dropout", "normalization", "overfitting", "underfitting", "bias", "variance"],
        "training": ["batch", "epoch", "cross-validation", "hyperparameter"],
        "methods": ["algorithm", "model", "classifier", "regression", "clustering", "ensemble", "boosting", "bagging"],
        "representation": ["embedding", "feature", "dimensionality", "kernel"],
        "theory": ["bayesian", "probability", "distribution", "inference", "generalization"],
    }
    for category, keywords in categories.items():
        if concept in keywords:
            return category
    return "general"


def index_chapter(
    chapter_info: dict,
    force: bool = False,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> dict:
    """Extract and index a single chapter. Returns status dict."""
    chapter_num = chapter_info["chapter_number"]
    title = chapter_info["title"]

    existing = database.get_all_chapters(db_path)
    already_indexed = any(c["chapter_number"] == chapter_num for c in existing)
    if already_indexed and not force:
        return {"status": "skipped", "chapter": chapter_num, "reason": "already indexed"}

    database.log_extraction(chapter_num, "extracting", f"Starting extraction of {title}", db_path)

    all_content = []
    for pdf_path in chapter_info["pdf_paths"]:
        content = extract_pdf_content(pdf_path)
        if content:
            all_content.append(content)

    if not all_content:
        database.log_extraction(chapter_num, "error", "No content extracted", db_path)
        return {"status": "error", "chapter": chapter_num, "reason": "no content extracted"}

    combined_content = "\n\n".join(all_content)

    chapter_id = database.store_chapter(
        chapter_number=chapter_num,
        title=title,
        source_path=chapter_info["folder_path"],
        markdown_content=combined_content,
        db_path=db_path,
    )

    concepts = extract_concepts(combined_content, title)
    database.store_concepts(chapter_id, concepts, db_path)

    num_chunks = embeddings.index_chapter(
        chapter_id=chapter_id,
        chapter_title=title,
        content=combined_content,
        chroma_path=chroma_path,
    )

    database.log_extraction(chapter_num, "done", f"Indexed {num_chunks} chunks, {len(concepts)} concepts", db_path)

    return {
        "status": "done",
        "chapter": chapter_num,
        "title": title,
        "chunks_indexed": num_chunks,
        "concepts_found": len(concepts),
        "word_count": len(combined_content.split()),
    }


def index_all_chapters(
    force: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
    chroma_path: Optional[str] = None,
) -> list[dict]:
    """Discover and index all chapters. Returns list of status dicts."""
    database.init_db(db_path)
    chapters = discover_chapters(base_path)

    if not chapters:
        return [{"status": "error", "reason": "No chapters found"}]

    results = []
    for chapter_info in chapters:
        result = index_chapter(chapter_info, force=force, db_path=db_path, chroma_path=chroma_path)
        results.append(result)
        logger.info(f"Chapter {chapter_info['chapter_number']}: {result['status']}")

    return results
