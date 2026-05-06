"""Background PDF extraction using docling."""

import os
import re
import logging
from pathlib import Path
from typing import Optional

from mlsyseng_mcp.database import store_chapter, log_extraction

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.path.expanduser(
    os.environ.get(
        "ML_PRINCIPLES_PATH",
        "~/Desktop/Machine Learning Principles - Chapters"
    )
)

CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "loss function", "regularization",
    "cross-validation", "ensemble", "boosting", "bagging", "random forest",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention", "embedding", "optimization", "hyperparameter",
    "feature engineering", "dimensionality reduction", "clustering",
    "classification", "regression", "overfitting", "underfitting",
    "bias-variance", "kernel", "support vector", "decision tree",
    "reinforcement learning", "generative", "discriminative",
    "bayesian", "markov", "monte carlo", "variational",
    "normalization", "batch norm", "dropout", "activation function",
    "learning rate", "momentum", "adam", "sgd", "inference",
    "transfer learning", "fine-tuning", "data augmentation",
    "precision", "recall", "f1 score", "auc", "roc",
]


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover PDF chapters in the ML Principles directory."""
    path = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not path.exists():
        logger.warning(f"ML Principles path does not exist: {path}")
        return chapters

    for item in sorted(path.iterdir()):
        if item.is_dir():
            match = re.match(r"(\d+)[_\s-]*(.*)", item.name)
            if match:
                chapter_num = match.group(1).zfill(2)
                title = match.group(2).replace("_", " ").strip()
                pdfs = list(item.glob("*.pdf"))
                if pdfs:
                    chapters.append({
                        "chapter_number": chapter_num,
                        "title": title or f"Chapter {chapter_num}",
                        "source_path": str(pdfs[0]),
                        "folder": str(item),
                    })
        elif item.suffix.lower() == ".pdf":
            match = re.match(r"(\d+)[_\s-]*(.*?)\.pdf", item.name, re.IGNORECASE)
            if match:
                chapter_num = match.group(1).zfill(2)
                title = match.group(2).replace("_", " ").strip()
                chapters.append({
                    "chapter_number": chapter_num,
                    "title": title or f"Chapter {chapter_num}",
                    "source_path": str(item),
                    "folder": str(item.parent),
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
        logger.warning("docling not installed, using fallback extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"Docling extraction failed for {pdf_path}: {e}")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Fallback PDF extraction without docling."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        return f"[PDF extraction unavailable for: {pdf_path}]"


def extract_concepts(content: str) -> list[str]:
    """Extract ML/AI concepts from chapter content."""
    content_lower = content.lower()
    found_concepts = []

    for keyword in CONCEPT_KEYWORDS:
        if keyword in content_lower:
            found_concepts.append(keyword)

    sentences = re.split(r'[.!?]\s+', content)
    for sentence in sentences:
        pattern = r'\b(?:defines?|introduces?|presents?)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        matches = re.findall(pattern, sentence)
        for match in matches:
            concept = match.lower()
            if concept not in found_concepts and len(concept) > 3:
                found_concepts.append(concept)

    return sorted(set(found_concepts))


def process_chapter(chapter_info: dict, force: bool = False,
                    db_path: Optional[str] = None) -> dict:
    """Process a single chapter: extract PDF, identify concepts, store."""
    chapter_num = chapter_info["chapter_number"]
    title = chapter_info["title"]
    source_path = chapter_info["source_path"]

    log_extraction(chapter_num, "in_progress", db_path=db_path)

    try:
        content = extract_pdf_content(source_path)

        if not content or content.startswith("[PDF extraction"):
            log_extraction(chapter_num, "failed",
                           error_message="No content extracted", db_path=db_path)
            return {"status": "failed", "chapter": chapter_num, "error": "No content extracted"}

        concepts = extract_concepts(content)

        chapter_id = store_chapter(
            chapter_number=chapter_num,
            title=title,
            source_path=source_path,
            content_md=content,
            concepts=concepts,
            db_path=db_path,
        )

        log_extraction(chapter_num, "completed", db_path=db_path)

        return {
            "status": "completed",
            "chapter": chapter_num,
            "title": title,
            "concepts_found": len(concepts),
            "content_length": len(content),
            "chapter_id": chapter_id,
        }

    except Exception as e:
        error_msg = str(e)
        log_extraction(chapter_num, "failed", error_message=error_msg, db_path=db_path)
        return {"status": "failed", "chapter": chapter_num, "error": error_msg}


def extract_all(force_reindex: bool = False, base_path: Optional[str] = None,
                db_path: Optional[str] = None) -> list[dict]:
    """Extract all chapters from ML Principles directory."""
    chapters = discover_chapters(base_path)

    if not chapters:
        return [{"status": "no_chapters", "message": "No PDF chapters found"}]

    results = []
    for chapter_info in chapters:
        result = process_chapter(chapter_info, force=force_reindex, db_path=db_path)
        results.append(result)

    return results
