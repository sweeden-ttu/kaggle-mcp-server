"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser(
        "~/Desktop/Machine Learning Principles - Chapters"
    ),
)

ML_CONCEPT_KEYWORDS = [
    "gradient descent",
    "backpropagation",
    "regularization",
    "overfitting",
    "underfitting",
    "bias-variance",
    "cross-validation",
    "ensemble",
    "bagging",
    "boosting",
    "neural network",
    "deep learning",
    "convolution",
    "recurrent",
    "transformer",
    "attention",
    "optimization",
    "loss function",
    "activation function",
    "feature engineering",
    "dimensionality reduction",
    "clustering",
    "classification",
    "regression",
    "reinforcement learning",
    "generative model",
    "discriminative model",
    "bayesian",
    "kernel",
    "support vector",
    "decision tree",
    "random forest",
    "hyperparameter",
    "batch normalization",
    "dropout",
    "learning rate",
    "momentum",
    "adam optimizer",
    "data augmentation",
    "transfer learning",
    "fine-tuning",
    "embedding",
    "tokenization",
]


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover chapter folders in ML Principles directory."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning(f"ML Principles path not found: {base}")
        return []

    chapters = []
    pattern = re.compile(r"^(\d+)[_\s\-]+(.+)$")

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if match:
            chapter_num = int(match.group(1))
            title = match.group(2).replace("_", " ").strip()
            pdfs = list(item.glob("*.pdf"))
            chapters.append({
                "chapter_number": chapter_num,
                "title": title,
                "folder_path": str(item),
                "pdf_path": str(pdfs[0]) if pdfs else None,
            })

    return chapters


def extract_pdf_content(pdf_path: str) -> list[dict]:
    """Extract content blocks from a PDF using docling.

    Falls back to basic PyPDF2/pdfplumber extraction if docling unavailable.
    """
    blocks = []

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)

        for i, item in enumerate(result.document.iterate_items()):
            text = item.text if hasattr(item, "text") else str(item)
            if text.strip():
                block_type = "text"
                if hasattr(item, "label"):
                    label = str(item.label).lower()
                    if "heading" in label or "title" in label:
                        block_type = "heading"
                    elif "table" in label:
                        block_type = "table"
                    elif "formula" in label or "equation" in label:
                        block_type = "formula"

                blocks.append({
                    "content": text.strip(),
                    "block_type": block_type,
                    "position": i,
                })

    except ImportError:
        logger.info("docling not available, trying pdfplumber fallback")
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        blocks.append({
                            "content": text.strip(),
                            "block_type": "text",
                            "page_number": page_num + 1,
                            "position": page_num,
                        })
        except ImportError:
            logger.warning("No PDF extraction library available")

    return blocks


def extract_concepts(text: str) -> list[dict]:
    """Extract ML/AI concepts from text content."""
    text_lower = text.lower()
    found = []

    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword in text_lower:
            idx = text_lower.index(keyword)
            start = max(0, idx - 50)
            end = min(len(text), idx + len(keyword) + 100)
            context = text[start:end].strip()

            found.append({
                "concept_name": keyword.title(),
                "description": context,
                "category": _categorize_concept(keyword),
            })

    return found


def _categorize_concept(keyword: str) -> str:
    """Categorize a concept into a broad ML category."""
    optimization_terms = {
        "gradient descent", "backpropagation", "optimization",
        "loss function", "learning rate", "momentum", "adam optimizer",
    }
    architecture_terms = {
        "neural network", "deep learning", "convolution", "recurrent",
        "transformer", "attention", "activation function",
    }
    training_terms = {
        "regularization", "overfitting", "underfitting", "batch normalization",
        "dropout", "data augmentation", "transfer learning", "fine-tuning",
        "cross-validation", "hyperparameter",
    }
    model_terms = {
        "ensemble", "bagging", "boosting", "decision tree", "random forest",
        "kernel", "support vector", "bayesian", "generative model",
        "discriminative model",
    }

    if keyword in optimization_terms:
        return "optimization"
    elif keyword in architecture_terms:
        return "architecture"
    elif keyword in training_terms:
        return "training"
    elif keyword in model_terms:
        return "models"
    else:
        return "general"


def process_chapter(
    chapter_info: dict,
    force: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Process a single chapter: extract PDF, store content, identify concepts."""
    chapter_num = chapter_info["chapter_number"]
    title = chapter_info["title"]
    folder_path = chapter_info["folder_path"]
    pdf_path = chapter_info.get("pdf_path")

    chapter_id = db.insert_chapter(
        chapter_number=chapter_num,
        title=title,
        folder_path=folder_path,
        pdf_path=pdf_path,
        db_path=db_path,
    )

    db.log_extraction(chapter_id, "in_progress", db_path=db_path)

    try:
        blocks = []
        if pdf_path and Path(pdf_path).exists():
            blocks = extract_pdf_content(pdf_path)

        all_text = ""
        for i, block in enumerate(blocks):
            db.insert_content_block(
                chapter_id=chapter_id,
                content=block["content"],
                block_type=block.get("block_type", "text"),
                page_number=block.get("page_number"),
                position=block.get("position", i),
                db_path=db_path,
            )
            all_text += " " + block["content"]

        concepts = extract_concepts(all_text)
        for concept in concepts:
            db.insert_concept(
                chapter_id=chapter_id,
                concept_name=concept["concept_name"],
                description=concept.get("description"),
                category=concept.get("category"),
                db_path=db_path,
            )

        db.log_extraction(chapter_id, "completed", db_path=db_path)

        return {
            "chapter_id": chapter_id,
            "chapter_number": chapter_num,
            "title": title,
            "blocks_extracted": len(blocks),
            "concepts_found": len(concepts),
            "status": "completed",
        }

    except Exception as e:
        db.log_extraction(chapter_id, "failed", str(e), db_path=db_path)
        logger.error(f"Failed to process chapter {chapter_num}: {e}")
        return {
            "chapter_id": chapter_id,
            "chapter_number": chapter_num,
            "title": title,
            "blocks_extracted": 0,
            "concepts_found": 0,
            "status": "failed",
            "error": str(e),
        }


def run_extraction(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Run full extraction pipeline on all discovered chapters."""
    db.init_db(db_path)
    chapters = discover_chapters(base_path)

    if not chapters:
        return [{
            "status": "no_chapters",
            "message": f"No chapters found at {base_path or ML_PRINCIPLES_PATH}",
        }]

    results = []
    for chapter_info in chapters:
        result = process_chapter(chapter_info, force=force_reindex, db_path=db_path)
        results.append(result)

    return results
