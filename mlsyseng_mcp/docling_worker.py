"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "regularization", "overfitting", "underfitting", "bias-variance",
    "cross-validation", "ensemble", "random forest", "decision tree",
    "support vector", "logistic regression", "linear regression",
    "clustering", "dimensionality reduction", "PCA", "feature engineering",
    "hyperparameter", "learning rate", "batch normalization",
    "convolutional", "recurrent", "attention", "transformer",
    "reinforcement learning", "Q-learning", "policy gradient",
    "generative", "discriminative", "autoencoder", "GAN",
    "Bayesian", "maximum likelihood", "loss function", "optimization",
    "stochastic", "mini-batch", "momentum", "Adam",
    "dropout", "data augmentation", "transfer learning", "fine-tuning",
    "embedding", "tokenization", "sequence model", "time series",
    "classification", "regression", "anomaly detection", "recommendation",
    "kernel method", "boosting", "bagging", "XGBoost", "LightGBM",
    "natural language processing", "computer vision", "model selection",
    "pipeline", "MLOps", "deployment", "inference", "training",
    "distributed training", "model parallelism", "data parallelism",
]


def _try_docling_extract(pdf_path: str) -> str:
    """Extract text from PDF using docling, with graceful fallback."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return _basic_pdf_extract(pdf_path)
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)
        return _basic_pdf_extract(pdf_path)


def _basic_pdf_extract(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2 or pdfminer."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        pass

    return f"[Could not extract text from {pdf_path} - install docling, PyPDF2, or pdfminer]"


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text content."""
    text_lower = text.lower()
    found = []
    for concept in CONCEPT_KEYWORDS:
        if concept.lower() in text_lower:
            found.append(concept)
    return sorted(set(found))


def _chapter_number_and_name(folder_name: str) -> Tuple[str, str]:
    """Parse chapter number and name from folder name."""
    match = re.match(r"(\d+)[_\s.-]*(.+)", folder_name)
    if match:
        num = match.group(1).zfill(2)
        name = match.group(2).strip().replace("_", " ").replace("-", " ").title()
        return num, name
    return "00", folder_name.replace("_", " ").title()


def scan_chapter_folders(
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan ML Principles folder for chapter directories with PDFs."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.info("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        num, name = _chapter_number_and_name(entry.name)
        chapters.append({
            "folder": str(entry),
            "folder_name": entry.name,
            "chapter_number": num,
            "chapter_name": f"{num}_{name}",
            "pdfs": [str(p) for p in pdfs],
        })
    return chapters


def extract_chapter(
    chapter_info: Dict[str, Any],
    db: Database,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract content from a single chapter's PDFs."""
    chapter_name = chapter_info["chapter_name"]

    existing = db.get_chapter(chapter_name)
    if existing and existing["status"] == "extracted" and not force:
        return {"chapter_name": chapter_name, "status": "skipped", "message": "already extracted"}

    db.log_extraction(chapter_name, "started")

    try:
        all_text = []
        for pdf_path in chapter_info["pdfs"]:
            text = _try_docling_extract(pdf_path)
            all_text.append(text)

        combined = "\n\n---\n\n".join(all_text)
        concepts = extract_concepts(combined)

        db.upsert_chapter(
            chapter_name=chapter_name,
            source_path=chapter_info["folder"],
            markdown_content=combined,
            concepts=concepts,
            status="extracted",
        )
        db.log_extraction(chapter_name, "completed", f"extracted {len(concepts)} concepts")

        return {
            "chapter_name": chapter_name,
            "status": "extracted",
            "concepts_count": len(concepts),
            "concepts": concepts,
            "text_length": len(combined),
        }
    except Exception as e:
        db.log_extraction(chapter_name, "failed", str(e))
        db.upsert_chapter(
            chapter_name=chapter_name,
            source_path=chapter_info["folder"],
            markdown_content="",
            concepts=[],
            status="failed",
        )
        return {"chapter_name": chapter_name, "status": "failed", "message": str(e)}


def extract_all_chapters(
    base_path: Optional[str] = None,
    db: Optional[Database] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Extract all chapters from ML Principles PDFs."""
    if db is None:
        db = Database()

    chapters = scan_chapter_folders(base_path)
    if not chapters:
        return [{"status": "no_chapters", "message": f"No chapter folders found at {base_path or DEFAULT_ML_PRINCIPLES_PATH}"}]

    results = []
    for ch in chapters:
        result = extract_chapter(ch, db, force=force)
        results.append(result)

    return results
