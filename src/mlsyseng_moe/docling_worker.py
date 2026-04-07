"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies concepts, and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "convolutional", "recurrent", "transformer", "attention mechanism",
    "regularization", "dropout", "batch normalization", "learning rate",
    "loss function", "cross-entropy", "mean squared error", "optimization",
    "overfitting", "underfitting", "bias-variance", "ensemble",
    "random forest", "decision tree", "support vector", "kernel",
    "clustering", "dimensionality reduction", "principal component",
    "reinforcement learning", "markov decision", "policy gradient",
    "generative adversarial", "autoencoder", "variational",
    "feature engineering", "feature selection", "hyperparameter",
    "cross-validation", "train-test split", "confusion matrix",
    "precision", "recall", "f1 score", "accuracy", "auc", "roc",
    "bayesian", "naive bayes", "logistic regression", "linear regression",
    "gradient boosting", "xgboost", "lightgbm", "catboost",
    "embedding", "word2vec", "bert", "gpt", "fine-tuning",
    "transfer learning", "data augmentation", "normalization",
    "standardization", "one-hot encoding", "label encoding",
    "time series", "sequence model", "lstm", "gru",
    "object detection", "image classification", "segmentation",
    "natural language processing", "sentiment analysis", "tokenization",
    "ml systems", "model deployment", "model serving", "mlops",
    "distributed training", "model compression", "quantization",
    "knowledge distillation", "pruning", "sparse model",
]


def _get_ml_principles_path() -> str:
    return os.environ.get(
        "ML_PRINCIPLES_PATH",
        str(Path.home() / "Desktop" / "Machine Learning Principles - Chapters"),
    )


def discover_chapter_folders(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Find chapter folders under the ML Principles directory."""
    base = Path(base_path or _get_ml_principles_path())
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for item in sorted(base.iterdir()):
        if item.is_dir():
            match = re.match(r"^(\d+)[_\s.-]+(.+)$", item.name)
            if match:
                chapter_id = match.group(1).zfill(2)
                title = match.group(2).replace("_", " ").strip()
            else:
                chapter_id = item.name[:2] if item.name[:2].isdigit() else item.name
                title = item.name

            pdfs = list(item.glob("*.pdf"))
            chapters.append({
                "chapter_id": chapter_id,
                "title": title,
                "folder_path": str(item),
                "pdf_path": str(pdfs[0]) if pdfs else None,
            })
    return chapters


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text from a PDF using docling, falling back to simple extraction."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, falling back to basic extraction")
        return _fallback_pdf_extract(pdf_path)
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)
        return _fallback_pdf_extract(pdf_path)


def _fallback_pdf_extract(pdf_path: str) -> str:
    """Basic PDF text extraction using PyPDF2 or pdfplumber if available."""
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

    logger.error("No PDF extraction library available (install docling, pdfplumber, or PyPDF2)")
    return ""


def extract_concepts(content: str) -> List[str]:
    """Identify ML/AI concepts present in the content."""
    if not content:
        return []

    lower = content.lower()
    found = []
    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword in lower:
            found.append(keyword)
    return sorted(set(found))


def process_chapter(
    chapter_info: Dict[str, str],
    db: Database,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract and store a single chapter."""
    chapter_id = chapter_info["chapter_id"]
    title = chapter_info["title"]
    pdf_path = chapter_info.get("pdf_path")

    existing = db.get_chapter(chapter_id)
    if existing and not force:
        return {
            "chapter_id": chapter_id,
            "status": "skipped",
            "message": "Already extracted",
        }

    db.set_extraction_status(chapter_id, "extracting", f"Processing {title}")

    content = ""
    if pdf_path and Path(pdf_path).exists():
        content = extract_pdf_content(pdf_path)
    else:
        md_files = list(Path(chapter_info["folder_path"]).glob("*.md"))
        txt_files = list(Path(chapter_info["folder_path"]).glob("*.txt"))
        for f in md_files + txt_files:
            content += f"\n\n# {f.stem}\n\n{f.read_text(errors='replace')}"

    if not content:
        db.set_extraction_status(chapter_id, "empty", "No extractable content found")
        return {
            "chapter_id": chapter_id,
            "status": "empty",
            "message": "No content found",
        }

    concepts = extract_concepts(content)

    db.upsert_chapter(
        chapter_id=chapter_id,
        title=title,
        folder_path=chapter_info["folder_path"],
        content_md=content,
        concepts=concepts,
        pdf_path=pdf_path,
    )
    db.set_extraction_status(chapter_id, "done", f"Extracted {len(concepts)} concepts")

    return {
        "chapter_id": chapter_id,
        "status": "done",
        "concepts_count": len(concepts),
        "content_length": len(content),
    }


def run_extraction(
    db: Database,
    base_path: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Extract all chapters and return per-chapter results."""
    chapters = discover_chapter_folders(base_path)
    if not chapters:
        return [{"status": "no_chapters", "message": "No chapter folders found"}]

    results = []
    for ch in chapters:
        try:
            r = process_chapter(ch, db, force=force)
            results.append(r)
        except Exception as e:
            logger.exception("Failed to process chapter %s", ch["chapter_id"])
            db.set_extraction_status(ch["chapter_id"], "error", str(e))
            results.append({
                "chapter_id": ch["chapter_id"],
                "status": "error",
                "message": str(e),
            })
    return results
