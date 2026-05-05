"""Background PDF extraction worker using docling for ML Principles chapters."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import ChapterRecord, MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "gradient descent",
    "backpropagation",
    "loss function",
    "regularization",
    "overfitting",
    "underfitting",
    "bias-variance",
    "cross-validation",
    "hyperparameter",
    "feature engineering",
    "dimensionality reduction",
    "ensemble",
    "bagging",
    "boosting",
    "random forest",
    "neural network",
    "deep learning",
    "convolutional",
    "recurrent",
    "transformer",
    "attention mechanism",
    "embedding",
    "optimization",
    "stochastic",
    "batch normalization",
    "dropout",
    "learning rate",
    "activation function",
    "softmax",
    "sigmoid",
    "relu",
    "kernel",
    "support vector",
    "decision tree",
    "clustering",
    "k-means",
    "pca",
    "svd",
    "bayesian",
    "naive bayes",
    "markov",
    "reinforcement learning",
    "q-learning",
    "policy gradient",
    "generative",
    "discriminative",
    "autoencoder",
    "gan",
    "vae",
    "transfer learning",
    "fine-tuning",
    "data augmentation",
    "normalization",
    "tokenization",
    "word2vec",
    "bert",
    "gpt",
    "mixture of experts",
    "sparse",
    "dense",
    "mlp",
    "residual",
    "skip connection",
    "layer normalization",
    "positional encoding",
    "self-attention",
    "multi-head attention",
    "beam search",
    "greedy decoding",
    "temperature",
    "top-k",
    "nucleus sampling",
    "perplexity",
    "bleu",
    "rouge",
    "f1 score",
    "precision",
    "recall",
    "auc",
    "roc",
    "confusion matrix",
    "accuracy",
    "mse",
    "mae",
    "rmse",
    "log loss",
    "cross entropy",
    "kl divergence",
]


def _extract_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling. Falls back to basic extraction if unavailable."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, using fallback PDF extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Fallback PDF text extraction using PyMuPDF or pdfplumber."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    return f"[PDF extraction unavailable for {pdf_path} - install docling, PyMuPDF, or pdfplumber]"


def extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts in extracted text."""
    text_lower = text.lower()
    found = []
    for keyword in CONCEPT_KEYWORDS:
        if keyword in text_lower:
            found.append(keyword)
    return sorted(set(found))


def _chapter_id_from_path(folder_name: str) -> str:
    """Derive a stable chapter_id from a folder name like '08_ML_Systems'."""
    match = re.match(r"(\d+)", folder_name)
    prefix = match.group(1) if match else "00"
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    return f"ch_{prefix}_{slug}"


def _chapter_title_from_folder(folder_name: str) -> str:
    """Derive a human-readable title from the folder name."""
    cleaned = re.sub(r"^\d+[_\s-]*", "", folder_name)
    return cleaned.replace("_", " ").strip() or folder_name


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf")) + list(entry.glob("*.PDF"))
        if not pdfs:
            continue
        chapters.append(
            {
                "folder": entry.name,
                "path": str(entry),
                "pdfs": [str(p) for p in sorted(pdfs)],
                "chapter_id": _chapter_id_from_path(entry.name),
                "title": _chapter_title_from_folder(entry.name),
            }
        )
    return chapters


def extract_chapter(chapter_info: Dict[str, Any]) -> ChapterRecord:
    """Extract content from all PDFs in a chapter folder."""
    all_text_parts = []
    for pdf_path in chapter_info["pdfs"]:
        text = _extract_with_docling(pdf_path)
        all_text_parts.append(text)

    combined_text = "\n\n---\n\n".join(all_text_parts)
    concepts = extract_concepts(combined_text)

    return ChapterRecord(
        chapter_id=chapter_info["chapter_id"],
        title=chapter_info["title"],
        source_pdf=";".join(chapter_info["pdfs"]),
        content_md=combined_text,
        concepts=concepts,
    )


def run_extraction(
    db: MLSysEngDatabase,
    base_path: Optional[str] = None,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """
    Full extraction pipeline: discover → extract → store.

    Returns summary statistics.
    """
    chapters = discover_chapters(base_path)
    if not chapters:
        return {
            "status": "no_chapters_found",
            "base_path": base_path or DEFAULT_ML_PRINCIPLES_PATH,
            "chapters_found": 0,
        }

    extracted = 0
    skipped = 0
    errors = []

    for ch_info in chapters:
        if not force_reindex:
            existing = db.get_chapter(ch_info["chapter_id"])
            if existing:
                skipped += 1
                continue

        try:
            record = extract_chapter(ch_info)
            db.upsert_chapter(record)
            extracted += 1
            logger.info("Extracted chapter: %s (%d words)", record.title, record.word_count)
        except Exception as e:
            errors.append({"chapter": ch_info["chapter_id"], "error": str(e)})
            logger.error("Failed to extract %s: %s", ch_info["chapter_id"], e)

    return {
        "status": "completed",
        "chapters_found": len(chapters),
        "extracted": extracted,
        "skipped": skipped,
        "errors": errors,
    }
