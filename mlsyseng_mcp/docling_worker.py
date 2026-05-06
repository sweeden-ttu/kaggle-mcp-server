"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts PDF content to markdown,
and identifies key ML/AI concepts from each chapter.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/ML-Principles-Chapters"),
)

CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "regularization", "overfitting", "underfitting", "bias-variance",
    "cross-validation", "ensemble", "random forest", "decision tree",
    "support vector", "svm", "logistic regression", "linear regression",
    "dimensionality reduction", "pca", "principal component",
    "clustering", "k-means", "reinforcement learning", "q-learning",
    "convolutional", "cnn", "recurrent", "rnn", "lstm", "transformer",
    "attention mechanism", "batch normalization", "dropout",
    "hyperparameter", "learning rate", "optimizer", "adam", "sgd",
    "loss function", "cross entropy", "mean squared error",
    "feature engineering", "feature selection", "data augmentation",
    "transfer learning", "fine-tuning", "pre-training",
    "generative", "discriminative", "bayesian", "maximum likelihood",
    "hypothesis testing", "confidence interval", "p-value",
    "precision", "recall", "f1 score", "auc", "roc",
    "embedding", "word2vec", "tokenization", "normalization",
    "activation function", "relu", "sigmoid", "softmax",
    "convex optimization", "stochastic", "mini-batch",
    "kernel", "gaussian process", "markov", "hidden markov",
    "autoencoder", "variational", "gan", "generative adversarial",
    "mixture of experts", "expert system", "knowledge distillation",
    "model compression", "pruning", "quantization",
    "federated learning", "differential privacy",
    "natural language processing", "nlp", "computer vision",
    "object detection", "segmentation", "classification",
    "regression", "anomaly detection", "time series",
]


def _try_docling_extract(pdf_path: str) -> str:
    """Extract text from PDF using docling if available, else fall back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic PDF extraction")
        return _basic_pdf_extract(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _basic_pdf_extract(pdf_path)


def _basic_pdf_extract(pdf_path: str) -> str:
    """Basic PDF text extraction using PyPDF2 or pdfminer as fallback."""
    try:
        import PyPDF2
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text
        return extract_text(pdf_path)
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available (install docling, PyPDF2, or pdfminer.six)"
    )
    return ""


def extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts present in the text."""
    text_lower = text.lower()
    found = []
    for kw in CONCEPT_KEYWORDS:
        if kw in text_lower:
            found.append(kw)
    return sorted(set(found))


def _parse_chapter_number(name: str) -> str:
    """Extract chapter number from folder/file name like '08_ML Systems'."""
    match = re.match(r"(\d+)", name)
    return match.group(1).zfill(2) if match else "00"


def scan_chapters(
    base_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs.

    Returns a list of dicts with chapter_id, title, folder_path, pdf_path.
    """
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            if entry.suffix.lower() == ".pdf":
                chap_num = _parse_chapter_number(entry.stem)
                chapters.append({
                    "chapter_id": f"ch_{chap_num}",
                    "title": entry.stem,
                    "folder_path": str(entry.parent),
                    "pdf_path": str(entry),
                })
            continue

        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue

        chap_num = _parse_chapter_number(entry.name)
        pdf_path = str(pdfs[0])

        chapters.append({
            "chapter_id": f"ch_{chap_num}",
            "title": entry.name,
            "folder_path": str(entry),
            "pdf_path": pdf_path,
        })

    return chapters


def extract_chapter(
    pdf_path: str,
) -> Tuple[str, List[str]]:
    """Extract content and concepts from a single PDF chapter.

    Returns (markdown_content, concept_list).
    """
    content = _try_docling_extract(pdf_path)
    concepts = extract_concepts(content)
    return content, concepts


def extract_all_chapters(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    db=None,
) -> List[Dict[str, Any]]:
    """Extract all chapters: scan, extract, store in DB.

    Args:
        base_path: Override for ML Principles folder.
        force_reindex: Re-extract even if already in DB.
        db: MLSysEngDatabase instance (optional).

    Returns:
        List of chapter info dicts with extraction results.
    """
    chapters = scan_chapters(base_path)
    results = []

    for chap in chapters:
        chapter_id = chap["chapter_id"]

        if db and not force_reindex and db.chapter_exists(chapter_id):
            logger.info("Skipping already-indexed chapter %s", chapter_id)
            results.append({**chap, "status": "skipped"})
            continue

        if db:
            db.log_extraction_event(chapter_id, "extraction_start", chap["pdf_path"])

        try:
            content, concepts = extract_chapter(chap["pdf_path"])
            chap["content_md"] = content
            chap["concepts"] = concepts

            if db:
                db.upsert_chapter(
                    chapter_id=chapter_id,
                    title=chap["title"],
                    folder_path=chap.get("folder_path", ""),
                    pdf_path=chap["pdf_path"],
                    content_md=content,
                    concepts=concepts,
                )
                db.log_extraction_event(
                    chapter_id,
                    "extraction_complete",
                    f"concepts={len(concepts)}, chars={len(content)}",
                )

            results.append({**chap, "status": "extracted"})
            logger.info(
                "Extracted chapter %s: %d concepts, %d chars",
                chapter_id, len(concepts), len(content),
            )
        except Exception as e:
            logger.error("Failed to extract chapter %s: %s", chapter_id, e)
            if db:
                db.log_extraction_event(chapter_id, "extraction_error", str(e))
            results.append({**chap, "status": "error", "error": str(e)})

    return results
