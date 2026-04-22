"""Background PDF extraction using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_principles_path() -> str:
    return os.environ.get(
        "ML_PRINCIPLES_PATH",
        os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
    )


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    root = Path(base_path or _default_principles_path())
    chapters: List[Dict[str, Any]] = []

    if not root.exists():
        logger.warning("ML Principles path does not exist: %s", root)
        return chapters

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = re.match(r"^(\d+)", entry.name)
        if not match:
            continue
        chapter_num = int(match.group(1))
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        chapters.append({
            "chapter_num": chapter_num,
            "title": entry.name,
            "folder": str(entry),
            "pdfs": [str(p) for p in pdfs],
        })

    return chapters


def extract_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text content from a PDF using docling.

    Returns (markdown_text, metadata_dict).
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        metadata = {
            "source": pdf_path,
            "pages": len(result.document.pages) if hasattr(result.document, "pages") else 0,
        }
        return markdown, metadata

    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Minimal fallback when docling is unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout, {"source": pdf_path, "method": "pdftotext"}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return f"[Could not extract content from {pdf_path}]", {"source": pdf_path, "method": "failed"}


def extract_concepts(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract key ML/AI concepts from chapter markdown content.

    Uses heuristics: headings, bold terms, and known ML vocabulary.
    """
    concepts: List[Dict[str, Any]] = []
    seen = set()

    ml_vocab = {
        "gradient descent", "backpropagation", "regularization", "cross-validation",
        "overfitting", "underfitting", "bias-variance", "ensemble", "bagging",
        "boosting", "random forest", "neural network", "deep learning",
        "convolutional", "recurrent", "transformer", "attention",
        "loss function", "optimization", "hyperparameter", "feature engineering",
        "dimensionality reduction", "clustering", "classification", "regression",
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "transfer learning", "fine-tuning", "batch normalization",
        "dropout", "learning rate", "momentum", "adam optimizer",
        "precision", "recall", "f1 score", "auc", "roc",
        "confusion matrix", "cross entropy", "mean squared error",
        "principal component", "singular value", "eigenvalue",
        "kernel", "support vector", "decision tree", "naive bayes",
        "logistic regression", "linear regression", "polynomial",
        "activation function", "sigmoid", "relu", "softmax",
        "embedding", "tokenization", "word2vec", "bert", "gpt",
        "mixture of experts", "gating network", "sparse",
        "distributed training", "data parallelism", "model parallelism",
        "inference", "serving", "deployment", "monitoring",
        "mlops", "pipeline", "experiment tracking", "model registry",
        "feature store", "data drift", "concept drift",
    }

    headings = re.findall(r'^#{1,3}\s+(.+)$', markdown, re.MULTILINE)
    for h in headings:
        clean = h.strip().lower()
        if clean not in seen and len(clean) > 2:
            seen.add(clean)
            concepts.append({
                "concept": h.strip(),
                "category": "heading",
                "relevance": 0.9,
            })

    bold_terms = re.findall(r'\*\*([^*]+)\*\*', markdown)
    for term in bold_terms:
        clean = term.strip().lower()
        if clean not in seen and len(clean) > 2 and len(clean) < 100:
            seen.add(clean)
            concepts.append({
                "concept": term.strip(),
                "category": "key_term",
                "relevance": 0.7,
            })

    text_lower = markdown.lower()
    for vocab in ml_vocab:
        if vocab in text_lower and vocab not in seen:
            seen.add(vocab)
            concepts.append({
                "concept": vocab.title(),
                "category": "ml_concept",
                "relevance": 0.8,
            })

    return concepts


def extract_chapter(
    chapter_info: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract all PDFs for a chapter and merge content.

    Returns (merged_markdown, concepts, metadata).
    """
    all_markdown: List[str] = []
    total_pages = 0

    for pdf_path in chapter_info["pdfs"]:
        md, meta = extract_pdf(pdf_path)
        all_markdown.append(md)
        total_pages += meta.get("pages", 0)

    merged = "\n\n---\n\n".join(all_markdown)
    concepts = extract_concepts(merged)
    metadata = {
        "chapter_num": chapter_info["chapter_num"],
        "title": chapter_info["title"],
        "pdf_count": len(chapter_info["pdfs"]),
        "total_pages": total_pages,
        "concept_count": len(concepts),
    }

    return merged, concepts, metadata
