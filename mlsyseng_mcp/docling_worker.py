"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.path.expanduser(
    os.environ.get(
        "ML_PRINCIPLES_PATH",
        "~/Desktop/Machine Learning Principles - Chapters",
    )
)

CHAPTER_SLUG_RE = re.compile(r"^(\d+)[_\s-]*(.*)", re.IGNORECASE)


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def discover_chapter_folders(base_path: str = ML_PRINCIPLES_PATH) -> List[Dict[str, str]]:
    """Find chapter folders and their PDFs under *base_path*."""
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters: List[Dict[str, str]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        m = CHAPTER_SLUG_RE.match(entry.name)
        if not m:
            continue
        number, rest = m.group(1), m.group(2)
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            pdfs = list(entry.glob("**/*.pdf"))
        chapters.append(
            {
                "number": number,
                "name": rest.strip() or f"Chapter {number}",
                "slug": f"{number}_{_slugify(rest or f'chapter_{number}')}",
                "folder": str(entry),
                "pdfs": [str(p) for p in pdfs],
            }
        )
    return chapters


def _extract_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, falling back to basic extraction")
        return _extract_basic(pdf_path)
    except Exception as exc:
        logger.warning("docling failed on %s: %s – falling back", pdf_path, exc)
        return _extract_basic(pdf_path)


def _extract_basic(pdf_path: str) -> str:
    """Minimal PDF-to-text using PyPDF2 or pdfplumber if available."""
    try:
        import pdfplumber

        pages: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

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

    return f"[Could not extract text from {pdf_path} – install docling, pdfplumber, or PyPDF2]"


# ── Concept extraction heuristics ─────────────────────────────────────

_ML_KEYWORDS = {
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "convolutional", "recurrent", "transformer", "attention", "embedding",
    "regularization", "dropout", "batch normalization", "loss function",
    "cross entropy", "mean squared error", "optimizer", "learning rate",
    "overfitting", "underfitting", "bias", "variance", "ensemble",
    "random forest", "decision tree", "support vector machine", "svm",
    "linear regression", "logistic regression", "bayesian", "markov",
    "hidden markov model", "principal component analysis", "pca",
    "feature engineering", "feature selection", "hyperparameter",
    "cross validation", "confusion matrix", "precision", "recall",
    "f1 score", "roc curve", "auc", "transfer learning", "fine tuning",
    "data augmentation", "normalization", "standardization",
    "activation function", "relu", "sigmoid", "softmax",
    "generative adversarial network", "gan", "autoencoder", "vae",
    "diffusion model", "mixture of experts", "moe",
}


def extract_concepts(markdown: str) -> List[Dict[str, str]]:
    """Pull ML/AI concepts from extracted markdown."""
    lower = markdown.lower()
    found: List[Dict[str, str]] = []
    for kw in _ML_KEYWORDS:
        if kw in lower:
            idx = lower.index(kw)
            start = max(0, idx - 80)
            end = min(len(markdown), idx + len(kw) + 120)
            snippet = markdown[start:end].replace("\n", " ").strip()
            found.append({"term": kw, "definition": snippet, "category": "ml_concept"})
    return found


def extract_chapter(
    chapter_info: Dict[str, str],
) -> Tuple[str, List[Dict[str, str]]]:
    """Extract markdown + concepts for a single chapter.

    Returns (markdown, concepts).
    """
    pdfs = chapter_info.get("pdfs", [])
    all_text: List[str] = []
    for pdf_path in pdfs:
        text = _extract_with_docling(pdf_path)
        all_text.append(text)
    markdown = "\n\n---\n\n".join(all_text) if all_text else ""
    concepts = extract_concepts(markdown)
    return markdown, concepts
