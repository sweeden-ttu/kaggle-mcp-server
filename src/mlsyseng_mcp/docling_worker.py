"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

_CONCEPT_KEYWORDS = [
    "regression",
    "classification",
    "neural network",
    "deep learning",
    "gradient descent",
    "backpropagation",
    "optimization",
    "regularization",
    "cross-validation",
    "ensemble",
    "random forest",
    "support vector",
    "svm",
    "bayesian",
    "probability",
    "loss function",
    "activation function",
    "convolution",
    "recurrent",
    "transformer",
    "attention",
    "embedding",
    "feature engineering",
    "dimensionality reduction",
    "pca",
    "clustering",
    "reinforcement learning",
    "generative",
    "discriminative",
    "overfitting",
    "underfitting",
    "bias-variance",
    "hyperparameter",
    "batch normalization",
    "dropout",
    "learning rate",
    "momentum",
    "adam",
    "sgd",
    "data augmentation",
    "transfer learning",
    "fine-tuning",
    "model selection",
    "evaluation metrics",
    "precision",
    "recall",
    "f1 score",
    "auc",
    "roc",
    "confusion matrix",
    "kernel",
    "markov",
    "monte carlo",
    "boosting",
    "bagging",
    "stacking",
    "decision tree",
    "logistic regression",
    "linear regression",
    "polynomial",
    "spline",
    "gaussian process",
    "mixture model",
    "expectation maximization",
    "variational inference",
    "information theory",
    "entropy",
    "mutual information",
    "kl divergence",
]


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover chapter folders containing PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "pdf_count": len(pdfs),
                        "pdfs": [str(p) for p in pdfs],
                    }
                )

    if not chapters:
        pdfs = list(base.glob("*.pdf"))
        for pdf in sorted(pdfs):
            chapters.append(
                {
                    "name": pdf.stem,
                    "path": str(pdf.parent),
                    "pdf_count": 1,
                    "pdfs": [str(pdf)],
                }
            )

    return chapters


def extract_pdf_content(pdf_path: str) -> Tuple[str, int]:
    """
    Extract text content from a PDF using docling.
    Falls back to basic extraction if docling is unavailable.

    Returns (markdown_content, page_count).
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 1
        return md, page_count

    except ImportError:
        logger.info("docling not available, using fallback extraction for %s", pdf_path)
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> Tuple[str, int]:
    """Fallback PDF extraction without docling."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages), len(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages), len(pages)
    except ImportError:
        pass

    return f"[PDF extraction unavailable for {Path(pdf_path).name}]", 0


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """Extract ML/AI concepts from chapter text content."""
    text_lower = text.lower()
    found = []
    seen = set()

    for keyword in _CONCEPT_KEYWORDS:
        if keyword in text_lower and keyword not in seen:
            seen.add(keyword)
            context = _find_context(text, keyword)
            found.append(
                {
                    "name": keyword.title(),
                    "description": context,
                    "category": _categorize_concept(keyword),
                }
            )

    title_pattern = re.compile(
        r"(?:^|\n)#{1,3}\s+(.+?)(?:\n|$)", re.MULTILINE
    )
    for match in title_pattern.finditer(text):
        heading = match.group(1).strip()
        if len(heading) > 3 and heading.lower() not in seen:
            seen.add(heading.lower())
            found.append(
                {
                    "name": heading,
                    "description": f"Section: {heading}",
                    "category": "section",
                }
            )

    return found


def _find_context(text: str, keyword: str, window: int = 200) -> str:
    """Find contextual snippet around a keyword."""
    idx = text.lower().find(keyword)
    if idx == -1:
        return ""
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(keyword) + window // 2)
    snippet = text[start:end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


def _categorize_concept(keyword: str) -> str:
    """Assign a high-level category to a concept keyword."""
    categories = {
        "model": [
            "regression", "classification", "neural network", "deep learning",
            "svm", "support vector", "decision tree", "random forest",
            "logistic regression", "linear regression", "gaussian process",
            "transformer", "recurrent", "convolution",
        ],
        "optimization": [
            "gradient descent", "backpropagation", "optimization", "learning rate",
            "momentum", "adam", "sgd", "loss function",
        ],
        "regularization": [
            "regularization", "dropout", "batch normalization",
            "overfitting", "underfitting", "bias-variance",
        ],
        "evaluation": [
            "cross-validation", "evaluation metrics", "precision", "recall",
            "f1 score", "auc", "roc", "confusion matrix", "model selection",
        ],
        "ensemble": [
            "ensemble", "boosting", "bagging", "stacking",
        ],
        "unsupervised": [
            "clustering", "dimensionality reduction", "pca",
            "mixture model", "expectation maximization",
        ],
        "probabilistic": [
            "bayesian", "probability", "markov", "monte carlo",
            "variational inference", "kl divergence",
        ],
        "information_theory": [
            "information theory", "entropy", "mutual information",
        ],
        "training": [
            "data augmentation", "transfer learning", "fine-tuning",
            "hyperparameter", "feature engineering",
        ],
    }

    for cat, keywords in categories.items():
        if keyword.lower() in keywords:
            return cat
    return "general"


def process_chapter(
    chapter_info: Dict[str, str],
) -> Dict[str, Any]:
    """
    Process a single chapter: extract PDF content and concepts.
    Returns a result dict with the extracted data.
    """
    chapter_name = chapter_info["name"]
    pdfs = chapter_info.get("pdfs", [])

    all_content = []
    total_pages = 0

    for pdf_path in pdfs:
        content, pages = extract_pdf_content(pdf_path)
        all_content.append(content)
        total_pages += pages

    merged_content = "\n\n---\n\n".join(all_content)
    concepts = extract_concepts(merged_content)

    return {
        "chapter_name": chapter_name,
        "source_path": chapter_info["path"],
        "markdown_content": merged_content,
        "concepts": concepts,
        "pages_extracted": total_pages,
    }
