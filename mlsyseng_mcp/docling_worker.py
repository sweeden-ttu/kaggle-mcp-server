"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CHAPTER_PATTERN = re.compile(r"^(\d+)[_\s-]+(.+)$")

ML_CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "loss function", "regularization",
    "overfitting", "underfitting", "cross-validation", "hyperparameter",
    "neural network", "deep learning", "convolutional", "recurrent",
    "transformer", "attention", "embedding", "feature engineering",
    "ensemble", "random forest", "boosting", "bagging",
    "support vector", "kernel", "dimensionality reduction", "PCA",
    "clustering", "k-means", "classification", "regression",
    "optimization", "learning rate", "batch normalization", "dropout",
    "activation function", "sigmoid", "relu", "softmax",
    "precision", "recall", "f1", "accuracy", "AUC", "ROC",
    "bias-variance", "maximum likelihood", "bayesian",
    "generative", "discriminative", "autoencoder", "GAN",
    "reinforcement learning", "policy gradient", "reward",
    "transfer learning", "fine-tuning", "pre-training",
    "data augmentation", "normalization", "standardization",
    "pipeline", "model selection", "inference", "deployment",
]


def discover_chapters(base_path: Optional[str] = None) -> List[Tuple[int, str, Path]]:
    """Discover chapter directories and return (number, name, path) tuples."""
    root = Path(base_path or ML_PRINCIPLES_PATH)
    if not root.exists():
        logger.warning("ML Principles path not found: %s", root)
        return []

    chapters = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = CHAPTER_PATTERN.match(entry.name)
        if match:
            num = int(match.group(1))
            name = match.group(2).replace("_", " ").strip()
            chapters.append((num, name, entry))
    return chapters


def extract_pdf_content(pdf_path: Path) -> str:
    """Extract text content from a PDF using docling (falls back to simple extraction)."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, using fallback PDF extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: Path) -> str:
    """Simple fallback PDF text extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    return f"[PDF extraction unavailable for {pdf_path.name}]"


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text content."""
    text_lower = text.lower()
    found = []
    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword.lower() in text_lower:
            found.append(keyword)
    return sorted(set(found))


def extract_chapter(chapter_path: Path) -> Tuple[str, List[str]]:
    """Extract content and concepts from all PDFs in a chapter directory."""
    pdfs = list(chapter_path.glob("*.pdf"))
    if not pdfs:
        pdfs = list(chapter_path.glob("*.PDF"))

    all_text = []
    for pdf in pdfs:
        text = extract_pdf_content(pdf)
        if text:
            all_text.append(f"## {pdf.stem}\n\n{text}")

    combined = "\n\n---\n\n".join(all_text) if all_text else ""
    concepts = extract_concepts(combined) if combined else []
    return combined, concepts
