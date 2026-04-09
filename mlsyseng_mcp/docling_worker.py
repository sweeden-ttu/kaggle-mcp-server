"""Background PDF extraction using docling (with graceful fallback)."""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

_CONCEPT_PATTERNS = [
    r"(?i)\b(gradient descent|sgd|adam|rmsprop)\b",
    r"(?i)\b(backpropagation|forward pass|backward pass)\b",
    r"(?i)\b(regularization|l1|l2|dropout|batch norm)\b",
    r"(?i)\b(convolutional neural network|cnn|convnet)\b",
    r"(?i)\b(recurrent neural network|rnn|lstm|gru)\b",
    r"(?i)\b(transformer|attention mechanism|self-attention)\b",
    r"(?i)\b(decision tree|random forest|xgboost|lightgbm|gradient boosting)\b",
    r"(?i)\b(support vector machine|svm|kernel trick)\b",
    r"(?i)\b(logistic regression|linear regression)\b",
    r"(?i)\b(cross.?validation|k-fold|train.?test split)\b",
    r"(?i)\b(feature engineering|feature selection|pca|dimensionality reduction)\b",
    r"(?i)\b(hyperparameter tuning|grid search|bayesian optimization)\b",
    r"(?i)\b(ensemble method|bagging|boosting|stacking)\b",
    r"(?i)\b(neural network|deep learning|perceptron|mlp)\b",
    r"(?i)\b(overfitting|underfitting|bias-variance)\b",
    r"(?i)\b(loss function|cross entropy|mse|mean squared error)\b",
    r"(?i)\b(activation function|relu|sigmoid|tanh|softmax)\b",
    r"(?i)\b(data augmentation|transfer learning|fine-tuning)\b",
    r"(?i)\b(clustering|k-means|dbscan|hierarchical)\b",
    r"(?i)\b(natural language processing|nlp|tokenization|embedding)\b",
    r"(?i)\b(generative adversarial|gan|vae|autoencoder)\b",
    r"(?i)\b(reinforcement learning|q-learning|policy gradient)\b",
    r"(?i)\b(model evaluation|precision|recall|f1.?score|auc|roc)\b",
]


def discover_chapter_folders(base_path: Optional[str] = None) -> List[Path]:
    """Return sorted list of chapter folders that contain at least one PDF."""
    root = Path(base_path or ML_PRINCIPLES_PATH)
    if not root.exists():
        logger.warning("ML Principles path does not exist: %s", root)
        return []
    folders = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*.pdf")):
            folders.append(child)
    return folders


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF, preferring docling then falling back to PyPDF2/pdfplumber."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not installed, trying PyPDF2")
    except Exception as exc:
        logger.warning("docling failed for %s: %s", pdf_path, exc)

    try:
        import PyPDF2

        text_parts: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.info("PyPDF2 not installed, trying pdfplumber")
    except Exception as exc:
        logger.warning("PyPDF2 failed for %s: %s", pdf_path, exc)

    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("No PDF extraction library available")
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", pdf_path, exc)

    return ""


def extract_concepts(text: str) -> List[str]:
    """Pull ML/AI concept mentions from text."""
    found: set = set()
    for pattern in _CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text):
            found.add(match.group(0).strip().lower())
    return sorted(found)


def extract_chapter(folder: Path) -> Dict:
    """Extract all PDFs in a chapter folder and return combined result."""
    pdfs = sorted(folder.glob("*.pdf"))
    parts: List[str] = []
    for pdf in pdfs:
        text = extract_pdf_text(pdf)
        if text:
            parts.append(text)
    combined = "\n\n".join(parts)
    title = folder.name
    number_match = re.match(r"(\d+)", folder.name)
    chapter_num = number_match.group(1) if number_match else "00"
    concepts = extract_concepts(combined)
    return {
        "folder_name": folder.name,
        "title": title,
        "chapter_num": chapter_num,
        "markdown": combined,
        "concepts": concepts,
        "pdf_count": len(pdfs),
    }


def extract_all_chapters(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    on_progress=None,
) -> List[Dict]:
    """Extract knowledge from every chapter folder.

    Args:
        base_path: Override for ML_PRINCIPLES_PATH.
        force_reindex: If True re-extract even if already done.
        on_progress: Optional callback(folder_name, status, message).

    Returns:
        List of chapter dicts.
    """
    folders = discover_chapter_folders(base_path)
    results: List[Dict] = []
    for folder in folders:
        if on_progress:
            on_progress(folder.name, "running", "Extracting...")
        try:
            chapter = extract_chapter(folder)
            results.append(chapter)
            if on_progress:
                on_progress(
                    folder.name,
                    "done",
                    f"Extracted {chapter['pdf_count']} PDFs, {len(chapter['concepts'])} concepts",
                )
        except Exception as exc:
            logger.error("Failed to extract %s: %s", folder.name, exc)
            if on_progress:
                on_progress(folder.name, "error", str(exc))
    return results
