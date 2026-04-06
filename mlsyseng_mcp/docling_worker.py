"""Background PDF extraction using docling for ML Principles chapters."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import Database

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH", os.path.expanduser("~/ML_Principles_Chapters")
)

ML_CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "loss function", "regularization", "dropout", "batch normalization",
    "convolutional", "recurrent", "transformer", "attention mechanism",
    "embedding", "feature engineering", "cross-validation", "hyperparameter",
    "overfitting", "underfitting", "bias-variance", "ensemble",
    "random forest", "gradient boosting", "support vector", "kernel",
    "dimensionality reduction", "PCA", "clustering", "classification",
    "regression", "optimization", "stochastic", "momentum", "learning rate",
    "activation function", "softmax", "sigmoid", "ReLU", "LSTM", "GRU",
    "autoencoder", "generative", "discriminative", "Bayesian",
    "reinforcement learning", "policy gradient", "Q-learning",
    "transfer learning", "fine-tuning", "data augmentation",
    "normalization", "standardization", "tokenization",
    "precision", "recall", "F1 score", "AUC", "ROC",
    "confusion matrix", "mean squared error", "cross entropy",
]


def _try_docling_extract(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF using docling if available, else fallback."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown()
        pages = result.document.num_pages if hasattr(result.document, "num_pages") else 0
        return md, pages
    except ImportError:
        logger.warning("docling not installed, using fallback text extraction")
        return _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> Tuple[str, int]:
    """Fallback extraction when docling is not available."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = len(doc)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(text_parts), pages
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text

        text = extract_text(pdf_path)
        return text, 0
    except ImportError:
        pass

    return f"[Could not extract text from {pdf_path} - install docling, PyMuPDF, or pdfminer]", 0


def extract_concepts(text: str) -> List[Dict[str, Any]]:
    """Extract ML/AI concepts from text content."""
    text_lower = text.lower()
    found = []
    seen = set()

    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword.lower() in text_lower and keyword.lower() not in seen:
            seen.add(keyword.lower())
            count = text_lower.count(keyword.lower())
            context = _find_context(text, keyword)
            found.append({
                "name": keyword,
                "description": context,
                "category": _categorize_concept(keyword),
                "confidence": min(1.0, count / 5.0),
            })

    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    for match in heading_pattern.finditer(text):
        heading = match.group(1).strip()
        if len(heading) > 3 and heading.lower() not in seen:
            seen.add(heading.lower())
            found.append({
                "name": heading,
                "description": f"Section heading: {heading}",
                "category": "topic",
                "confidence": 0.8,
            })

    return found


def _find_context(text: str, keyword: str, window: int = 200) -> str:
    """Find surrounding context for a keyword."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(keyword) + window // 2)
    snippet = text[start:end].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


def _categorize_concept(keyword: str) -> str:
    """Categorize a concept by its keyword."""
    categories = {
        "architecture": [
            "neural network", "convolutional", "recurrent", "transformer",
            "LSTM", "GRU", "autoencoder", "attention mechanism",
        ],
        "optimization": [
            "gradient descent", "backpropagation", "optimization", "stochastic",
            "momentum", "learning rate", "loss function",
        ],
        "regularization": [
            "regularization", "dropout", "batch normalization", "overfitting",
            "underfitting", "bias-variance",
        ],
        "evaluation": [
            "precision", "recall", "F1 score", "AUC", "ROC",
            "confusion matrix", "mean squared error", "cross entropy",
            "cross-validation",
        ],
        "technique": [
            "feature engineering", "dimensionality reduction", "PCA",
            "clustering", "ensemble", "random forest", "gradient boosting",
            "transfer learning", "fine-tuning", "data augmentation",
            "normalization", "standardization", "tokenization",
        ],
    }
    kl = keyword.lower()
    for cat, keywords in categories.items():
        if any(kl == k.lower() for k in keywords):
            return cat
    return "general"


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover chapter folders and PDFs in the ML Principles directory."""
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append({
                    "name": item.name,
                    "path": str(item),
                    "pdf_path": str(pdfs[0]),
                    "pdf_count": len(pdfs),
                })
        elif item.suffix.lower() == ".pdf":
            chapters.append({
                "name": item.stem,
                "path": str(item.parent),
                "pdf_path": str(item),
                "pdf_count": 1,
            })

    return chapters


def extract_chapter(
    chapter_info: Dict[str, str], db: Database, force: bool = False
) -> Dict[str, Any]:
    """Extract a single chapter's content and store in database."""
    name = chapter_info["name"]
    pdf_path = chapter_info["pdf_path"]

    existing = [c for c in db.get_chapters() if c["chapter_name"] == name]
    if existing and not force:
        return {
            "chapter": name,
            "status": "skipped",
            "message": "Already extracted (use force=True to re-extract)",
        }

    db.log_extraction(name, "started")

    try:
        markdown, page_count = _try_docling_extract(pdf_path)
        chapter_id = db.upsert_chapter(name, pdf_path, markdown, page_count)
        concepts = extract_concepts(markdown)
        concept_count = db.add_concepts(chapter_id, concepts)

        db.log_extraction(name, "completed", f"{concept_count} concepts extracted")
        return {
            "chapter": name,
            "status": "success",
            "page_count": page_count,
            "word_count": len(markdown.split()),
            "concepts_extracted": concept_count,
        }
    except Exception as e:
        db.log_extraction(name, "failed", str(e))
        return {"chapter": name, "status": "error", "message": str(e)}


def extract_all_chapters(
    base_path: Optional[str] = None,
    db: Optional[Database] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Extract all discovered chapters."""
    if db is None:
        db = Database()

    chapters = discover_chapters(base_path)
    if not chapters:
        return [{"status": "no_chapters", "message": f"No chapters found at {base_path or DEFAULT_ML_PRINCIPLES_PATH}"}]

    results = []
    for ch in chapters:
        result = extract_chapter(ch, db, force=force)
        results.append(result)

    return results
