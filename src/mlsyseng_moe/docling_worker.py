"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts text from PDFs,
identifies concepts, and stores results in SQLite.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CHAPTER_PATTERN = re.compile(r"^(\d+)[_\s\-]+(.+)$")

ML_CONCEPT_KEYWORDS = [
    "gradient descent",
    "backpropagation",
    "regularization",
    "overfitting",
    "underfitting",
    "bias-variance",
    "cross-validation",
    "hyperparameter",
    "loss function",
    "optimization",
    "neural network",
    "deep learning",
    "convolutional",
    "recurrent",
    "transformer",
    "attention",
    "embedding",
    "feature engineering",
    "dimensionality reduction",
    "ensemble",
    "bagging",
    "boosting",
    "random forest",
    "decision tree",
    "support vector",
    "kernel",
    "clustering",
    "classification",
    "regression",
    "reinforcement learning",
    "generative",
    "discriminative",
    "bayesian",
    "probabilistic",
    "maximum likelihood",
    "posterior",
    "prior",
    "markov",
    "monte carlo",
    "batch normalization",
    "dropout",
    "learning rate",
    "momentum",
    "adam optimizer",
    "sgd",
    "convex optimization",
    "non-convex",
    "saddle point",
    "local minima",
    "global minimum",
    "activation function",
    "relu",
    "sigmoid",
    "softmax",
    "normalization",
    "standardization",
    "data augmentation",
    "transfer learning",
    "fine-tuning",
    "pre-training",
    "tokenization",
    "word2vec",
    "glove",
    "bert",
    "gpt",
    "variational",
    "autoencoder",
    "gan",
    "diffusion",
    "mixture of experts",
    "sparse",
    "dense",
    "model selection",
    "information criterion",
    "aic",
    "bic",
    "f1 score",
    "precision",
    "recall",
    "roc",
    "auc",
    "confusion matrix",
    "accuracy",
]


def discover_chapters(base_path: str | None = None) -> list[dict]:
    """Scan the ML Principles folder for chapter directories."""
    root = Path(base_path or ML_PRINCIPLES_PATH)
    if not root.exists():
        logger.warning("ML Principles path does not exist: %s", root)
        return []

    chapters = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = CHAPTER_PATTERN.match(entry.name)
        if match:
            chapters.append(
                {
                    "chapter_number": match.group(1).zfill(2),
                    "title": match.group(2).replace("_", " ").strip(),
                    "path": str(entry),
                }
            )
    return chapters


def find_pdfs(chapter_path: str) -> list[str]:
    """Find all PDFs within a chapter directory."""
    p = Path(chapter_path)
    return sorted(str(f) for f in p.rglob("*.pdf"))


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using docling, with fallback to PyPDF2."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, trying PyPDF2 fallback")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

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
        logger.warning("Neither docling nor PyPDF2 available")
        return ""
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)
        return ""


def extract_concepts(text: str) -> list[dict]:
    """Identify ML/AI concepts from chapter text content."""
    text_lower = text.lower()
    found = []
    for keyword in ML_CONCEPT_KEYWORDS:
        if keyword in text_lower:
            count = text_lower.count(keyword)
            found.append(
                {
                    "name": keyword.title(),
                    "description": f"Concept '{keyword}' found {count} time(s) in chapter content",
                    "category": _categorize_concept(keyword),
                }
            )
    return found


def _categorize_concept(keyword: str) -> str:
    categories = {
        "optimization": [
            "gradient descent", "backpropagation", "loss function", "optimization",
            "learning rate", "momentum", "adam optimizer", "sgd", "convex optimization",
            "non-convex", "saddle point", "local minima", "global minimum",
        ],
        "architecture": [
            "neural network", "deep learning", "convolutional", "recurrent",
            "transformer", "attention", "autoencoder", "gan", "diffusion",
            "mixture of experts",
        ],
        "training": [
            "regularization", "overfitting", "underfitting", "batch normalization",
            "dropout", "data augmentation", "transfer learning", "fine-tuning",
            "pre-training",
        ],
        "evaluation": [
            "cross-validation", "f1 score", "precision", "recall", "roc", "auc",
            "confusion matrix", "accuracy", "bias-variance", "model selection",
            "information criterion", "aic", "bic",
        ],
        "representation": [
            "embedding", "feature engineering", "dimensionality reduction",
            "tokenization", "word2vec", "glove", "bert", "gpt", "normalization",
            "standardization",
        ],
        "models": [
            "ensemble", "bagging", "boosting", "random forest", "decision tree",
            "support vector", "kernel", "clustering", "classification", "regression",
            "reinforcement learning",
        ],
        "probabilistic": [
            "bayesian", "probabilistic", "maximum likelihood", "posterior", "prior",
            "markov", "monte carlo", "generative", "discriminative", "variational",
        ],
    }
    for cat, keywords in categories.items():
        if keyword in keywords:
            return cat
    return "general"


def extract_chapter(
    chapter_info: dict,
    db_conn=None,
    force: bool = False,
) -> dict:
    """Extract content from a single chapter.

    Returns dict with chapter_number, title, content, concepts, page_count.
    Optionally writes to database if db_conn is provided.
    """
    from . import database as db

    chapter_number = chapter_info["chapter_number"]
    title = chapter_info["title"]
    path = chapter_info["path"]

    pdfs = find_pdfs(path)
    if not pdfs:
        logger.info("No PDFs found for chapter %s", chapter_number)
        return {
            "chapter_number": chapter_number,
            "title": title,
            "content": "",
            "concepts": [],
            "page_count": 0,
        }

    all_text = []
    total_pages = 0
    for pdf in pdfs:
        text = extract_pdf_text(pdf)
        all_text.append(text)
        try:
            from PyPDF2 import PdfReader
            total_pages += len(PdfReader(pdf).pages)
        except Exception:
            total_pages += text.count("\n\n") // 2 or 1

    content = "\n\n---\n\n".join(all_text)
    concepts = extract_concepts(content)

    if db_conn:
        chapter_id = db.upsert_chapter(
            db_conn, chapter_number, title, path, content, total_pages
        )
        db.log_extraction(db_conn, chapter_id, "running")
        try:
            for concept in concepts:
                db.upsert_concept(
                    db_conn,
                    chapter_id,
                    concept["name"],
                    concept["description"],
                    concept["category"],
                )
            db.log_extraction(db_conn, chapter_id, "done")
        except Exception as e:
            db.log_extraction(db_conn, chapter_id, "error", str(e))
            raise

    return {
        "chapter_number": chapter_number,
        "title": title,
        "content": content,
        "concepts": concepts,
        "page_count": total_pages,
    }


def extract_all(
    db_conn=None,
    base_path: str | None = None,
    force_reindex: bool = False,
) -> list[dict]:
    """Extract all chapters from ML Principles PDFs."""
    chapters = discover_chapters(base_path)
    if not chapters:
        logger.warning("No chapters discovered at %s", base_path or ML_PRINCIPLES_PATH)
        return []

    results = []
    for ch in chapters:
        try:
            result = extract_chapter(ch, db_conn, force=force_reindex)
            results.append(result)
            logger.info(
                "Extracted chapter %s: %s (%d concepts)",
                result["chapter_number"],
                result["title"],
                len(result["concepts"]),
            )
        except Exception as e:
            logger.error("Failed to extract chapter %s: %s", ch["chapter_number"], e)
            results.append(
                {
                    "chapter_number": ch["chapter_number"],
                    "title": ch["title"],
                    "content": "",
                    "concepts": [],
                    "page_count": 0,
                    "error": str(e),
                }
            )
    return results
