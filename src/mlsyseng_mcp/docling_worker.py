"""Background PDF extraction worker using docling for MLSysEng MoE."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH", os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters")
)

CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "loss function", "regularization", "optimization", "convolution",
    "recurrent", "transformer", "attention", "embedding", "feature",
    "classification", "regression", "clustering", "dimensionality reduction",
    "ensemble", "boosting", "bagging", "random forest", "decision tree",
    "support vector", "kernel", "bayesian", "probabilistic", "inference",
    "generative", "discriminative", "reinforcement learning", "policy",
    "reward", "value function", "q-learning", "cross-validation",
    "hyperparameter", "overfitting", "underfitting", "bias", "variance",
    "activation function", "batch normalization", "dropout", "learning rate",
    "momentum", "adam", "sgd", "data augmentation", "transfer learning",
    "fine-tuning", "pre-training", "tokenization", "word2vec", "glove",
    "BERT", "GPT", "diffusion", "GAN", "autoencoder", "VAE",
    "mixture of experts", "gating network", "sparse", "dense",
    "model selection", "evaluation", "precision", "recall", "f1",
    "ROC", "AUC", "confusion matrix", "feature engineering",
    "feature selection", "PCA", "t-SNE", "UMAP", "normalization",
    "standardization", "imputation", "pipeline", "deployment",
    "monitoring", "MLOps", "experiment tracking", "model registry",
]


def discover_chapter_folders(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover chapter folders in the ML Principles directory.

    Expected structure: folders named like '01_Introduction', '02_Linear_Regression', etc.
    Each folder may contain one or more PDF files.
    """
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    chapter_pattern = re.compile(r"^(\d+)[_\s-](.+)$")

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            if item.suffix.lower() == ".pdf":
                match = chapter_pattern.match(item.stem)
                if match:
                    chapters.append({
                        "chapter_num": int(match.group(1)),
                        "title": match.group(2).replace("_", " ").strip(),
                        "path": str(item),
                        "pdfs": [str(item)],
                    })
            continue

        match = chapter_pattern.match(item.name)
        if not match:
            continue

        pdfs = sorted(str(p) for p in item.glob("*.pdf"))
        if not pdfs:
            pdfs = sorted(str(p) for p in item.glob("**/*.pdf"))

        chapters.append({
            "chapter_num": int(match.group(1)),
            "title": match.group(2).replace("_", " ").strip(),
            "path": str(item),
            "pdfs": pdfs,
        })

    return chapters


def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    """Extract text from a PDF using docling, falling back to simpler methods.

    Returns (markdown_text, page_count).
    """
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        md = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 0
        return md, page_count
    except ImportError:
        logger.info("docling not available, trying PyPDF2")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        import PyPDF2
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_count = len(reader.pages)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts), page_count
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfplumber")
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts), page_count
    except ImportError:
        logger.warning("No PDF extraction library available (docling/PyPDF2/pdfplumber)")
        return "", 0
    except Exception as e:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, e)
        return "", 0


def extract_concepts_from_text(text: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """Extract ML/AI concepts from text using keyword matching and frequency analysis."""
    text_lower = text.lower()
    found = []

    for keyword in CONCEPT_KEYWORDS:
        count = text_lower.count(keyword.lower())
        if count > 0:
            category = _categorize_concept(keyword)
            importance = min(1.0, count / 10.0)
            found.append({
                "concept_name": keyword,
                "description": f"Mentioned {count} time(s) in chapter content",
                "category": category,
                "importance": round(importance, 3),
                "count": count,
            })

    found.sort(key=lambda x: x["count"], reverse=True)
    return found[:top_n]


def _categorize_concept(concept: str) -> str:
    """Categorize a concept into a broad ML category."""
    categories = {
        "architecture": ["neural network", "transformer", "convolution", "recurrent", "attention",
                         "autoencoder", "GAN", "VAE", "mixture of experts", "gating network"],
        "optimization": ["gradient descent", "backpropagation", "loss function", "optimization",
                         "adam", "sgd", "learning rate", "momentum"],
        "regularization": ["regularization", "dropout", "batch normalization", "overfitting",
                           "underfitting", "bias", "variance"],
        "evaluation": ["cross-validation", "precision", "recall", "f1", "ROC", "AUC",
                       "confusion matrix", "evaluation", "model selection"],
        "feature_engineering": ["feature", "PCA", "t-SNE", "UMAP", "normalization",
                                "standardization", "imputation", "feature engineering",
                                "feature selection", "dimensionality reduction"],
        "nlp": ["tokenization", "word2vec", "glove", "BERT", "GPT", "embedding"],
        "ensemble": ["ensemble", "boosting", "bagging", "random forest", "decision tree"],
        "reinforcement_learning": ["reinforcement learning", "policy", "reward",
                                   "value function", "q-learning"],
        "infrastructure": ["pipeline", "deployment", "monitoring", "MLOps",
                           "experiment tracking", "model registry"],
    }
    concept_lower = concept.lower()
    for cat, keywords in categories.items():
        if any(k.lower() in concept_lower or concept_lower in k.lower() for k in keywords):
            return cat
    return "general"


def extract_chapter(chapter_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all content from a chapter (all its PDFs).

    Returns a dict with markdown_content, page_count, and concepts.
    """
    all_text = []
    total_pages = 0

    for pdf_path in chapter_info.get("pdfs", []):
        if not os.path.exists(pdf_path):
            logger.warning("PDF not found: %s", pdf_path)
            continue
        text, pages = extract_pdf_text(pdf_path)
        all_text.append(text)
        total_pages += pages

    combined_text = "\n\n---\n\n".join(all_text) if all_text else ""
    concepts = extract_concepts_from_text(combined_text)

    return {
        "chapter_num": chapter_info["chapter_num"],
        "title": chapter_info["title"],
        "source_path": chapter_info["path"],
        "markdown_content": combined_text,
        "page_count": total_pages,
        "concepts": concepts,
    }
