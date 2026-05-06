"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies concepts, and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import database as db

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.path.expanduser(
    "~/Desktop/Machine Learning Principles - Chapters"
)

ML_CONCEPT_PATTERNS = [
    r"(?i)\b(gradient descent|stochastic gradient descent|SGD|Adam|AdaGrad)\b",
    r"(?i)\b(backpropagation|forward pass|backward pass)\b",
    r"(?i)\b(overfitting|underfitting|regularization|dropout|batch norm)\b",
    r"(?i)\b(cross[- ]?validation|train[- ]?test split|holdout)\b",
    r"(?i)\b(neural network|deep learning|convolutional|recurrent|transformer)\b",
    r"(?i)\b(loss function|cost function|objective function)\b",
    r"(?i)\b(hyperparameter|learning rate|batch size|epoch)\b",
    r"(?i)\b(feature engineering|feature selection|dimensionality reduction)\b",
    r"(?i)\b(ensemble|bagging|boosting|random forest|XGBoost)\b",
    r"(?i)\b(support vector machine|SVM|kernel trick)\b",
    r"(?i)\b(decision tree|CART|pruning)\b",
    r"(?i)\b(logistic regression|linear regression|polynomial regression)\b",
    r"(?i)\b(clustering|k-means|DBSCAN|hierarchical)\b",
    r"(?i)\b(PCA|principal component|SVD|eigenvalue)\b",
    r"(?i)\b(Bayesian|prior|posterior|likelihood|MAP|MLE)\b",
    r"(?i)\b(attention mechanism|self-attention|multi-head attention)\b",
    r"(?i)\b(embedding|word2vec|GloVe|BERT|GPT)\b",
    r"(?i)\b(reinforcement learning|Q-learning|policy gradient|reward)\b",
    r"(?i)\b(GAN|generative adversarial|VAE|variational autoencoder)\b",
    r"(?i)\b(precision|recall|F1[- ]?score|AUC|ROC)\b",
    r"(?i)\b(bias[- ]?variance tradeoff|model complexity)\b",
    r"(?i)\b(data augmentation|transfer learning|fine[- ]?tuning)\b",
    r"(?i)\b(normalization|standardization|min-max scaling)\b",
    r"(?i)\b(activation function|ReLU|sigmoid|tanh|softmax)\b",
    r"(?i)\b(convolution|pooling|stride|padding|filter)\b",
]


def _get_ml_principles_path() -> str:
    return os.environ.get("ML_PRINCIPLES_PATH", DEFAULT_ML_PRINCIPLES_PATH)


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, falling back to basic extraction")
        return _extract_pdf_fallback(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _extract_pdf_fallback(pdf_path)


def _extract_pdf_fallback(pdf_path: str) -> str:
    """Fallback PDF text extraction using PyPDF2 or pdfplumber."""
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

    return f"[PDF extraction unavailable for {pdf_path} — install docling or pdfplumber]"


def _extract_concepts_from_text(text: str) -> List[Dict[str, str]]:
    """Identify ML/AI concepts from extracted text using pattern matching."""
    found = {}
    for pattern in ML_CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text):
            concept = match.group(1).strip()
            normalized = concept.lower().replace("-", " ").replace("  ", " ")
            if normalized not in found:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].replace("\n", " ").strip()
                category = _categorize_concept(normalized)
                found[normalized] = {
                    "concept": concept,
                    "description": context,
                    "category": category,
                }
    return list(found.values())


def _categorize_concept(concept: str) -> str:
    """Assign a category to a concept."""
    categories = {
        "optimization": ["gradient", "sgd", "adam", "adagrad", "learning rate",
                         "loss function", "cost function", "objective"],
        "architecture": ["neural network", "cnn", "rnn", "transformer",
                         "convolutional", "recurrent", "attention", "embedding",
                         "activation", "relu", "sigmoid", "pooling", "convolution"],
        "regularization": ["overfitting", "underfitting", "regularization",
                           "dropout", "batch norm", "bias variance"],
        "evaluation": ["cross validation", "precision", "recall", "f1",
                       "auc", "roc", "train test"],
        "ensemble": ["ensemble", "bagging", "boosting", "random forest", "xgboost"],
        "unsupervised": ["clustering", "k means", "dbscan", "pca",
                         "principal component", "svd", "dimensionality"],
        "generative": ["gan", "generative adversarial", "vae", "variational"],
        "nlp": ["word2vec", "glove", "bert", "gpt", "self attention"],
        "classical": ["svm", "support vector", "decision tree", "logistic",
                      "linear regression", "polynomial", "cart"],
        "preprocessing": ["feature engineering", "feature selection",
                          "normalization", "standardization", "data augmentation"],
        "training": ["backpropagation", "forward pass", "backward pass",
                     "hyperparameter", "batch size", "epoch"],
        "transfer": ["transfer learning", "fine tuning"],
        "rl": ["reinforcement learning", "q learning", "policy gradient", "reward"],
        "probabilistic": ["bayesian", "prior", "posterior", "likelihood", "map", "mle"],
    }
    concept_lower = concept.lower()
    for cat, keywords in categories.items():
        if any(kw in concept_lower for kw in keywords):
            return cat
    return "general"


def discover_chapters(base_path: Optional[str] = None) -> List[Tuple[str, str]]:
    """Find chapter folders and their PDFs.

    Returns list of (folder_name, pdf_path) tuples.
    """
    base = Path(base_path or _get_ml_principles_path())
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        pdfs = list(item.glob("*.pdf"))
        if pdfs:
            chapters.append((item.name, str(pdfs[0])))
        else:
            logger.debug("No PDF found in chapter folder: %s", item.name)
    return chapters


def extract_chapter(folder_name: str, pdf_path: str,
                    force: bool = False) -> Dict:
    """Extract a single chapter PDF and store results.

    Returns extraction result dict.
    """
    db.init_db()

    existing = [c for c in db.get_all_chapters()
                if c["folder_name"] == folder_name]
    if existing and not force:
        return {
            "status": "skipped",
            "folder": folder_name,
            "reason": "already extracted (use force_reindex=true to re-extract)",
        }

    chapter_id = db.upsert_chapter(
        folder_name=folder_name,
        title=folder_name,
        pdf_path=pdf_path,
        markdown_content="",
        word_count=0,
    )
    db.log_extraction(chapter_id, "started")

    try:
        markdown = _extract_pdf_with_docling(pdf_path)
        word_count = len(markdown.split())

        db.upsert_chapter(
            folder_name=folder_name,
            title=folder_name,
            pdf_path=pdf_path,
            markdown_content=markdown,
            word_count=word_count,
        )

        concepts = _extract_concepts_from_text(markdown)
        db.add_concepts(chapter_id, concepts)
        db.log_extraction(chapter_id, "completed")

        return {
            "status": "completed",
            "folder": folder_name,
            "word_count": word_count,
            "concepts_found": len(concepts),
            "concepts": [c["concept"] for c in concepts[:10]],
        }
    except Exception as e:
        db.log_extraction(chapter_id, "failed", str(e))
        return {
            "status": "failed",
            "folder": folder_name,
            "error": str(e),
        }


def extract_all(force_reindex: bool = False) -> List[Dict]:
    """Extract all discovered chapters."""
    db.init_db()
    chapters = discover_chapters()
    if not chapters:
        return [{
            "status": "error",
            "message": f"No chapters found at {_get_ml_principles_path()}",
        }]
    return [extract_chapter(folder, pdf, force=force_reindex)
            for folder, pdf in chapters]
