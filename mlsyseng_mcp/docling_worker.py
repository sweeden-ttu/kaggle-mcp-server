"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content, and stores
the results in the SQLite database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from . import database as db

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

_CHAPTER_RE = re.compile(r"^(\d+)[\s_\-]+(.+)$")


def _try_import_docling():
    """Attempt to import docling; return converter class or None."""
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter
    except ImportError:
        logger.warning(
            "docling not installed — falling back to basic text extraction. "
            "Install with: pip install docling"
        )
        return None


def _basic_pdf_extract(pdf_path: str) -> str:
    """Fallback PDF text extraction when docling is unavailable."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available. "
        "Install docling, PyMuPDF, or pypdf."
    )
    return ""


def extract_pdf(pdf_path: str) -> str:
    """Extract markdown text from a PDF file."""
    DocumentConverter = _try_import_docling()
    if DocumentConverter is not None:
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    return _basic_pdf_extract(pdf_path)


def extract_concepts(text: str) -> list[str]:
    """Extract key ML/AI concepts from chapter text.

    Uses keyword pattern matching to identify concepts. Returns a
    deduplicated list of concept strings found in the text.
    """
    concept_patterns = [
        r"(?i)\b(neural network|deep learning|convolutional|recurrent|transformer)\w*\b",
        r"(?i)\b(gradient descent|backpropagation|forward pass|loss function)\b",
        r"(?i)\b(regularization|dropout|batch normalization|weight decay)\b",
        r"(?i)\b(cross[- ]?validation|train[- ]?test split|holdout|k-fold)\b",
        r"(?i)\b(overfitting|underfitting|bias[- ]?variance|generalization)\b",
        r"(?i)\b(feature engineering|feature selection|dimensionality reduction)\b",
        r"(?i)\b(decision tree|random forest|gradient boosting|xgboost|lightgbm)\b",
        r"(?i)\b(support vector machine|svm|kernel trick)\b",
        r"(?i)\b(logistic regression|linear regression|polynomial regression)\b",
        r"(?i)\b(clustering|k-means|dbscan|hierarchical clustering)\b",
        r"(?i)\b(ensemble|bagging|boosting|stacking)\b",
        r"(?i)\b(attention mechanism|self[- ]?attention|multi[- ]?head attention)\b",
        r"(?i)\b(embedding|word2vec|tokenization|vocabulary)\b",
        r"(?i)\b(optimizer|adam|sgd|learning rate|momentum)\b",
        r"(?i)\b(hyperparameter|tuning|grid search|bayesian optimization)\b",
        r"(?i)\b(precision|recall|f1[- ]?score|accuracy|auc|roc)\b",
        r"(?i)\b(data augmentation|preprocessing|normalization|standardization)\b",
        r"(?i)\b(mixture of experts|gating network|sparse activation)\b",
        r"(?i)\b(reinforcement learning|reward function|policy gradient|q-learning)\b",
        r"(?i)\b(generative adversarial|gan|variational autoencoder|vae)\b",
        r"(?i)\b(transfer learning|fine[- ]?tuning|pre[- ]?training)\b",
        r"(?i)\b(distributed training|model parallelism|data parallelism)\b",
        r"(?i)\b(inference|deployment|serving|latency|throughput)\b",
    ]
    found: set[str] = set()
    for pattern in concept_patterns:
        matches = re.findall(pattern, text[:50000])
        for m in matches:
            found.add(m.strip().lower())
    return sorted(found)


def discover_chapters(
    base_path: Optional[str] = None,
) -> list[dict]:
    """Scan the ML Principles directory for chapter folders.

    Returns a list of dicts with keys: folder_name, chapter_number,
    title, pdf_path.
    """
    base = Path(base_path or ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        match = _CHAPTER_RE.match(entry.name)
        if not match:
            continue
        chapter_num = int(match.group(1))
        title = match.group(2).replace("_", " ").strip()

        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue

        chapters.append({
            "folder_name": entry.name,
            "chapter_number": chapter_num,
            "title": title,
            "pdf_path": str(pdfs[0]),
        })
    return chapters


def extract_chapter(
    chapter_info: dict,
    db_path: Optional[str] = None,
    force: bool = False,
) -> dict:
    """Extract a single chapter PDF and store results.

    Returns a dict with extraction status and metadata.
    """
    folder = chapter_info["folder_name"]

    if not force:
        existing = db.get_chapter(folder, db_path=db_path)
        if existing and existing.get("markdown_content"):
            return {
                "folder_name": folder,
                "status": "skipped",
                "reason": "already extracted",
            }

    job_id = db.create_extraction_job(folder, db_path=db_path)
    db.update_extraction_job(job_id, "running", db_path=db_path)

    try:
        markdown = extract_pdf(chapter_info["pdf_path"])
        concepts = extract_concepts(markdown)

        db.upsert_chapter(
            folder_name=folder,
            chapter_number=chapter_info["chapter_number"],
            title=chapter_info["title"],
            pdf_path=chapter_info["pdf_path"],
            markdown_content=markdown,
            concepts=concepts,
            db_path=db_path,
        )

        db.update_extraction_job(job_id, "done", db_path=db_path)
        return {
            "folder_name": folder,
            "status": "done",
            "word_count": len(markdown.split()),
            "concepts_found": len(concepts),
        }
    except Exception as e:
        db.update_extraction_job(job_id, "error", str(e), db_path=db_path)
        logger.error("Failed to extract %s: %s", folder, e)
        return {
            "folder_name": folder,
            "status": "error",
            "error": str(e),
        }


def extract_all(
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
    force: bool = False,
) -> list[dict]:
    """Extract all discovered chapters. Returns list of per-chapter results."""
    db.init_db(db_path)
    chapters = discover_chapters(base_path)
    if not chapters:
        return [{"status": "no_chapters", "path": base_path or ML_PRINCIPLES_PATH}]

    results = []
    for ch in chapters:
        result = extract_chapter(ch, db_path=db_path, force=force)
        results.append(result)
    return results
