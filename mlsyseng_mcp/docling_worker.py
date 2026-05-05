"""Background PDF extraction worker using docling."""

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlsyseng_mcp.database import MoEDatabase

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_PATTERNS = [
    r"(?i)\b(gradient descent|SGD|stochastic gradient)\b",
    r"(?i)\b(backpropagation|back-propagation)\b",
    r"(?i)\b(regularization|L1|L2|dropout|weight decay)\b",
    r"(?i)\b(cross[ -]?validation|k-fold)\b",
    r"(?i)\b(bias[ -]?variance|overfitting|underfitting)\b",
    r"(?i)\b(neural network|deep learning|CNN|RNN|LSTM|transformer)\b",
    r"(?i)\b(random forest|decision tree|XGBoost|LightGBM|gradient boosting)\b",
    r"(?i)\b(support vector|SVM|kernel)\b",
    r"(?i)\b(dimensionality reduction|PCA|t-SNE|UMAP)\b",
    r"(?i)\b(clustering|k-means|DBSCAN|hierarchical)\b",
    r"(?i)\b(feature engineering|feature selection|feature extraction)\b",
    r"(?i)\b(hyperparameter|learning rate|batch size|epoch)\b",
    r"(?i)\b(loss function|objective function|cost function)\b",
    r"(?i)\b(attention mechanism|self-attention|multi-head)\b",
    r"(?i)\b(ensemble|bagging|boosting|stacking)\b",
    r"(?i)\b(optimization|Adam|RMSProp|momentum)\b",
    r"(?i)\b(normalization|batch norm|layer norm)\b",
    r"(?i)\b(data augmentation|preprocessing|imputation)\b",
    r"(?i)\b(model selection|evaluation metric|AUC|F1|precision|recall)\b",
    r"(?i)\b(reinforcement learning|Q-learning|policy gradient)\b",
]


def _extract_concepts_from_text(text: str) -> List[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    found: set = set()
    for pattern in CONCEPT_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            cleaned = m.strip().lower()
            if len(cleaned) > 2:
                found.add(cleaned)
    return sorted(found)


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic PDF extraction")
        return _extract_pdf_basic(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _extract_pdf_basic(pdf_path)


def _extract_pdf_basic(pdf_path: str) -> str:
    """Basic PDF text extraction fallback using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    return f"[Could not extract text from {pdf_path} - install docling, pdfplumber, or PyPDF2]"


def _chapter_id_from_path(folder_path: Path) -> str:
    """Generate a chapter ID from a folder name like '08_ML_Systems'."""
    name = folder_path.name
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def discover_chapters(base_path: str = DEFAULT_ML_PRINCIPLES_PATH) -> List[Dict[str, Any]]:
    """Discover chapter folders and their PDFs."""
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base_path)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        chapters.append(
            {
                "folder": str(entry),
                "chapter_id": _chapter_id_from_path(entry),
                "title": entry.name,
                "pdfs": [str(p) for p in pdfs],
            }
        )
    return chapters


def extract_chapter(
    chapter_info: Dict[str, Any], db: MoEDatabase
) -> Dict[str, Any]:
    """Extract a single chapter's PDFs and store in the database."""
    all_text = []
    for pdf_path in chapter_info["pdfs"]:
        logger.info("Extracting %s", pdf_path)
        text = _extract_pdf_with_docling(pdf_path)
        all_text.append(text)

    combined_markdown = "\n\n---\n\n".join(all_text)
    concepts = _extract_concepts_from_text(combined_markdown)

    db.upsert_chapter(
        chapter_id=chapter_info["chapter_id"],
        title=chapter_info["title"],
        source_path=chapter_info["folder"],
        markdown=combined_markdown,
        concepts=concepts,
    )

    return {
        "chapter_id": chapter_info["chapter_id"],
        "title": chapter_info["title"],
        "concepts": concepts,
        "text_length": len(combined_markdown),
        "pdf_count": len(chapter_info["pdfs"]),
    }


def run_extraction(
    db: MoEDatabase,
    force_reindex: bool = False,
    base_path: str = DEFAULT_ML_PRINCIPLES_PATH,
) -> Dict[str, Any]:
    """Run full extraction pipeline: discover → extract → store."""
    job_id = str(uuid.uuid4())
    chapters = discover_chapters(base_path)

    if not chapters:
        return {
            "job_id": job_id,
            "status": "no_chapters_found",
            "base_path": base_path,
            "chapters_processed": 0,
        }

    if not force_reindex:
        existing = {c["chapter_id"] for c in db.list_chapters()}
        chapters = [c for c in chapters if c["chapter_id"] not in existing]
        if not chapters:
            return {
                "job_id": job_id,
                "status": "already_indexed",
                "message": "All chapters already indexed. Use force_reindex=True to re-extract.",
            }

    db.create_extraction_job(job_id, total_files=len(chapters))
    results = []
    errors = []

    for i, chapter in enumerate(chapters):
        try:
            result = extract_chapter(chapter, db)
            results.append(result)
            db.update_extraction_job(job_id, processed=i + 1)
        except Exception as e:
            error_msg = f"Error extracting {chapter['title']}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            db.update_extraction_job(job_id, error=error_msg)

    status = "completed" if not errors else "completed_with_errors"
    db.update_extraction_job(job_id, status=status)

    return {
        "job_id": job_id,
        "status": status,
        "chapters_processed": len(results),
        "errors": errors,
        "results": results,
    }
