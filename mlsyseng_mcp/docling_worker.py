"""Background PDF extraction using docling for ML Principles chapters."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import MLSysEngDatabase

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.path.expanduser(
    os.getenv(
        "ML_PRINCIPLES_PATH",
        "~/Desktop/Machine Learning Principles - Chapters",
    )
)

CONCEPT_PATTERNS = [
    r"(?:gradient\s+descent|backpropagation|forward\s+pass)",
    r"(?:loss\s+function|objective\s+function|cost\s+function)",
    r"(?:regularization|dropout|batch\s+norm(?:alization)?)",
    r"(?:convolutional|recurrent|transformer|attention)",
    r"(?:overfitting|underfitting|bias[\s-]variance)",
    r"(?:cross[\s-]validation|train[\s-]test\s+split)",
    r"(?:hyperparameter|learning\s+rate|epoch|batch\s+size)",
    r"(?:ensemble|bagging|boosting|random\s+forest)",
    r"(?:supervised|unsupervised|reinforcement)\s+learning",
    r"(?:neural\s+network|deep\s+learning|perceptron)",
    r"(?:feature\s+engineering|feature\s+selection|dimensionality\s+reduction)",
    r"(?:precision|recall|f1[\s-]score|accuracy|AUC|ROC)",
    r"(?:PCA|SVD|t-SNE|UMAP)",
    r"(?:SVM|support\s+vector\s+machine)",
    r"(?:decision\s+tree|XGBoost|LightGBM|CatBoost)",
    r"(?:embedding|word2vec|GloVe|BERT|GPT)",
    r"(?:optimization|Adam|SGD|momentum|RMSprop)",
    r"(?:activation\s+function|ReLU|sigmoid|softmax|tanh)",
    r"(?:data\s+augmentation|normalization|standardization)",
    r"(?:inference|prediction|classification|regression)",
]

COMPILED_CONCEPT_RE = re.compile("|".join(CONCEPT_PATTERNS), re.IGNORECASE)


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from a PDF using docling, falling back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic PDF extraction")
        return _extract_pdf_fallback(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _extract_pdf_fallback(pdf_path)


def _extract_pdf_fallback(pdf_path: str) -> str:
    """Basic PDF text extraction without docling."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("Neither docling nor PyMuPDF available, returning empty content")
        return ""


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text content."""
    found = set()
    for match in COMPILED_CONCEPT_RE.finditer(text):
        concept = match.group(0).strip().lower()
        concept = re.sub(r"\s+", " ", concept)
        found.add(concept)
    return sorted(found)


def discover_chapters(base_path: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Discover chapter folders containing PDFs.

    Returns list of (chapter_name, pdf_path) tuples.
    """
    base = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append((item.name, str(pdfs[0])))
        elif item.suffix.lower() == ".pdf":
            chapters.append((item.stem, str(item)))

    return chapters


def extract_chapter(
    chapter_name: str,
    pdf_path: str,
    db: MLSysEngDatabase,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Extract a single chapter PDF and store results in the database.

    Returns dict with extraction results.
    """
    existing = db.get_chapter(chapter_name)
    if existing and existing["status"] == "extracted" and not force:
        return {
            "chapter_name": chapter_name,
            "status": "skipped",
            "message": "Already extracted (use force=True to re-extract)",
        }

    db.log_event(chapter_name, "extraction_started", f"Path: {pdf_path}")

    try:
        markdown_content = _extract_pdf_with_docling(pdf_path)
        concepts = extract_concepts(markdown_content)

        db.upsert_chapter(
            chapter_name=chapter_name,
            source_path=pdf_path,
            markdown_content=markdown_content,
            concepts=concepts,
            status="extracted",
        )

        db.log_event(
            chapter_name,
            "extraction_complete",
            f"Concepts: {len(concepts)}, Content length: {len(markdown_content)}",
        )

        return {
            "chapter_name": chapter_name,
            "status": "extracted",
            "concepts_count": len(concepts),
            "concepts": concepts,
            "content_length": len(markdown_content),
        }
    except Exception as e:
        db.log_event(chapter_name, "extraction_failed", str(e))
        return {
            "chapter_name": chapter_name,
            "status": "failed",
            "error": str(e),
        }


def extract_all_chapters(
    db: MLSysEngDatabase,
    base_path: Optional[str] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Extract all chapters from the ML Principles directory.

    Returns summary dict with results for each chapter.
    """
    chapters = discover_chapters(base_path)

    if not chapters:
        return {
            "status": "no_chapters_found",
            "message": f"No PDF chapters found in {base_path or DEFAULT_ML_PRINCIPLES_PATH}",
        }

    results = []
    for i, (name, path) in enumerate(chapters):
        if progress_callback:
            progress_callback(name, i + 1, len(chapters))

        result = extract_chapter(name, path, db, force=force)
        results.append(result)

    extracted = sum(1 for r in results if r["status"] == "extracted")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    return {
        "status": "complete",
        "total": len(chapters),
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
        "results": results,
    }
