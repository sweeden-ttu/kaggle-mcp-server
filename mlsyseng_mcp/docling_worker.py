"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies concepts, and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_CONCEPT_PATTERNS = [
    r"\b(gradient descent|backpropagation|forward pass)\b",
    r"\b(loss function|objective function|cost function)\b",
    r"\b(neural network|deep learning|convolutional)\b",
    r"\b(regularization|dropout|batch norm)\b",
    r"\b(hyperparameter|learning rate|momentum)\b",
    r"\b(cross.?validation|train.?test split|overfitting|underfitting)\b",
    r"\b(feature engineering|feature selection|dimensionality reduction)\b",
    r"\b(ensemble|bagging|boosting|random forest)\b",
    r"\b(attention mechanism|transformer|self.?attention)\b",
    r"\b(reinforcement learning|reward function|policy gradient)\b",
    r"\b(generative model|discriminative model|GAN)\b",
    r"\b(optimization|SGD|Adam|RMSProp)\b",
    r"\b(precision|recall|F1.?score|accuracy|AUC|ROC)\b",
    r"\b(bias.?variance|model complexity|capacity)\b",
    r"\b(kernel|SVM|support vector)\b",
    r"\b(clustering|k.?means|hierarchical)\b",
    r"\b(PCA|SVD|eigenvalue|eigenvector)\b",
    r"\b(Bayesian|posterior|prior|likelihood)\b",
    r"\b(MoE|mixture of experts|gating network)\b",
    r"\b(distributed training|data parallel|model parallel)\b",
]


def _default_ml_principles_path() -> str:
    return os.environ.get(
        "ML_PRINCIPLES_PATH",
        str(Path.home() / "Desktop" / "Machine Learning Principles - Chapters"),
    )


def extract_concepts_from_text(text: str) -> List[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    concepts = set()
    text_lower = text.lower()
    for pattern in ML_CONCEPT_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            concepts.add(match.strip())
    return sorted(concepts)


def _parse_chapter_folder_name(folder_name: str) -> Tuple[Optional[int], str]:
    """Parse chapter number and title from folder name.

    Expected formats:
        '01_Introduction'
        '08 ML Systems'
        'Chapter 3 - Deep Learning'
    """
    patterns = [
        r"^(\d+)[_\s\-]+(.+)$",
        r"^Chapter\s+(\d+)[_\s\-]+(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, folder_name, re.IGNORECASE)
        if m:
            return int(m.group(1)), m.group(2).strip()
    return None, folder_name


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, falling back to PyPDF2/pdfplumber")
        return _fallback_pdf_extract(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        return _fallback_pdf_extract(pdf_path)


def _fallback_pdf_extract(pdf_path: str) -> str:
    """Fallback PDF extraction using pdfplumber or PyPDF2."""
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

    return f"[PDF extraction unavailable for: {pdf_path}]"


def scan_chapter_folders(base_path: Optional[str] = None) -> List[Dict]:
    """Scan the ML Principles directory for chapter folders."""
    base = Path(base_path or _default_ml_principles_path())
    chapters = []

    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return chapters

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        chapter_num, title = _parse_chapter_folder_name(item.name)
        if chapter_num is None:
            continue

        pdfs = list(item.glob("*.pdf")) + list(item.glob("*.PDF"))
        if pdfs:
            chapters.append({
                "chapter_number": chapter_num,
                "title": title,
                "folder_path": str(item),
                "pdf_files": [str(p) for p in pdfs],
            })

    return chapters


def extract_chapter(
    chapter_info: Dict,
    db: Database,
    force: bool = False,
) -> Dict:
    """Extract a single chapter's content and store in database.

    Returns a status dict with extraction results.
    """
    chapter_num = chapter_info["chapter_number"]
    title = chapter_info["title"]

    if not force:
        existing = db.get_chapter(chapter_num)
        if existing and existing.get("markdown_content"):
            return {
                "chapter_number": chapter_num,
                "status": "skipped",
                "reason": "already extracted",
            }

    db.set_extraction_status(chapter_num, "extracting")

    all_text_parts = []
    total_pages = 0

    for pdf_path in chapter_info["pdf_files"]:
        try:
            text = _extract_pdf_with_docling(pdf_path)
            if text:
                all_text_parts.append(text)
                total_pages += text.count("\n\n") + 1
        except Exception as e:
            logger.error(f"Failed to extract {pdf_path}: {e}")
            db.set_extraction_status(chapter_num, "failed", error_message=str(e))
            return {
                "chapter_number": chapter_num,
                "status": "failed",
                "error": str(e),
            }

    combined_text = "\n\n---\n\n".join(all_text_parts)
    concepts = extract_concepts_from_text(combined_text)

    db.upsert_chapter(
        chapter_number=chapter_num,
        title=title,
        source_path=chapter_info["folder_path"],
        markdown_content=combined_text,
        concepts=concepts,
    )
    db.set_extraction_status(chapter_num, "completed", pages_extracted=total_pages)

    return {
        "chapter_number": chapter_num,
        "title": title,
        "status": "completed",
        "word_count": len(combined_text.split()),
        "concepts_found": len(concepts),
        "concepts": concepts[:10],
    }


def extract_all_chapters(
    db: Database,
    base_path: Optional[str] = None,
    force: bool = False,
) -> List[Dict]:
    """Extract all chapters from the ML Principles directory."""
    chapters = scan_chapter_folders(base_path)
    results = []

    for chapter_info in chapters:
        result = extract_chapter(chapter_info, db, force=force)
        results.append(result)
        logger.info(
            f"Chapter {result['chapter_number']}: {result['status']}"
        )

    return results
