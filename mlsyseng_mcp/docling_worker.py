"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts text from PDFs,
identifies key concepts, and stores results in SQLite.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

# Concept patterns used to extract ML/AI terms from text
_CONCEPT_PATTERNS = [
    r"\b(?:gradient\s+descent|backpropagation|forward\s+pass)\b",
    r"\b(?:loss\s+function|cost\s+function|objective\s+function)\b",
    r"\b(?:regularization|dropout|batch\s+norm(?:alization)?)\b",
    r"\b(?:convolutional|recurrent|transformer|attention)\b",
    r"\b(?:overfitting|underfitting|bias[- ]variance)\b",
    r"\b(?:cross[- ]validation|hyperparameter|learning\s+rate)\b",
    r"\b(?:feature\s+engineering|data\s+augmentation|transfer\s+learning)\b",
    r"\b(?:ensemble|bagging|boosting|random\s+forest)\b",
    r"\b(?:neural\s+network|deep\s+learning|machine\s+learning)\b",
    r"\b(?:support\s+vector|decision\s+tree|k-nearest|naive\s+bayes)\b",
    r"\b(?:precision|recall|f1[- ]score|accuracy|auc|roc)\b",
    r"\b(?:embedding|tokenization|vocabulary|softmax)\b",
    r"\b(?:optimization|convergence|epoch|mini[- ]batch)\b",
    r"\b(?:supervised|unsupervised|reinforcement|semi-supervised)\b",
    r"\b(?:PCA|SVD|dimensionality\s+reduction)\b",
    r"\b(?:mixture\s+of\s+experts|gating\s+network|MoE)\b",
    r"\b(?:LSTM|GRU|seq2seq|encoder[- ]decoder)\b",
    r"\b(?:generative|discriminative|GAN|VAE|diffusion)\b",
]

_COMPILED_CONCEPTS = [re.compile(p, re.IGNORECASE) for p in _CONCEPT_PATTERNS]


def _extract_concepts(text: str) -> List[str]:
    """Pull unique ML/AI concept mentions from text."""
    found = set()
    for pattern in _COMPILED_CONCEPTS:
        for match in pattern.finditer(text):
            found.add(match.group(0).lower().strip())
    return sorted(found)


def _chapter_id_from_folder(folder_name: str) -> str:
    """Deterministic chapter ID from folder name."""
    return hashlib.sha256(folder_name.encode()).hexdigest()[:12]


def _title_from_folder(folder_name: str) -> str:
    """Human-readable title from a folder name like '08_ML Systems'."""
    name = folder_name.lstrip("0123456789_- ")
    return name.replace("_", " ").strip() or folder_name


def scan_chapter_folders(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover chapter folders that contain PDFs.

    Returns a list of dicts with keys: folder_name, title, pdf_path, chapter_id.
    """
    base = Path(base_path or ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        folder_name = entry.name
        chapters.append(
            {
                "folder_name": folder_name,
                "title": _title_from_folder(folder_name),
                "pdf_path": str(pdfs[0]),
                "chapter_id": _chapter_id_from_folder(folder_name),
            }
        )
    return chapters


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using docling (falls back to basic extraction)."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not installed; falling back to basic PDF extraction")
    except Exception as exc:
        logger.warning("docling extraction failed: %s; trying fallback", exc)

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_text

        return extract_text(pdf_path)
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available. Install docling, PyMuPDF, or pdfminer."
    )
    return ""


def extract_chapter(
    chapter_info: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    """Extract content and concepts from a single chapter.

    Returns (content_md, pdf_path, concepts).
    """
    pdf_path = chapter_info["pdf_path"]
    content = extract_pdf_text(pdf_path)
    concepts = _extract_concepts(content)
    return content, pdf_path, concepts


def extract_all_chapters(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    db=None,
) -> List[Dict[str, Any]]:
    """Scan, extract, and store all chapter folders.

    Args:
        base_path: Override for the ML Principles path.
        force_reindex: Re-extract even if already in DB.
        db: An MLSysEngDB instance (created if None).

    Returns:
        List of chapter dicts that were processed.
    """
    if db is None:
        from mlsyseng_mcp.database import MLSysEngDB

        db = MLSysEngDB()

    chapters = scan_chapter_folders(base_path)
    processed = []

    for ch in chapters:
        existing = db.get_chapter(ch["chapter_id"])
        if existing and not force_reindex:
            logger.info("Skipping already-indexed chapter: %s", ch["folder_name"])
            processed.append(existing)
            continue

        logger.info("Extracting chapter: %s", ch["folder_name"])
        content, pdf_path, concepts = extract_chapter(ch)

        row = db.upsert_chapter(
            chapter_id=ch["chapter_id"],
            folder_name=ch["folder_name"],
            title=ch["title"],
            content_md=content,
            concepts=concepts,
            pdf_path=pdf_path,
        )
        processed.append(row)

    return processed
