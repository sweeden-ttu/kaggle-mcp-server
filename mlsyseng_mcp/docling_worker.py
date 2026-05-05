"""
Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies key concepts, and stores results in the database.
"""

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

_CONCEPT_PATTERNS = [
    r"(?:gradient\s+descent|SGD|stochastic\s+gradient)",
    r"(?:back\s*propagation|backprop)",
    r"(?:regularization|L1|L2|dropout|weight\s+decay)",
    r"(?:neural\s+network|deep\s+learning|CNN|RNN|LSTM|transformer)",
    r"(?:decision\s+tree|random\s+forest|XGBoost|LightGBM)",
    r"(?:support\s+vector\s+machine|SVM|kernel)",
    r"(?:cross[\s-]?validation|k-fold|stratified)",
    r"(?:feature\s+engineering|feature\s+selection|PCA|dimensionality)",
    r"(?:ensemble|bagging|boosting|stacking)",
    r"(?:hyperparameter|grid\s+search|Bayesian\s+optimization)",
    r"(?:loss\s+function|objective\s+function|cost\s+function)",
    r"(?:activation\s+function|ReLU|sigmoid|tanh|softmax)",
    r"(?:batch\s+normalization|layer\s+normalization)",
    r"(?:attention\s+mechanism|self-attention|multi-head)",
    r"(?:transfer\s+learning|fine-tuning|pre-training)",
    r"(?:data\s+augmentation|oversampling|SMOTE)",
    r"(?:bias[\s-]?variance\s+tradeoff|overfitting|underfitting)",
    r"(?:precision|recall|F1[\s-]?score|AUC|ROC)",
    r"(?:confusion\s+matrix|classification\s+report)",
    r"(?:clustering|k-means|DBSCAN|hierarchical)",
]


def discover_chapters(base_path: str = ML_PRINCIPLES_PATH) -> List[Dict[str, Any]]:
    """
    Scan the chapter folder structure and return a list of chapter metadata.

    Expected layout:
        <base_path>/
          01 - Introduction/
            chapter01.pdf
          02 - Supervised Learning/
            chapter02.pdf
          ...
    """
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base_path)
        return []

    chapters: List[Dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        match = re.match(r"^(\d+)", entry.name)
        if not match:
            continue
        chapter_num = int(match.group(1))
        title = re.sub(r"^\d+\s*[-–_]\s*", "", entry.name).strip()
        pdfs = list(entry.glob("*.pdf"))
        if not pdfs:
            continue
        chapters.append({
            "chapter_num": chapter_num,
            "title": title,
            "dir_path": str(entry),
            "pdf_paths": [str(p) for p in pdfs],
        })
    return chapters


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using docling.
    Falls back to PyPDF2/pdfplumber if docling is unavailable.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, trying PyPDF2")
    except Exception as exc:
        logger.warning("docling extraction failed for %s: %s", pdf_path, exc)

    try:
        import PyPDF2

        text_parts: List[str] = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfplumber")
    except Exception as exc:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, exc)

    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("No PDF extraction library available")
        return ""
    except Exception as exc:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, exc)
        return ""


def extract_concepts(text: str) -> List[str]:
    """
    Extract ML/AI concepts from text using regex-based pattern matching.
    """
    found: List[str] = []
    lower = text.lower()
    for pattern in _CONCEPT_PATTERNS:
        matches = re.findall(pattern, lower)
        for m in matches:
            concept = m.strip().title()
            if concept and concept not in found:
                found.append(concept)
    return found


def extract_chapter(
    chapter_meta: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """
    Extract markdown and concepts from a chapter's PDFs.

    Returns:
        (combined_markdown, concepts)
    """
    all_text: List[str] = []
    for pdf_path in chapter_meta.get("pdf_paths", []):
        text = extract_pdf_text(pdf_path)
        if text:
            all_text.append(text)

    combined = "\n\n---\n\n".join(all_text)
    concepts = extract_concepts(combined)
    return combined, concepts


def run_extraction(
    db,
    base_path: str = ML_PRINCIPLES_PATH,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """
    Run the full extraction pipeline:
    1. Discover chapters
    2. Extract PDFs
    3. Store in database
    4. Return status

    Args:
        db: MoEDatabase instance
        base_path: Path to ML Principles chapters
        force_reindex: Re-extract even if already done
    """
    chapters = discover_chapters(base_path)
    results = {"discovered": len(chapters), "extracted": 0, "skipped": 0, "failed": 0}

    for ch in chapters:
        chapter_id = f"ch_{ch['chapter_num']:02d}"
        existing = db.get_chapter(chapter_id)
        if existing and existing["status"] == "extracted" and not force_reindex:
            results["skipped"] += 1
            continue

        db.upsert_chapter(
            chapter_id=chapter_id,
            chapter_num=ch["chapter_num"],
            title=ch["title"],
            source_path=ch["dir_path"],
            status="extracting",
        )
        db.log_event(chapter_id, "extraction_started", ch["dir_path"])

        try:
            markdown, concepts = extract_chapter(ch)
            db.upsert_chapter(
                chapter_id=chapter_id,
                chapter_num=ch["chapter_num"],
                title=ch["title"],
                source_path=ch["dir_path"],
                markdown=markdown,
                concepts=concepts,
                status="extracted",
            )
            db.log_event(
                chapter_id,
                "extraction_complete",
                f"concepts={len(concepts)}",
            )
            results["extracted"] += 1
        except Exception as exc:
            db.upsert_chapter(
                chapter_id=chapter_id,
                chapter_num=ch["chapter_num"],
                title=ch["title"],
                source_path=ch["dir_path"],
                status="failed",
            )
            db.log_event(chapter_id, "extraction_failed", str(exc))
            results["failed"] += 1
            logger.error("Failed to extract chapter %s: %s", chapter_id, exc)

    return results
