"""Background PDF extraction using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CHAPTER_RE = re.compile(r"^(\d+)[_\s\-]+(.+)$")


def _try_import_docling():
    """Lazy-import docling so the rest of the system works without it."""
    try:
        from docling.document_converter import DocumentConverter
        return DocumentConverter
    except ImportError:
        return None


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders containing PDFs."""
    root = Path(base_path or ML_PRINCIPLES_PATH)
    if not root.is_dir():
        logger.warning("ML Principles path does not exist: %s", root)
        return []

    chapters: List[Dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        m = CHAPTER_RE.match(entry.name)
        if not m:
            continue
        chapter_num = int(m.group(1))
        title = m.group(2).replace("_", " ").strip()
        pdfs = list(entry.glob("*.pdf"))
        if pdfs:
            chapters.append(
                {
                    "chapter_num": chapter_num,
                    "title": title,
                    "source_path": str(entry),
                    "pdfs": [str(p) for p in pdfs],
                }
            )
    return chapters


def extract_pdf(pdf_path: str) -> str:
    """Extract text from a single PDF using docling, falling back to basic extraction."""
    DocumentConverter = _try_import_docling()
    if DocumentConverter is not None:
        try:
            converter = DocumentConverter()
            result = converter.convert(pdf_path)
            return result.document.export_to_markdown()
        except Exception as exc:
            logger.warning("Docling extraction failed for %s: %s", pdf_path, exc)

    try:
        import fitz  # PyMuPDF fallback

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    return f"[PDF extraction unavailable for {pdf_path}]"


def extract_concepts(text: str) -> List[str]:
    """Extract key ML/AI concepts from chapter text using heuristics."""
    concept_patterns = [
        r"(?:gradient\s+descent|backpropagation|forward\s+pass)",
        r"(?:neural\s+network|deep\s+learning|convolutional|recurrent|transformer)",
        r"(?:regularization|dropout|batch\s+norm|layer\s+norm)",
        r"(?:loss\s+function|cross[- ]entropy|mean\s+squared\s+error|mse)",
        r"(?:optimization|adam|sgd|learning\s+rate|momentum)",
        r"(?:overfitting|underfitting|bias[- ]variance|generalization)",
        r"(?:feature\s+engineering|feature\s+selection|dimensionality\s+reduction)",
        r"(?:ensemble|bagging|boosting|random\s+forest|xgboost)",
        r"(?:hyperparameter|grid\s+search|bayesian\s+optimization)",
        r"(?:reinforcement\s+learning|policy\s+gradient|q[- ]learning)",
        r"(?:attention\s+mechanism|self[- ]attention|multi[- ]head)",
        r"(?:embedding|word2vec|representation\s+learning)",
        r"(?:data\s+augmentation|preprocessing|normalization|standardization)",
        r"(?:model\s+selection|cross[- ]validation|k[- ]fold)",
        r"(?:precision|recall|f1[- ]score|auc|roc)",
    ]

    text_lower = text.lower()
    found = set()
    for pattern in concept_patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            concept = m.strip().replace("-", " ").title()
            found.add(concept)

    heading_re = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    for m in heading_re.finditer(text):
        heading = m.group(1).strip()
        if 3 <= len(heading) <= 80:
            found.add(heading)

    return sorted(found)


def run_extraction(
    db: Database,
    force_reindex: bool = False,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full extraction pipeline: discover chapters -> extract PDFs -> store in DB.

    Returns summary dict with counts and any errors.
    """
    chapters = discover_chapters(base_path)
    if not chapters:
        return {"status": "no_chapters", "message": "No chapter folders found"}

    results: Dict[str, Any] = {"total": len(chapters), "extracted": 0, "skipped": 0, "errors": []}

    for ch in chapters:
        chapter_num = ch["chapter_num"]
        existing = db.get_chapter(chapter_num)
        if existing and existing.get("content_md") and not force_reindex:
            results["skipped"] += 1
            continue

        db.set_extraction_status(chapter_num, "running")
        try:
            all_text: List[str] = []
            for pdf_path in ch["pdfs"]:
                text = extract_pdf(pdf_path)
                all_text.append(text)

            content_md = "\n\n---\n\n".join(all_text)
            concepts = extract_concepts(content_md)

            db.upsert_chapter(
                chapter_num=chapter_num,
                title=ch["title"],
                source_path=ch["source_path"],
                content_md=content_md,
                concepts=concepts,
            )
            db.set_extraction_status(chapter_num, "done")
            results["extracted"] += 1

        except Exception as exc:
            error_msg = str(exc)
            db.set_extraction_status(chapter_num, "error", error=error_msg)
            results["errors"].append({"chapter": chapter_num, "error": error_msg})
            logger.error("Extraction failed for chapter %d: %s", chapter_num, exc)

    return results
