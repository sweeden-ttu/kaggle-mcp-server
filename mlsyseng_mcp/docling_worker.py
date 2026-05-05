"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts text from PDFs,
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

# Common ML/AI concept patterns
_CONCEPT_PATTERNS = [
    re.compile(r"\b(gradient\s+descent)\b", re.IGNORECASE),
    re.compile(r"\b(back\s*propagation)\b", re.IGNORECASE),
    re.compile(r"\b(regularization)\b", re.IGNORECASE),
    re.compile(r"\b(cross[- ]?validation)\b", re.IGNORECASE),
    re.compile(r"\b(bias[- ]variance\s+trade[- ]?off)\b", re.IGNORECASE),
    re.compile(r"\b(overfitting)\b", re.IGNORECASE),
    re.compile(r"\b(underfitting)\b", re.IGNORECASE),
    re.compile(r"\b(neural\s+network)\b", re.IGNORECASE),
    re.compile(r"\b(convolutional\s+neural\s+network|CNN)\b", re.IGNORECASE),
    re.compile(r"\b(recurrent\s+neural\s+network|RNN|LSTM|GRU)\b", re.IGNORECASE),
    re.compile(r"\b(transformer)\b", re.IGNORECASE),
    re.compile(r"\b(attention\s+mechanism)\b", re.IGNORECASE),
    re.compile(r"\b(support\s+vector\s+machine|SVM)\b", re.IGNORECASE),
    re.compile(r"\b(decision\s+tree)\b", re.IGNORECASE),
    re.compile(r"\b(random\s+forest)\b", re.IGNORECASE),
    re.compile(r"\b(ensemble\s+method)\b", re.IGNORECASE),
    re.compile(r"\b(boosting)\b", re.IGNORECASE),
    re.compile(r"\b(bagging)\b", re.IGNORECASE),
    re.compile(r"\b(feature\s+engineering)\b", re.IGNORECASE),
    re.compile(r"\b(dimensionality\s+reduction)\b", re.IGNORECASE),
    re.compile(r"\b(principal\s+component\s+analysis|PCA)\b", re.IGNORECASE),
    re.compile(r"\b(k[- ]?means)\b", re.IGNORECASE),
    re.compile(r"\b(reinforcement\s+learning)\b", re.IGNORECASE),
    re.compile(r"\b(loss\s+function)\b", re.IGNORECASE),
    re.compile(r"\b(activation\s+function)\b", re.IGNORECASE),
    re.compile(r"\b(learning\s+rate)\b", re.IGNORECASE),
    re.compile(r"\b(batch\s+normalization)\b", re.IGNORECASE),
    re.compile(r"\b(dropout)\b", re.IGNORECASE),
    re.compile(r"\b(hyperparameter)\b", re.IGNORECASE),
    re.compile(r"\b(mixture\s+of\s+experts|MoE)\b", re.IGNORECASE),
    re.compile(r"\b(transfer\s+learning)\b", re.IGNORECASE),
    re.compile(r"\b(data\s+augmentation)\b", re.IGNORECASE),
    re.compile(r"\b(generative\s+adversarial|GAN)\b", re.IGNORECASE),
    re.compile(r"\b(variational\s+auto[- ]?encoder|VAE)\b", re.IGNORECASE),
    re.compile(r"\b(Bayesian\s+optimization)\b", re.IGNORECASE),
]


def discover_chapter_folders(base_path: str = ML_PRINCIPLES_PATH) -> List[Path]:
    """Return sorted list of chapter folders found under *base_path*."""
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return []

    folders = sorted(
        p for p in base.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return folders


def find_pdfs(folder: Path) -> List[Path]:
    """Return all PDF files inside *folder* (non-recursive)."""
    return sorted(folder.glob("*.pdf"))


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF using docling (falls back to basic extraction)."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not installed; attempting fallback PDF extraction")
    except Exception as exc:
        logger.warning("docling extraction failed for %s: %s", pdf_path, exc)

    # Fallback: try PyPDF2 / pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Last resort: pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract

        return pdfminer_extract(str(pdf_path))
    except ImportError:
        pass

    logger.error(
        "No PDF extraction library available. Install docling, pypdf, or pdfminer.six."
    )
    return ""


def extract_concepts(text: str) -> List[Dict[str, str]]:
    """Identify ML/AI concepts in extracted text using pattern matching."""
    found: Dict[str, Dict[str, str]] = {}
    lines = text.split("\n")

    for line_no, line in enumerate(lines):
        for pattern in _CONCEPT_PATTERNS:
            for match in pattern.finditer(line):
                term = match.group(0).strip()
                normalised = term.lower()
                if normalised not in found:
                    start = max(0, line_no - 2)
                    end = min(len(lines), line_no + 3)
                    context = "\n".join(lines[start:end])
                    found[normalised] = {
                        "term": term,
                        "context": context,
                        "line": line_no,
                    }

    return list(found.values())


def chapter_id_from_folder(folder: Path) -> str:
    """Derive a stable chapter_id from the folder name."""
    name = folder.name
    # Strip leading numbers and underscores: "08_ML Systems" -> "08_ml_systems"
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


def process_chapter(
    folder: Path,
    db: Any,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract PDFs from a chapter folder and persist to the database.

    Returns a summary dict with extraction results.
    """
    chapter_id = chapter_id_from_folder(folder)
    title = folder.name

    existing = db.get_chapter(chapter_id) if not force else None
    if existing and existing.get("content_md"):
        return {"chapter_id": chapter_id, "status": "skipped", "reason": "already extracted"}

    pdfs = find_pdfs(folder)
    if not pdfs:
        return {"chapter_id": chapter_id, "status": "skipped", "reason": "no PDFs found"}

    db.set_extraction_status(chapter_id, "running", progress=0.0)

    all_text_parts: List[str] = []
    all_concepts: List[Dict[str, str]] = []

    for i, pdf_path in enumerate(pdfs):
        try:
            text = extract_pdf_text(pdf_path)
            all_text_parts.append(text)
            concepts = extract_concepts(text)
            all_concepts.extend(concepts)
            progress = (i + 1) / len(pdfs)
            db.set_extraction_status(chapter_id, "running", progress=progress)
        except Exception as exc:
            logger.error("Failed to extract %s: %s", pdf_path, exc)
            db.set_extraction_status(chapter_id, "error", error=str(exc))
            return {"chapter_id": chapter_id, "status": "error", "error": str(exc)}

    combined_md = "\n\n---\n\n".join(all_text_parts)
    db.upsert_chapter(
        chapter_id=chapter_id,
        folder_name=folder.name,
        title=title,
        content_md=combined_md,
        pdf_path=str(pdfs[0]) if pdfs else "",
    )

    for concept in all_concepts:
        db.add_concept(
            chapter_id=chapter_id,
            term=concept["term"],
            context=concept.get("context", ""),
        )

    db.set_extraction_status(chapter_id, "done", progress=1.0)

    return {
        "chapter_id": chapter_id,
        "status": "done",
        "pages_extracted": len(pdfs),
        "concepts_found": len(all_concepts),
        "content_length": len(combined_md),
    }


def extract_all_chapters(
    db: Any,
    base_path: str = ML_PRINCIPLES_PATH,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Scan all chapter folders and extract content.

    Returns list of per-chapter result summaries.
    """
    folders = discover_chapter_folders(base_path)
    if not folders:
        return [{"status": "error", "error": f"No chapter folders found at {base_path}"}]

    results = []
    for folder in folders:
        result = process_chapter(folder, db, force=force)
        results.append(result)
    return results
