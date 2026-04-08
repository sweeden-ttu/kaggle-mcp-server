"""Docling worker for PDF extraction from ML Principles chapters.

Scans chapter directories, extracts PDF content using docling,
and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)


def _extract_chapter_number(name: str) -> int | None:
    """Extract chapter number from folder or filename like '08_ML_Systems'."""
    m = re.match(r"^(\d+)", name)
    return int(m.group(1)) if m else None


def _extract_title(name: str) -> str:
    """Turn '08_ML_Systems' into 'ML Systems'."""
    cleaned = re.sub(r"^\d+[_\s]*", "", name)
    return cleaned.replace("_", " ").strip() or name


def _find_chapter_dirs(base_path: str) -> list[dict]:
    """Find chapter directories or PDFs under the base path."""
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base_path)
        return []

    chapters = []
    for entry in sorted(base.iterdir()):
        num = _extract_chapter_number(entry.name)
        if num is None:
            continue

        if entry.is_dir():
            pdfs = list(entry.glob("*.pdf"))
            if pdfs:
                chapters.append(
                    {
                        "number": num,
                        "title": _extract_title(entry.name),
                        "pdf_path": str(pdfs[0]),
                        "dir_path": str(entry),
                    }
                )
        elif entry.suffix.lower() == ".pdf":
            chapters.append(
                {
                    "number": num,
                    "title": _extract_title(entry.stem),
                    "pdf_path": str(entry),
                    "dir_path": str(entry.parent),
                }
            )
    return chapters


def extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text content from a PDF using docling.

    Falls back to a simpler extraction method if docling is not available.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.info("docling not available, trying PyPDF2 fallback")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        import PyPDF2

        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except ImportError:
        logger.info("PyPDF2 not available, trying pdfminer fallback")
    except Exception as e:
        logger.warning("PyPDF2 extraction failed for %s: %s", pdf_path, e)

    try:
        from pdfminer.high_level import extract_text

        return extract_text(pdf_path)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("pdfminer extraction failed for %s: %s", pdf_path, e)

    return f"[PDF extraction unavailable for {pdf_path}]"


def extract_concepts_from_text(text: str) -> list[dict[str, str]]:
    """Extract ML/AI concepts from chapter text using heuristics.

    Identifies section headings, bold terms, and common ML vocabulary.
    """
    concepts = []
    seen = set()

    ml_keywords = {
        "gradient descent",
        "backpropagation",
        "regularization",
        "cross-validation",
        "ensemble",
        "bagging",
        "boosting",
        "dropout",
        "batch normalization",
        "attention mechanism",
        "transformer",
        "convolution",
        "recurrent",
        "lstm",
        "gru",
        "autoencoder",
        "generative adversarial",
        "reinforcement learning",
        "transfer learning",
        "fine-tuning",
        "hyperparameter",
        "loss function",
        "activation function",
        "feature engineering",
        "dimensionality reduction",
        "principal component",
        "support vector",
        "decision tree",
        "random forest",
        "neural network",
        "deep learning",
        "optimization",
        "stochastic",
        "learning rate",
        "momentum",
        "adam optimizer",
        "bias-variance",
        "overfitting",
        "underfitting",
        "embedding",
        "tokenization",
        "normalization",
        "data augmentation",
        "confusion matrix",
        "precision",
        "recall",
        "f1 score",
        "roc curve",
        "auc",
    }

    headings = re.findall(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)
    for h in headings:
        clean = h.strip().strip("#").strip()
        if clean.lower() not in seen and len(clean) > 2:
            concepts.append(
                {"concept": clean, "description": f"Section: {clean}", "category": "heading"}
            )
            seen.add(clean.lower())

    bold_terms = re.findall(r"\*\*([^*]+)\*\*", text)
    for term in bold_terms:
        term = term.strip()
        if term.lower() not in seen and 2 < len(term) < 80:
            concepts.append(
                {"concept": term, "description": f"Key term: {term}", "category": "term"}
            )
            seen.add(term.lower())

    text_lower = text.lower()
    for kw in ml_keywords:
        if kw in text_lower and kw not in seen:
            concepts.append(
                {"concept": kw.title(), "description": f"ML concept: {kw}", "category": "ml_concept"}
            )
            seen.add(kw)

    return concepts


def run_extraction(
    db: "Database",
    base_path: str | None = None,
    force_reindex: bool = False,
) -> dict:
    """Run full extraction pipeline on all chapter directories.

    Returns a summary dict with counts.
    """
    from .database import Database as DB

    if not isinstance(db, DB):
        raise TypeError("db must be a Database instance")

    path = base_path or ML_PRINCIPLES_PATH
    chapter_dirs = _find_chapter_dirs(path)

    if not chapter_dirs:
        return {
            "status": "no_chapters_found",
            "path": path,
            "chapters_processed": 0,
        }

    results = {"chapters_processed": 0, "concepts_extracted": 0, "errors": []}

    for ch in chapter_dirs:
        existing = db.get_chapter_content(ch["number"])
        if existing and not force_reindex:
            logger.info("Chapter %d already indexed, skipping", ch["number"])
            continue

        chapter_id = db.upsert_chapter(
            chapter_number=ch["number"],
            title=ch["title"],
            source_path=ch["pdf_path"],
            markdown_content="",
            page_count=0,
        )
        db.log_extraction(chapter_id, "running", f"Extracting {ch['pdf_path']}")

        try:
            markdown = extract_pdf_with_docling(ch["pdf_path"])
            page_count = markdown.count("\n\n---\n\n") + 1

            db.upsert_chapter(
                chapter_number=ch["number"],
                title=ch["title"],
                source_path=ch["pdf_path"],
                markdown_content=markdown,
                page_count=page_count,
            )

            concepts = extract_concepts_from_text(markdown)
            db.add_concepts(chapter_id, concepts)
            results["concepts_extracted"] += len(concepts)

            db.log_extraction(
                chapter_id,
                "done",
                f"Extracted {len(concepts)} concepts, {page_count} pages",
            )
            results["chapters_processed"] += 1
        except Exception as e:
            logger.error("Failed to extract chapter %d: %s", ch["number"], e)
            db.log_extraction(chapter_id, "error", str(e))
            results["errors"].append(
                {"chapter": ch["number"], "error": str(e)}
            )

    results["status"] = "complete"
    return results
