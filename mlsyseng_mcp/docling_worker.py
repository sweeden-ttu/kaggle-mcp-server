"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies key concepts, and stores everything in SQLite.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from mlsyseng_mcp.database import Database, ChapterRecord

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

# Common ML/AI concept keywords for extraction
_CONCEPT_PATTERNS = [
    r"(?i)\b(gradient\s+descent)\b",
    r"(?i)\b(backpropagation)\b",
    r"(?i)\b(regularization)\b",
    r"(?i)\b(cross[\-\s]?validation)\b",
    r"(?i)\b(overfitting|underfitting)\b",
    r"(?i)\b(bias[\-\s]?variance)\b",
    r"(?i)\b(ensemble\s+(?:methods?|learning))\b",
    r"(?i)\b(neural\s+network)\b",
    r"(?i)\b(deep\s+learning)\b",
    r"(?i)\b(convolutional\s+neural)\b",
    r"(?i)\b(recurrent\s+neural)\b",
    r"(?i)\b(transformer)\b",
    r"(?i)\b(attention\s+mechanism)\b",
    r"(?i)\b(reinforcement\s+learning)\b",
    r"(?i)\b(supervised\s+learning)\b",
    r"(?i)\b(unsupervised\s+learning)\b",
    r"(?i)\b(feature\s+engineering)\b",
    r"(?i)\b(hyperparameter\s+tuning)\b",
    r"(?i)\b(loss\s+function)\b",
    r"(?i)\b(activation\s+function)\b",
    r"(?i)\b(batch\s+normalization)\b",
    r"(?i)\b(dropout)\b",
    r"(?i)\b(learning\s+rate)\b",
    r"(?i)\b(optimization)\b",
    r"(?i)\b(stochastic\s+gradient)\b",
    r"(?i)\b(decision\s+tree)\b",
    r"(?i)\b(random\s+forest)\b",
    r"(?i)\b(support\s+vector)\b",
    r"(?i)\b(principal\s+component)\b",
    r"(?i)\b(dimensionality\s+reduction)\b",
    r"(?i)\b(clustering)\b",
    r"(?i)\b(k[\-\s]?means)\b",
    r"(?i)\b(bayesian)\b",
    r"(?i)\b(maximum\s+likelihood)\b",
    r"(?i)\b(information\s+gain)\b",
    r"(?i)\b(entropy)\b",
    r"(?i)\b(precision|recall|f1[\-\s]?score)\b",
    r"(?i)\b(roc\s+curve|auc)\b",
    r"(?i)\b(confusion\s+matrix)\b",
    r"(?i)\b(data\s+augmentation)\b",
    r"(?i)\b(transfer\s+learning)\b",
    r"(?i)\b(fine[\-\s]?tuning)\b",
    r"(?i)\b(embedding)\b",
    r"(?i)\b(tokenization)\b",
    r"(?i)\b(normalization)\b",
    r"(?i)\b(softmax)\b",
    r"(?i)\b(sigmoid)\b",
    r"(?i)\b(relu)\b",
]


def _extract_concepts(text: str) -> list[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    found = set()
    for pattern in _CONCEPT_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            concept = m.strip().lower()
            concept = re.sub(r"\s+", " ", concept)
            found.add(concept)
    return sorted(found)


def _slugify(name: str) -> str:
    """Convert a chapter folder name to a clean slug."""
    slug = re.sub(r"^(\d+)[\s_\-\.]+", r"\1_", name)
    slug = re.sub(r"[^\w]+", "_", slug)
    slug = slug.strip("_").lower()
    return slug


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling, falling back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, attempting fallback extraction")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not installed, attempting pdfplumber")
    except Exception as e:
        logger.warning("PyMuPDF extraction failed: %s", e)

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("No PDF extraction library available")
    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)

    return ""


class DoclingWorker:
    """Background worker for extracting ML Principles PDFs into the database."""

    def __init__(
        self,
        db: Database,
        ml_principles_path: str = DEFAULT_ML_PRINCIPLES_PATH,
    ):
        self.db = db
        self.ml_principles_path = Path(ml_principles_path)

    def discover_chapters(self) -> list[dict]:
        """Scan the ML Principles directory for chapter folders with PDFs."""
        if not self.ml_principles_path.exists():
            logger.warning("ML Principles path not found: %s", self.ml_principles_path)
            return []

        chapters = []
        for item in sorted(self.ml_principles_path.iterdir()):
            if not item.is_dir():
                continue
            pdfs = list(item.glob("*.pdf")) + list(item.glob("*.PDF"))
            if not pdfs:
                continue
            chapter_id = _slugify(item.name)
            chapters.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_name": item.name,
                    "source_path": str(item),
                    "pdfs": [str(p) for p in pdfs],
                }
            )
        return chapters

    def extract_chapter(
        self, chapter_info: dict, force: bool = False
    ) -> Optional[ChapterRecord]:
        """Extract a single chapter's PDFs and store in the database."""
        chapter_id = chapter_info["chapter_id"]

        if not force and self.db.chapter_exists(chapter_id):
            logger.info("Chapter %s already indexed, skipping", chapter_id)
            return self.db.get_chapter(chapter_id)

        self.db.log_extraction(chapter_id, "started", "Beginning extraction")

        all_text = []
        for pdf_path in chapter_info["pdfs"]:
            logger.info("Extracting %s", pdf_path)
            text = _extract_pdf_with_docling(pdf_path)
            if text:
                all_text.append(text)
            else:
                self.db.log_extraction(
                    chapter_id, "warning", f"Empty extraction: {pdf_path}"
                )

        if not all_text:
            self.db.log_extraction(chapter_id, "failed", "No text extracted from PDFs")
            return None

        content_md = "\n\n---\n\n".join(all_text)
        concepts = _extract_concepts(content_md)
        word_count = len(content_md.split())

        record = ChapterRecord(
            chapter_id=chapter_id,
            chapter_name=chapter_info["chapter_name"],
            source_path=chapter_info["source_path"],
            content_md=content_md,
            concepts=concepts,
            extracted_at=time.time(),
            word_count=word_count,
        )

        self.db.upsert_chapter(record)
        self.db.log_extraction(
            chapter_id,
            "completed",
            f"Extracted {word_count} words, {len(concepts)} concepts",
        )

        logger.info(
            "Chapter %s: %d words, %d concepts",
            chapter_id,
            word_count,
            len(concepts),
        )
        return record

    def extract_all(self, force_reindex: bool = False) -> list[ChapterRecord]:
        """Extract all discovered chapters."""
        chapters = self.discover_chapters()
        if not chapters:
            logger.warning("No chapters discovered at %s", self.ml_principles_path)
            return []

        results = []
        for chapter_info in chapters:
            record = self.extract_chapter(chapter_info, force=force_reindex)
            if record:
                results.append(record)

        logger.info(
            "Extraction complete: %d/%d chapters processed",
            len(results),
            len(chapters),
        )
        return results

    def get_status(self) -> dict:
        """Get extraction status summary."""
        chapters = self.discover_chapters()
        indexed = self.db.list_chapters()
        indexed_ids = {c.chapter_id for c in indexed}

        return {
            "discovered": len(chapters),
            "indexed": len(indexed),
            "pending": len([c for c in chapters if c["chapter_id"] not in indexed_ids]),
            "ml_principles_path": str(self.ml_principles_path),
            "path_exists": self.ml_principles_path.exists(),
            "chapters": [
                {
                    "chapter_id": c["chapter_id"],
                    "name": c["chapter_name"],
                    "pdf_count": len(c["pdfs"]),
                    "indexed": c["chapter_id"] in indexed_ids,
                }
                for c in chapters
            ],
        }
