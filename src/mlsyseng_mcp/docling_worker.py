"""
Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies key concepts, and stores results in the database.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ML/AI concept keywords used for extraction heuristics
_CONCEPT_KEYWORDS = [
    "neural network", "deep learning", "gradient descent", "backpropagation",
    "regularization", "dropout", "batch normalization", "convolution",
    "recurrent", "transformer", "attention", "embedding",
    "loss function", "cross-entropy", "mean squared error",
    "optimization", "learning rate", "momentum", "adam",
    "overfitting", "underfitting", "bias-variance",
    "feature engineering", "dimensionality reduction", "pca",
    "ensemble", "random forest", "gradient boosting", "xgboost",
    "support vector", "svm", "kernel",
    "clustering", "k-means", "dbscan",
    "bayesian", "prior", "posterior", "likelihood",
    "reinforcement learning", "reward", "policy", "q-learning",
    "generative", "discriminative", "gan", "vae",
    "hyperparameter", "grid search", "cross-validation",
    "precision", "recall", "f1", "auc", "roc",
    "data augmentation", "transfer learning", "fine-tuning",
    "tokenization", "word2vec", "bert", "gpt",
    "pipeline", "mlops", "deployment", "inference",
    "distributed", "parallel", "gpu", "tpu",
    "normalization", "standardization", "scaling",
    "classification", "regression", "segmentation",
]


def _try_docling_extract(pdf_path: str) -> str:
    """Extract text from a PDF using docling, falling back to simple text extraction."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return _basic_pdf_extract(pdf_path)
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)
        return _basic_pdf_extract(pdf_path)


def _basic_pdf_extract(pdf_path: str) -> str:
    """Minimal PDF text extraction using PyMuPDF or pdfminer as fallback."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
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
        "No PDF extraction library available. Install docling, PyMuPDF, or pdfminer.six"
    )
    return ""


def extract_concepts(text: str) -> List[str]:
    """Identify ML/AI concepts present in the text."""
    text_lower = text.lower()
    found = []
    for concept in _CONCEPT_KEYWORDS:
        if concept in text_lower:
            found.append(concept)
    return sorted(set(found))


def _chapter_number_from_name(name: str) -> int:
    """Try to parse a chapter number from the folder/file name."""
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return 0


def _chapter_id(path: str) -> str:
    """Deterministic chapter ID from file path."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def scan_chapter_folders(base_path: str) -> List[Dict[str, Any]]:
    """
    Scan the ML Principles base directory for chapter folders containing PDFs.

    Returns a list of dicts: {chapter_name, chapter_number, pdf_path}
    """
    base = Path(base_path)
    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base_path)
        return []

    chapters: List[Dict[str, Any]] = []

    if base.is_file() and base.suffix.lower() == ".pdf":
        chapters.append(
            {
                "chapter_name": base.stem,
                "chapter_number": _chapter_number_from_name(base.stem),
                "pdf_path": str(base),
            }
        )
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf")) + list(item.glob("*.PDF"))
            for pdf in pdfs:
                chapters.append(
                    {
                        "chapter_name": item.name,
                        "chapter_number": _chapter_number_from_name(item.name),
                        "pdf_path": str(pdf),
                    }
                )
        elif item.is_file() and item.suffix.lower() == ".pdf":
            chapters.append(
                {
                    "chapter_name": item.stem,
                    "chapter_number": _chapter_number_from_name(item.stem),
                    "pdf_path": str(item),
                }
            )

    return chapters


def extract_chapter(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract content and concepts from a single PDF.

    Returns (markdown_content, concepts_list).
    """
    content = _try_docling_extract(pdf_path)
    concepts = extract_concepts(content)
    return content, concepts


class DoclingWorker:
    """Manages the extraction pipeline for all ML Principles chapters."""

    def __init__(self, db, ml_principles_path: Optional[str] = None):
        self.db = db
        self.ml_principles_path = ml_principles_path or os.environ.get(
            "ML_PRINCIPLES_PATH",
            str(Path.home() / "Desktop" / "Machine Learning Principles - Chapters"),
        )
        self._progress: Dict[str, str] = {}

    @property
    def progress(self) -> Dict[str, str]:
        return dict(self._progress)

    def run(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Run the full extraction pipeline.

        1. Scan chapter folders
        2. Extract PDF content
        3. Identify concepts
        4. Store in database
        """
        from .database import ChapterRecord

        chapters_found = scan_chapter_folders(self.ml_principles_path)

        if not chapters_found:
            return {
                "status": "no_chapters",
                "message": f"No PDF chapters found at {self.ml_principles_path}",
                "chapters_processed": 0,
            }

        self.db.set_status("extraction_status", "running")
        self.db.set_status("extraction_total", str(len(chapters_found)))

        processed = 0
        errors = []

        for ch_info in chapters_found:
            pdf_path = ch_info["pdf_path"]
            ch_id = _chapter_id(pdf_path)

            self._progress[ch_info["chapter_name"]] = "extracting"

            if not force_reindex and self.db.get_chapter(ch_id) is not None:
                self._progress[ch_info["chapter_name"]] = "skipped"
                processed += 1
                continue

            try:
                content, concepts = extract_chapter(pdf_path)

                record = ChapterRecord(
                    chapter_id=ch_id,
                    chapter_name=ch_info["chapter_name"],
                    chapter_number=ch_info["chapter_number"],
                    source_path=pdf_path,
                    content_md=content,
                    concepts=concepts,
                )
                self.db.upsert_chapter(record)

                self._progress[ch_info["chapter_name"]] = "done"
                processed += 1
            except Exception as e:
                logger.error("Failed to extract %s: %s", pdf_path, e)
                errors.append({"chapter": ch_info["chapter_name"], "error": str(e)})
                self._progress[ch_info["chapter_name"]] = f"error: {e}"

        self.db.set_status("extraction_processed", str(processed))
        self.db.set_status(
            "extraction_status", "completed" if not errors else "completed_with_errors"
        )

        return {
            "status": "completed" if not errors else "completed_with_errors",
            "chapters_found": len(chapters_found),
            "chapters_processed": processed,
            "errors": errors,
        }
