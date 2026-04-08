"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content, identifies concepts,
and stores results in the database.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_CONCEPT_PATTERNS = [
    r"(?i)\b(neural\s+network|deep\s+learning|machine\s+learning)\b",
    r"(?i)\b(gradient\s+descent|backpropagation|forward\s+pass)\b",
    r"(?i)\b(loss\s+function|objective\s+function|cost\s+function)\b",
    r"(?i)\b(regularization|dropout|batch\s+normalization)\b",
    r"(?i)\b(convolutional|recurrent|transformer|attention)\b",
    r"(?i)\b(overfitting|underfitting|bias-variance|generalization)\b",
    r"(?i)\b(cross-validation|hyperparameter|learning\s+rate)\b",
    r"(?i)\b(feature\s+engineering|feature\s+selection|dimensionality)\b",
    r"(?i)\b(ensemble|bagging|boosting|random\s+forest)\b",
    r"(?i)\b(optimization|SGD|Adam|momentum)\b",
    r"(?i)\b(classification|regression|clustering|anomaly)\b",
    r"(?i)\b(precision|recall|F1|AUC|ROC)\b",
    r"(?i)\b(embedding|tokenization|word2vec|BERT)\b",
    r"(?i)\b(reinforcement\s+learning|reward|policy|Q-learning)\b",
    r"(?i)\b(Bayesian|prior|posterior|likelihood)\b",
    r"(?i)\b(decision\s+tree|support\s+vector|SVM|kernel)\b",
    r"(?i)\b(PCA|t-SNE|autoencoder|variational)\b",
    r"(?i)\b(data\s+augmentation|transfer\s+learning|fine-tuning)\b",
    r"(?i)\b(model\s+selection|model\s+evaluation|validation)\b",
    r"(?i)\b(pipeline|preprocessing|normalization|scaling)\b",
]


def _get_ml_principles_path() -> str:
    return os.environ.get(
        "ML_PRINCIPLES_PATH",
        os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
    )


def _parse_chapter_folder_name(folder_name: str) -> Optional[Tuple[int, str]]:
    """Extract chapter number and title from folder name like '08_ML_Systems'."""
    match = re.match(r"(\d+)[_\s-]+(.+)", folder_name)
    if match:
        return int(match.group(1)), match.group(2).replace("_", " ").strip()
    return None


def extract_concepts_from_text(text: str) -> List[Dict[str, Any]]:
    """Identify ML/AI concepts from extracted text."""
    found = {}
    for pattern in ML_CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text):
            concept = match.group(0).strip().lower()
            concept = re.sub(r"\s+", " ", concept)
            if concept not in found:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                found[concept] = {
                    "name": concept,
                    "description": context,
                    "category": _categorize_concept(concept),
                    "confidence": 1.0,
                }
    return list(found.values())


def _categorize_concept(concept: str) -> str:
    categories = {
        "architecture": ["neural network", "convolutional", "recurrent", "transformer",
                         "attention", "autoencoder", "variational"],
        "optimization": ["gradient descent", "backpropagation", "SGD", "Adam",
                         "momentum", "learning rate", "optimization"],
        "regularization": ["regularization", "dropout", "batch normalization",
                           "overfitting", "underfitting"],
        "evaluation": ["cross-validation", "precision", "recall", "F1", "AUC",
                        "ROC", "model evaluation", "validation"],
        "feature_engineering": ["feature engineering", "feature selection",
                                "dimensionality", "PCA", "t-SNE"],
        "ensemble": ["ensemble", "bagging", "boosting", "random forest"],
        "nlp": ["embedding", "tokenization", "word2vec", "BERT"],
        "supervised": ["classification", "regression", "decision tree",
                       "support vector", "SVM"],
        "unsupervised": ["clustering", "anomaly"],
        "reinforcement": ["reinforcement learning", "reward", "policy", "Q-learning"],
        "probabilistic": ["Bayesian", "prior", "posterior", "likelihood"],
        "practical": ["pipeline", "preprocessing", "normalization", "scaling",
                      "data augmentation", "transfer learning", "fine-tuning"],
    }
    concept_lower = concept.lower()
    for cat, keywords in categories.items():
        if any(kw.lower() in concept_lower for kw in keywords):
            return cat
    return "general"


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text content from PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, falling back to basic extraction")
        return _extract_pdf_basic(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        return _extract_pdf_basic(pdf_path)


def _extract_pdf_basic(pdf_path: str) -> str:
    """Fallback PDF extraction without docling."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("Neither docling nor PyMuPDF available; returning empty text")
        return ""


def _count_pdf_pages(pdf_path: str) -> int:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except ImportError:
        return 0


class DoclingWorker:
    """Background worker that scans and extracts ML Principles PDFs."""

    def __init__(self, db: Database, principles_path: Optional[str] = None):
        self.db = db
        self.principles_path = principles_path or _get_ml_principles_path()
        self._thread: Optional[threading.Thread] = None
        self._progress_callback: Optional[Callable] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def discover_chapters(self) -> List[Dict[str, Any]]:
        """Scan the principles directory for chapter folders containing PDFs."""
        chapters = []
        base = Path(self.principles_path)
        if not base.exists():
            logger.warning(f"ML Principles path does not exist: {base}")
            return chapters

        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            parsed = _parse_chapter_folder_name(entry.name)
            if not parsed:
                continue
            chapter_num, title = parsed
            pdfs = list(entry.glob("*.pdf")) + list(entry.glob("*.PDF"))
            if pdfs:
                chapters.append({
                    "chapter_num": chapter_num,
                    "title": title,
                    "folder_path": str(entry),
                    "pdf_files": [str(p) for p in pdfs],
                })
        return chapters

    def extract_chapter(self, chapter_info: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        """Extract content from a single chapter's PDFs."""
        chapter_num = chapter_info["chapter_num"]
        title = chapter_info["title"]

        if not force:
            existing = self.db.get_chapter(chapter_num)
            if existing and existing.get("markdown_content"):
                return {
                    "chapter_num": chapter_num,
                    "title": title,
                    "status": "skipped",
                    "message": "Already extracted",
                }

        all_text = []
        total_pages = 0
        for pdf_path in chapter_info["pdf_files"]:
            text = _extract_pdf_with_docling(pdf_path)
            pages = _count_pdf_pages(pdf_path)
            all_text.append(text)
            total_pages += pages

        combined_text = "\n\n---\n\n".join(all_text)
        chapter_id = self.db.upsert_chapter(
            chapter_num=chapter_num,
            title=title,
            source_path=chapter_info["folder_path"],
            markdown_content=combined_text,
            page_count=total_pages,
        )

        concepts = extract_concepts_from_text(combined_text)
        if concepts:
            self.db.add_concepts(chapter_id, concepts)

        self.db.log_extraction(chapter_id, "completed", f"Extracted {total_pages} pages, {len(concepts)} concepts")

        return {
            "chapter_num": chapter_num,
            "title": title,
            "status": "completed",
            "pages": total_pages,
            "concepts_found": len(concepts),
            "chapter_id": chapter_id,
        }

    def extract_all(self, force_reindex: bool = False, callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Extract all discovered chapters."""
        self._is_running = True
        results = []
        try:
            chapters = self.discover_chapters()
            total = len(chapters)
            for i, chapter in enumerate(chapters):
                try:
                    result = self.extract_chapter(chapter, force=force_reindex)
                    results.append(result)
                    if callback:
                        callback(i + 1, total, result)
                except Exception as e:
                    logger.error(f"Failed to extract chapter {chapter['chapter_num']}: {e}")
                    results.append({
                        "chapter_num": chapter["chapter_num"],
                        "title": chapter["title"],
                        "status": "error",
                        "message": str(e),
                    })
        finally:
            self._is_running = False
        return results

    def extract_all_async(self, force_reindex: bool = False, callback: Optional[Callable] = None):
        """Run extraction in a background thread."""
        self._thread = threading.Thread(
            target=self.extract_all,
            args=(force_reindex, callback),
            daemon=True,
        )
        self._thread.start()
