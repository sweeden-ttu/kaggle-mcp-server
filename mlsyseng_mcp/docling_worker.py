"""Background PDF extraction worker using docling.

Scans ML Principles chapter folders, extracts PDF content,
identifies concepts, and stores results in the database.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH", os.path.expanduser("~/ML_Principles_Chapters")
)

CONCEPT_PATTERNS = [
    r"\b(neural network|deep learning|convolutional|recurrent|transformer)\b",
    r"\b(gradient descent|backpropagation|optimization|loss function)\b",
    r"\b(regularization|dropout|batch normalization|weight decay)\b",
    r"\b(ensemble|bagging|boosting|random forest|xgboost)\b",
    r"\b(support vector|kernel|svm|svr)\b",
    r"\b(decision tree|classification|regression|clustering)\b",
    r"\b(feature engineering|feature selection|dimensionality reduction|pca)\b",
    r"\b(cross.?validation|train.?test split|overfitting|underfitting|bias.?variance)\b",
    r"\b(reinforcement learning|q.?learning|policy gradient|reward)\b",
    r"\b(natural language processing|nlp|tokenization|embedding|attention)\b",
    r"\b(generative adversarial|gan|variational autoencoder|vae)\b",
    r"\b(hyperparameter|grid search|bayesian optimization|learning rate)\b",
    r"\b(data augmentation|preprocessing|normalization|standardization)\b",
    r"\b(activation function|relu|sigmoid|softmax|tanh)\b",
    r"\b(precision|recall|f1.?score|auc|roc|confusion matrix|accuracy)\b",
    r"\b(time series|forecasting|arima|lstm|sequence)\b",
    r"\b(computer vision|image classification|object detection|segmentation)\b",
    r"\b(model selection|model evaluation|model deployment|mlops)\b",
]


def _extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    concepts = set()
    text_lower = text.lower()
    for pattern in CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            concept = match.group(0).strip()
            concept = re.sub(r"\s+", " ", concept)
            concepts.add(concept.title())
    return sorted(concepts)


def _parse_chapter_info(folder_name: str) -> Optional[Tuple[int, str]]:
    """Extract chapter number and title from folder name.

    Expected formats:
        '01 - Introduction to ML'
        '02_Deep_Learning'
        'Chapter 3 - Optimization'
    """
    patterns = [
        r"^(\d+)\s*[-_]\s*(.+)$",
        r"^[Cc]hapter\s*(\d+)\s*[-_:]\s*(.+)$",
        r"^(\d+)\.?\s+(.+)$",
    ]
    for pat in patterns:
        m = re.match(pat, folder_name)
        if m:
            return int(m.group(1)), m.group(2).strip().replace("_", " ")
    return None


def _extract_pdf_with_docling(pdf_path: str) -> Tuple[str, int]:
    """Extract text from PDF using docling. Returns (markdown_text, page_count)."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        page_count = len(result.document.pages) if hasattr(result.document, "pages") else 0
        return markdown, page_count
    except ImportError:
        logger.warning("docling not installed, falling back to basic PDF extraction")
        return _extract_pdf_fallback(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _extract_pdf_fallback(pdf_path)


def _extract_pdf_fallback(pdf_path: str) -> Tuple[str, int]:
    """Fallback PDF extraction using pdfplumber or PyPDF2."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts), page_count
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        page_count = len(reader.pages)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts), page_count
    except ImportError:
        pass

    logger.error("No PDF extraction library available. Install docling, pdfplumber, or PyPDF2.")
    return "", 0


def scan_chapters(
    principles_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan the ML Principles directory for chapter folders with PDFs."""
    base = Path(principles_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not base.exists():
        logger.warning("ML Principles path not found: %s", base)
        return []

    chapters = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        info = _parse_chapter_info(item.name)
        if info is None:
            continue
        chapter_num, title = info
        pdfs = list(item.glob("*.pdf"))
        if not pdfs:
            pdfs = list(item.glob("*.PDF"))
        if pdfs:
            chapters.append({
                "chapter_num": chapter_num,
                "title": title,
                "folder_path": str(item),
                "pdf_path": str(pdfs[0]),
            })
    return chapters


def extract_chapter(
    chapter_info: Dict[str, Any], db: Database
) -> Dict[str, Any]:
    """Extract a single chapter's PDF and store in database."""
    chapter_num = chapter_info["chapter_num"]
    title = chapter_info["title"]
    pdf_path = chapter_info["pdf_path"]

    db.set_extraction_status(chapter_num, "started")

    try:
        markdown, page_count = _extract_pdf_with_docling(pdf_path)
        if not markdown.strip():
            raise ValueError(f"Empty extraction result for {pdf_path}")

        concepts = _extract_concepts(markdown)

        chapter_id = db.upsert_chapter(
            chapter_num=chapter_num,
            title=title,
            source_path=pdf_path,
            markdown_content=markdown,
            concepts=concepts,
            page_count=page_count,
        )

        db.set_extraction_status(chapter_num, "completed")

        return {
            "chapter_num": chapter_num,
            "title": title,
            "chapter_id": chapter_id,
            "page_count": page_count,
            "concepts": concepts,
            "status": "success",
        }
    except Exception as e:
        db.set_extraction_status(chapter_num, "failed", str(e))
        logger.error("Failed to extract chapter %d (%s): %s", chapter_num, title, e)
        return {
            "chapter_num": chapter_num,
            "title": title,
            "status": "failed",
            "error": str(e),
        }


def extract_all_chapters(
    principles_path: Optional[str] = None,
    db: Optional[Database] = None,
    force_reindex: bool = False,
) -> List[Dict[str, Any]]:
    """Scan and extract all chapters. Skip already-extracted unless force_reindex."""
    if db is None:
        db = Database()

    chapters = scan_chapters(principles_path)
    if not chapters:
        return [{"status": "no_chapters", "message": "No chapter folders found"}]

    results = []
    for ch in chapters:
        if not force_reindex:
            existing = db.get_chapter(ch["chapter_num"])
            if existing and existing.get("markdown_content"):
                results.append({
                    "chapter_num": ch["chapter_num"],
                    "title": ch["title"],
                    "status": "skipped",
                    "message": "Already indexed",
                })
                continue

        result = extract_chapter(ch, db)
        results.append(result)

    return results
