"""Background PDF extraction using docling for MLSysEng MoE."""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_PATTERNS = [
    r"(?:gradient\s+descent|backpropagation|forward\s+pass)",
    r"(?:loss\s+function|objective\s+function|cost\s+function)",
    r"(?:regularization|dropout|batch\s+normalization)",
    r"(?:convolutional|recurrent|transformer|attention)",
    r"(?:overfitting|underfitting|bias-variance)",
    r"(?:hyperparameter|learning\s+rate|momentum)",
    r"(?:cross-validation|train-test\s+split)",
    r"(?:ensemble|bagging|boosting|stacking)",
    r"(?:neural\s+network|deep\s+learning|machine\s+learning)",
    r"(?:feature\s+engineering|feature\s+selection)",
    r"(?:dimensionality\s+reduction|PCA|t-SNE)",
    r"(?:clustering|k-means|hierarchical)",
    r"(?:classification|regression|supervised|unsupervised)",
    r"(?:reinforcement\s+learning|Q-learning|policy\s+gradient)",
    r"(?:optimization|Adam|SGD|RMSProp)",
    r"(?:activation\s+function|ReLU|sigmoid|softmax)",
    r"(?:embedding|word2vec|GloVe)",
    r"(?:generative|discriminative|GAN|VAE)",
    r"(?:Bayesian|probabilistic|posterior|prior)",
    r"(?:kernel|SVM|support\s+vector)",
]


def discover_chapters(base_path: Optional[str] = None) -> List[Dict[str, str]]:
    """Discover chapter folders containing PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return chapters

    for item in sorted(base.iterdir()):
        if item.is_dir():
            pdfs = list(item.glob("*.pdf"))
            if pdfs:
                chapters.append(
                    {
                        "chapter_name": item.name,
                        "folder_path": str(item),
                        "pdf_path": str(pdfs[0]),
                    }
                )

    return chapters


def extract_pdf_content(pdf_path: str) -> Tuple[str, List[str]]:
    """
    Extract text content and concepts from a PDF using docling.

    Returns (markdown_content, list_of_concepts).
    Falls back to basic extraction if docling unavailable.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_content = result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, using fallback extraction")
        markdown_content = _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"docling extraction failed for {pdf_path}: {e}")
        markdown_content = _fallback_extract(pdf_path)

    concepts = extract_concepts(markdown_content)
    return markdown_content, concepts


def _fallback_extract(pdf_path: str) -> str:
    """Fallback PDF extraction using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    logger.error(f"No PDF extraction library available for {pdf_path}")
    return ""


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text content."""
    if not text:
        return []

    text_lower = text.lower()
    found_concepts = set()

    for pattern in CONCEPT_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            concept = match.strip()
            if len(concept) > 3:
                found_concepts.add(concept)

    return sorted(found_concepts)


def chunk_content(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split content into overlapping chunks for embedding."""
    if not text:
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap

    return chunks


def extract_all_chapters(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    db=None,
) -> Dict[str, any]:
    """
    Extract all chapters from ML Principles PDFs.

    Returns extraction statistics.
    """
    from .database import Database

    if db is None:
        db = Database()

    chapters = discover_chapters(base_path)
    stats = {"discovered": len(chapters), "extracted": 0, "skipped": 0, "errors": 0}

    for chapter_info in chapters:
        chapter_name = chapter_info["chapter_name"]

        if not force_reindex:
            existing = db.get_chapter(chapter_name)
            if existing and existing.get("status") == "extracted":
                stats["skipped"] += 1
                continue

        db.upsert_chapter(
            chapter_name=chapter_name,
            folder_path=chapter_info["folder_path"],
            pdf_path=chapter_info["pdf_path"],
            status="extracting",
        )

        try:
            content, concepts = extract_pdf_content(chapter_info["pdf_path"])

            chapter_id = db.upsert_chapter(
                chapter_name=chapter_name,
                folder_path=chapter_info["folder_path"],
                pdf_path=chapter_info["pdf_path"],
                content_md=content,
                concepts=concepts,
                status="extracted",
            )

            chunks = chunk_content(content)
            for idx, chunk in enumerate(chunks):
                db.add_embedding_chunk(chapter_id, chunk, idx)

            stats["extracted"] += 1
            logger.info(f"Extracted: {chapter_name} ({len(concepts)} concepts, {len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to extract {chapter_name}: {e}")
            db.upsert_chapter(
                chapter_name=chapter_name,
                folder_path=chapter_info["folder_path"],
                pdf_path=chapter_info.get("pdf_path"),
                status="error",
            )
            stats["errors"] += 1

    return stats
