"""Background PDF extraction worker using docling."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

from mlsyseng_moe.database import (
    get_connection,
    init_db,
    insert_chapter,
    insert_concept,
    insert_content,
    log_extraction,
)

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_KEYWORDS = [
    "gradient descent", "backpropagation", "regularization", "cross-validation",
    "overfitting", "underfitting", "bias-variance", "ensemble", "bagging",
    "boosting", "neural network", "deep learning", "convolutional",
    "recurrent", "transformer", "attention", "optimization", "loss function",
    "activation function", "batch normalization", "dropout", "learning rate",
    "hyperparameter", "feature engineering", "dimensionality reduction",
    "clustering", "classification", "regression", "reinforcement learning",
    "supervised learning", "unsupervised learning", "semi-supervised",
    "transfer learning", "data augmentation", "normalization",
    "standardization", "embedding", "tokenization", "fine-tuning",
    "pre-training", "inference", "model selection", "cross-entropy",
    "mean squared error", "precision", "recall", "f1 score", "auc",
    "roc curve", "confusion matrix", "kernel", "support vector",
    "decision tree", "random forest", "gradient boosting",
    "principal component", "singular value decomposition",
    "matrix factorization", "bayesian", "markov", "monte carlo",
    "generative adversarial", "autoencoder", "variational",
]


def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Scan the ML Principles directory for chapter folders and PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning(f"ML Principles path does not exist: {base}")
        return chapters

    chapter_pattern = re.compile(r"(\d+)[_\s\-]*(.*)")

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        match = chapter_pattern.match(item.name)
        if not match:
            continue

        chapter_num = int(match.group(1))
        title = match.group(2).replace("_", " ").strip()

        pdfs = list(item.glob("*.pdf"))
        pdf_path = str(pdfs[0]) if pdfs else None

        chapters.append({
            "chapter_number": chapter_num,
            "title": title or f"Chapter {chapter_num}",
            "folder_path": str(item),
            "pdf_path": pdf_path,
        })

    return chapters


def extract_pdf_content(pdf_path: str) -> list[dict]:
    """Extract content from a PDF using docling or fallback methods."""
    sections = []

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        for i, element in enumerate(doc.texts):
            sections.append({
                "section_title": getattr(element, "label", f"Section {i}"),
                "content": element.text,
                "page_number": getattr(element, "page_no", None),
                "content_type": "text",
            })

    except ImportError:
        logger.info("Docling not available, using fallback PDF extraction")
        sections = _fallback_extract(pdf_path)
    except Exception as e:
        logger.error(f"Docling extraction failed for {pdf_path}: {e}")
        sections = _fallback_extract(pdf_path)

    return sections


def _fallback_extract(pdf_path: str) -> list[dict]:
    """Fallback PDF extraction using PyPDF2 or pdfplumber."""
    sections = []

    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    sections.append({
                        "section_title": f"Page {i + 1}",
                        "content": text.strip(),
                        "page_number": i + 1,
                        "content_type": "text",
                    })
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(pdf_path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    sections.append({
                        "section_title": f"Page {i + 1}",
                        "content": text.strip(),
                        "page_number": i + 1,
                        "content_type": "text",
                    })
        except ImportError:
            logger.error("No PDF extraction library available (docling, pdfplumber, or PyPDF2)")

    return sections


def extract_concepts(content_blocks: list[dict]) -> list[dict]:
    """Identify ML/AI concepts from extracted content."""
    all_text = " ".join(block["content"].lower() for block in content_blocks)
    found_concepts = []

    for keyword in CONCEPT_KEYWORDS:
        if keyword in all_text:
            count = all_text.count(keyword)
            context_idx = all_text.find(keyword)
            context = all_text[max(0, context_idx - 50):context_idx + len(keyword) + 100]
            found_concepts.append({
                "concept_name": keyword.title(),
                "description": f"Mentioned {count} time(s). Context: ...{context.strip()}...",
                "category": _categorize_concept(keyword),
            })

    return found_concepts


def _categorize_concept(concept: str) -> str:
    """Categorize a concept into a broad ML category."""
    optimization = ["gradient descent", "optimization", "learning rate", "loss function", "backpropagation"]
    architecture = ["neural network", "deep learning", "convolutional", "recurrent", "transformer", "attention"]
    regularization = ["regularization", "dropout", "batch normalization", "overfitting", "underfitting"]
    evaluation = ["cross-validation", "precision", "recall", "f1 score", "auc", "roc curve", "confusion matrix"]
    methods = ["ensemble", "bagging", "boosting", "random forest", "gradient boosting", "decision tree"]
    unsupervised = ["clustering", "dimensionality reduction", "principal component", "autoencoder"]

    if concept in optimization:
        return "optimization"
    elif concept in architecture:
        return "architecture"
    elif concept in regularization:
        return "regularization"
    elif concept in evaluation:
        return "evaluation"
    elif concept in methods:
        return "ensemble_methods"
    elif concept in unsupervised:
        return "unsupervised"
    return "general"


def process_chapter(
    chapter_info: dict,
    force_reindex: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Process a single chapter: extract PDF, store content, identify concepts."""
    pdf_path = chapter_info.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        return {"status": "skipped", "reason": "no PDF found"}

    content_hash = compute_file_hash(pdf_path)

    if not force_reindex:
        conn = get_connection(db_path)
        try:
            existing = conn.execute(
                "SELECT content_hash FROM chapters WHERE chapter_number = ?",
                (chapter_info["chapter_number"],),
            ).fetchone()
            if existing and existing["content_hash"] == content_hash:
                return {"status": "skipped", "reason": "already indexed (hash match)"}
        finally:
            conn.close()

    chapter_id = insert_chapter(
        chapter_number=chapter_info["chapter_number"],
        title=chapter_info["title"],
        folder_path=chapter_info["folder_path"],
        pdf_path=pdf_path,
        db_path=db_path,
    )

    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE chapters SET content_hash = ? WHERE id = ?", (content_hash, chapter_id))
        conn.commit()
    finally:
        conn.close()

    log_extraction(chapter_id, "started", db_path=db_path)

    try:
        content_blocks = extract_pdf_content(pdf_path)

        for block in content_blocks:
            insert_content(
                chapter_id=chapter_id,
                content=block["content"],
                section_title=block.get("section_title"),
                page_number=block.get("page_number"),
                content_type=block.get("content_type", "text"),
                db_path=db_path,
            )

        concepts = extract_concepts(content_blocks)
        for concept in concepts:
            insert_concept(
                chapter_id=chapter_id,
                concept_name=concept["concept_name"],
                description=concept.get("description"),
                category=concept.get("category"),
                db_path=db_path,
            )

        log_extraction(chapter_id, "completed", f"Extracted {len(content_blocks)} blocks, {len(concepts)} concepts", db_path=db_path)

        return {
            "status": "completed",
            "chapter_id": chapter_id,
            "content_blocks": len(content_blocks),
            "concepts": len(concepts),
        }

    except Exception as e:
        log_extraction(chapter_id, "failed", str(e), db_path=db_path)
        logger.error(f"Failed to process chapter {chapter_info['title']}: {e}")
        return {"status": "failed", "error": str(e)}


def extract_all(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict:
    """Extract knowledge from all chapters."""
    init_db(db_path)
    chapters = discover_chapters(base_path)

    results = {
        "total_chapters": len(chapters),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }

    for chapter in chapters:
        result = process_chapter(chapter, force_reindex=force_reindex, db_path=db_path)
        results["details"].append({
            "chapter": chapter["title"],
            "result": result,
        })

        if result["status"] == "completed":
            results["processed"] += 1
        elif result["status"] == "skipped":
            results["skipped"] += 1
        else:
            results["failed"] += 1

    return results
