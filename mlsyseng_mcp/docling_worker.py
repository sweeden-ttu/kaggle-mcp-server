"""Background PDF extraction worker using docling."""

import os
import re
from pathlib import Path
from typing import Optional

try:
    from docling.document_converter import DocumentConverter

    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

from mlsyseng_mcp.database import (
    insert_chapter,
    insert_concept,
    log_extraction,
)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

ML_CONCEPTS = [
    "supervised learning",
    "unsupervised learning",
    "reinforcement learning",
    "neural network",
    "deep learning",
    "gradient descent",
    "backpropagation",
    "regularization",
    "cross-validation",
    "bias-variance tradeoff",
    "ensemble methods",
    "random forest",
    "support vector machine",
    "decision tree",
    "linear regression",
    "logistic regression",
    "clustering",
    "dimensionality reduction",
    "feature engineering",
    "hyperparameter tuning",
    "convolutional neural network",
    "recurrent neural network",
    "transformer",
    "attention mechanism",
    "generative adversarial network",
    "autoencoder",
    "bayesian inference",
    "optimization",
    "loss function",
    "activation function",
    "batch normalization",
    "dropout",
    "transfer learning",
    "data augmentation",
    "model selection",
    "overfitting",
    "underfitting",
    "feature selection",
    "principal component analysis",
    "natural language processing",
    "computer vision",
    "time series",
    "anomaly detection",
    "recommendation system",
    "kernel methods",
    "boosting",
    "bagging",
    "stochastic gradient descent",
    "learning rate",
    "momentum",
]


def discover_chapters(base_path: Optional[str] = None) -> list[dict]:
    """Discover chapter folders and their PDFs."""
    path = Path(base_path or DEFAULT_ML_PRINCIPLES_PATH)
    if not path.exists():
        return []

    chapters = []
    pattern = re.compile(r"^(\d+)[_\s\-]+(.+)$")

    for item in sorted(path.iterdir()):
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                chapter_number = int(match.group(1))
                title = match.group(2).replace("_", " ").strip()
                pdfs = list(item.glob("*.pdf"))
                if pdfs:
                    chapters.append({
                        "chapter_number": chapter_number,
                        "title": title,
                        "path": str(item),
                        "pdfs": [str(p) for p in pdfs],
                    })

    if not chapters:
        for pdf in sorted(path.glob("*.pdf")):
            match = re.match(r"^(\d+)[_\s\-]+(.+)\.pdf$", pdf.name)
            if match:
                chapters.append({
                    "chapter_number": int(match.group(1)),
                    "title": match.group(2).replace("_", " ").strip(),
                    "path": str(pdf.parent),
                    "pdfs": [str(pdf)],
                })

    return chapters


def extract_pdf_content(pdf_path: str) -> str:
    """Extract text content from a PDF using docling."""
    if HAS_DOCLING:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    else:
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
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
        return f"[PDF extraction unavailable for: {pdf_path}]"


def extract_concepts(content: str) -> list[dict]:
    """Extract ML/AI concepts from text content."""
    content_lower = content.lower()
    found = []

    for concept in ML_CONCEPTS:
        if concept in content_lower:
            count = content_lower.count(concept)
            sentences = [
                s.strip()
                for s in content.split(".")
                if concept in s.lower()
            ]
            description = sentences[0] if sentences else ""
            category = _categorize_concept(concept)
            confidence = min(1.0, count / 5.0)

            found.append({
                "concept_name": concept,
                "description": description[:500],
                "category": category,
                "confidence": confidence,
            })

    return found


def _categorize_concept(concept: str) -> str:
    """Categorize a concept into a high-level category."""
    categories = {
        "algorithm": [
            "random forest", "support vector machine", "decision tree",
            "linear regression", "logistic regression", "clustering",
            "boosting", "bagging", "kernel methods",
        ],
        "deep_learning": [
            "neural network", "deep learning", "convolutional neural network",
            "recurrent neural network", "transformer", "attention mechanism",
            "generative adversarial network", "autoencoder",
        ],
        "optimization": [
            "gradient descent", "stochastic gradient descent", "learning rate",
            "momentum", "optimization", "loss function", "backpropagation",
        ],
        "regularization": [
            "regularization", "dropout", "batch normalization",
            "overfitting", "underfitting", "bias-variance tradeoff",
        ],
        "methodology": [
            "supervised learning", "unsupervised learning",
            "reinforcement learning", "cross-validation",
            "feature engineering", "hyperparameter tuning",
            "model selection", "transfer learning", "data augmentation",
            "feature selection", "ensemble methods",
        ],
        "application": [
            "natural language processing", "computer vision",
            "time series", "anomaly detection", "recommendation system",
        ],
    }
    for cat, concepts in categories.items():
        if concept in concepts:
            return cat
    return "general"


def run_extraction(
    base_path: Optional[str] = None,
    force_reindex: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Run the full extraction pipeline."""
    chapters = discover_chapters(base_path)
    results = {
        "chapters_found": len(chapters),
        "chapters_processed": 0,
        "concepts_extracted": 0,
        "errors": [],
    }

    for chapter_info in chapters:
        try:
            combined_content = ""
            for pdf_path in chapter_info["pdfs"]:
                content = extract_pdf_content(pdf_path)
                combined_content += content + "\n\n"

            chapter_id = insert_chapter(
                chapter_number=chapter_info["chapter_number"],
                title=chapter_info["title"],
                source_path=chapter_info["path"],
                content_md=combined_content,
                db_path=db_path,
            )

            log_extraction(chapter_id, "completed", f"Extracted {len(combined_content)} chars", db_path)

            concepts = extract_concepts(combined_content)
            for concept in concepts:
                insert_concept(
                    chapter_id=chapter_id,
                    concept_name=concept["concept_name"],
                    description=concept["description"],
                    category=concept["category"],
                    confidence=concept["confidence"],
                    db_path=db_path,
                )

            results["chapters_processed"] += 1
            results["concepts_extracted"] += len(concepts)

        except Exception as e:
            results["errors"].append({
                "chapter": chapter_info["title"],
                "error": str(e),
            })
            if chapter_info.get("chapter_number"):
                log_extraction(
                    chapter_info["chapter_number"],
                    "error",
                    str(e),
                    db_path,
                )

    return results
