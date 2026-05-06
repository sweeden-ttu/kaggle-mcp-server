"""Background PDF extraction worker using docling."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def slugify(name: str) -> str:
    """Convert chapter name to a URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return slug


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_concepts(text: str) -> List[str]:
    """Extract key ML/AI concepts from text using pattern matching."""
    concept_patterns = [
        r"\b(neural network|deep learning|machine learning|reinforcement learning)\b",
        r"\b(gradient descent|backpropagation|forward pass)\b",
        r"\b(convolutional|recurrent|transformer|attention)\b",
        r"\b(regularization|dropout|batch normalization)\b",
        r"\b(loss function|objective function|cost function)\b",
        r"\b(overfitting|underfitting|bias-variance)\b",
        r"\b(cross[- ]validation|train[- ]test split)\b",
        r"\b(hyperparameter|learning rate|momentum)\b",
        r"\b(ensemble|bagging|boosting|random forest)\b",
        r"\b(feature engineering|feature selection|dimensionality reduction)\b",
        r"\b(classification|regression|clustering)\b",
        r"\b(precision|recall|f1[- ]score|accuracy|AUC|ROC)\b",
        r"\b(SVM|support vector|kernel)\b",
        r"\b(decision tree|gradient boosting|XGBoost|LightGBM)\b",
        r"\b(embedding|word2vec|tokenization)\b",
        r"\b(optimizer|Adam|SGD|RMSProp)\b",
        r"\b(activation function|ReLU|sigmoid|softmax)\b",
        r"\b(generative|discriminative|GAN|VAE)\b",
        r"\b(Bayesian|prior|posterior|likelihood)\b",
        r"\b(time series|ARIMA|LSTM)\b",
    ]

    concepts = set()
    text_lower = text.lower()
    for pattern in concept_patterns:
        matches = re.findall(pattern, text_lower)
        concepts.update(matches)

    return sorted(concepts)


class DoclingWorker:
    """Worker that extracts text and structure from ML Principles PDFs."""

    def __init__(self, ml_principles_path: Optional[str] = None):
        self.ml_principles_path = Path(
            ml_principles_path or DEFAULT_ML_PRINCIPLES_PATH
        )
        self._docling_available = None

    @property
    def docling_available(self) -> bool:
        """Check if docling is available."""
        if self._docling_available is None:
            try:
                from docling.document_converter import DocumentConverter  # noqa: F401
                self._docling_available = True
            except ImportError:
                self._docling_available = False
        return self._docling_available

    def discover_chapters(self) -> List[Dict[str, Any]]:
        """Discover chapter folders and PDF files."""
        chapters = []
        if not self.ml_principles_path.exists():
            logger.warning(f"ML Principles path not found: {self.ml_principles_path}")
            return chapters

        for item in sorted(self.ml_principles_path.iterdir()):
            if item.is_dir():
                pdfs = list(item.glob("*.pdf"))
                if pdfs:
                    chapters.append({
                        "name": item.name,
                        "slug": slugify(item.name),
                        "path": str(item),
                        "pdf_files": [str(p) for p in pdfs],
                    })
            elif item.suffix.lower() == ".pdf":
                chapters.append({
                    "name": item.stem,
                    "slug": slugify(item.stem),
                    "path": str(item),
                    "pdf_files": [str(item)],
                })

        return chapters

    def extract_pdf_docling(self, pdf_path: str) -> str:
        """Extract text from a PDF using docling."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()

    def extract_pdf_fallback(self, pdf_path: str) -> str:
        """Fallback PDF extraction using PyPDF2 or pdfplumber."""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            pass

        logger.error(
            "No PDF extraction library available. "
            "Install docling, pdfplumber, or PyPDF2."
        )
        return ""

    def extract_chapter(self, chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from a chapter's PDF files."""
        all_text = []
        for pdf_path in chapter_info["pdf_files"]:
            if self.docling_available:
                text = self.extract_pdf_docling(pdf_path)
            else:
                text = self.extract_pdf_fallback(pdf_path)
            if text:
                all_text.append(text)

        full_text = "\n\n".join(all_text)
        concepts = extract_concepts(full_text)
        chunks = chunk_text(full_text)

        return {
            "name": chapter_info["name"],
            "slug": chapter_info["slug"],
            "text": full_text,
            "concepts": concepts,
            "chunks": [
                {"content": c, "metadata": {"chapter": chapter_info["name"], "index": i}}
                for i, c in enumerate(chunks)
            ],
        }

    def extract_all(
        self, force_reindex: bool = False, db=None
    ) -> List[Dict[str, Any]]:
        """Extract all chapters, optionally skipping already-extracted ones."""
        chapters = self.discover_chapters()
        results = []

        for chapter_info in chapters:
            if not force_reindex and db:
                existing = db.get_chapter(chapter_info["slug"])
                if existing and existing.get("extraction_status") == "completed":
                    logger.info(f"Skipping already extracted: {chapter_info['name']}")
                    continue

            if db:
                db.upsert_chapter(
                    chapter_name=chapter_info["name"],
                    slug=chapter_info["slug"],
                    pdf_path=chapter_info["pdf_files"][0] if chapter_info["pdf_files"] else None,
                    extraction_status="extracting",
                )

            try:
                result = self.extract_chapter(chapter_info)
                results.append(result)

                if db:
                    chapter_id = db.upsert_chapter(
                        chapter_name=result["name"],
                        slug=result["slug"],
                        extracted_text=result["text"][:10000],
                        concepts=result["concepts"],
                        extraction_status="completed",
                    )
                    db.store_knowledge_chunks(chapter_id, result["chunks"])

            except Exception as e:
                logger.error(f"Failed to extract {chapter_info['name']}: {e}")
                if db:
                    db.upsert_chapter(
                        chapter_name=chapter_info["name"],
                        slug=chapter_info["slug"],
                        extraction_status="failed",
                    )

        return results

    def get_extraction_status(self, db) -> Dict[str, Any]:
        """Get the current extraction status."""
        chapters = db.list_chapters()
        status_counts = {"pending": 0, "extracting": 0, "completed": 0, "failed": 0}
        for ch in chapters:
            s = ch.get("extraction_status", "pending")
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_discovered": len(self.discover_chapters()),
            "total_in_db": len(chapters),
            "status_counts": status_counts,
            "chapters": chapters,
        }
