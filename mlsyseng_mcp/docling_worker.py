"""Background PDF extraction worker using docling."""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_ml_principles_path() -> Path:
    return Path(os.environ.get(
        "ML_PRINCIPLES_PATH",
        os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters")
    ))


def _extract_chapter_number(folder_name: str) -> Optional[int]:
    """Extract chapter number from folder name like '08_ML_Systems'."""
    match = re.match(r"^(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return None


def _extract_chapter_title(folder_name: str) -> str:
    """Extract clean title from folder name."""
    cleaned = re.sub(r"^\d+[_\s]*", "", folder_name)
    return cleaned.replace("_", " ").strip()


class DoclingWorker:
    """Extracts text and structured data from ML Principles PDFs using docling."""

    def __init__(self, ml_principles_path: Optional[Path] = None):
        self.ml_principles_path = ml_principles_path or _default_ml_principles_path()
        self._docling_available = self._check_docling()

    def _check_docling(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            return True
        except ImportError:
            logger.warning("docling not available, falling back to basic PDF extraction")
            return False

    def scan_chapters(self) -> List[Dict[str, Any]]:
        """Scan the ML Principles directory for chapter folders."""
        if not self.ml_principles_path.exists():
            logger.warning(f"ML Principles path not found: {self.ml_principles_path}")
            return []

        chapters = []
        for item in sorted(self.ml_principles_path.iterdir()):
            if item.is_dir():
                chapter_num = _extract_chapter_number(item.name)
                if chapter_num is not None:
                    pdfs = list(item.glob("*.pdf"))
                    chapters.append({
                        "chapter_number": chapter_num,
                        "title": _extract_chapter_title(item.name),
                        "folder_path": str(item),
                        "pdf_files": [str(p) for p in pdfs],
                        "pdf_count": len(pdfs),
                    })

        return chapters

    def extract_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """
        Extract text from a PDF file.

        Returns:
            Tuple of (markdown_content, list_of_concepts)
        """
        if self._docling_available:
            return self._extract_with_docling(pdf_path)
        return self._extract_fallback(pdf_path)

    def _extract_with_docling(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract using docling for high-quality structured output."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        concepts = self._extract_concepts_from_text(markdown)
        return markdown, concepts

    def _extract_fallback(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Fallback extraction using PyPDF2 or pdfplumber."""
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            except ImportError:
                logger.error("No PDF extraction library available (install docling, pdfplumber, or PyPDF2)")
                return "", []

        concepts = self._extract_concepts_from_text(text)
        return text, concepts

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract ML/AI concepts from text using pattern matching."""
        concept_patterns = [
            r'\b(neural network|deep learning|gradient descent|backpropagation)\b',
            r'\b(convolutional|recurrent|transformer|attention mechanism)\b',
            r'\b(regularization|dropout|batch normalization|layer normalization)\b',
            r'\b(loss function|cross[- ]entropy|mean squared error)\b',
            r'\b(optimization|adam|sgd|learning rate|momentum)\b',
            r'\b(overfitting|underfitting|bias[- ]variance|generalization)\b',
            r'\b(feature engineering|feature selection|dimensionality reduction)\b',
            r'\b(ensemble|bagging|boosting|random forest|xgboost)\b',
            r'\b(hyperparameter|grid search|bayesian optimization)\b',
            r'\b(cross[- ]validation|train[- ]test split|k[- ]fold)\b',
            r'\b(precision|recall|f1[- ]score|auc|roc)\b',
            r'\b(clustering|k[- ]means|hierarchical|dbscan)\b',
            r'\b(reinforcement learning|policy gradient|q[- ]learning)\b',
            r'\b(natural language processing|nlp|tokenization|embedding)\b',
            r'\b(computer vision|image classification|object detection)\b',
            r'\b(transfer learning|fine[- ]tuning|pre[- ]training)\b',
            r'\b(data augmentation|synthetic data|data pipeline)\b',
            r'\b(model selection|model evaluation|model deployment)\b',
            r'\b(distributed training|model parallelism|data parallelism)\b',
            r'\b(mixture of experts|gating network|sparse activation)\b',
        ]

        concepts = set()
        text_lower = text.lower()
        for pattern in concept_patterns:
            matches = re.findall(pattern, text_lower)
            concepts.update(matches)

        return sorted(concepts)

    def extract_chapter(self, chapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all PDFs from a chapter and combine results."""
        all_text = []
        all_concepts = set()

        for pdf_path in chapter_info.get("pdf_files", []):
            try:
                text, concepts = self.extract_pdf(pdf_path)
                if text:
                    all_text.append(text)
                all_concepts.update(concepts)
            except Exception as e:
                logger.error(f"Error extracting {pdf_path}: {e}")

        combined_text = "\n\n---\n\n".join(all_text)
        return {
            "chapter_number": chapter_info["chapter_number"],
            "title": chapter_info["title"],
            "markdown_content": combined_text,
            "concepts": sorted(all_concepts),
            "word_count": len(combined_text.split()),
            "source_path": chapter_info["folder_path"],
        }
