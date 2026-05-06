"""Docling-based PDF extraction worker for MLSysEng MoE.

Scans chapter folders, extracts PDF content to markdown, and
identifies key ML/AI concepts from extracted text.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from .database import ChapterRecord, Database

logger = logging.getLogger(__name__)

ML_CONCEPT_PATTERNS = [
    r"\b(neural network|deep learning|convolutional|recurrent|transformer)\b",
    r"\b(gradient descent|backpropagation|optimization|loss function)\b",
    r"\b(regularization|dropout|batch normalization|weight decay)\b",
    r"\b(cross[- ]?validation|overfitting|underfitting|bias[- ]?variance)\b",
    r"\b(supervised|unsupervised|reinforcement|semi[- ]?supervised)\b",
    r"\b(classification|regression|clustering|dimensionality reduction)\b",
    r"\b(decision tree|random forest|gradient boosting|xgboost|lightgbm)\b",
    r"\b(support vector|kernel|SVM|logistic regression)\b",
    r"\b(bayesian|prior|posterior|likelihood|maximum likelihood)\b",
    r"\b(feature engineering|feature selection|feature extraction)\b",
    r"\b(ensemble|bagging|boosting|stacking)\b",
    r"\b(attention mechanism|self[- ]?attention|multi[- ]?head)\b",
    r"\b(embedding|word2vec|GloVe|BERT|GPT)\b",
    r"\b(hyperparameter|learning rate|batch size|epoch)\b",
    r"\b(precision|recall|f1[- ]?score|accuracy|AUC|ROC)\b",
    r"\b(data augmentation|preprocessing|normalization|standardization)\b",
    r"\b(activation function|ReLU|sigmoid|softmax|tanh)\b",
    r"\b(autoencoder|variational|GAN|generative)\b",
    r"\b(transfer learning|fine[- ]?tuning|pre[- ]?training)\b",
    r"\b(convolution|pooling|stride|padding|filter)\b",
    r"\b(LSTM|GRU|sequence model|time series)\b",
    r"\b(PCA|t-SNE|UMAP|manifold learning)\b",
    r"\b(k-means|DBSCAN|hierarchical clustering)\b",
    r"\b(monte carlo|sampling|markov chain)\b",
    r"\b(information gain|entropy|mutual information)\b",
    r"\b(model selection|cross[- ]?validation|grid search)\b",
    r"\b(pipeline|workflow|MLOps|model deployment)\b",
    r"\b(distributed training|data parallel|model parallel)\b",
    r"\b(mixture of experts|gating network|sparse model)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ML_CONCEPT_PATTERNS]


def extract_concepts(text: str) -> list[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    concepts = set()
    for pattern in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            concept = match.group(0).strip().lower()
            concept = re.sub(r"\s+", " ", concept)
            concepts.add(concept)
    return sorted(concepts)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_")


def _try_docling_extract(pdf_path: str) -> Optional[str]:
    """Try extracting PDF content via docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, falling back to basic extraction")
        return None
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return None


def _try_pymupdf_extract(pdf_path: str) -> Optional[str]:
    """Fallback: extract text with PyMuPDF (fitz)."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("PyMuPDF not installed, falling back to pdfplumber")
        return None
    except Exception as e:
        logger.error("PyMuPDF extraction failed for %s: %s", pdf_path, e)
        return None


def _try_pdfplumber_extract(pdf_path: str) -> Optional[str]:
    """Fallback: extract text with pdfplumber."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("pdfplumber not installed")
        return None
    except Exception as e:
        logger.error("pdfplumber extraction failed for %s: %s", pdf_path, e)
        return None


def extract_pdf(pdf_path: str) -> str:
    """Extract text from a PDF, trying multiple backends in order."""
    for extractor in [_try_docling_extract, _try_pymupdf_extract, _try_pdfplumber_extract]:
        result = extractor(pdf_path)
        if result and result.strip():
            return result

    logger.error("All PDF extraction methods failed for %s", pdf_path)
    return ""


class DoclingWorker:
    """Scans chapter directories and extracts PDF content."""

    def __init__(self, db: Database, ml_principles_path: Optional[str] = None):
        self.db = db
        self.ml_principles_path = ml_principles_path or os.environ.get(
            "ML_PRINCIPLES_PATH",
            os.path.expanduser(
                "~/Desktop/Machine Learning Principles - Chapters"
            ),
        )
        self._extraction_progress: dict[str, str] = {}

    def scan_chapters(self) -> list[dict]:
        """Scan the ML Principles directory for chapter folders."""
        base = Path(self.ml_principles_path)
        if not base.exists():
            logger.warning("ML Principles path does not exist: %s", base)
            return []

        chapters = []
        for item in sorted(base.iterdir()):
            if not item.is_dir():
                continue
            pdfs = list(item.glob("*.pdf"))
            if not pdfs:
                continue
            chapters.append({
                "name": item.name,
                "path": str(item),
                "pdfs": [str(p) for p in pdfs],
                "chapter_id": _slugify(item.name),
            })
        return chapters

    def extract_chapter(
        self, chapter_info: dict, force: bool = False
    ) -> Optional[ChapterRecord]:
        """Extract a single chapter's PDF content."""
        chapter_id = chapter_info["chapter_id"]
        self._extraction_progress[chapter_id] = "extracting"

        if not force:
            existing = self.db.get_chapter(chapter_id)
            if existing and existing.content_md:
                self._extraction_progress[chapter_id] = "skipped (exists)"
                return existing

        all_content = []
        for pdf_path in chapter_info["pdfs"]:
            logger.info("Extracting: %s", pdf_path)
            content = extract_pdf(pdf_path)
            if content:
                all_content.append(content)

        if not all_content:
            self._extraction_progress[chapter_id] = "failed"
            return None

        combined = "\n\n---\n\n".join(all_content)
        concepts = extract_concepts(combined)
        word_count = len(combined.split())

        record = ChapterRecord(
            chapter_id=chapter_id,
            title=chapter_info["name"],
            slug=_slugify(chapter_info["name"]),
            source_path=chapter_info["path"],
            content_md=combined,
            concepts=concepts,
            extracted_at=time.time(),
            word_count=word_count,
        )
        self.db.upsert_chapter(record)
        self._extraction_progress[chapter_id] = "done"
        return record

    def extract_all(self, force: bool = False) -> list[ChapterRecord]:
        """Extract all chapters found in the ML Principles directory."""
        chapters_info = self.scan_chapters()
        results = []
        for info in chapters_info:
            record = self.extract_chapter(info, force=force)
            if record:
                results.append(record)
        return results

    def get_progress(self) -> dict[str, str]:
        return dict(self._extraction_progress)
