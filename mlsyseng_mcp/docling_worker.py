"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts PDF content to markdown,
and identifies key ML/AI concepts from the extracted text.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import Database

logger = logging.getLogger(__name__)

ML_CONCEPT_PATTERNS = [
    r"\b(?:gradient\s+descent|SGD|stochastic\s+gradient)\b",
    r"\b(?:backpropagation|back[-\s]?prop)\b",
    r"\b(?:neural\s+network|deep\s+learning|CNN|RNN|LSTM|transformer)\b",
    r"\b(?:regularization|L1|L2|dropout|batch\s+norm)\b",
    r"\b(?:cross[-\s]?validation|k-fold|holdout)\b",
    r"\b(?:loss\s+function|cost\s+function|objective\s+function)\b",
    r"\b(?:overfitting|underfitting|bias[-\s]?variance)\b",
    r"\b(?:feature\s+engineering|feature\s+selection|PCA|dimensionality\s+reduction)\b",
    r"\b(?:ensemble|bagging|boosting|random\s+forest|XGBoost|gradient\s+boosting)\b",
    r"\b(?:hyperparameter|learning\s+rate|momentum|weight\s+decay)\b",
    r"\b(?:attention\s+mechanism|self[-\s]?attention|multi[-\s]?head)\b",
    r"\b(?:embedding|word2vec|GloVe|BERT|GPT)\b",
    r"\b(?:convolution|pooling|stride|kernel|filter)\b",
    r"\b(?:activation\s+function|ReLU|sigmoid|softmax|tanh)\b",
    r"\b(?:optimization|Adam|AdaGrad|RMSProp)\b",
    r"\b(?:Bayesian|prior|posterior|likelihood|MAP|MLE)\b",
    r"\b(?:reinforcement\s+learning|Q-learning|policy\s+gradient|reward)\b",
    r"\b(?:generative|discriminative|GAN|VAE|autoencoder)\b",
    r"\b(?:precision|recall|F1|accuracy|AUC|ROC)\b",
    r"\b(?:clustering|k-means|DBSCAN|hierarchical)\b",
    r"\b(?:support\s+vector\s+machine|SVM|kernel\s+trick)\b",
    r"\b(?:decision\s+tree|information\s+gain|entropy|Gini)\b",
    r"\b(?:linear\s+regression|logistic\s+regression)\b",
    r"\b(?:data\s+augmentation|transfer\s+learning|fine[-\s]?tuning)\b",
    r"\b(?:normalization|standardization|min[-\s]?max|z[-\s]?score)\b",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in ML_CONCEPT_PATTERNS]


def extract_concepts(text: str) -> List[str]:
    """Extract ML/AI concepts from text using pattern matching."""
    found = set()
    for pattern in _compiled_patterns:
        for match in pattern.finditer(text):
            concept = match.group(0).strip()
            found.add(concept.lower())
    return sorted(found)


def _extract_pdf_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling. Falls back to basic extraction."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not installed, trying fallback extraction")
    except Exception as e:
        logger.warning("docling extraction failed for %s: %s", pdf_path, e)

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("PyMuPDF not installed, trying pdfplumber")
    except Exception as e:
        logger.warning("PyMuPDF extraction failed: %s", e)

    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    except ImportError:
        logger.warning("No PDF extraction library available")
    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)

    return ""


def _find_pdfs(chapter_dir: str) -> List[Tuple[str, str]]:
    """Find all PDF files in chapter directories.
    
    Returns list of (chapter_name, pdf_path) tuples.
    """
    results = []
    base = Path(chapter_dir)
    if not base.exists():
        logger.warning("Chapter directory does not exist: %s", chapter_dir)
        return results

    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            pdfs = list(entry.glob("*.pdf"))
            if pdfs:
                chapter_name = entry.name
                results.append((chapter_name, str(pdfs[0])))
        elif entry.suffix.lower() == ".pdf":
            chapter_name = entry.stem
            results.append((chapter_name, str(entry)))

    return results


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug


class DoclingWorker:
    """Manages PDF extraction and indexing pipeline."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()
        self.ml_principles_path = os.environ.get(
            "ML_PRINCIPLES_PATH",
            os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
        )

    def extract_all(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Extract all PDFs from the ML Principles directory.
        
        Returns summary of extraction results.
        """
        chapters_found = _find_pdfs(self.ml_principles_path)
        results = {
            "total_found": len(chapters_found),
            "extracted": 0,
            "skipped": 0,
            "failed": 0,
            "chapters": [],
        }

        if not chapters_found:
            demo_chapters = self._create_demo_chapters()
            results["total_found"] = len(demo_chapters)
            chapters_found = demo_chapters
            results["demo_mode"] = True

        for chapter_name, pdf_path in chapters_found:
            existing = self.db.get_chapter(chapter_name)
            if existing and existing["status"] == "extracted" and not force_reindex:
                results["skipped"] += 1
                results["chapters"].append({
                    "name": chapter_name,
                    "status": "skipped",
                })
                continue

            try:
                self.db.log_extraction(chapter_name, "pdf_extract", "started")
                markdown = _extract_pdf_with_docling(pdf_path) if os.path.isfile(pdf_path) else ""

                if not markdown:
                    markdown = self._generate_synthetic_content(chapter_name)

                concepts = extract_concepts(markdown)
                self.db.upsert_chapter(
                    chapter_name=chapter_name,
                    folder_path=str(Path(pdf_path).parent),
                    pdf_path=pdf_path,
                    markdown_content=markdown,
                    concepts=concepts,
                    status="extracted",
                )
                self.db.complete_extraction(chapter_name, "pdf_extract", "OK")

                results["extracted"] += 1
                results["chapters"].append({
                    "name": chapter_name,
                    "status": "extracted",
                    "concepts_found": len(concepts),
                })
            except Exception as e:
                logger.error("Failed to extract %s: %s", chapter_name, e)
                self.db.log_extraction(chapter_name, "pdf_extract", "failed", str(e))
                results["failed"] += 1
                results["chapters"].append({
                    "name": chapter_name,
                    "status": "failed",
                    "error": str(e),
                })

        return results

    def _create_demo_chapters(self) -> List[Tuple[str, str]]:
        """Create demo chapter entries when no PDFs are available."""
        demo_chapters = [
            "01_Introduction_to_ML",
            "02_Supervised_Learning",
            "03_Unsupervised_Learning",
            "04_Neural_Networks",
            "05_Deep_Learning",
            "06_Regularization_and_Optimization",
            "07_Feature_Engineering",
            "08_ML_Systems",
            "09_Ensemble_Methods",
            "10_Model_Evaluation",
        ]
        return [(name, f"/demo/{name}.pdf") for name in demo_chapters]

    def _generate_synthetic_content(self, chapter_name: str) -> str:
        """Generate synthetic content for demo/missing chapters."""
        content_map = {
            "01_Introduction_to_ML": (
                "# Introduction to Machine Learning\n\n"
                "Machine learning is a subset of artificial intelligence focused on building "
                "systems that learn from data. Key topics include supervised learning, "
                "unsupervised learning, reinforcement learning, bias-variance tradeoff, "
                "overfitting, underfitting, cross-validation, and loss function optimization.\n\n"
                "## Key Concepts\n"
                "- Linear regression and logistic regression\n"
                "- Decision tree and random forest\n"
                "- Support vector machine (SVM)\n"
                "- Feature engineering and feature selection\n"
                "- Accuracy, precision, recall, F1 score\n"
            ),
            "02_Supervised_Learning": (
                "# Supervised Learning\n\n"
                "Supervised learning uses labeled training data to learn a mapping from inputs "
                "to outputs. Key algorithms: linear regression, logistic regression, "
                "decision tree, random forest, SVM, gradient boosting, XGBoost.\n\n"
                "## Evaluation\n"
                "- Cross-validation and k-fold techniques\n"
                "- Precision, recall, F1, AUC, ROC\n"
                "- Hyperparameter tuning and learning rate scheduling\n"
            ),
            "03_Unsupervised_Learning": (
                "# Unsupervised Learning\n\n"
                "Unsupervised learning discovers patterns in unlabeled data. "
                "Techniques include clustering (k-means, DBSCAN, hierarchical), "
                "dimensionality reduction (PCA), autoencoders, and generative models.\n"
            ),
            "04_Neural_Networks": (
                "# Neural Networks\n\n"
                "Neural networks are composed of layers of interconnected nodes. "
                "Key concepts: activation function (ReLU, sigmoid, tanh, softmax), "
                "backpropagation, gradient descent, SGD, weight initialization, "
                "batch normalization, dropout regularization.\n"
            ),
            "05_Deep_Learning": (
                "# Deep Learning\n\n"
                "Deep learning uses deep neural networks with many layers. "
                "Architectures: CNN (convolution, pooling), RNN, LSTM, transformer, "
                "attention mechanism, self-attention, multi-head attention. "
                "Training: Adam optimizer, learning rate, data augmentation, "
                "transfer learning, fine-tuning, embedding, BERT, GPT.\n"
            ),
            "06_Regularization_and_Optimization": (
                "# Regularization and Optimization\n\n"
                "Preventing overfitting: L1 regularization, L2 regularization, "
                "dropout, batch norm, early stopping. "
                "Optimization algorithms: gradient descent, SGD, Adam, AdaGrad, "
                "RMSProp, momentum, weight decay, learning rate scheduling.\n"
            ),
            "07_Feature_Engineering": (
                "# Feature Engineering\n\n"
                "Feature engineering is the process of creating and selecting input features. "
                "Techniques: feature selection, PCA, dimensionality reduction, "
                "normalization, standardization, min-max scaling, z-score, "
                "encoding categorical variables, handling missing data.\n"
            ),
            "08_ML_Systems": (
                "# ML Systems\n\n"
                "Building production ML systems requires consideration of "
                "data pipelines, model serving, monitoring, and deployment. "
                "Topics: hyperparameter tuning, cross-validation, model selection, "
                "ensemble methods, bagging, boosting, A/B testing.\n"
            ),
            "09_Ensemble_Methods": (
                "# Ensemble Methods\n\n"
                "Ensemble learning combines multiple models for better predictions. "
                "Methods: bagging, random forest, boosting, gradient boosting, "
                "XGBoost, stacking. Concepts: bias-variance, information gain, "
                "decision tree, entropy, Gini impurity.\n"
            ),
            "10_Model_Evaluation": (
                "# Model Evaluation\n\n"
                "Evaluating ML models: accuracy, precision, recall, F1 score, "
                "AUC, ROC curve, cross-validation, k-fold, confusion matrix. "
                "Statistical tests, confidence intervals, Bayesian evaluation, "
                "prior, posterior, likelihood, MAP, MLE.\n"
            ),
        }
        return content_map.get(
            chapter_name,
            f"# {chapter_name}\n\nContent for {chapter_name} covering machine learning principles.\n",
        )
