"""Background PDF extraction using docling.

Scans ML Principles chapter folders, extracts text from PDFs,
and identifies key ML/AI concepts.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

try:
    from . import database as db
except ImportError:
    import database as db

logger = logging.getLogger(__name__)

ML_PRINCIPLES_PATH = os.environ.get(
    "ML_PRINCIPLES_PATH",
    os.path.expanduser("~/Desktop/Machine Learning Principles - Chapters"),
)

CONCEPT_PATTERNS = [
    r"(?:gradient\s+descent|backpropagation|learning\s+rate)",
    r"(?:neural\s+network|deep\s+learning|convolutional)",
    r"(?:regularization|dropout|batch\s+normalization)",
    r"(?:loss\s+function|cross[- ]entropy|mean\s+squared\s+error)",
    r"(?:optimization|adam|sgd|momentum)",
    r"(?:overfitting|underfitting|bias[- ]variance)",
    r"(?:feature\s+engineering|dimensionality\s+reduction|PCA)",
    r"(?:ensemble|bagging|boosting|random\s+forest)",
    r"(?:support\s+vector|kernel\s+trick|SVM)",
    r"(?:decision\s+tree|information\s+gain|entropy)",
    r"(?:clustering|k-means|DBSCAN|hierarchical)",
    r"(?:recurrent|LSTM|GRU|attention|transformer)",
    r"(?:generative|GAN|VAE|diffusion)",
    r"(?:reinforcement\s+learning|Q-learning|policy\s+gradient)",
    r"(?:hyperparameter|grid\s+search|bayesian\s+optimization)",
    r"(?:cross[- ]validation|train[- ]test\s+split|stratified)",
    r"(?:precision|recall|F1[- ]score|ROC|AUC)",
    r"(?:embedding|word2vec|GloVe|tokenization)",
    r"(?:data\s+augmentation|transfer\s+learning|fine[- ]tuning)",
    r"(?:model\s+selection|model\s+evaluation|confusion\s+matrix)",
]


def _extract_with_docling(pdf_path: str) -> str:
    """Extract text from PDF using docling."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        return result.document.export_to_markdown()
    except ImportError:
        logger.warning("docling not available, falling back to basic extraction")
        return _extract_basic(pdf_path)
    except Exception as e:
        logger.error("docling extraction failed for %s: %s", pdf_path, e)
        return _extract_basic(pdf_path)


def _extract_basic(pdf_path: str) -> str:
    """Fallback text extraction without docling."""
    try:
        import subprocess

        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return f"[PDF extraction unavailable for {Path(pdf_path).name}]"


def extract_concepts(text: str) -> list[str]:
    """Identify ML/AI concepts from extracted text."""
    found = set()
    text_lower = text.lower()
    for pattern in CONCEPT_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            concept = match.group(0).strip()
            concept = re.sub(r"\s+", " ", concept)
            found.add(concept.title())
    general_terms = [
        "Machine Learning",
        "Artificial Intelligence",
        "Data Science",
        "Statistical Learning",
        "Supervised Learning",
        "Unsupervised Learning",
        "Semi-Supervised Learning",
        "Classification",
        "Regression",
        "Inference",
        "Prediction",
    ]
    for term in general_terms:
        if term.lower() in text_lower:
            found.add(term)

    return sorted(found)


def scan_chapter_folders(
    base_path: Optional[str] = None,
) -> list[dict]:
    """Scan for chapter folders containing PDFs."""
    base = Path(base_path or ML_PRINCIPLES_PATH)
    chapters = []

    if not base.exists():
        logger.warning("ML Principles path does not exist: %s", base)
        return chapters

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        pdfs = list(entry.glob("*.pdf"))
        if pdfs:
            chapters.append({
                "name": entry.name,
                "folder_path": str(entry),
                "pdf_path": str(pdfs[0]),
            })
        else:
            chapters.append({
                "name": entry.name,
                "folder_path": str(entry),
                "pdf_path": None,
            })

    return chapters


def extract_chapter(
    chapter_name: str,
    folder_path: str,
    pdf_path: Optional[str],
    force: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Extract content from a single chapter."""
    chapter_id = db.upsert_chapter(chapter_name, folder_path, pdf_path, db_path)

    if not force:
        existing = db.get_chapters_by_status("completed", db_path)
        for ch in existing:
            if ch["chapter_name"] == chapter_name:
                return {
                    "chapter_id": chapter_id,
                    "status": "skipped",
                    "message": "Already extracted",
                }

    if not pdf_path or not Path(pdf_path).exists():
        concepts = _infer_concepts_from_name(chapter_name)
        db.update_chapter_extraction(
            chapter_id,
            f"[No PDF available for {chapter_name}]",
            concepts,
            "completed",
            db_path,
        )
        return {
            "chapter_id": chapter_id,
            "status": "completed",
            "message": f"No PDF; inferred {len(concepts)} concepts from name",
            "concepts": concepts,
        }

    try:
        text = _extract_with_docling(pdf_path)
        concepts = extract_concepts(text)
        db.update_chapter_extraction(chapter_id, text, concepts, "completed", db_path)
        return {
            "chapter_id": chapter_id,
            "status": "completed",
            "message": f"Extracted {len(text)} chars, {len(concepts)} concepts",
            "concepts": concepts,
        }
    except Exception as e:
        db.mark_chapter_failed(chapter_id, str(e), db_path)
        return {
            "chapter_id": chapter_id,
            "status": "failed",
            "message": str(e),
        }


def _infer_concepts_from_name(chapter_name: str) -> list[str]:
    """Infer concepts from chapter folder name when no PDF is available."""
    name_lower = chapter_name.lower()
    concepts = []
    keyword_map = {
        "neural": ["Neural Network", "Deep Learning"],
        "deep learning": ["Deep Learning", "Neural Network", "Backpropagation"],
        "optimization": ["Optimization", "Gradient Descent", "Learning Rate"],
        "regularization": ["Regularization", "Overfitting", "Dropout"],
        "cnn": ["Convolutional", "Neural Network", "Feature Engineering"],
        "rnn": ["Recurrent", "LSTM", "Sequence Modeling"],
        "transformer": ["Transformer", "Attention", "Self-Attention"],
        "ensemble": ["Ensemble", "Bagging", "Boosting", "Random Forest"],
        "svm": ["Support Vector", "Kernel Trick", "SVM"],
        "tree": ["Decision Tree", "Information Gain", "Entropy"],
        "cluster": ["Clustering", "K-Means", "Unsupervised Learning"],
        "bayes": ["Bayesian Optimization", "Probability", "Statistical Learning"],
        "feature": ["Feature Engineering", "Dimensionality Reduction"],
        "evaluation": ["Model Evaluation", "Cross-Validation", "Precision"],
        "reinforcement": ["Reinforcement Learning", "Q-Learning", "Policy Gradient"],
        "generative": ["Generative", "GAN", "VAE"],
        "nlp": ["Embedding", "Tokenization", "Natural Language Processing"],
        "system": ["Machine Learning", "Model Selection", "Data Science"],
        "introduction": ["Machine Learning", "Artificial Intelligence"],
        "linear": ["Regression", "Classification", "Linear Model"],
        "probability": ["Probability", "Statistical Learning", "Bayesian"],
    }
    for keyword, mapped_concepts in keyword_map.items():
        if keyword in name_lower:
            concepts.extend(mapped_concepts)
    if not concepts:
        concepts = ["Machine Learning", "Data Science"]
    return sorted(set(concepts))


def extract_all(
    force_reindex: bool = False,
    base_path: Optional[str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Extract all chapters from ML Principles."""
    chapters = scan_chapter_folders(base_path)
    results = []
    for ch in chapters:
        result = extract_chapter(
            ch["name"], ch["folder_path"], ch["pdf_path"], force_reindex, db_path
        )
        results.append(result)
        logger.info("Chapter %s: %s", ch["name"], result["status"])
    return results
