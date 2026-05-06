"""MLSysEng MoE - Machine Learning Systems Expert Mixture of Experts.

A Mixture of Experts system that extracts knowledge from ML Principles PDFs,
registers chapter experts with skills/strategy/formulas, and builds Kaggle
competition entries using RAG-informed skill selection with state convergence loops.
"""

__version__ = "0.1.0"

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.expert_registry import ExpertRegistry
from mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_mcp.loop_controller import LoopController
from mlsyseng_mcp.docling_worker import DoclingWorker

__all__ = [
    "Database",
    "ExpertRegistry",
    "EmbeddingEngine",
    "LoopController",
    "DoclingWorker",
]
