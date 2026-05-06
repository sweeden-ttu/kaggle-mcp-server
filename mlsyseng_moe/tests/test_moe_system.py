"""Integration tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import numpy as np
import pytest

from mlsyseng_moe.mlsyseng_mcp.database import Database
from mlsyseng_moe.mlsyseng_mcp.docling_worker import (
    DoclingWorker,
    chunk_text,
    extract_concepts,
    slugify,
)
from mlsyseng_moe.mlsyseng_mcp.embeddings import EmbeddingEngine
from mlsyseng_moe.mlsyseng_mcp.expert_registry import (
    ExpertRegistry,
    infer_capabilities,
    infer_formula,
    infer_skills_from_concepts,
)
from mlsyseng_moe.mlsyseng_mcp.loop_controller import LoopController, l2_norm


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield Database(db_path=db_path)


@pytest.fixture
def tmp_chroma():
    """Create a temporary ChromaDB for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_path = os.path.join(tmpdir, "chroma")
        yield EmbeddingEngine(chroma_path=chroma_path)


class TestDatabase:
    def test_chapter_upsert_and_get(self, tmp_db):
        ch_id = tmp_db.upsert_chapter(
            "Test Chapter", "test_chapter", extraction_status="completed",
            concepts=["neural network"]
        )
        assert ch_id > 0

        chapter = tmp_db.get_chapter("test_chapter")
        assert chapter is not None
        assert chapter["chapter_name"] == "Test Chapter"
        assert chapter["concepts"] == ["neural network"]

    def test_expert_upsert_and_get(self, tmp_db):
        expert_data = {
            "expert_name": "Test Expert",
            "slug": "test_expert",
            "capabilities": ["cap1", "cap2"],
            "skills": ["/path/skill1"],
            "strategy": "Strategy A",
            "formula": {"objective": "min_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        exp_id = tmp_db.upsert_expert(expert_data)
        assert exp_id > 0

        expert = tmp_db.get_expert("test_expert")
        assert expert["expert_name"] == "Test Expert"
        assert expert["capabilities"] == ["cap1", "cap2"]
        assert expert["formula"]["objective"] == "min_loss"

    def test_knowledge_chunks(self, tmp_db):
        ch_id = tmp_db.upsert_chapter("Ch1", "ch1", extraction_status="completed")
        chunks = [
            {"content": "chunk 1 content", "metadata": {"page": 1}},
            {"content": "chunk 2 content", "metadata": {"page": 2}},
        ]
        tmp_db.store_knowledge_chunks(ch_id, chunks)

        retrieved = tmp_db.get_knowledge_chunks(ch_id)
        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "chunk 1 content"

    def test_competition_state(self, tmp_db):
        tmp_db.save_competition_state(
            "titanic", 0, [0.5, 0.3, 0.8], {"loss": 0.5}, ["expert1"]
        )
        history = tmp_db.get_competition_history("titanic")
        assert len(history) == 1
        assert history[0]["state_vector"] == [0.5, 0.3, 0.8]

    def test_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert "total_chapters" in stats
        assert "total_experts" in stats


class TestDoclingWorker:
    def test_slugify(self):
        assert slugify("08_ML Systems") == "08_ml_systems"
        assert slugify("Hello World!") == "hello_world"

    def test_chunk_text(self):
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3
        assert len(chunks[0]) == 1000

    def test_chunk_text_empty(self):
        assert chunk_text("") == []

    def test_extract_concepts(self):
        text = "neural network with gradient descent for classification"
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "gradient descent" in concepts
        assert "classification" in concepts

    def test_discover_chapters_missing_path(self):
        worker = DoclingWorker("/nonexistent/path")
        chapters = worker.discover_chapters()
        assert chapters == []


class TestEmbeddingEngine:
    def test_generate_embeddings(self, tmp_chroma):
        embeddings = tmp_chroma.generate_embeddings(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_index_and_search(self, tmp_chroma):
        chunks = [
            {"content": "deep learning optimization", "metadata": {}},
            {"content": "random forest classification", "metadata": {}},
        ]
        tmp_chroma.index_chunks(chunks, "test_ch")

        results = tmp_chroma.search("neural network training", n_results=2)
        assert len(results) == 2
        assert results[0]["similarity"] > 0

    def test_stats(self, tmp_chroma):
        stats = tmp_chroma.get_stats()
        assert stats["model"] == "all-MiniLM-L6-v2"


class TestExpertRegistry:
    def test_infer_skills(self):
        concepts = ["deep learning", "feature engineering"]
        skills = infer_skills_from_concepts(concepts, "/tmp/skills")
        assert "/tmp/skills/kaggle-deep-learning" in skills
        assert "/tmp/skills/kaggle-feature-engineer" in skills

    def test_infer_capabilities(self):
        concepts = ["neural network", "ensemble"]
        caps = infer_capabilities(concepts)
        assert "Build baseline models quickly" in caps
        assert "Design and train neural network architectures" in caps

    def test_infer_formula_classification(self):
        formula = infer_formula(["classification"], "Test")
        assert "accuracy" in formula["metrics"]

    def test_register_expert(self, tmp_db):
        registry = ExpertRegistry(tmp_db, skills_path="/tmp/skills")
        expert = registry.register_expert_from_chapter(
            "Test", "test", ["deep learning"]
        )
        assert expert["expert_name"] == "Test"
        assert len(expert["skills"]) > 0

        experts = registry.list_experts()
        assert len(experts) == 1


class TestLoopController:
    def test_l2_norm(self):
        assert abs(l2_norm([1, 0, 0], [0, 0, 0]) - 1.0) < 1e-6
        assert abs(l2_norm([1, 1, 1], [1, 1, 1])) < 1e-6

    def test_check_convergence(self, tmp_db):
        controller = LoopController(tmp_db, epsilon=0.1)
        converged, diff = controller.check_convergence([0.5, 0.5], [0.5, 0.5])
        assert converged is True
        assert diff < 0.1

        converged, diff = controller.check_convergence([0.0, 0.0], [1.0, 1.0])
        assert converged is False

    def test_run_loop_converges(self, tmp_db):
        controller = LoopController(tmp_db, epsilon=0.1, max_iterations=50, patience=2)
        result = controller.run_loop(
            "test_comp",
            experts=[{"slug": "exp1"}],
        )
        assert result["status"] in ("converged", "max_iterations_reached")
        assert result["iterations"] > 0

    def test_get_status_not_started(self, tmp_db):
        controller = LoopController(tmp_db)
        status = controller.get_competition_status("nonexistent")
        assert status["status"] == "not_started"
