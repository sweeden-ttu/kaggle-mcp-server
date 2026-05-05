"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from src.mlsyseng_mcp.database import MLSysEngDB
from src.mlsyseng_mcp.docling_worker import extract_concepts, scan_chapters
from src.mlsyseng_mcp.expert_registry import ExpertRegistry
from src.mlsyseng_mcp.loop_controller import LoopController, StateVector


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = MLSysEngDB(db_path)
    yield database
    os.unlink(db_path)


@pytest.fixture
def populated_db(db):
    ch_id = db.upsert_chapter(
        1, "ML Fundamentals", "/tmp/ch1",
        "Content about gradient descent and neural network optimization",
        10,
    )
    db.add_concepts(ch_id, [
        {"concept": "gradient descent", "description": "Optimization algo", "category": "optimization"},
        {"concept": "neural network", "description": "Architecture", "category": "architecture"},
    ])
    ch_id2 = db.upsert_chapter(
        2, "Deep Learning", "/tmp/ch2",
        "Content about deep learning, transformers, and attention mechanisms",
        15,
    )
    db.add_concepts(ch_id2, [
        {"concept": "transformer", "description": "Attention architecture", "category": "architecture"},
        {"concept": "dropout", "description": "Regularization", "category": "regularization"},
    ])
    return db


class TestDatabase:
    def test_create_and_get_chapter(self, db):
        ch_id = db.upsert_chapter(1, "Test Chapter", "/tmp/test", "Content", 5)
        assert ch_id > 0
        ch = db.get_chapter(1)
        assert ch is not None
        assert ch["title"] == "Test Chapter"

    def test_upsert_chapter_updates(self, db):
        db.upsert_chapter(1, "Original", "/tmp/orig", "Old content", 5)
        db.upsert_chapter(1, "Updated", "/tmp/new", "New content", 10)
        ch = db.get_chapter(1)
        assert ch["title"] == "Updated"
        assert ch["page_count"] == 10

    def test_concepts(self, db):
        ch_id = db.upsert_chapter(1, "Ch", "/tmp", "content", 1)
        db.add_concepts(ch_id, [
            {"concept": "gradient descent", "category": "optimization"},
            {"concept": "backpropagation", "category": "training"},
        ])
        concepts = db.get_concepts(ch_id)
        assert len(concepts) == 2
        names = {c["concept"] for c in concepts}
        assert "gradient descent" in names

    def test_expert_upsert(self, db):
        expert_id = db.upsert_expert({
            "expert_name": "Test Expert",
            "slug": "test_expert",
            "capabilities": ["cap1", "cap2"],
            "skills": ["/skills/test"],
            "strategy": "Test strategy",
            "formula": {"objective": "test"},
            "loop_config": {"epsilon": 0.01},
        })
        assert expert_id > 0
        expert = db.get_expert("test_expert")
        assert expert is not None
        assert expert["capabilities"] == ["cap1", "cap2"]

    def test_stats(self, populated_db):
        stats = populated_db.get_stats()
        assert stats["chapters"] == 2
        assert stats["concepts"] == 4

    def test_get_all_chapters(self, populated_db):
        chapters = populated_db.get_all_chapters()
        assert len(chapters) == 2
        assert chapters[0]["chapter_number"] == 1


class TestConceptExtraction:
    def test_extract_known_concepts(self):
        text = "This discusses gradient descent and backpropagation for neural networks."
        concepts = extract_concepts(text)
        names = {c["concept"] for c in concepts}
        assert "gradient descent" in names
        assert "backpropagation" in names

    def test_extract_categories(self):
        text = "Regularization prevents overfitting. Cross-validation evaluates models."
        concepts = extract_concepts(text)
        categories = {c["concept"]: c["category"] for c in concepts}
        assert categories.get("regularization") == "regularization"
        assert categories.get("cross-validation") == "evaluation"

    def test_empty_text(self):
        concepts = extract_concepts("")
        assert concepts == []


class TestExpertRegistry:
    def test_create_expert(self, populated_db):
        registry = ExpertRegistry(populated_db, "/skills")
        concepts = populated_db.get_concepts(1)
        expert = registry.create_expert_from_chapter(1, "ML Fundamentals", 1, concepts)
        assert expert["expert_name"] == "01_ML Fundamentals"
        assert expert["slug"] == "01_ml_fundamentals"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0

    def test_register_from_db(self, populated_db):
        registry = ExpertRegistry(populated_db, "/skills")
        created = registry.register_experts_from_db()
        assert len(created) == 2

    def test_ask_expert(self, populated_db):
        registry = ExpertRegistry(populated_db, "/skills")
        registry.register_experts_from_db()
        result = registry.ask_expert("01_ml_fundamentals", "How to tune learning rate?")
        assert "error" not in result
        assert result["expert"] == "01_ML Fundamentals"

    def test_ask_missing_expert(self, populated_db):
        registry = ExpertRegistry(populated_db, "/skills")
        result = registry.ask_expert("nonexistent", "question")
        assert "error" in result

    def test_build_entry(self, populated_db):
        registry = ExpertRegistry(populated_db, "/skills")
        registry.register_experts_from_db()
        experts = registry.list_experts()
        matched = [{"expert": e} for e in experts]
        entry = registry.build_competition_entry("titanic", matched)
        assert entry["competition"] == "titanic"
        assert "primary_expert" in entry
        assert "pipeline" in entry


class TestLoopController:
    def test_state_vector_l2(self):
        s1 = StateVector({"a": 1.0, "b": 2.0})
        s2 = StateVector({"a": 1.0, "b": 2.0})
        assert s1.l2_distance(s2) == 0.0

        s3 = StateVector({"a": 4.0, "b": 6.0})
        assert abs(s1.l2_distance(s3) - 5.0) < 1e-6

    def test_convergence_detection(self):
        controller = LoopController(epsilon=0.1, max_iterations=20, patience=2)
        controller.update({"loss": 1.0})
        controller.update({"loss": 0.5})
        r = controller.update({"loss": 0.49})
        assert r["convergence_count"] == 1
        r = controller.update({"loss": 0.489})
        assert r["converged"] is True

    def test_max_iterations(self):
        controller = LoopController(epsilon=0.0001, max_iterations=3, patience=2)
        for i in range(3):
            r = controller.update({"loss": 1.0 / (i + 1)})
        assert not r["should_continue"]

    def test_run_loop(self):
        controller = LoopController(epsilon=0.01, max_iterations=50, patience=3)

        def step_fn(iteration, prev):
            if prev is None:
                return {"loss": 1.0}
            return {"loss": prev["loss"] * 0.5}

        result = controller.run_loop(step_fn)
        assert result["status"] in ("converged", "max_iterations")
        assert result["total_iterations"] > 0

    def test_history(self):
        controller = LoopController(epsilon=0.1, max_iterations=5)
        controller.update({"x": 1.0})
        controller.update({"x": 0.9})
        history = controller.get_history()
        assert len(history) == 2
        assert "l2_distance" in history[1]


class TestScanChapters:
    def test_nonexistent_path(self):
        chapters = scan_chapters("/nonexistent/path")
        assert chapters == []

    def test_empty_directory(self, tmp_path):
        chapters = scan_chapters(str(tmp_path))
        assert chapters == []
