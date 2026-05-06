"""Tests for MLSysEng MoE system."""

import json
import os
import tempfile

import numpy as np
import pytest

from mlsyseng_moe import database as db
from mlsyseng_moe.docling_worker import extract_concepts, discover_chapters
from mlsyseng_moe.expert_registry import (
    create_expert_from_chapter,
    list_experts,
    _make_slug,
    _determine_primary_category,
)
from mlsyseng_moe.loop_controller import (
    LoopController,
    ConvergenceResult,
    create_competition_state,
    run_competition_loop,
)


@pytest.fixture
def tmp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db.init_db(db_path)
    yield db_path
    os.unlink(db_path)


class TestDatabase:
    def test_init_db(self, tmp_db):
        conn = db.get_connection(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "chapters" in table_names
        assert "content_blocks" in table_names
        assert "concepts" in table_names
        assert "experts" in table_names
        conn.close()

    def test_insert_chapter(self, tmp_db):
        ch_id = db.insert_chapter(1, "Test Chapter", "/tmp/ch1", db_path=tmp_db)
        assert ch_id == 1

        ch_id2 = db.insert_chapter(1, "Updated Chapter", "/tmp/ch1", db_path=tmp_db)
        assert ch_id2 == 1

    def test_insert_content_block(self, tmp_db):
        ch_id = db.insert_chapter(1, "Test", "/tmp", db_path=tmp_db)
        block_id = db.insert_content_block(
            ch_id, "Test content", "text", 1, 0, db_path=tmp_db
        )
        assert block_id > 0

        blocks = db.get_chapter_content(ch_id, tmp_db)
        assert len(blocks) == 1
        assert blocks[0]["content"] == "Test content"

    def test_insert_concept(self, tmp_db):
        ch_id = db.insert_chapter(1, "Test", "/tmp", db_path=tmp_db)
        concept_id = db.insert_concept(
            ch_id, "Neural Network", "A model", "architecture", db_path=tmp_db
        )
        assert concept_id > 0

        concepts = db.get_chapter_concepts(ch_id, tmp_db)
        assert len(concepts) == 1
        assert concepts[0]["concept_name"] == "Neural Network"

    def test_insert_expert(self, tmp_db):
        ch_id = db.insert_chapter(1, "Test", "/tmp", db_path=tmp_db)
        expert_id = db.insert_expert(
            expert_name="01_Test",
            slug="01_test",
            chapter_id=ch_id,
            capabilities=["cap1", "cap2"],
            skills=["/skills/s1"],
            strategy="Baseline → Submit",
            formula={"objective": "minimize_loss"},
            loop_config={"epsilon": 0.001},
            db_path=tmp_db,
        )
        assert expert_id > 0

        expert = db.get_expert_by_slug("01_test", tmp_db)
        assert expert["expert_name"] == "01_Test"
        assert expert["capabilities"] == ["cap1", "cap2"]
        assert expert["formula"]["objective"] == "minimize_loss"

    def test_get_stats(self, tmp_db):
        stats = db.get_stats(tmp_db)
        assert stats["chapters"] == 0
        assert stats["experts"] == 0

        db.insert_chapter(1, "Ch1", "/tmp", db_path=tmp_db)
        stats = db.get_stats(tmp_db)
        assert stats["chapters"] == 1


class TestConceptExtraction:
    def test_extract_concepts(self):
        text = "Neural network optimization using gradient descent and backpropagation"
        concepts = extract_concepts(text)
        names = {c["concept_name"] for c in concepts}
        assert "Neural Network" in names
        assert "Gradient Descent" in names
        assert "Backpropagation" in names

    def test_extract_concepts_empty(self):
        concepts = extract_concepts("This text has no ML concepts.")
        assert concepts == []

    def test_concept_categories(self):
        text = "Apply regularization to prevent overfitting using cross-validation"
        concepts = extract_concepts(text)
        categories = {c["category"] for c in concepts}
        assert "training" in categories


class TestExpertRegistry:
    def test_make_slug(self):
        assert _make_slug(8, "ML Systems") == "08_ml_systems"
        assert _make_slug(1, "Deep Learning") == "01_deep_learning"

    def test_determine_category(self):
        concepts = [
            {"category": "optimization"},
            {"category": "optimization"},
            {"category": "training"},
        ]
        assert _determine_primary_category(concepts) == "optimization"

    def test_determine_category_empty(self):
        assert _determine_primary_category([]) == "general"

    def test_create_expert(self, tmp_db):
        ch_id = db.insert_chapter(1, "Optimization", "/tmp", db_path=tmp_db)
        concepts = [
            {"concept_name": "Gradient Descent", "category": "optimization"},
        ]
        expert = create_expert_from_chapter(
            chapter_id=ch_id,
            chapter_number=1,
            title="Optimization",
            concepts=concepts,
            db_path=tmp_db,
        )
        assert expert["slug"] == "01_optimization"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0


class TestLoopController:
    def test_convergence(self):
        controller = LoopController(epsilon=0.1, max_iterations=20, patience=2)
        state = np.array([1.0, 2.0, 3.0])

        def step(s, i):
            return s + np.random.normal(0, 0.01 * (0.5 ** i), size=s.shape), {"i": i}

        result = controller.run(state, step)
        assert result.converged is True
        assert result.iterations_run <= 20

    def test_max_iterations(self):
        controller = LoopController(epsilon=0.0001, max_iterations=3, patience=2)
        state = np.array([1.0, 2.0])

        def step(s, i):
            return s + np.random.normal(0, 0.5, size=s.shape), {"i": i}

        result = controller.run(state, step)
        assert result.converged is False
        assert result.exit_reason == "max_iterations_reached"

    def test_create_competition_state(self):
        state = create_competition_state(expert_count=5, feature_dim=8)
        assert state.shape == (5 + 8 + 3,)
        assert np.isclose(state[:5].sum(), 1.0)

    def test_run_competition_loop(self):
        experts = [
            {"expert_name": "expert_1", "capabilities": []},
            {"expert_name": "expert_2", "capabilities": []},
        ]
        result = run_competition_loop(
            experts=experts,
            competition_name="test",
            epsilon=0.01,
            max_iterations=20,
            patience=3,
        )
        assert "competition" in result
        assert "converged" in result
        assert "delta_history" in result
        assert result["experts_used"] == ["expert_1", "expert_2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
