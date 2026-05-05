"""Tests for the MLSysEng MoE system."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mlsyseng_mcp.database import Chapter, ConceptEntry, Expert, MLSysEngDB
from mlsyseng_mcp.docling_worker import _extract_concepts_from_text, _make_slug, _parse_chapter_number
from mlsyseng_mcp.expert_registry import (
    _build_formula,
    _build_loop_config,
    _infer_capabilities,
    _infer_skills,
    register_experts_from_chapters,
)
from mlsyseng_mcp.loop_controller import (
    LoopConfig,
    LoopController,
    build_competition_state_vector,
    l2_norm,
)


def _make_test_db():
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    return MLSysEngDB(db_path)


class TestDatabase:
    def test_init_creates_tables(self):
        db = _make_test_db()
        stats = db.get_stats()
        assert stats["total_chapters"] == 0
        assert stats["total_experts"] == 0

    def test_upsert_and_get_chapter(self):
        db = _make_test_db()
        ch = Chapter(
            chapter_number=1, title="Test Chapter", slug="test_chapter",
            source_path="/tmp/test", status="extracted",
        )
        ch_id = db.upsert_chapter(ch)
        assert ch_id is not None

        retrieved = db.get_chapter("test_chapter")
        assert retrieved is not None
        assert retrieved.title == "Test Chapter"

    def test_upsert_updates_existing(self):
        db = _make_test_db()
        ch = Chapter(chapter_number=1, title="V1", slug="ch1", source_path="/tmp")
        db.upsert_chapter(ch)

        ch.title = "V2"
        db.upsert_chapter(ch)

        retrieved = db.get_chapter("ch1")
        assert retrieved.title == "V2"

    def test_list_chapters(self):
        db = _make_test_db()
        for i in range(3):
            db.upsert_chapter(Chapter(
                chapter_number=i, title=f"Ch {i}", slug=f"ch_{i}", source_path=f"/tmp/{i}",
            ))
        chapters = db.list_chapters()
        assert len(chapters) == 3

    def test_upsert_and_get_expert(self):
        db = _make_test_db()
        expert = Expert(expert_name="Test Expert", slug="test_expert")
        eid = db.upsert_expert(expert)
        assert eid is not None

        retrieved = db.get_expert("test_expert")
        assert retrieved.expert_name == "Test Expert"

    def test_add_and_search_concepts(self):
        db = _make_test_db()
        db.add_concept(ConceptEntry(concept="Neural Network", description="A type of model"))
        db.add_concept(ConceptEntry(concept="Decision Tree", description="Tree-based model"))

        results = db.search_concepts("neural")
        assert len(results) == 1
        assert results[0].concept == "Neural Network"


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = "This covers gradient descent and neural network training with regularization"
        concepts = _extract_concepts_from_text(text)
        assert "Gradient Descent" in concepts
        assert "Neural Network" in concepts
        assert "Regularization" in concepts

    def test_parse_chapter_number(self):
        assert _parse_chapter_number("08_ML_Systems") == 8
        assert _parse_chapter_number("01_Introduction") == 1
        assert _parse_chapter_number("no_number") == 0

    def test_make_slug(self):
        assert _make_slug("08 ML Systems") == "08_ml_systems"
        assert _make_slug("Hello World!") == "hello_world"


class TestExpertRegistry:
    def test_infer_skills(self):
        concepts = ["Gradient Descent", "Neural Network"]
        skills = _infer_skills(concepts, "/tmp/skills")
        assert len(skills) > 0
        assert any("model-trainer" in s for s in skills)

    def test_infer_capabilities(self):
        concepts = ["Deep Learning", "Feature Engineering"]
        caps = _infer_capabilities(concepts)
        assert "Build baseline models quickly" in caps
        assert any("deep learning" in c.lower() for c in caps)

    def test_build_formula(self):
        formula = _build_formula(["accuracy", "F1 Score"])
        assert formula["objective"] == "minimize_validation_loss"
        assert "accuracy" in formula["metrics"]

    def test_build_loop_config(self):
        config = _build_loop_config(epsilon=0.01, max_iterations=5)
        assert config["epsilon"] == 0.01
        assert config["max_iterations"] == 5

    def test_register_experts_from_chapters(self):
        db = _make_test_db()
        db.upsert_chapter(Chapter(
            chapter_number=1, title="Test", slug="test",
            source_path="/tmp", concepts='["Neural Network"]',
            status="extracted",
        ))

        result = register_experts_from_chapters(db)
        assert result["registered"] == 1

        experts = db.list_experts()
        assert len(experts) == 1


class TestLoopController:
    def test_l2_norm(self):
        assert l2_norm([0, 0], [3, 4]) == 5.0
        assert l2_norm([1, 1], [1, 1]) == 0.0

    def test_l2_norm_different_lengths(self):
        result = l2_norm([1, 2, 3], [1, 2])
        assert result == 0.0

    def test_convergence(self):
        lc = LoopController(LoopConfig(epsilon=0.1, max_iterations=20, patience=2))

        for i in range(20):
            state = lc.record_state([1.0, 1.0], {"accuracy": 0.9})
            if not lc.should_continue():
                break

        summary = lc.get_summary()
        assert summary["converged"] is True
        assert summary["total_iterations"] <= 20

    def test_max_iterations(self):
        lc = LoopController(LoopConfig(epsilon=0.0001, max_iterations=3, patience=10))
        import random
        random.seed(42)

        for i in range(10):
            state_vec = [random.random() for _ in range(4)]
            lc.record_state(state_vec)
            if not lc.should_continue():
                break

        assert lc.get_summary()["total_iterations"] <= 3

    def test_build_state_vector(self):
        metrics = {"accuracy": 0.9, "f1_score": 0.85, "loss": 0.1}
        vec = build_competition_state_vector(metrics)
        assert len(vec) == 6
        assert vec[0] == 0.9
        assert vec[5] == 0.1

    def test_run_with_function(self):
        lc = LoopController(LoopConfig(epsilon=0.01, max_iterations=5, patience=2))

        def iterate_fn(iteration, prev):
            return [1.0, 0.9, 0.8], {"accuracy": 0.9}

        summary = lc.run(iterate_fn)
        assert summary["converged"] is True

    def test_from_expert_config(self):
        config_json = json.dumps({
            "objective": "minimize_loss",
            "epsilon": 0.005,
            "max_iterations": 8,
            "patience": 4,
        })
        lc = LoopController.from_expert_config(config_json)
        assert lc.config.epsilon == 0.005
        assert lc.config.max_iterations == 8


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
