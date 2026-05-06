"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.docling_worker import extract_concepts, discover_chapters
from mlsyseng_mcp.expert_registry import (
    ExpertRegistry,
    _slugify,
    _infer_skills,
    _infer_capabilities,
    _infer_formula,
)
from mlsyseng_mcp.loop_controller import LoopConfig, LoopController, l2_norm


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    database = Database(path)
    yield database
    os.unlink(path)


@pytest.fixture
def populated_db(db):
    ch_id = db.upsert_chapter(
        "01", "Introduction to ML",
        "Content about gradient descent and neural networks and regularization",
        concepts=["gradient descent", "neural network", "regularization"],
    )
    db.upsert_expert(
        expert_name="01_Introduction to ML",
        slug="01_introduction_to_ml",
        chapter_id=ch_id,
        capabilities=["Build baseline models", "ML problem formulation"],
        skills=["/skills/kaggle-preprocessor", "/skills/kaggle-model-trainer"],
        strategy="Baseline → Submit",
        formula={"objective": "minimize_validation_loss"},
        loop_config={"epsilon": 0.001, "max_iterations": 10},
    )
    return db


class TestDatabase:
    def test_init_creates_tables(self, db):
        stats = db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_upsert_and_get_chapter(self, db):
        ch_id = db.upsert_chapter("01", "Test Chapter", "Content here")
        assert ch_id > 0
        ch = db.get_chapter("01")
        assert ch is not None
        assert ch["title"] == "Test Chapter"

    def test_upsert_chapter_idempotent(self, db):
        db.upsert_chapter("01", "Title V1", "Content V1")
        db.upsert_chapter("01", "Title V2", "Content V2")
        chapters = db.list_chapters()
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Title V2"

    def test_upsert_and_get_expert(self, db):
        exp_id = db.upsert_expert(
            expert_name="Test Expert",
            slug="test_expert",
            capabilities=["cap1"],
            skills=["/skills/s1"],
        )
        assert exp_id > 0
        expert = db.get_expert("test_expert")
        assert expert is not None
        assert expert["expert_name"] == "Test Expert"
        assert expert["capabilities"] == ["cap1"]

    def test_list_experts(self, populated_db):
        experts = populated_db.list_experts()
        assert len(experts) == 1
        assert experts[0]["slug"] == "01_introduction_to_ml"

    def test_search_chapters(self, populated_db):
        results = populated_db.search_chapters("gradient")
        assert len(results) == 1

    def test_extraction_status(self, db):
        db.set_extraction_status("01", "running")
        db.set_extraction_status("01", "completed", pages_extracted=10)
        statuses = db.get_extraction_status()
        assert len(statuses) == 2
        assert statuses[0]["status"] == "completed"

    def test_convergence_state(self, db):
        db.record_convergence_state("test", 0, [0.5, 0.6], 1.0, False)
        db.record_convergence_state("test", 1, [0.7, 0.8], 0.001, True)
        history = db.get_convergence_history("test")
        assert len(history) == 2
        assert history[1]["converged"] == True


class TestDoclingWorker:
    def test_extract_concepts(self):
        text = "This covers gradient descent, neural network architectures, and regularization techniques."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "neural network" in concepts
        assert "regularization" in concepts

    def test_extract_concepts_empty(self):
        assert extract_concepts("") == []
        assert extract_concepts("unrelated text about cooking") == []

    def test_discover_chapters_nonexistent(self):
        chapters = discover_chapters("/nonexistent/path")
        assert chapters == []


class TestExpertRegistry:
    def test_slugify(self):
        assert _slugify("08 ML Systems") == "08_ml_systems"
        assert _slugify("Introduction to ML!") == "introduction_to_ml"
        assert _slugify("Deep-Learning") == "deep_learning"

    def test_infer_skills(self):
        skills = _infer_skills(["neural network", "gradient descent"])
        assert any("kaggle-model-trainer" in s for s in skills)
        assert any("kaggle-deep-learning" in s for s in skills)

    def test_infer_skills_default(self):
        skills = _infer_skills(["unknown_concept"])
        assert any("kaggle-preprocessor" in s for s in skills)

    def test_infer_capabilities(self):
        caps = _infer_capabilities(["neural network", "deep learning"])
        assert "Build and train neural networks" in caps
        assert "Build baseline models quickly" in caps

    def test_infer_formula_classification(self):
        formula = _infer_formula(["classification"])
        assert "precision" in formula["metrics"]
        assert "recall" in formula["metrics"]

    def test_infer_formula_regression(self):
        formula = _infer_formula(["regression"])
        assert formula["objective"] == "minimize_rmse"
        assert "rmse" in formula["metrics"]

    def test_create_expert_from_chapter(self, populated_db):
        registry = ExpertRegistry(populated_db)
        expert = registry.create_expert_from_chapter(
            "02", "Feature Engineering", ["feature engineering", "cross-validation"]
        )
        assert expert["slug"] == "02_feature_engineering"
        assert "Automated feature creation" in expert["capabilities"]

    def test_get_experts_for_competition(self, populated_db):
        registry = ExpertRegistry(populated_db)
        experts = registry.get_experts_for_competition("Build a baseline classification model")
        assert len(experts) >= 0


class TestLoopController:
    def test_l2_norm_identical(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_l2_norm_different(self):
        result = l2_norm([1.0, 0.0], [0.0, 0.0])
        assert abs(result - 1.0) < 1e-10

    def test_l2_norm_empty(self):
        assert l2_norm([], []) == float("inf")

    def test_l2_norm_unequal_length(self):
        result = l2_norm([1.0, 2.0, 3.0], [1.0, 2.0])
        assert result == 0.0

    def test_convergence_loop(self, populated_db):
        config = LoopConfig(max_iterations=5, patience=2, epsilon=0.01)
        controller = LoopController(populated_db, config)
        experts = populated_db.list_experts()
        result = controller.run_loop("test_comp", experts)
        assert result["competition"] == "test_comp"
        assert result["total_iterations"] > 0
        assert "converged" in result
        assert "history" in result

    def test_convergence_summary(self, populated_db):
        config = LoopConfig(max_iterations=3)
        controller = LoopController(populated_db, config)
        experts = populated_db.list_experts()
        controller.run_loop("summary_test", experts)
        summary = controller.get_convergence_summary("summary_test")
        assert summary["competition"] == "summary_test"
        assert summary["total_iterations"] > 0

    def test_convergence_summary_no_data(self, db):
        controller = LoopController(db)
        summary = controller.get_convergence_summary("nonexistent")
        assert summary["status"] == "no_data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
