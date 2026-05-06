"""Tests for the MLSysEng MoE system.

Tests database operations, expert registry, loop controller,
and docling worker (without actual PDF files).
"""

import json
import math
import os
import tempfile

import pytest

from src.mlsyseng_mcp.database import (
    ChapterRecord,
    Database,
    ExpertRecord,
    LoopState,
)
from src.mlsyseng_mcp.docling_worker import DoclingWorker, extract_concepts
from src.mlsyseng_mcp.expert_registry import ExpertRegistry
from src.mlsyseng_mcp.loop_controller import (
    ConvergenceConfig,
    LoopController,
    l2_norm,
)


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "test.db")
        yield Database(db_path)


@pytest.fixture
def sample_chapter():
    return ChapterRecord(
        chapter_id="ch01",
        title="01 Introduction to ML",
        slug="01_introduction_to_ml",
        source_path="/tmp/chapters/01",
        content_md="This chapter covers neural network optimization and gradient descent.",
        concepts=["neural network", "optimization", "gradient descent"],
        extracted_at=1000.0,
        word_count=100,
    )


@pytest.fixture
def sample_expert():
    return ExpertRecord(
        expert_name="01 Introduction to ML",
        slug="01_introduction_to_ml",
        chapter_id="ch01",
        capabilities=["Build baseline models quickly"],
        skills=["/tmp/skills/kaggle-model-trainer"],
        strategy="Baseline → Submit",
        formula={
            "objective": "minimize_validation_loss",
            "function": "L = f(X, θ, α)",
            "metrics": ["accuracy"],
        },
        loop_config={
            "objective": "minimize_validation_loss",
            "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
            "epsilon": 0.001,
            "max_iterations": 10,
            "patience": 3,
        },
    )


class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_upsert_and_get_chapter(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        got = tmp_db.get_chapter("ch01")
        assert got is not None
        assert got.title == "01 Introduction to ML"
        assert got.word_count == 100
        assert "neural network" in got.concepts

    def test_chapter_by_slug(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        got = tmp_db.get_chapter_by_slug("01_introduction_to_ml")
        assert got is not None
        assert got.chapter_id == "ch01"

    def test_list_chapters(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        chapters = tmp_db.list_chapters()
        assert len(chapters) == 1

    def test_upsert_and_get_expert(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        got = tmp_db.get_expert("01 Introduction to ML")
        assert got is not None
        assert got.slug == "01_introduction_to_ml"

    def test_expert_by_slug(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        got = tmp_db.get_expert_by_slug("01_introduction_to_ml")
        assert got is not None

    def test_loop_state(self, tmp_db):
        state = LoopState(
            loop_id="test_loop",
            competition="titanic",
            iteration=0,
            state_vector=[1.0, 0.5, 0.6],
            metrics={"loss": 0.5},
            converged=False,
            timestamp=1000.0,
        )
        tmp_db.save_loop_state(state)
        got = tmp_db.get_latest_loop_state("test_loop")
        assert got is not None
        assert got.iteration == 0
        assert got.state_vector == [1.0, 0.5, 0.6]

    def test_loop_history(self, tmp_db):
        for i in range(3):
            state = LoopState(
                loop_id="test_loop",
                competition="titanic",
                iteration=i,
                state_vector=[1.0 / (i + 1)],
                metrics={"loss": 1.0 / (i + 1)},
                converged=i >= 2,
                timestamp=1000.0 + i,
            )
            tmp_db.save_loop_state(state)

        history = tmp_db.get_loop_history("test_loop")
        assert len(history) == 3
        assert history[0].iteration == 0
        assert history[2].converged is True

    def test_stats(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        stats = tmp_db.get_stats()
        assert stats["chapters_indexed"] == 1
        assert stats["experts_registered"] == 1
        assert stats["total_words_extracted"] == 100


class TestConceptExtraction:
    def test_extract_neural_network(self):
        text = "Neural network architectures are fundamental to deep learning."
        concepts = extract_concepts(text)
        assert "neural network" in concepts
        assert "deep learning" in concepts

    def test_extract_optimization(self):
        text = "Gradient descent is an optimization algorithm."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "optimization" in concepts

    def test_extract_empty(self):
        concepts = extract_concepts("This is a random text with no ML concepts.")
        assert isinstance(concepts, list)

    def test_extract_multiple(self):
        text = (
            "Cross-validation and regularization are key to preventing "
            "overfitting. Random forest and gradient boosting are ensemble methods."
        )
        concepts = extract_concepts(text)
        assert any("cross" in c for c in concepts)
        assert any("regularization" in c for c in concepts)
        assert any("random forest" in c for c in concepts)
        assert any("gradient boosting" in c for c in concepts)


class TestExpertRegistry:
    def test_create_expert_from_chapter(self, tmp_db, sample_chapter):
        tmp_db.upsert_chapter(sample_chapter)
        registry = ExpertRegistry(tmp_db, skills_base_path="/tmp/skills")
        expert = registry.create_expert_from_chapter(sample_chapter)
        assert expert.expert_name == sample_chapter.title
        assert len(expert.skills) > 0
        assert len(expert.capabilities) > 0

    def test_query_expert(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        registry = ExpertRegistry(tmp_db)
        result = registry.query_expert("01_introduction_to_ml", "What is ML?")
        assert result["expert_name"] == "01 Introduction to ML"
        assert result["question"] == "What is ML?"

    def test_query_nonexistent_expert(self, tmp_db):
        registry = ExpertRegistry(tmp_db)
        result = registry.query_expert("nonexistent", "test")
        assert "error" in result

    def test_list_experts(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        registry = ExpertRegistry(tmp_db)
        experts = registry.list_experts()
        assert len(experts) == 1

    def test_export_expert(self, tmp_db, sample_chapter, sample_expert):
        tmp_db.upsert_chapter(sample_chapter)
        tmp_db.upsert_expert(sample_expert)
        registry = ExpertRegistry(tmp_db)
        data = registry.export_expert("01_introduction_to_ml")
        assert data is not None
        assert data["slug"] == "01_introduction_to_ml"


class TestL2Norm:
    def test_zero_distance(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_distance(self):
        result = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_different_lengths(self):
        result = l2_norm([1.0], [1.0, 2.0])
        assert abs(result - 2.0) < 1e-10

    def test_single_element(self):
        result = l2_norm([5.0], [3.0])
        assert abs(result - 2.0) < 1e-10


class TestLoopController:
    def test_create_loop(self, tmp_db):
        controller = LoopController(tmp_db)
        loop_id = controller.create_loop("titanic")
        assert "titanic" in loop_id

    def test_record_iteration(self, tmp_db):
        controller = LoopController(tmp_db)
        loop_id = controller.create_loop("titanic")

        result = controller.record_iteration(
            loop_id, "titanic", 0,
            [1.0, 0.5], {"loss": 0.5},
        )
        assert result.iteration == 0
        assert result.l2_norm == float("inf")
        assert result.converged is False

    def test_convergence_detection(self, tmp_db):
        config = ConvergenceConfig(epsilon=0.1, max_iterations=10, patience=1)
        controller = LoopController(tmp_db, config)
        loop_id = controller.create_loop("titanic")

        controller.record_iteration(
            loop_id, "titanic", 0,
            [1.0, 0.5], {"loss": 0.5},
        )
        result = controller.record_iteration(
            loop_id, "titanic", 1,
            [1.0, 0.5], {"loss": 0.5},
        )
        assert result.l2_norm == 0.0
        assert result.converged is True

    def test_run_loop(self, tmp_db):
        config = ConvergenceConfig(
            epsilon=0.01,
            max_iterations=5,
            patience=2,
        )
        controller = LoopController(tmp_db, config)

        call_count = [0]

        def step_fn(iteration, prev_state):
            call_count[0] += 1
            val = 1.0 / (iteration + 1)
            return [val, val * 0.5], {"loss": val}

        result = controller.run_loop("test_comp", step_fn, config)
        assert result["competition"] == "test_comp"
        assert result["total_iterations"] > 0
        assert call_count[0] > 0

    def test_max_iterations_stop(self, tmp_db):
        config = ConvergenceConfig(
            epsilon=0.0001,
            max_iterations=3,
            patience=10,
        )
        controller = LoopController(tmp_db, config)

        def step_fn(iteration, prev_state):
            return [float(iteration)], {"loss": float(iteration)}

        result = controller.run_loop("test", step_fn, config)
        assert result["total_iterations"] == 3

    def test_patience_stop(self, tmp_db):
        config = ConvergenceConfig(
            epsilon=1.0,
            max_iterations=20,
            patience=2,
        )
        controller = LoopController(tmp_db, config)

        def step_fn(iteration, prev_state):
            return [1.0], {"loss": 1.0}

        result = controller.run_loop("test", step_fn, config)
        assert result["converged"] is True
        assert result["total_iterations"] <= 4

    def test_loop_summary(self, tmp_db):
        controller = LoopController(tmp_db)
        loop_id = controller.create_loop("titanic")

        controller.record_iteration(
            loop_id, "titanic", 0,
            [1.0], {"loss": 1.0},
        )

        summary = controller.get_loop_summary(loop_id)
        assert summary["loop_id"] == loop_id
        assert summary["total_iterations"] == 1


class TestDoclingWorker:
    def test_scan_empty_directory(self, tmp_db):
        with tempfile.TemporaryDirectory() as d:
            worker = DoclingWorker(tmp_db, ml_principles_path=d)
            chapters = worker.scan_chapters()
            assert chapters == []

    def test_scan_nonexistent_directory(self, tmp_db):
        worker = DoclingWorker(tmp_db, ml_principles_path="/nonexistent/path")
        chapters = worker.scan_chapters()
        assert chapters == []

    def test_progress_tracking(self, tmp_db):
        with tempfile.TemporaryDirectory() as d:
            worker = DoclingWorker(tmp_db, ml_principles_path=d)
            progress = worker.get_progress()
            assert isinstance(progress, dict)
