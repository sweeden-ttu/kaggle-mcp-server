"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from mlsyseng_mcp import database
from mlsyseng_mcp.docling_worker import extract_concepts, discover_chapters
from mlsyseng_mcp.expert_registry import (
    create_expert_from_chapter,
    register_all_experts,
    get_expert_for_query,
    select_experts_for_competition,
)
from mlsyseng_mcp.loop_controller import (
    LoopController,
    StateVector,
    run_convergence_loop,
)


@pytest.fixture
def db_path():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    database.init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def populated_db(db_path):
    """Create a database with sample chapters and experts."""
    database.store_chapter(
        chapter_number=1,
        title="Introduction to ML",
        source_path="/test/ch1.pdf",
        extracted_text="Machine learning is about gradient descent and optimization.",
        markdown_content="# Introduction\n\nML basics.",
        concepts=["gradient descent", "optimization"],
        db_path=db_path,
    )
    database.store_chapter(
        chapter_number=2,
        title="Deep Learning",
        source_path="/test/ch2.pdf",
        extracted_text="Deep learning uses neural networks with backpropagation for training.",
        markdown_content="# Deep Learning\n\nNeural networks.",
        concepts=["neural network", "deep learning", "backpropagation"],
        db_path=db_path,
    )
    database.store_chapter(
        chapter_number=3,
        title="Feature Engineering",
        source_path="/test/ch3.pdf",
        extracted_text="Feature engineering creates better features for model performance.",
        markdown_content="# Feature Engineering\n\nFeature creation.",
        concepts=["feature engineering", "normalization"],
        db_path=db_path,
    )
    return db_path


class TestDatabase:
    def test_init_db(self, db_path):
        stats = database.get_stats(db_path=db_path)
        assert stats["total_chapters"] == 0
        assert stats["total_experts"] == 0

    def test_store_and_retrieve_chapter(self, db_path):
        database.store_chapter(
            chapter_number=5,
            title="Test Chapter",
            source_path="/test.pdf",
            extracted_text="Some text content.",
            markdown_content="# Test",
            concepts=["concept1", "concept2"],
            db_path=db_path,
        )
        ch = database.get_chapter(5, db_path=db_path)
        assert ch is not None
        assert ch["title"] == "Test Chapter"
        assert ch["concepts"] == ["concept1", "concept2"]

    def test_store_and_retrieve_expert(self, db_path):
        expert_data = {
            "expert_name": "Test Expert",
            "slug": "test_expert",
            "capabilities": ["cap1", "cap2"],
            "skills": ["/path/to/skill"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001},
        }
        database.store_expert(expert_data, db_path=db_path)
        expert = database.get_expert("test_expert", db_path=db_path)
        assert expert is not None
        assert expert["expert_name"] == "Test Expert"
        assert expert["capabilities"] == ["cap1", "cap2"]

    def test_get_all_chapters(self, populated_db):
        chapters = database.get_all_chapters(db_path=populated_db)
        assert len(chapters) == 3

    def test_get_stats(self, populated_db):
        stats = database.get_stats(db_path=populated_db)
        assert stats["total_chapters"] == 3

    def test_store_competition_entry(self, db_path):
        entry_id = database.store_competition_entry(
            competition_name="titanic",
            experts_used=["expert_a", "expert_b"],
            skills_applied=["/skill1", "/skill2"],
            state_history=[[1.0, 2.0], [1.1, 2.1]],
            converged=True,
            final_score=0.85,
            db_path=db_path,
        )
        assert entry_id > 0


class TestConceptExtraction:
    def test_extract_concepts_basic(self):
        text = "We use gradient descent and backpropagation for training."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "backpropagation" in concepts

    def test_extract_concepts_empty(self):
        concepts = extract_concepts("No ML terms here.")
        assert concepts == []

    def test_extract_concepts_deduplication(self):
        text = "Gradient descent. More gradient descent. Even more gradient descent."
        concepts = extract_concepts(text)
        assert concepts.count("gradient descent") == 1


class TestExpertRegistry:
    def test_create_expert_from_chapter(self, populated_db):
        ch = database.get_chapter(2, db_path=populated_db)
        expert = create_expert_from_chapter(ch, db_path=populated_db)
        assert expert["slug"] == "02_deep_learning"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0

    def test_register_all_experts(self, populated_db):
        experts = register_all_experts(db_path=populated_db)
        assert len(experts) == 3

    def test_get_expert_for_query(self, populated_db):
        register_all_experts(db_path=populated_db)
        expert = get_expert_for_query("deep learning neural", db_path=populated_db)
        assert expert is not None
        assert "deep" in expert["expert_name"].lower() or "neural" in expert["expert_name"].lower()


class TestLoopController:
    def test_state_vector_l2(self):
        a = StateVector([1.0, 2.0, 3.0])
        b = StateVector([1.0, 2.0, 3.0])
        assert a.l2_distance(b) == 0.0

    def test_state_vector_l2_nonzero(self):
        a = StateVector([0.0, 0.0])
        b = StateVector([3.0, 4.0])
        assert abs(a.l2_distance(b) - 5.0) < 1e-10

    def test_convergence(self):
        controller = LoopController(epsilon=0.01, max_iterations=100, patience=2)
        for i in range(20):
            state = [1.0 + 0.5**i, 2.0 + 0.5**i]
            status = controller.update_state(state, {"loss": 1.0 / (i + 1)})
            if not controller.should_continue():
                break
        assert controller.converged

    def test_max_iterations_exit(self):
        controller = LoopController(epsilon=0.0001, max_iterations=3, patience=3)
        for i in range(10):
            state = [float(i), float(i * 2)]
            controller.update_state(state)
            if not controller.should_continue():
                break
        assert controller.current_iteration == 3
        assert not controller.converged

    def test_run_convergence_loop(self):
        def step_fn(iteration, current):
            if current:
                base = current.to_list()
            else:
                base = [1.0, 2.0, 3.0]
            decay = 0.3**iteration
            return [v + decay * 0.001 for v in base], {"loss": 1.0 / (iteration + 1)}

        summary = run_convergence_loop(
            step_fn=step_fn, epsilon=0.001, max_iterations=20, patience=2
        )
        assert summary["converged"]
        assert summary["total_iterations"] < 20

    def test_summary_structure(self):
        controller = LoopController()
        controller.update_state([1.0, 2.0])
        controller.update_state([1.1, 2.1])
        summary = controller.get_summary()
        assert "total_iterations" in summary
        assert "converged" in summary
        assert "distances" in summary
        assert len(summary["distances"]) == 1
