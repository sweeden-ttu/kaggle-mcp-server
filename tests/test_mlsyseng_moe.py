"""Tests for MLSysEng MoE system."""

import json
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def temp_chroma():
    """Create a temporary ChromaDB directory."""
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestDatabase:
    def test_init_db(self, temp_db):
        from mlsyseng_moe import database

        database.init_db(temp_db)
        conn = database.get_connection(temp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "chapters" in table_names
        assert "concepts" in table_names
        assert "experts" in table_names
        assert "extraction_log" in table_names
        conn.close()

    def test_store_and_retrieve_chapter(self, temp_db):
        from mlsyseng_moe import database

        database.init_db(temp_db)
        chapter_id = database.store_chapter(
            chapter_number=1,
            title="Introduction to ML",
            source_path="/path/to/chapter1",
            markdown_content="This is chapter 1 content about machine learning.",
            db_path=temp_db,
        )
        assert chapter_id > 0

        chapters = database.get_all_chapters(temp_db)
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Introduction to ML"
        assert chapters[0]["chapter_number"] == 1

    def test_store_and_retrieve_concepts(self, temp_db):
        from mlsyseng_moe import database

        database.init_db(temp_db)
        chapter_id = database.store_chapter(1, "Test", "/path", "content", temp_db)
        concepts = [
            {"name": "Gradient Descent", "description": "Optimization algo", "category": "optimization"},
            {"name": "Neural Network", "description": "Deep learning model", "category": "architecture"},
        ]
        database.store_concepts(chapter_id, concepts, temp_db)

        retrieved = database.get_concepts_for_chapter(chapter_id, temp_db)
        assert len(retrieved) == 2
        assert retrieved[0]["concept_name"] == "Gradient Descent"

    def test_store_and_retrieve_expert(self, temp_db):
        from mlsyseng_moe import database

        database.init_db(temp_db)
        expert_def = {
            "expert_name": "01_Introduction",
            "slug": "01_introduction",
            "chapter_id": None,
            "capabilities": ["Build models", "Optimize"],
            "skills": ["/path/skill1", "/path/skill2"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
        }
        expert_id = database.store_expert(expert_def, temp_db)
        assert expert_id > 0

        experts = database.get_all_experts(temp_db)
        assert len(experts) == 1
        assert experts[0]["expert_name"] == "01_Introduction"
        assert experts[0]["capabilities"] == ["Build models", "Optimize"]

    def test_get_stats(self, temp_db):
        from mlsyseng_moe import database

        database.init_db(temp_db)
        stats = database.get_stats(temp_db)
        assert stats["chapters_indexed"] == 0
        assert stats["concepts_extracted"] == 0
        assert stats["experts_registered"] == 0


class TestLoopController:
    def test_convergence(self):
        from mlsyseng_moe.loop_controller import LoopController, LoopState

        controller = LoopController(epsilon=0.01, max_iterations=20, patience=2)

        target = np.array([1.0, 0.5, 0.8])

        def step_fn(iteration, prev_state):
            if prev_state is None:
                return np.array([0.0, 0.0, 0.0])
            alpha = 0.5
            return prev_state.values + alpha * (target - prev_state.values)

        result = controller.run(step_fn)
        assert result.converged
        assert result.iterations_run < 20
        np.testing.assert_allclose(result.final_state.values, target, atol=0.05)

    def test_max_iterations(self):
        from mlsyseng_moe.loop_controller import LoopController

        controller = LoopController(epsilon=0.0001, max_iterations=3, patience=2)

        def step_fn(iteration, prev_state):
            return np.random.rand(4) * iteration

        result = controller.run(step_fn)
        assert not result.converged
        assert result.iterations_run == 3
        assert result.exit_reason == "max_iterations_reached"

    def test_l2_distance(self):
        from mlsyseng_moe.loop_controller import LoopController

        controller = LoopController()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert controller.compute_l2_distance(a, b) == 0.0

        c = np.array([4.0, 6.0, 3.0])
        dist = controller.compute_l2_distance(a, c)
        assert dist == pytest.approx(5.0)

    def test_run_competition_loop(self):
        from mlsyseng_moe.loop_controller import run_competition_loop

        result = run_competition_loop(
            competition_name="titanic",
            expert_skills=["skill1", "skill2"],
            max_iterations=5,
        )
        assert "converged" in result
        assert "iterations_run" in result
        assert "final_state" in result
        assert result["iterations_run"] <= 5


class TestExpertRegistry:
    def test_slugify(self):
        from mlsyseng_moe.expert_registry import _slugify

        assert _slugify("01_ML Systems") == "01_ml_systems"
        assert _slugify("Deep Learning") == "deep_learning"

    def test_create_expert_from_chapter(self, temp_db):
        from mlsyseng_moe import database
        from mlsyseng_moe.expert_registry import create_expert_from_chapter

        database.init_db(temp_db)
        chapter_id = database.store_chapter(8, "ML Systems", "/path", "content about ML", temp_db)
        concepts = [
            {"name": "Gradient Descent", "description": "", "category": "optimization"},
            {"name": "Neural Network", "description": "", "category": "architecture"},
        ]

        expert = create_expert_from_chapter(chapter_id, 8, "ML Systems", concepts, temp_db)
        assert expert["slug"] == "08_ml_systems"
        assert "Optimize model parameters" in " ".join(expert["capabilities"])
        assert len(expert["skills"]) > 0
        assert expert["formula"]["objective"] == "minimize_validation_loss"


class TestDoclingWorker:
    def test_extract_concepts(self):
        from mlsyseng_moe.docling_worker import extract_concepts

        content = """
        This chapter covers gradient descent optimization algorithms.
        We discuss neural network architectures and regularization
        techniques including dropout and batch normalization.
        The gradient is computed using backpropagation. Gradient
        methods are central to optimization in deep learning.
        Neural network layers include convolution and attention mechanisms.
        """
        concepts = extract_concepts(content, "Test Chapter")
        concept_names = [c["name"] for c in concepts]
        assert "Gradient" in concept_names
        assert "Optimization" in concept_names

    def test_discover_chapters_empty(self):
        from mlsyseng_moe.docling_worker import discover_chapters

        with tempfile.TemporaryDirectory() as d:
            chapters = discover_chapters(d)
            assert chapters == []


class TestEmbeddings:
    def test_chunk_text(self):
        from mlsyseng_moe.embeddings import chunk_text

        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_index_and_search(self, temp_chroma):
        from mlsyseng_moe.embeddings import get_collection_stats, index_chapter, search

        num_chunks = index_chapter(
            chapter_id=1,
            chapter_title="Optimization Methods",
            content="Gradient descent is an optimization algorithm used to minimize loss functions. "
            "Stochastic gradient descent uses random samples. "
            "Adam optimizer combines momentum and adaptive learning rates. " * 20,
            chroma_path=temp_chroma,
        )
        assert num_chunks > 0

        results = search("optimization algorithm", n_results=3, chroma_path=temp_chroma)
        assert len(results) > 0
        assert "Optimization" in results[0]["chapter_title"]

        stats = get_collection_stats(temp_chroma)
        assert stats["total_chunks"] == num_chunks


class TestSkillGenerator:
    def test_load_skills_yaml(self):
        from mlsyseng_moe.skill_generator import load_skills_yaml

        config = load_skills_yaml()
        assert config["name"] == "mlsyseng-moe"
        assert "tools" in config
        assert "platforms" in config
        assert len(config["tools"]) >= 7

    def test_generate_skill_md(self):
        from mlsyseng_moe.skill_generator import generate_skill_md, load_skills_yaml

        config = load_skills_yaml()
        md = generate_skill_md(config)
        assert "# MLSysEng MoE" in md
        assert "extract-knowledge" in md
        assert "evolve" in md

    def test_generate_mcp_config(self):
        from mlsyseng_moe.skill_generator import generate_mcp_config, load_skills_yaml

        config = load_skills_yaml()
        mcp_config = generate_mcp_config(config)
        assert mcp_config["command"] == "python"
        assert "-m" in mcp_config["args"]
        assert "mlsyseng_moe.server" in mcp_config["args"]
