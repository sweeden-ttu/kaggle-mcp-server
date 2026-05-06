"""Tests for the MLSysEng MoE system."""

import os
import json
import tempfile
import pytest
import numpy as np

from src.mlsyseng_mcp.database import (
    init_db,
    store_chapter,
    store_concepts,
    store_expert,
    get_all_experts,
    get_expert_by_slug,
    get_chapters,
    get_chapter_content,
    get_stats,
    store_competition_entry,
)
from src.mlsyseng_mcp.docling_worker import extract_concepts, discover_chapters
from src.mlsyseng_mcp.expert_registry import (
    create_expert_from_chapter,
    list_experts,
    get_expert,
    get_experts_for_competition,
)
from src.mlsyseng_mcp.loop_controller import (
    ConvergenceLoop,
    LoopConfig,
    LoopState,
    run_convergence_loop,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    os.unlink(db_path)


class TestDatabase:
    def test_init_db(self, temp_db):
        stats = get_stats(temp_db)
        assert stats["chapters_indexed"] == 0
        assert stats["concepts_extracted"] == 0
        assert stats["experts_registered"] == 0

    def test_store_and_retrieve_chapter(self, temp_db):
        chapter_id = store_chapter(
            chapter_number=1,
            title="Introduction to ML",
            source_path="/tmp/ch1",
            markdown_content="# Introduction\n\nThis is about machine learning.",
            db_path=temp_db,
        )
        assert chapter_id > 0

        content = get_chapter_content(1, temp_db)
        assert "machine learning" in content

        chapters = get_chapters(temp_db)
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Introduction to ML"

    def test_store_concepts(self, temp_db):
        chapter_id = store_chapter(1, "Test", "/tmp", "Content", db_path=temp_db)
        concepts = [
            {"name": "Gradient Descent", "description": "Optimization method", "category": "optimization"},
            {"name": "Neural Network", "description": "Deep learning model", "category": "model"},
        ]
        store_concepts(chapter_id, concepts, temp_db)
        stats = get_stats(temp_db)
        assert stats["concepts_extracted"] == 2

    def test_store_expert(self, temp_db):
        expert_data = {
            "expert_name": "01_Introduction",
            "slug": "01_introduction",
            "chapter_id": 1,
            "capabilities": ["Build baseline models"],
            "skills": ["/path/to/skill"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
        }
        store_expert(expert_data, temp_db)

        experts = get_all_experts(temp_db)
        assert len(experts) == 1
        assert experts[0]["slug"] == "01_introduction"
        assert isinstance(experts[0]["capabilities"], list)

    def test_get_expert_by_slug(self, temp_db):
        expert_data = {
            "expert_name": "08_ML_Systems",
            "slug": "08_ml_systems",
            "capabilities": ["System design"],
            "skills": [],
            "strategy": "Baseline → Submit",
            "formula": {},
            "loop_config": {},
        }
        store_expert(expert_data, temp_db)

        expert = get_expert_by_slug("08_ml_systems", temp_db)
        assert expert is not None
        assert expert["expert_name"] == "08_ML_Systems"

        missing = get_expert_by_slug("nonexistent", temp_db)
        assert missing is None

    def test_competition_entry(self, temp_db):
        entry_id = store_competition_entry(
            competition_name="titanic",
            experts_used=["expert_1", "expert_2"],
            skills_applied=["skill_a"],
            state_history=[{"iteration": 0, "values": [0, 0]}],
            converged=True,
            final_score=0.85,
            db_path=temp_db,
        )
        assert entry_id > 0
        stats = get_stats(temp_db)
        assert stats["competition_entries"] == 1


class TestDoclingWorker:
    def test_extract_concepts_finds_ml_keywords(self):
        content = """
        In this chapter we discuss gradient descent and backpropagation.
        Neural networks use cross-entropy loss and regularization to prevent overfitting.
        Ensemble methods combine multiple models.
        """
        concepts = extract_concepts(content)
        names = {c["name"].lower() for c in concepts}
        assert "gradient descent" in names
        assert "backpropagation" in names
        assert "neural network" in names

    def test_extract_concepts_finds_headings(self):
        content = """
        # Deep Learning Fundamentals

        ## Convolutional Neural Networks

        Content here about CNNs.
        """
        concepts = extract_concepts(content)
        names = [c["name"] for c in concepts]
        assert any("Deep Learning" in n for n in names)

    def test_discover_chapters_missing_path(self):
        chapters = discover_chapters("/nonexistent/path")
        assert chapters == []


class TestExpertRegistry:
    def test_create_expert_from_chapter(self, temp_db):
        concepts = [
            {"name": "Neural Network", "description": "Deep model"},
            {"name": "Gradient Descent", "description": "Optimizer"},
            {"name": "Cross-Validation", "description": "Evaluation"},
        ]
        expert = create_expert_from_chapter(
            chapter_number=5,
            title="Deep Learning",
            concepts=concepts,
            db_path=temp_db,
        )
        assert expert["slug"] == "05_deep_learning"
        assert len(expert["capabilities"]) > 0
        assert expert["strategy"] == "Baseline → EDA → Feature Engineering → Model Selection → Submit"
        assert "objective" in expert["formula"]

    def test_list_experts_empty(self, temp_db):
        experts = list_experts(temp_db)
        assert experts == []

    def test_get_experts_for_competition(self, temp_db):
        expert_data = {
            "expert_name": "03_Feature_Engineering",
            "slug": "03_feature_engineering",
            "chapter_id": 3,
            "capabilities": ["Create features"],
            "skills": [],
            "strategy": "Baseline → Submit",
            "formula": {},
            "loop_config": {},
        }
        store_expert(expert_data, temp_db)

        relevant = [{"chapter_number": 3, "relevance_score": 0.9}]
        selected = get_experts_for_competition("titanic", relevant, temp_db)
        assert len(selected) == 1
        assert selected[0]["slug"] == "03_feature_engineering"


class TestLoopController:
    def test_convergence_loop_basic(self):
        config = LoopConfig(epsilon=0.1, max_iterations=20, patience=2)
        loop = ConvergenceLoop(config)
        loop.initialize_state(5)

        scale = 1.0
        for i in range(20):
            scale *= 0.3
            new_values = loop.current_state.values + np.random.randn(5) * scale
            result = loop.step(new_values)
            if result["converged"]:
                break

        assert loop.converged or loop.current_iteration == 20

    def test_convergence_loop_converges(self):
        config = LoopConfig(epsilon=0.01, max_iterations=50, patience=3)
        loop = ConvergenceLoop(config)
        loop.initialize_state(3)

        target = np.array([1.0, 2.0, 3.0])
        for i in range(50):
            current = loop.current_state.values
            new_values = current + (target - current) * 0.5
            result = loop.step(new_values)
            if result["converged"]:
                break

        assert loop.converged
        assert loop.convergence_iteration is not None

    def test_max_iterations_reached(self):
        config = LoopConfig(epsilon=0.0001, max_iterations=3, patience=2)
        loop = ConvergenceLoop(config)
        loop.initialize_state(5)

        for i in range(5):
            new_values = np.random.randn(5) * 10
            result = loop.step(new_values)
            if result["status"] == "max_iterations_reached":
                break

        assert not loop.converged

    def test_run_convergence_loop(self):
        experts = [
            {"slug": "expert_1", "expert_name": "Expert 1"},
            {"slug": "expert_2", "expert_name": "Expert 2"},
        ]
        config = LoopConfig(epsilon=0.1, max_iterations=20, patience=2)
        report = run_convergence_loop(experts=experts, config=config)

        assert "total_iterations" in report
        assert "converged" in report
        assert "state_history" in report
        assert "experts_used" in report
        assert report["experts_used"] == ["Expert 1", "Expert 2"]

    def test_state_history_serializable(self):
        config = LoopConfig(epsilon=0.01, max_iterations=5, patience=2)
        loop = ConvergenceLoop(config)
        loop.initialize_state(3)
        loop.step(np.array([0.1, 0.2, 0.3]))

        history = loop.get_state_history()
        serialized = json.dumps(history)
        assert serialized is not None

    def test_convergence_report(self):
        config = LoopConfig(epsilon=0.5, max_iterations=10, patience=2)
        loop = ConvergenceLoop(config)
        loop.initialize_state(3)
        loop.step(np.array([0.01, 0.01, 0.01]))
        loop.step(np.array([0.011, 0.011, 0.011]))

        report = loop.get_convergence_report()
        assert "deltas" in report
        assert len(report["deltas"]) == 2
        assert report["epsilon"] == 0.5


class TestSkillGenerator:
    def test_skills_yaml_exists(self):
        assert os.path.exists("skills.yaml")

    def test_skills_yaml_valid(self):
        import yaml
        with open("skills.yaml") as f:
            config = yaml.safe_load(f)
        assert config["name"] == "mlsyseng-moe"
        assert "tools" in config
        assert len(config["tools"]) > 0
        assert "platforms" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
