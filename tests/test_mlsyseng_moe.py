"""Tests for MLSysEng MoE system."""

import json
import os
import tempfile

import numpy as np
import pytest

from mlsyseng_mcp.database import (
    get_all_chapters,
    get_all_experts,
    get_chapter_concepts,
    get_extraction_status,
    get_expert_by_slug,
    get_stats,
    init_db,
    insert_chapter,
    insert_concept,
    insert_expert,
    log_extraction,
)
from mlsyseng_mcp.docling_worker import extract_concepts
from mlsyseng_mcp.expert_registry import (
    create_expert_from_chapter,
    list_experts,
    query_expert,
    register_all_experts,
    slugify,
)
from mlsyseng_mcp.loop_controller import LoopController, LoopState


@pytest.fixture
def db_path():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    init_db(path)
    yield path
    os.unlink(path)


class TestDatabase:
    def test_init_db(self, db_path):
        stats = get_stats(db_path)
        assert stats["chapters"] == 0
        assert stats["concepts"] == 0
        assert stats["experts"] == 0

    def test_insert_chapter(self, db_path):
        ch_id = insert_chapter(1, "Test Chapter", "/path/test", "Content here", db_path)
        assert ch_id > 0
        chapters = get_all_chapters(db_path)
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Test Chapter"
        assert chapters[0]["chapter_number"] == 1

    def test_insert_concept(self, db_path):
        ch_id = insert_chapter(1, "Test", "/p", "text", db_path)
        c_id = insert_concept(ch_id, "neural network", "A computational model", "deep_learning", 0.9, db_path)
        assert c_id > 0
        concepts = get_chapter_concepts(ch_id, db_path)
        assert len(concepts) == 1
        assert concepts[0]["concept_name"] == "neural network"

    def test_insert_expert(self, db_path):
        expert_data = {
            "expert_name": "01_Test_Expert",
            "slug": "01_test_expert",
            "chapter_id": None,
            "capabilities": ["cap1", "cap2"],
            "skills": ["/path/skill1"],
            "strategy": "Baseline → Submit",
            "formula": {"objective": "minimize_loss"},
            "loop_config": {"epsilon": 0.001, "max_iterations": 10},
        }
        eid = insert_expert(expert_data, db_path)
        assert eid > 0
        expert = get_expert_by_slug("01_test_expert", db_path)
        assert expert is not None
        assert expert["expert_name"] == "01_Test_Expert"
        assert expert["capabilities"] == ["cap1", "cap2"]

    def test_extraction_status(self, db_path):
        status = get_extraction_status(db_path)
        assert "chapters_indexed" in status
        assert status["chapters_indexed"] == 0

    def test_log_extraction(self, db_path):
        ch_id = insert_chapter(1, "Test", "/p", "text", db_path)
        log_id = log_extraction(ch_id, "completed", "Test message", db_path)
        assert log_id > 0


class TestDoclingWorker:
    def test_extract_concepts(self):
        content = "This covers supervised learning and neural network architecture with gradient descent."
        concepts = extract_concepts(content)
        names = [c["concept_name"] for c in concepts]
        assert "supervised learning" in names
        assert "neural network" in names
        assert "gradient descent" in names

    def test_extract_concepts_empty(self):
        concepts = extract_concepts("")
        assert concepts == []

    def test_concept_categories(self):
        content = "deep learning with convolutional neural network and regularization"
        concepts = extract_concepts(content)
        categories = {c["concept_name"]: c["category"] for c in concepts}
        assert categories.get("deep learning") == "deep_learning"
        assert categories.get("regularization") == "regularization"


class TestExpertRegistry:
    def test_slugify(self):
        assert slugify("08 ML Systems") == "08_ml_systems"
        assert slugify("Hello-World!") == "hello_world"

    def test_create_expert_from_chapter(self, db_path):
        ch_id = insert_chapter(1, "ML Basics", "/p", "content", db_path)
        concepts = [
            {"concept_name": "neural network", "category": "deep_learning"},
            {"concept_name": "gradient descent", "category": "optimization"},
        ]
        expert = create_expert_from_chapter(1, "ML Basics", ch_id, concepts, db_path)
        assert expert["expert_name"] == "01_ML_Basics"
        assert expert["slug"] == "01_ml_basics"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0
        assert "minimize_validation_loss" in expert["formula"]["objective"]

    def test_query_expert(self, db_path):
        ch_id = insert_chapter(1, "Test", "/p", "content", db_path)
        concepts = [{"concept_name": "supervised learning", "category": "methodology"}]
        create_expert_from_chapter(1, "Test", ch_id, concepts, db_path)
        result = query_expert("01_test", "How to train?", db_path)
        assert "expert" in result
        assert "response" in result

    def test_query_nonexistent_expert(self, db_path):
        result = query_expert("nonexistent", "question", db_path)
        assert "error" in result


class TestLoopController:
    def test_initialization(self):
        controller = LoopController(epsilon=0.01, max_iterations=5, patience=2)
        assert controller.epsilon == 0.01
        assert controller.max_iterations == 5
        assert controller.patience == 2
        assert controller.current_iteration == 0
        assert not controller.converged

    def test_single_step(self):
        controller = LoopController()
        result = controller.step([1.0, 2.0, 3.0], score=0.5)
        assert result["iteration"] == 0
        assert result["l2_norm"] is None
        assert not result["should_stop"]

    def test_convergence(self):
        controller = LoopController(epsilon=0.1, patience=2)
        state = [1.0, 2.0, 3.0]
        controller.step(state, score=0.5)

        for _ in range(5):
            state = [s + 0.01 for s in state]
            result = controller.step(state, score=0.4)
            if result["should_stop"]:
                break

        assert controller.converged

    def test_max_iterations(self):
        controller = LoopController(epsilon=0.0001, max_iterations=3, patience=10)
        for i in range(5):
            state = np.random.randn(5).tolist()
            result = controller.step(state, score=float(i))
            if result["should_stop"]:
                break
        assert result["at_max_iterations"]

    def test_from_config(self):
        config = {"epsilon": 0.005, "max_iterations": 20, "patience": 5, "objective": "maximize_auc"}
        controller = LoopController.from_config(config)
        assert controller.epsilon == 0.005
        assert controller.max_iterations == 20
        assert controller.patience == 5
        assert controller.objective == "maximize_auc"

    def test_get_history(self):
        controller = LoopController()
        controller.step([1.0], score=0.1)
        controller.step([1.1], score=0.2)
        history = controller.get_history()
        assert len(history) == 2
        assert history[0]["score"] == 0.1
        assert history[1]["score"] == 0.2

    def test_reset(self):
        controller = LoopController()
        controller.step([1.0], score=0.1)
        controller.reset()
        assert controller.current_iteration == 0
        assert len(controller.history) == 0

    def test_loop_state_serialization(self):
        state = LoopState(
            iteration=5,
            state_vector=[1.0, 2.0],
            score=0.85,
            metrics={"accuracy": 0.9},
        )
        d = state.to_dict()
        restored = LoopState.from_dict(d)
        assert restored.iteration == 5
        assert restored.state_vector == [1.0, 2.0]
        assert restored.score == 0.85


class TestMCPServer:
    def test_server_tools_exist(self):
        from mlsyseng_mcp.server import (
            ask_expert,
            build_entry,
            evolve,
            extract_knowledge,
            get_extraction_progress,
            get_system_stats,
            list_all_experts,
            run_rdagent,
            search_concepts,
        )
        assert callable(extract_knowledge)
        assert callable(evolve)
        assert callable(search_concepts)
        assert callable(list_all_experts)
        assert callable(build_entry)
        assert callable(run_rdagent)
        assert callable(ask_expert)
        assert callable(get_extraction_progress)
        assert callable(get_system_stats)

    def test_get_system_stats(self):
        from mlsyseng_mcp.server import get_system_stats

        result = json.loads(get_system_stats())
        assert "chapters" in result
        assert "experts" in result

    def test_list_all_experts(self):
        from mlsyseng_mcp.server import list_all_experts

        result = json.loads(list_all_experts())
        assert "experts" in result
        assert "total" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
