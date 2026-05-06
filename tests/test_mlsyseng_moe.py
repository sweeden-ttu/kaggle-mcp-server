"""Tests for the MLSysEng MoE system."""

import json
import os
import tempfile

import pytest

from src.mlsyseng_mcp.database import Database
from src.mlsyseng_mcp.docling_worker import extract_concepts, discover_chapters
from src.mlsyseng_mcp.expert_registry import ExpertRegistry
from src.mlsyseng_mcp.loop_controller import LoopController, LoopState, l2_norm


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    yield db
    os.unlink(db_path)


@pytest.fixture
def sample_concepts():
    return [
        {"concept": "gradient descent", "description": "Optimization method",
         "category": "optimization"},
        {"concept": "neural network", "description": "Deep learning model",
         "category": "model_architecture"},
        {"concept": "dropout", "description": "Regularization",
         "category": "regularization"},
    ]


class TestDatabase:
    def test_upsert_chapter(self, tmp_db):
        ch_id = tmp_db.upsert_chapter(1, "Intro", "/path", "content here", 5)
        assert ch_id > 0
        chapter = tmp_db.get_chapter(1)
        assert chapter["title"] == "Intro"
        assert chapter["word_count"] == 2

    def test_upsert_chapter_idempotent(self, tmp_db):
        id1 = tmp_db.upsert_chapter(1, "V1", "/p", "old", 1)
        id2 = tmp_db.upsert_chapter(1, "V2", "/p", "new content", 2)
        assert id1 == id2
        chapter = tmp_db.get_chapter(1)
        assert chapter["title"] == "V2"

    def test_add_concepts(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "Ch1", "/p", "text", 1)
        tmp_db.add_concepts(ch_id, sample_concepts)
        concepts = tmp_db.get_concepts(ch_id)
        assert len(concepts) == 3

    def test_upsert_expert(self, tmp_db):
        exp_id = tmp_db.upsert_expert(
            slug="01_intro", expert_name="01_Intro",
            chapter_id=None, capabilities=["cap1"],
            skills=["/skills/a"], strategy="Baseline → Submit",
            formula={"objective": "min_loss"},
            loop_config={"epsilon": 0.001},
        )
        assert exp_id > 0
        expert = tmp_db.get_expert("01_intro")
        assert expert["expert_name"] == "01_Intro"
        assert expert["capabilities"] == ["cap1"]

    def test_get_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["chapters_indexed"] == 0
        assert stats["experts_registered"] == 0

    def test_extraction_log(self, tmp_db):
        tmp_db.log_extraction(1, "started")
        tmp_db.log_extraction(1, "completed")
        status = tmp_db.get_extraction_status()
        assert len(status) >= 1


class TestDoclingWorker:
    def test_extract_concepts_from_text(self):
        text = ("We apply gradient descent with a specific learning rate "
                "to train a neural network with dropout regularization.")
        concepts = extract_concepts(text)
        names = {c["concept"] for c in concepts}
        assert "gradient descent" in names
        assert "neural network" in names
        assert "dropout" in names
        assert "learning rate" in names

    def test_extract_concepts_empty(self):
        assert extract_concepts("") == []
        assert extract_concepts("nothing relevant here") == []

    def test_discover_chapters_missing_path(self):
        chapters = discover_chapters("/nonexistent/path")
        assert chapters == []


class TestExpertRegistry:
    def test_register_from_chapter(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "Test", "/p", "content", 1)
        reg = ExpertRegistry(tmp_db)
        expert = reg.register_from_chapter(1, "Test", sample_concepts, ch_id)
        assert expert["slug"] == "01_test"
        assert len(expert["capabilities"]) > 0
        assert len(expert["skills"]) > 0

    def test_list_experts(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "Ch1", "/p", "c", 1)
        reg = ExpertRegistry(tmp_db)
        reg.register_from_chapter(1, "Ch1", sample_concepts, ch_id)
        experts = reg.list_experts()
        assert len(experts) == 1

    def test_query_expert(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "Ch1", "/p", "content text", 1)
        tmp_db.add_concepts(ch_id, sample_concepts)
        reg = ExpertRegistry(tmp_db)
        reg.register_from_chapter(1, "Ch1", sample_concepts, ch_id)
        result = reg.query_expert("01_ch1", "What is gradient descent?")
        assert "expert" in result
        assert result["expert"]["slug"] == "01_ch1"

    def test_build_entry(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "ML", "/p", "c", 1)
        reg = ExpertRegistry(tmp_db)
        reg.register_from_chapter(1, "ML", sample_concepts, ch_id)
        entry = reg.build_entry("titanic")
        assert entry["competition"] == "titanic"
        assert len(entry["selected_experts"]) > 0

    def test_export_expert_json(self, tmp_db, sample_concepts):
        ch_id = tmp_db.upsert_chapter(1, "Ch1", "/p", "c", 1)
        reg = ExpertRegistry(tmp_db)
        reg.register_from_chapter(1, "Ch1", sample_concepts, ch_id)
        exported = reg.export_expert_json("01_ch1")
        assert exported is not None
        parsed = json.loads(exported)
        assert parsed["slug"] == "01_ch1"


class TestLoopController:
    def test_l2_norm(self):
        assert l2_norm([0, 0], [0, 0]) == 0.0
        assert abs(l2_norm([1, 0], [0, 0]) - 1.0) < 1e-10
        assert abs(l2_norm([3, 4], [0, 0]) - 5.0) < 1e-10

    def test_l2_norm_mismatch(self):
        with pytest.raises(ValueError):
            l2_norm([1, 2], [1])

    def test_convergence_check(self):
        ctrl = LoopController(epsilon=0.1)
        s1 = LoopState(0, metrics={"a": 1.0})
        s2 = LoopState(1, metrics={"a": 1.05})
        assert ctrl.check_convergence(s2, s1) is True

    def test_convergence_not_met(self):
        ctrl = LoopController(epsilon=0.001)
        s1 = LoopState(0, metrics={"a": 1.0})
        s2 = LoopState(1, metrics={"a": 2.0})
        assert ctrl.check_convergence(s2, s1) is False

    def test_run_converges(self):
        ctrl = LoopController(epsilon=0.01, max_iterations=20, patience=2)

        def step(i, prev):
            val = 1.0 / (1 + i)
            return LoopState(i, metrics={"loss": val})

        result = ctrl.run(step)
        assert result.iterations <= 20

    def test_run_max_iterations(self):
        ctrl = LoopController(epsilon=0.0001, max_iterations=3, patience=5)

        def step(i, prev):
            return LoopState(i, metrics={"loss": float(i)})

        result = ctrl.run(step)
        assert result.converged is False
        assert result.iterations == 3

    def test_from_config(self):
        config = {"epsilon": 0.01, "max_iterations": 5, "patience": 2}
        ctrl = LoopController.from_config(config)
        assert ctrl.epsilon == 0.01
        assert ctrl.max_iterations == 5

    def test_get_summary(self):
        ctrl = LoopController(epsilon=0.1, max_iterations=3, patience=1)

        def step(i, prev):
            return LoopState(i, metrics={"x": 1.0 + 0.001 * i})

        ctrl.run(step)
        summary = ctrl.get_summary()
        assert "total_iterations" in summary
        assert "delta_history" in summary
