"""Tests for the MLSysEng MoE system core components."""

import json
import math
import os
import tempfile

import pytest

from mlsyseng_mcp.database import Database
from mlsyseng_mcp.loop_controller import LoopController, l2_norm, initial_state_vector
from mlsyseng_mcp.expert_registry import (
    build_expert_from_chapter,
    save_expert_json,
    load_expert_json,
    load_all_experts_from_disk,
    _slug_from_name,
)
from mlsyseng_mcp.docling_worker import extract_concepts


class TestL2Norm:
    def test_identical_vectors(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_distance(self):
        assert math.isclose(l2_norm([0.0, 0.0], [3.0, 4.0]), 5.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            l2_norm([1.0], [1.0, 2.0])


class TestLoopController:
    def test_converges(self):
        lc = LoopController(epsilon=0.01, max_iterations=100, patience=3)
        state = [1.0, 0.0, 0.0, 0.0]
        result = lc.step(state)
        assert not result["should_stop"]

        for i in range(1, 50):
            state = [s * 0.5 for s in state] if any(s > 0.01 for s in state) else state
            result = lc.step(state)
            if result["should_stop"]:
                break

        assert result["converged"]

    def test_max_iterations(self):
        lc = LoopController(epsilon=0.0001, max_iterations=5, patience=10)
        for i in range(6):
            state = [float(i), float(i * 2), 0.0, 0.0]
            result = lc.step(state)
        assert result["should_stop"]
        assert not result["converged"]

    def test_run_method(self):
        lc = LoopController(epsilon=0.01, max_iterations=20, patience=2)

        def step_fn(iteration, current):
            return [c * 0.3 for c in current]

        result = lc.run(step_fn, [1.0, 1.0, 1.0, 1.0])
        assert result["should_stop"]
        assert result["total_iterations"] > 1


class TestDatabase:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)

    def teardown_method(self):
        self.db.close()

    def test_upsert_chapter(self):
        cid = self.db.upsert_chapter("ch01", "Chapter 1", "# Hello", ["gradient descent"])
        assert cid > 0
        chapters = self.db.list_chapters()
        assert len(chapters) == 1
        assert chapters[0]["title"] == "Chapter 1"

    def test_upsert_expert(self):
        expert = {
            "slug": "test_expert",
            "expert_name": "Test Expert",
            "capabilities": ["testing"],
            "skills": ["/path/skill"],
            "strategy": "test",
            "formula": {"objective": "test"},
            "loop_config": {"epsilon": 0.01},
        }
        eid = self.db.upsert_expert(expert)
        assert eid > 0
        experts = self.db.list_experts()
        assert len(experts) == 1
        assert experts[0]["slug"] == "test_expert"

    def test_get_expert(self):
        expert = {
            "slug": "lookup_test",
            "expert_name": "Lookup",
            "capabilities": [],
            "skills": [],
            "strategy": "",
            "formula": {},
            "loop_config": {},
        }
        self.db.upsert_expert(expert)
        found = self.db.get_expert("lookup_test")
        assert found is not None
        assert found["expert_name"] == "Lookup"
        assert self.db.get_expert("nonexistent") is None

    def test_stats(self):
        stats = self.db.get_stats()
        assert stats["chapters"] == 0
        assert stats["experts"] == 0
        assert stats["entries"] == 0

    def test_save_competition_entry(self):
        entry = {
            "competition": "titanic",
            "expert_slug": "test",
            "state_vector": [0.5, 0.5, 0.5, 0.5],
            "iteration": 5,
            "converged": True,
        }
        eid = self.db.save_competition_entry(entry)
        assert eid > 0
        stats = self.db.get_stats()
        assert stats["entries"] == 1


class TestExpertRegistry:
    def test_slug_from_name(self):
        assert _slug_from_name("08_ML Systems") == "08_ml_systems"
        assert _slug_from_name("Deep Learning!") == "deep_learning"

    def test_build_expert_from_chapter(self):
        chapter = {
            "title": "03 Deep Learning",
            "concepts": ["neural network", "deep learning", "gradient descent"],
            "folder_name": "03_Deep_Learning",
            "markdown": "test content",
        }
        expert = build_expert_from_chapter(chapter, chapter_id=1)
        assert expert["slug"] == "03_deep_learning"
        assert expert["chapter_id"] == 1
        assert len(expert["capabilities"]) > 0
        assert expert["formula"]["objective"] == "minimize_validation_loss"

    def test_save_and_load_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert = {
                "expert_name": "Test",
                "slug": "test",
                "capabilities": ["a"],
                "skills": [],
                "strategy": "test",
                "formula": {},
                "loop_config": {},
            }
            path = save_expert_json(expert, tmpdir)
            assert os.path.exists(path)

            loaded = load_expert_json("test", tmpdir)
            assert loaded is not None
            assert loaded["expert_name"] == "Test"

    def test_load_all_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a", "b", "c"]:
                save_expert_json({"expert_name": name, "slug": name}, tmpdir)
            all_exp = load_all_experts_from_disk(tmpdir)
            assert len(all_exp) == 3


class TestConceptExtraction:
    def test_finds_concepts(self):
        text = "We use gradient descent and backpropagation to train a CNN with dropout regularization."
        concepts = extract_concepts(text)
        assert "gradient descent" in concepts
        assert "backpropagation" in concepts
        assert "cnn" in concepts
        assert "dropout" in concepts

    def test_empty_text(self):
        assert extract_concepts("") == []

    def test_case_insensitive(self):
        concepts = extract_concepts("Random Forest and LSTM models")
        assert "random forest" in concepts
        assert "lstm" in concepts


class TestInitialStateVector:
    def test_returns_list(self):
        expert = {"slug": "test", "loop_config": {}}
        state = initial_state_vector(expert)
        assert isinstance(state, list)
        assert len(state) == 4
