"""Tests for MLSysEng MoE loop controller module."""

import pytest

from src.mlsyseng_mcp.database import Database
from src.mlsyseng_mcp.expert_registry import ExpertRegistry
from src.mlsyseng_mcp.loop_controller import LoopController, l2_norm, _encode_state


class TestL2Norm:
    def test_identical_vectors(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_different_vectors(self):
        result = l2_norm([1.0, 0.0], [0.0, 0.0])
        assert abs(result - 1.0) < 1e-10

    def test_known_distance(self):
        result = l2_norm([3.0, 0.0], [0.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            l2_norm([1.0], [1.0, 2.0])


class TestEncodeState:
    def test_basic_encoding(self):
        experts = [{"slug": "a"}, {"slug": "b"}]
        scores = {"a": 0.5, "b": 0.3}
        state = _encode_state(experts, scores, 1)
        assert len(state) == 4  # 2 expert scores + iteration + avg
        assert state[0] == 0.5
        assert state[1] == 0.3
        assert state[2] == 0.01  # iteration/100

    def test_empty_experts(self):
        state = _encode_state([], {}, 0)
        assert len(state) == 2  # iteration + avg


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture
def registry(db, tmp_path):
    return ExpertRegistry(db, str(tmp_path / "experts"))


@pytest.fixture
def setup_experts(db, registry):
    for i, name in enumerate(["ML_Basics", "Deep_Learning", "Feature_Eng"]):
        ch_id = db.upsert_chapter(name, name.lower(), f"/p/{i}", f"content about {name}", 1)
        db.add_concepts(ch_id, [
            {"concept_name": name.replace("_", " ")},
        ])
        registry.create_expert_from_chapter(ch_id)


class TestLoopController:
    def test_run_no_experts(self, db, registry):
        loop = LoopController(db, registry)
        result = loop.run("titanic")
        assert result["status"] == "no_experts"

    def test_run_converges(self, db, registry, setup_experts):
        loop = LoopController(
            db, registry,
            epsilon=100.0,  # Very large epsilon so it converges quickly
            max_iterations=10,
            patience=2,
        )
        result = loop.run("titanic", "classification problem")
        assert result["status"] == "converged"
        assert result["iterations"] > 0
        assert len(result["convergence_history"]) > 0

    def test_run_max_iterations(self, db, registry, setup_experts):
        loop = LoopController(
            db, registry,
            epsilon=0.0000001,  # Very small epsilon, won't converge
            max_iterations=3,
            patience=5,
        )
        result = loop.run("titanic")
        assert result["status"] == "max_iterations_reached"
        assert result["iterations"] == 3

    def test_run_with_callback(self, db, registry, setup_experts):
        iterations_seen = []

        def callback(iteration, info):
            iterations_seen.append(iteration)

        loop = LoopController(db, registry, max_iterations=3, patience=5, epsilon=1e-9)
        loop.run("test", on_iteration=callback)
        assert len(iterations_seen) == 3

    def test_convergence_history(self, db, registry, setup_experts):
        loop = LoopController(db, registry, max_iterations=3, patience=5, epsilon=1e-9)
        loop.run("history_test")

        history = loop.get_convergence_history("history_test")
        assert len(history) == 3
        assert history[0]["iteration"] == 1

    def test_result_has_top_experts(self, db, registry, setup_experts):
        loop = LoopController(db, registry, max_iterations=3, patience=5, epsilon=1e-9)
        result = loop.run("ml basics test", "machine learning basics problem")
        assert "top_experts" in result

    def test_competition_field(self, db, registry, setup_experts):
        loop = LoopController(db, registry, max_iterations=2, patience=5, epsilon=1e-9)
        result = loop.run("titanic")
        assert result["competition"] == "titanic"
