"""Tests for mlsyseng_moe.loop_controller module."""

import math

import pytest

from src.mlsyseng_moe.database import Database
from src.mlsyseng_moe.loop_controller import LoopController, l2_norm


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    d = Database(db_path=db_path)
    d.initialize()
    yield d
    d.close()


class TestL2Norm:
    def test_identical_vectors(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_distance(self):
        result = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_single_element(self):
        assert abs(l2_norm([5.0], [2.0]) - 3.0) < 1e-10

    def test_different_lengths(self):
        result = l2_norm([1.0], [1.0, 2.0])
        assert abs(result - 2.0) < 1e-10

    def test_zero_vectors(self):
        assert l2_norm([0.0, 0.0], [0.0, 0.0]) == 0.0


class TestLoopController:
    def test_no_experts(self, db):
        loop = LoopController(db)
        result = loop.run("test_comp", [])
        assert not result["converged"]
        assert result["iterations"] == 0

    def test_converges_with_default_step(self, db):
        experts = [
            {"expert_name": "e1", "concepts": ["a"], "skill_names": ["s1"]},
            {"expert_name": "e2", "concepts": ["b"], "skill_names": ["s2"]},
        ]
        loop = LoopController(db, epsilon=0.01, max_iterations=50, patience=3)
        result = loop.run("test_conv", experts)
        assert result["converged"]
        assert result["iterations"] > 0
        assert len(result["history"]) == result["iterations"]

    def test_respects_max_iterations(self, db):
        def divergent_step(iteration, state, experts):
            return [v + 1.0 for v in state]

        experts = [{"expert_name": "e1", "concepts": [], "skill_names": []}]
        loop = LoopController(db, max_iterations=5, patience=3)
        result = loop.run("test_max", experts, step_fn=divergent_step)
        assert not result["converged"]
        assert result["iterations"] == 5

    def test_custom_step_function(self, db):
        call_count = [0]

        def custom_step(iteration, state, experts):
            call_count[0] += 1
            return state

        experts = [{"expert_name": "e1", "concepts": [], "skill_names": []}]
        loop = LoopController(db, epsilon=0.01, patience=2)
        result = loop.run("test_custom", experts, step_fn=custom_step)
        assert result["converged"]
        assert call_count[0] == 2  # patience=2 iterations needed

    def test_patience_resets_on_divergence(self, db):
        iteration_counter = [0]

        def wobbling_step(iteration, state, experts):
            iteration_counter[0] += 1
            if iteration_counter[0] % 3 == 0:
                return [v + 1.0 for v in state]
            return state

        experts = [{"expert_name": "e1", "concepts": [], "skill_names": []}]
        loop = LoopController(db, epsilon=0.01, max_iterations=15, patience=3)
        result = loop.run("test_patience", experts, step_fn=wobbling_step)
        assert not result["converged"] or result["iterations"] > 3

    def test_saves_loop_states_to_db(self, db):
        experts = [{"expert_name": "e1", "concepts": ["a"], "skill_names": ["s1"]}]
        loop = LoopController(db, max_iterations=3, patience=5)
        loop.run("test_save", experts)
        states = db.get_loop_states("test_save")
        assert len(states) == 3

    def test_initial_state(self, db):
        experts = [{"expert_name": "e1", "concepts": ["a"], "skill_names": ["s1"]}]
        loop = LoopController(db, max_iterations=2, patience=5)
        result = loop.run("test_init", experts, initial_state=[1.0, 2.0, 3.0])
        assert result["iterations"] == 2

    def test_get_loop_history(self, db):
        experts = [{"expert_name": "e1", "concepts": ["a"], "skill_names": ["s1"]}]
        loop = LoopController(db, max_iterations=3, patience=5)
        loop.run("test_history", experts)
        history = loop.get_loop_history("test_history")
        assert len(history) == 3
