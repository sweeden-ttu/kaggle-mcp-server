"""Tests for MLSysEng MoE loop controller module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlsyseng_mcp.database import MLSysEngDatabase
from mlsyseng_mcp.loop_controller import LoopController, _l2_distance, _l2_norm


class TestL2Math:
    def test_l2_norm_zero(self):
        assert _l2_norm([0.0, 0.0, 0.0]) == 0.0

    def test_l2_norm_unit(self):
        assert abs(_l2_norm([1.0, 0.0, 0.0]) - 1.0) < 1e-9

    def test_l2_norm_3_4_5(self):
        assert abs(_l2_norm([3.0, 4.0]) - 5.0) < 1e-9

    def test_l2_distance_same(self):
        v = [1.0, 2.0, 3.0]
        assert _l2_distance(v, v) == 0.0

    def test_l2_distance_known(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        assert abs(_l2_distance(a, b) - 5.0) < 1e-9

    def test_l2_distance_mismatch(self):
        with pytest.raises(ValueError):
            _l2_distance([1.0], [1.0, 2.0])


class TestLoopController:
    @pytest.fixture
    def db(self, tmp_path):
        return MLSysEngDatabase(db_path=str(tmp_path / "test.db"))

    def test_converges_immediately(self, db):
        loop = LoopController(db, epsilon=0.1, max_iterations=10, patience=1)

        def constant_step(iteration, prev_state):
            return [0.5, 0.5, 0.5]

        result = loop.run("test_comp", constant_step, initial_state=[0.5, 0.5, 0.5])
        assert result["status"] == "converged"
        assert result["iterations"] <= 2

    def test_max_iterations_reached(self, db):
        loop = LoopController(db, epsilon=0.0001, max_iterations=3, patience=3)
        call_count = 0

        def diverging_step(iteration, prev_state):
            nonlocal call_count
            call_count += 1
            if prev_state is None:
                return [0.1, 0.2]
            return [v + 0.5 for v in prev_state]

        result = loop.run("test_comp", diverging_step)
        assert result["status"] == "max_iterations_reached"
        assert result["iterations"] == 3

    def test_patience_required(self, db):
        loop = LoopController(db, epsilon=0.01, max_iterations=20, patience=3)

        def slowly_converging(iteration, prev_state):
            if prev_state is None:
                return [1.0]
            if iteration < 5:
                return [prev_state[0] + 0.1]
            return [prev_state[0] + 0.001]

        result = loop.run("test_comp", slowly_converging)
        assert result["status"] == "converged"
        assert result["iterations"] >= 5

    def test_history_saved(self, db):
        loop = LoopController(db, epsilon=0.1, max_iterations=5, patience=5)

        def step(iteration, prev_state):
            return [float(iteration)]

        loop.run("hist_comp", step)
        history = loop.get_history("hist_comp")
        assert len(history) == 5

    def test_default_step_fn(self, db):
        experts = [{"expert_name": "test", "formula": {"metrics": ["accuracy"]}}]
        step_fn = LoopController.default_step_fn(experts)

        state0 = step_fn(0, None)
        assert len(state0) > 0
        assert all(0.0 <= v <= 1.0 for v in state0)

        state1 = step_fn(1, state0)
        assert len(state1) == len(state0)
