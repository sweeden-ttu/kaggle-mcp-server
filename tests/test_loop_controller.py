"""Tests for mlsyseng_mcp.loop_controller module."""

import pytest

from mlsyseng_mcp.loop_controller import (
    LoopConfig,
    LoopController,
    LoopState,
    check_convergence,
    l2_norm_diff,
)


class TestL2NormDiff:
    def test_identical_vectors(self):
        assert l2_norm_diff([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_simple_diff(self):
        diff = l2_norm_diff([1.0, 0.0], [0.0, 0.0])
        assert abs(diff - 1.0) < 1e-10

    def test_different_lengths(self):
        diff = l2_norm_diff([1.0], [1.0, 1.0])
        assert abs(diff - 1.0) < 1e-10

    def test_empty_vectors(self):
        assert l2_norm_diff([], []) == 0.0


class TestCheckConvergence:
    def test_converged(self):
        assert check_convergence([1.0, 2.0], [1.0, 2.0], epsilon=0.01)

    def test_not_converged(self):
        assert not check_convergence([1.0, 2.0], [0.0, 0.0], epsilon=0.01)

    def test_empty_previous(self):
        assert not check_convergence([1.0], [], epsilon=0.01)

    def test_near_convergence(self):
        assert check_convergence([1.0, 2.0], [1.0005, 2.0005], epsilon=0.001)


class TestLoopConfig:
    def test_defaults(self):
        config = LoopConfig()
        assert config.epsilon == 0.001
        assert config.max_iterations == 10
        assert config.patience == 3

    def test_from_dict(self):
        config = LoopConfig.from_dict({
            "epsilon": 0.01,
            "max_iterations": 5,
            "patience": 2,
        })
        assert config.epsilon == 0.01
        assert config.max_iterations == 5


class TestLoopController:
    def test_start_loop(self):
        lc = LoopController()
        state = lc.start_loop("titanic")
        assert state.competition == "titanic"
        assert state.iteration == 0

    def test_step(self):
        lc = LoopController()
        lc.start_loop("titanic")
        result = lc.step("titanic", [0.5, 0.6, 0.7])
        assert result["iteration"] == 1
        assert "l2_diff" in result
        assert "should_stop" in result

    def test_convergence_detection(self):
        config = LoopConfig(epsilon=0.01, patience=2, max_iterations=20)
        lc = LoopController(config=config)
        lc.start_loop("test")

        lc.step("test", [1.0, 2.0, 3.0])

        result = lc.step("test", [1.0, 2.0, 3.0])
        assert result["is_converging"]

        result = lc.step("test", [1.0, 2.0, 3.0])
        assert result["converged"]
        assert result["should_stop"]

    def test_max_iterations(self):
        config = LoopConfig(max_iterations=3, epsilon=1e-10)
        lc = LoopController(config=config)

        state = [0.0]
        lc.start_loop("test")
        for i in range(5):
            result = lc.step("test", [float(i)])
            if result["should_stop"]:
                break

        assert result["should_stop"]
        assert "Max iterations" in result["reason"]

    def test_run_loop(self):
        config = LoopConfig(epsilon=0.01, max_iterations=20, patience=3)
        lc = LoopController(config=config)

        call_count = [0]

        def step_fn(iteration, current):
            call_count[0] += 1
            decay = 0.5 ** iteration
            return [c + decay * (1.0 - c) for c in current]

        summary = lc.run_loop("test", step_fn, initial_state=[0.0, 0.0])
        assert summary["total_iterations"] > 0
        assert call_count[0] > 0

    def test_get_loop_summary(self):
        lc = LoopController()
        lc.start_loop("test")
        lc.step("test", [1.0])
        summary = lc.get_loop_summary("test")
        assert summary["competition"] == "test"
        assert summary["total_iterations"] == 1
        assert len(summary["history"]) == 1

    def test_reset_loop(self):
        lc = LoopController()
        lc.start_loop("test")
        lc.step("test", [1.0])
        lc.reset_loop("test")
        assert lc.get_loop("test") is None

    def test_nonexistent_loop_summary(self):
        lc = LoopController()
        summary = lc.get_loop_summary("nonexistent")
        assert "error" in summary
