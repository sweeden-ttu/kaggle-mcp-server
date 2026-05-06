"""Tests for mlsyseng_mcp.loop_controller module."""

import pytest

from mlsyseng_mcp.loop_controller import ConvergenceState, l2_norm, run_convergence_loop


class TestL2Norm:
    def test_identical_vectors(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_distance(self):
        result = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_single_element(self):
        assert abs(l2_norm([5.0], [2.0]) - 3.0) < 1e-10

    def test_mismatched_lengths(self):
        result = l2_norm([1.0, 2.0, 3.0], [1.0, 2.0])
        assert result == 0.0


class TestConvergenceState:
    def test_initial_state(self):
        state = ConvergenceState("titanic", epsilon=0.01, max_iterations=5)
        assert state.should_continue()
        assert state.iteration == 0
        assert not state.converged

    def test_max_iterations_exit(self):
        state = ConvergenceState("titanic", epsilon=0.0001, max_iterations=3)
        for i in range(3):
            state.update([float(i), float(i) * 0.5], {"acc": 0.5 + i * 0.1})
        assert state.converged
        assert "Max iterations" in state.exit_reason

    def test_convergence_with_patience(self):
        state = ConvergenceState("titanic", epsilon=0.1, max_iterations=100, patience=2)
        state.update([1.0, 1.0], {"acc": 0.8})
        state.update([1.01, 1.01], {"acc": 0.81})
        state.update([1.011, 1.011], {"acc": 0.811})
        assert state.converged
        assert "Converged" in state.exit_reason

    def test_summary(self):
        state = ConvergenceState("titanic", max_iterations=2)
        state.update([1.0], {"m": 0.5})
        state.update([2.0], {"m": 0.8})
        summary = state.summary()
        assert summary["competition"] == "titanic"
        assert summary["total_iterations"] == 2
        assert len(summary["history"]) == 2


class TestRunConvergenceLoop:
    def _make_experts(self, n=2):
        return [{
            "slug": f"expert_{i}",
            "expert_name": f"Expert {i}",
            "capabilities": ["cap"],
            "skills": ["/skill"],
            "strategy": "test",
            "formula": {"objective": "min_loss", "metrics": ["acc"]},
            "loop_config": {"epsilon": 0.001, "max_iterations": 5, "patience": 3},
        } for i in range(n)]

    def test_empty_experts(self):
        result = run_convergence_loop("titanic", [])
        assert result["status"] == "error"

    def test_runs_to_completion(self, tmp_path, monkeypatch):
        import os
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        from mlsyseng_mcp import database as db
        db.init_db()

        experts = self._make_experts()
        result = run_convergence_loop(
            "titanic", experts,
            epsilon=0.001, max_iterations=5, patience=3,
        )
        assert result["status"] in ("converged", "max_iterations")
        assert "experts_used" in result
        assert len(result["experts_used"]) == 2

    def test_with_callback(self, tmp_path, monkeypatch):
        import os
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        from mlsyseng_mcp import database as db
        db.init_db()

        calls = []
        def cb(expert, iteration, sv):
            calls.append((expert["slug"], iteration))
            return sv

        experts = self._make_experts(1)
        run_convergence_loop("titanic", experts, max_iterations=3,
                             step_callback=cb)
        assert len(calls) == 3
