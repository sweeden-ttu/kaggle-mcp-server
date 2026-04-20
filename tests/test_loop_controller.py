"""Tests for mlsyseng_moe.loop_controller module."""

import math
from typing import Optional

import pytest

from mlsyseng_moe.loop_controller import (
    ConvergenceResult,
    LoopConfig,
    LoopController,
    StateVector,
    l2_norm,
)


class TestL2Norm:
    def test_zero_distance(self):
        assert l2_norm([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_unit_distance(self):
        result = l2_norm([0.0], [1.0])
        assert result == 1.0

    def test_known_distance(self):
        result = l2_norm([0.0, 0.0], [3.0, 4.0])
        assert abs(result - 5.0) < 1e-10

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            l2_norm([1.0, 2.0], [1.0])


class TestStateVector:
    def test_to_vector(self):
        sv = StateVector(
            validation_loss=0.5,
            accuracy=0.8,
            f1_score=0.75,
            features_count=10,
            model_score=0.85,
        )
        vec = sv.to_vector()
        assert len(vec) == 5
        assert vec[0] == 0.5
        assert vec[1] == 0.8

    def test_from_dict(self):
        d = {"validation_loss": 0.3, "accuracy": 0.9, "iteration": 5}
        sv = StateVector.from_dict(d)
        assert sv.validation_loss == 0.3
        assert sv.accuracy == 0.9
        assert sv.iteration == 5

    def test_to_dict_roundtrip(self):
        sv = StateVector(
            iteration=3,
            validation_loss=0.2,
            accuracy=0.95,
            f1_score=0.93,
            features_count=20,
            model_score=0.94,
        )
        d = sv.to_dict()
        sv2 = StateVector.from_dict(d)
        assert sv2.validation_loss == sv.validation_loss
        assert sv2.accuracy == sv.accuracy


class TestLoopConfig:
    def test_defaults(self):
        config = LoopConfig()
        assert config.epsilon == 0.001
        assert config.max_iterations == 10
        assert config.patience == 3

    def test_from_dict(self):
        config = LoopConfig.from_dict({"epsilon": 0.01, "max_iterations": 5})
        assert config.epsilon == 0.01
        assert config.max_iterations == 5


class TestLoopController:
    def test_single_step_does_not_converge(self):
        controller = LoopController()
        state = StateVector(validation_loss=1.0, accuracy=0.5)
        should_stop, reason = controller.step(state)
        assert not should_stop
        assert "need_more_data" in reason

    def test_convergence_detection(self):
        config = LoopConfig(epsilon=0.1, patience=1)
        controller = LoopController(config)

        s1 = StateVector(validation_loss=0.5, accuracy=0.8, f1_score=0.75,
                         features_count=10, model_score=0.8)
        controller.step(s1)

        s2 = StateVector(validation_loss=0.5, accuracy=0.8, f1_score=0.75,
                         features_count=10, model_score=0.8)
        should_stop, reason = controller.step(s2)
        assert should_stop
        assert "converged" in reason

    def test_max_iterations_stop(self):
        config = LoopConfig(max_iterations=3, epsilon=0.0001)
        controller = LoopController(config)

        for i in range(3):
            state = StateVector(validation_loss=1.0 - i * 0.3, accuracy=0.5 + i * 0.1)
            should_stop, reason = controller.step(state)

        assert should_stop
        assert "max_iterations" in reason

    def test_patience_counting(self):
        config = LoopConfig(epsilon=1.0, patience=3)
        controller = LoopController(config)

        s1 = StateVector(validation_loss=0.5, accuracy=0.8)
        controller.step(s1)

        s2 = StateVector(validation_loss=0.5, accuracy=0.8)
        should_stop, _ = controller.step(s2)
        assert not should_stop

        s3 = StateVector(validation_loss=0.5, accuracy=0.8)
        should_stop, _ = controller.step(s3)
        assert not should_stop

        s4 = StateVector(validation_loss=0.5, accuracy=0.8)
        should_stop, reason = controller.step(s4)
        assert should_stop
        assert "converged" in reason

    def test_run_converges(self):
        config = LoopConfig(epsilon=0.01, max_iterations=20, patience=2)
        controller = LoopController(config)

        def step_fn(iteration: int, prev: Optional[StateVector]) -> StateVector:
            if prev is None:
                return StateVector(validation_loss=1.0, accuracy=0.5)
            factor = 0.5 ** (iteration + 1)
            return StateVector(
                validation_loss=prev.validation_loss * (1 - factor * 0.1),
                accuracy=min(prev.accuracy + factor * 0.05, 0.99),
            )

        result = controller.run(step_fn)
        assert isinstance(result, ConvergenceResult)
        assert result.iterations_run > 1
        assert result.final_state is not None

    def test_run_max_iterations(self):
        config = LoopConfig(epsilon=0.0, max_iterations=3, patience=10)
        controller = LoopController(config)

        def step_fn(iteration: int, prev: Optional[StateVector]) -> StateVector:
            return StateVector(
                validation_loss=1.0 / (iteration + 1),
                accuracy=float(iteration) / 10,
            )

        result = controller.run(step_fn)
        assert result.iterations_run == 3
        assert "max_iterations" in result.exit_reason

    def test_reset(self):
        controller = LoopController()
        controller.step(StateVector(validation_loss=0.5))
        assert len(controller.history) == 1

        controller.reset()
        assert len(controller.history) == 0
        assert len(controller.convergence_deltas) == 0
