"""Tests for the loop controller module."""

import os
import tempfile

import numpy as np
import pytest

import database as db
import loop_controller as lc


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    db.init_db(path)
    yield path
    os.unlink(path)


def test_l2_norm_diff():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert lc.l2_norm_diff(a, b) == 0.0

    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 0.0])
    assert abs(lc.l2_norm_diff(c, d) - 1.0) < 1e-10


def test_initialize_state():
    expert = {"slug": "test_expert", "loop_config": {}}
    state = lc.initialize_state("titanic", expert)
    assert state.competition_slug == "titanic"
    assert state.expert_slug == "test_expert"
    assert state.iteration == 0
    assert len(state.history) == 1


def test_step():
    expert = {"slug": "test_expert", "loop_config": {}}
    state = lc.initialize_state("titanic", expert)
    metrics = {"loss": 0.5, "primary_metric": 0.7}
    state = lc.step(state, metrics)
    assert state.iteration == 1
    assert len(state.history) == 2
    assert state.state_vector[0] == 0.5


def test_check_convergence_not_converged():
    expert = {"slug": "test_expert", "loop_config": {}}
    state = lc.initialize_state("titanic", expert)
    result = lc.check_convergence(state)
    assert result["should_exit"] is False
    assert result["reason"] == "insufficient_history"


def test_check_convergence_max_iterations():
    expert = {"slug": "test_expert", "loop_config": {}}
    state = lc.initialize_state("titanic", expert)
    state.iteration = 10
    result = lc.check_convergence(state, max_iterations=10)
    assert result["should_exit"] is True
    assert result["reason"] == "max_iterations_reached"


def test_check_convergence_converged():
    expert = {"slug": "test_expert", "loop_config": {}}
    state = lc.initialize_state("titanic", expert)
    identical = [1.0, 0.5, 0.3, 0.01, 0.5]
    state.history = [identical, identical, identical, identical]
    state.iteration = 4
    result = lc.check_convergence(state, epsilon=0.001, patience=3)
    assert result["converged"] is True
    assert result["should_exit"] is True


def test_run_loop(temp_db):
    expert = {
        "slug": "test_expert",
        "expert_name": "Test Expert",
        "loop_config": {
            "epsilon": 0.1,
            "max_iterations": 5,
            "patience": 2,
        },
    }

    def fast_converge(iteration, state):
        return {
            "loss": state[0] * 0.5,
            "primary_metric": 0.9,
            "secondary_metric": 0.8,
            "learning_rate": 0.001,
        }

    result = lc.run_loop("titanic", expert, metric_generator=fast_converge, db_path=temp_db)
    assert result["competition"] == "titanic"
    assert result["expert"] == "test_expert"
    assert result["iterations"] > 0
    assert isinstance(result["final_state"], list)
