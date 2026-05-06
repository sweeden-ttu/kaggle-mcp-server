"""State convergence loop controller for MLSysEng MoE.

Implements the iterative convergence loop where:
  exit_condition: ||state[n] - state[n-1]||_2 < epsilon

Each iteration refines the competition state vector until convergence
or max_iterations is reached.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from . import database as db
except ImportError:
    import database as db

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceState:
    """Tracks the state of a convergence loop."""

    competition_slug: str
    expert_slug: str
    state_vector: np.ndarray
    iteration: int = 0
    converged: bool = False
    history: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


def l2_norm_diff(state_current: np.ndarray, state_previous: np.ndarray) -> float:
    """Compute L2 norm of state difference: ||state[n] - state[n-1]||_2"""
    diff = state_current - state_previous
    return float(np.linalg.norm(diff, ord=2))


def initialize_state(
    competition_slug: str,
    expert: dict,
    initial_metrics: Optional[dict] = None,
) -> ConvergenceState:
    """Initialize state vector for a convergence loop.

    The state vector encodes:
      [loss, primary_metric, secondary_metric, learning_rate, iteration_progress]
    """
    if initial_metrics:
        state = np.array([
            initial_metrics.get("loss", 1.0),
            initial_metrics.get("primary_metric", 0.0),
            initial_metrics.get("secondary_metric", 0.0),
            initial_metrics.get("learning_rate", 0.01),
            0.0,
        ], dtype=np.float64)
    else:
        state = np.array([1.0, 0.0, 0.0, 0.01, 0.0], dtype=np.float64)

    return ConvergenceState(
        competition_slug=competition_slug,
        expert_slug=expert["slug"],
        state_vector=state,
        iteration=0,
        history=[state.tolist()],
    )


def step(
    conv_state: ConvergenceState,
    new_metrics: dict,
) -> ConvergenceState:
    """Advance the convergence loop by one step.

    Updates the state vector with new metrics and checks convergence.
    """
    previous = conv_state.state_vector.copy()

    new_state = np.array([
        new_metrics.get("loss", previous[0] * 0.95),
        new_metrics.get("primary_metric", previous[1] + 0.02),
        new_metrics.get("secondary_metric", previous[2] + 0.01),
        new_metrics.get("learning_rate", previous[3]),
        (conv_state.iteration + 1) / 10.0,
    ], dtype=np.float64)

    conv_state.state_vector = new_state
    conv_state.iteration += 1
    conv_state.history.append(new_state.tolist())

    return conv_state


def check_convergence(
    conv_state: ConvergenceState,
    epsilon: float = 0.001,
    patience: int = 3,
    max_iterations: int = 10,
) -> dict:
    """Check if the convergence loop should exit.

    Returns convergence status and metrics.
    """
    if conv_state.iteration >= max_iterations:
        return {
            "should_exit": True,
            "reason": "max_iterations_reached",
            "iteration": conv_state.iteration,
            "converged": False,
        }

    if len(conv_state.history) < 2:
        return {
            "should_exit": False,
            "reason": "insufficient_history",
            "iteration": conv_state.iteration,
            "converged": False,
        }

    current = np.array(conv_state.history[-1])
    previous = np.array(conv_state.history[-2])
    norm_diff = l2_norm_diff(current, previous)

    converging_count = 0
    for i in range(len(conv_state.history) - 1, max(0, len(conv_state.history) - patience - 1), -1):
        if i < 1:
            break
        s_curr = np.array(conv_state.history[i])
        s_prev = np.array(conv_state.history[i - 1])
        if l2_norm_diff(s_curr, s_prev) < epsilon:
            converging_count += 1
        else:
            break

    converged = converging_count >= patience

    return {
        "should_exit": converged,
        "reason": "converged" if converged else "not_converged",
        "iteration": conv_state.iteration,
        "converged": converged,
        "l2_norm_diff": norm_diff,
        "epsilon": epsilon,
        "consecutive_converging": converging_count,
        "patience": patience,
    }


def run_loop(
    competition_slug: str,
    expert: dict,
    metric_generator=None,
    db_path: Optional[str] = None,
) -> dict:
    """Run the full convergence loop for a competition/expert pair.

    Args:
        competition_slug: Competition identifier
        expert: Expert definition dict
        metric_generator: Optional callable(iteration, state) -> metrics dict.
                         If None, uses simulated improvement.
        db_path: Database path override

    Returns:
        Final convergence report
    """
    loop_config = expert.get("loop_config", {})
    epsilon = loop_config.get("epsilon", 0.001)
    max_iterations = loop_config.get("max_iterations", 10)
    patience = loop_config.get("patience", 3)

    conv_state = initialize_state(competition_slug, expert)
    logger.info(
        "Starting convergence loop for %s/%s (eps=%s, max=%d, patience=%d)",
        competition_slug,
        expert["slug"],
        epsilon,
        max_iterations,
        patience,
    )

    while True:
        if metric_generator:
            metrics = metric_generator(conv_state.iteration, conv_state.state_vector)
        else:
            metrics = _simulate_metrics(conv_state.iteration, conv_state.state_vector)

        conv_state = step(conv_state, metrics)

        convergence = check_convergence(conv_state, epsilon, patience, max_iterations)
        logger.info(
            "Iteration %d: l2_diff=%.6f, converging=%d/%d",
            conv_state.iteration,
            convergence.get("l2_norm_diff", float("inf")),
            convergence.get("consecutive_converging", 0),
            patience,
        )

        db.upsert_competition_entry(
            competition_slug,
            expert["slug"],
            state_vector=conv_state.state_vector.tolist(),
            iteration=conv_state.iteration,
            converged=convergence["converged"],
            db_path=db_path,
        )

        if convergence["should_exit"]:
            conv_state.converged = convergence["converged"]
            break

    elapsed = time.time() - conv_state.start_time
    return {
        "competition": competition_slug,
        "expert": expert["slug"],
        "converged": conv_state.converged,
        "iterations": conv_state.iteration,
        "final_state": conv_state.state_vector.tolist(),
        "final_l2_diff": convergence.get("l2_norm_diff", None),
        "elapsed_seconds": round(elapsed, 2),
        "history_length": len(conv_state.history),
    }


def _simulate_metrics(iteration: int, state: np.ndarray) -> dict:
    """Generate simulated improving metrics for demonstration."""
    decay = 0.85 ** (iteration + 1)
    noise = np.random.normal(0, 0.001)
    return {
        "loss": max(0.01, state[0] * decay + noise),
        "primary_metric": min(1.0, state[1] + 0.05 * decay),
        "secondary_metric": min(1.0, state[2] + 0.03 * decay),
        "learning_rate": state[3] * 0.95,
    }


def evolve_all_experts(
    competition_slug: str,
    experts: list[dict],
    db_path: Optional[str] = None,
) -> list[dict]:
    """Run convergence loops for all provided experts."""
    results = []
    for expert in experts:
        result = run_loop(competition_slug, expert, db_path=db_path)
        results.append(result)
    return results
