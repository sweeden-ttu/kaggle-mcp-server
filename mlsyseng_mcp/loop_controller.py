"""State convergence loop controller for expert-driven competition solving."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class LoopState:
    """Represents the state of a convergence loop iteration."""

    iteration: int = 0
    state_vector: list[float] = field(default_factory=list)
    score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    expert_contributions: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "state_vector": self.state_vector,
            "score": self.score,
            "metrics": self.metrics,
            "expert_contributions": self.expert_contributions,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoopState":
        return cls(
            iteration=data.get("iteration", 0),
            state_vector=data.get("state_vector", []),
            score=data.get("score", 0.0),
            metrics=data.get("metrics", {}),
            expert_contributions=data.get("expert_contributions", []),
            timestamp=data.get("timestamp", time.time()),
        )


class LoopController:
    """Controls the state convergence loop for competition solving.

    Exit condition: ||state[n] - state[n-1]||_2 < epsilon
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        max_iterations: int = 10,
        patience: int = 3,
        objective: str = "minimize_validation_loss",
    ):
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.patience = patience
        self.objective = objective

        self.history: list[LoopState] = []
        self.convergence_count = 0
        self.converged = False

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    @property
    def current_state(self) -> Optional[LoopState]:
        return self.history[-1] if self.history else None

    @property
    def previous_state(self) -> Optional[LoopState]:
        return self.history[-2] if len(self.history) >= 2 else None

    def step(
        self,
        state_vector: list[float],
        score: float = 0.0,
        metrics: Optional[dict[str, float]] = None,
        expert_contributions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Advance the loop by one iteration.

        Returns status dict with convergence info.
        """
        new_state = LoopState(
            iteration=self.current_iteration,
            state_vector=state_vector,
            score=score,
            metrics=metrics or {},
            expert_contributions=expert_contributions or [],
        )
        self.history.append(new_state)

        l2_norm = self._compute_l2_norm()
        is_converging = l2_norm is not None and l2_norm < self.epsilon

        if is_converging:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        if self.convergence_count >= self.patience:
            self.converged = True

        at_max = self.current_iteration >= self.max_iterations
        should_stop = self.converged or at_max

        return {
            "iteration": new_state.iteration,
            "l2_norm": l2_norm,
            "epsilon": self.epsilon,
            "is_converging": is_converging,
            "convergence_count": self.convergence_count,
            "patience": self.patience,
            "converged": self.converged,
            "at_max_iterations": at_max,
            "should_stop": should_stop,
            "score": score,
            "objective": self.objective,
        }

    def _compute_l2_norm(self) -> Optional[float]:
        """Compute ||state[n] - state[n-1]||_2."""
        if len(self.history) < 2:
            return None

        current = np.array(self.history[-1].state_vector)
        previous = np.array(self.history[-2].state_vector)

        if current.shape != previous.shape:
            return None

        if current.size == 0:
            return 0.0

        return float(np.linalg.norm(current - previous))

    def get_status(self) -> dict:
        """Get current loop status."""
        return {
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "epsilon": self.epsilon,
            "patience": self.patience,
            "convergence_count": self.convergence_count,
            "converged": self.converged,
            "objective": self.objective,
            "history_length": len(self.history),
            "last_l2_norm": self._compute_l2_norm(),
        }

    def get_history(self) -> list[dict]:
        """Get full iteration history."""
        return [state.to_dict() for state in self.history]

    def reset(self) -> None:
        """Reset the loop controller."""
        self.history = []
        self.convergence_count = 0
        self.converged = False

    @classmethod
    def from_config(cls, config: dict) -> "LoopController":
        """Create a LoopController from expert loop_config."""
        return cls(
            epsilon=config.get("epsilon", 0.001),
            max_iterations=config.get("max_iterations", 10),
            patience=config.get("patience", 3),
            objective=config.get("objective", "minimize_validation_loss"),
        )


def build_competition_entry(
    competition: str,
    experts: list[dict],
    embedding_engine=None,
    db_path: Optional[str] = None,
) -> dict:
    """Build a competition entry using expert knowledge and RAG.

    Selects relevant experts, combines their strategies, and produces
    a competition plan with notebook structure.
    """
    relevant_experts = experts
    if embedding_engine and len(experts) > 3:
        relevant_chapters = embedding_engine.get_relevant_experts(
            competition, n_results=3
        )
        relevant_experts = [
            e for e in experts
            if e.get("chapter_id") in relevant_chapters
        ] or experts[:3]

    combined_capabilities = set()
    combined_skills = set()
    all_metrics = set()

    for expert in relevant_experts:
        for cap in expert.get("capabilities", []):
            combined_capabilities.add(cap)
        for skill in expert.get("skills", []):
            combined_skills.add(skill)
        formula = expert.get("formula", {})
        for metric in formula.get("metrics", []):
            all_metrics.add(metric)

    loop_config = relevant_experts[0].get("loop_config", {}) if relevant_experts else {}

    entry = {
        "competition": competition,
        "experts_used": [e["expert_name"] for e in relevant_experts],
        "capabilities": sorted(combined_capabilities),
        "skills": sorted(combined_skills),
        "strategy": STRATEGY_TEMPLATE_FULL.format(competition=competition),
        "metrics": sorted(all_metrics),
        "loop_config": loop_config,
        "notebook_plan": _generate_notebook_plan(competition, relevant_experts),
    }

    return entry


STRATEGY_TEMPLATE_FULL = """Competition: {competition}
Phase 1: Data Loading & EDA
Phase 2: Feature Engineering (expert-guided)
Phase 3: Baseline Model
Phase 4: Advanced Models (expert-recommended)
Phase 5: Hyperparameter Tuning
Phase 6: Ensemble & Stacking
Phase 7: Submission Generation
"""


def _generate_notebook_plan(competition: str, experts: list[dict]) -> list[dict]:
    """Generate a notebook cell plan for a competition."""
    cells = [
        {"type": "markdown", "content": f"# {competition} - Expert MoE Solution"},
        {"type": "code", "content": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split"},
        {"type": "markdown", "content": "## Data Loading"},
        {"type": "code", "content": f"train = pd.read_csv('/kaggle/input/{competition}/train.csv')\ntest = pd.read_csv('/kaggle/input/{competition}/test.csv')"},
        {"type": "markdown", "content": "## EDA"},
        {"type": "code", "content": "train.info()\ntrain.describe()"},
        {"type": "markdown", "content": "## Feature Engineering"},
    ]

    for expert in experts:
        cells.append({
            "type": "markdown",
            "content": f"### Expert: {expert.get('expert_name', 'Unknown')}",
        })
        cells.append({
            "type": "code",
            "content": f"# Strategy: {expert.get('strategy', 'baseline')}\n# Capabilities: {', '.join(expert.get('capabilities', [])[:3])}",
        })

    cells.extend([
        {"type": "markdown", "content": "## Model Training"},
        {"type": "code", "content": "from sklearn.ensemble import GradientBoostingClassifier\nmodel = GradientBoostingClassifier()\nmodel.fit(X_train, y_train)"},
        {"type": "markdown", "content": "## Submission"},
        {"type": "code", "content": f"submission = pd.DataFrame({{'Id': test.index, 'Target': predictions}})\nsubmission.to_csv('submission.csv', index=False)"},
    ])

    return cells
