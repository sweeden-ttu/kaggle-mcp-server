"""Kaggriculture RL: primary stack is BC → PPO (multi-discrete) + HER milestones.

Hierarchical Dueling Double DQN (Path B) is retained as an ablation / offline
baseline only — not the submission ceiling.
"""

from .action_space import (
    ABLATION_STACK,
    IGNORE_FOR_ENV,
    PRIMARY_STACK,
    kaggriculture_action_spec,
)

__version__ = "0.2.0"
__all__ = [
    "ABLATION_STACK",
    "IGNORE_FOR_ENV",
    "PRIMARY_STACK",
    "kaggriculture_action_spec",
]
