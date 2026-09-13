"""Hindsight Experience Replay (HER) for sparse Kaggriculture seasons.

Terminal prize is end-of-season money. Many intermediate plans “fail” relative
to an ambitious cash target but still achieve useful milestones (feed the herd,
harvest wheat, liquidate by day 28). HER relabels those trajectories toward
*achieved* goals so early seasons are not zero-learning until day 30.

HER here is a **milestone / goal buffer** used alongside PPO — not a replacement
for the on-policy clipped update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class MilestoneGoal:
    """Sparse season goal used for hindsight relabeling."""

    name: str
    # Target final cash (coins). Achieved goal becomes the hindsight target.
    cash_target: float
    # Optional: require survival of livestock through this day index (0-based).
    herd_alive_through_day: Optional[int] = None
    # Optional: require liquidation bias after this fraction of the season.
    liquidate_after_frac: Optional[float] = None


def default_milestones() -> List[MilestoneGoal]:
    """Default milestone ladder for a 30-day / 720-step season."""
    return [
        MilestoneGoal("survive_feed", cash_target=3_000.0, herd_alive_through_day=10),
        MilestoneGoal("cash_10k", cash_target=10_000.0),
        MilestoneGoal("cash_50k", cash_target=50_000.0),
        MilestoneGoal("cash_100k", cash_target=100_000.0),
        MilestoneGoal("liquidate_late", cash_target=20_000.0, liquidate_after_frac=0.85),
    ]


@dataclass
class EpisodeMilestoneInfo:
    """Per-step info needed to decide which milestones were achieved."""

    money: Sequence[float]
    day: Sequence[int]
    herd_alive: Sequence[bool]
    bank_delta: Sequence[float]


def achieved_milestones(
    info: EpisodeMilestoneInfo,
    milestones: Optional[Sequence[MilestoneGoal]] = None,
) -> List[MilestoneGoal]:
    milestones = list(milestones or default_milestones())
    if not info.money:
        return []
    final_cash = float(info.money[-1])
    hit: List[MilestoneGoal] = []
    for m in milestones:
        if final_cash < m.cash_target:
            continue
        if m.herd_alive_through_day is not None:
            ok = True
            for t, day in enumerate(info.day):
                if day <= m.herd_alive_through_day and not info.herd_alive[t]:
                    ok = False
                    break
            if not ok:
                continue
        if m.liquidate_after_frac is not None and info.bank_delta:
            n = len(info.bank_delta)
            start = int(m.liquidate_after_frac * n)
            late = info.bank_delta[start:]
            if not late or sum(late) <= 0:
                continue
        hit.append(m)
    return hit


def her_bonus_schedule(
    n_steps: int,
    milestones_hit: Sequence[MilestoneGoal],
    *,
    base_scale: float = 1.0,
) -> torch.Tensor:
    """Dense hindsight bonus that grows toward the end of a successful season."""
    bonus = torch.zeros(n_steps, dtype=torch.float32)
    if n_steps <= 0 or not milestones_hit:
        return bonus
    # Stronger credit for harder (higher cash) milestones.
    weight = sum(max(m.cash_target, 1.0) for m in milestones_hit) / 50_000.0
    weight = min(max(weight, 0.25), 4.0)
    for t in range(n_steps):
        progress = (t + 1) / n_steps
        bonus[t] = base_scale * weight * (1.0 + progress)
    return bonus


def relabel_rewards_with_her(
    rewards: torch.Tensor,
    info: EpisodeMilestoneInfo,
    milestones: Optional[Sequence[MilestoneGoal]] = None,
    *,
    base_scale: float = 1.0,
) -> Dict[str, object]:
    """Return rewards + HER bonus and metadata for logging / aux buffers."""
    hit = achieved_milestones(info, milestones)
    bonus = her_bonus_schedule(int(rewards.numel()), hit, base_scale=base_scale)
    if bonus.device != rewards.device:
        bonus = bonus.to(rewards.device)
    augmented = rewards + bonus
    return {
        "rewards": augmented,
        "her_bonus": bonus,
        "milestones_hit": [m.name for m in hit],
        "n_milestones": len(hit),
    }


def synthetic_milestone_info(
    horizon: int,
    *,
    final_cash: float = 55_000.0,
    device: Optional[torch.device] = None,
) -> EpisodeMilestoneInfo:
    """Build a smooth cash ramp for dry-run HER plumbing tests."""
    money = [final_cash * ((t + 1) / horizon) for t in range(horizon)]
    day = [min(29, t // 24) for t in range(horizon)]
    herd_alive = [True] * horizon
    bank_delta = [money[0]] + [money[t] - money[t - 1] for t in range(1, horizon)]
    # Late liquidation: push positive deltas in the last 15%.
    start = int(0.85 * horizon)
    for t in range(start, horizon):
        bank_delta[t] = abs(bank_delta[t]) + 10.0
    return EpisodeMilestoneInfo(
        money=money,
        day=day,
        herd_alive=herd_alive,
        bank_delta=bank_delta,
    )
