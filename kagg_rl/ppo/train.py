"""PPO fine-tune starting from an imitation-learning (BC) checkpoint.

What is PPO?
  Proximal Policy Optimization updates a policy with a *clipped* probability
  ratio so each step cannot move too far from the behavior that collected the
  batch. That stability is why it is a default for discrete game agents.

This module expects a BC checkpoint from ``kagg_rl.il.train_bc`` and fine-tunes
it against a callable environment factory. When no real Kaggriculture env is
installed, use ``--dry-run`` to verify the clipped objective on synthetic rolls.

Wire a real env later::

    def make_env():
        from kaggle_environments import make
        return make("kaggriculture")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from kagg_rl.il.model import MultiHeadPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ppo")


def load_bc_policy(path: Path, device: torch.device) -> MultiHeadPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiHeadPolicy(
        obs_dim=ckpt.get("obs_dim"),
        hidden=ckpt.get("hidden", 256),
        depth=ckpt.get("depth", 3),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    logger.info("Loaded BC checkpoint %s (meta=%s)", path, ckpt.get("meta"))
    return model


def ppo_update(
    model: MultiHeadPolicy,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: Dict[str, torch.Tensor],
    old_logprob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Standard clipped PPO on a batch of on-policy transitions."""
    n = obs.size(0)
    idx = torch.arange(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, batch_size):
            mb = perm[start : start + batch_size]
            mb_obs = obs[mb]
            mb_actions = {k: v[mb] for k, v in actions.items()}
            mb_old = old_logprob[mb]
            mb_ret = returns[mb]
            mb_adv = advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            logprob, value, entropy = model.evaluate_actions(mb_obs, mb_actions)
            ratio = torch.exp(logprob - mb_old)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, mb_ret)
            ent = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(ent.detach())
            updates += 1

    if updates:
        for k in stats:
            stats[k] /= updates
    return stats


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last = delta + gamma * lam * next_nonterminal * last
        advantages[t] = last
    returns = advantages + values[: len(rewards)]
    return advantages, returns


def synthetic_rollout(model: MultiHeadPolicy, device, horizon=128):
    """Dry-run rollout with random obs / bank-delta rewards for plumbing tests."""
    obs_dim = model.backbone[0].in_features
    obs_list, logps, rewards, values, dones = [], [], [], [], []
    act_lists = {k: [] for k in ("farmer_op", "farmer_item", "market_op", "market_item")}

    for t in range(horizon):
        o = torch.randn(1, obs_dim, device=device)
        with torch.no_grad():
            actions, logprob, value, _ = model.act_for_ppo(o)
        obs_list.append(o.squeeze(0))
        for k in act_lists:
            act_lists[k].append(actions[k].squeeze(0))
        logps.append(logprob.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(torch.tensor(float(np.random.randn() * 0.01), device=device))
        dones.append(torch.tensor(1.0 if t == horizon - 1 else 0.0, device=device))

    obs_t = torch.stack(obs_list)
    actions_t = {k: torch.stack(v) for k, v in act_lists.items()}
    logp_t = torch.stack(logps)
    rew_t = torch.stack(rewards)
    val_t = torch.stack(values)
    done_t = torch.stack(dones)
    adv, ret = compute_gae(rew_t, val_t, done_t)
    return obs_t, actions_t, logp_t.detach(), ret.detach(), adv.detach()


def parse_args():
    p = argparse.ArgumentParser(description="PPO fine-tune from BC checkpoint")
    p.add_argument("--bc-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/ppo_ft.pt"))
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run PPO math on synthetic rollouts (no kaggle-environments needed)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_bc_policy(args.bc_checkpoint, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.dry_run:
        logger.warning(
            "No online Kaggriculture env wired in this entrypoint yet. "
            "Use --dry-run to exercise PPO, or extend main() with kaggle_environments."
        )
        args.dry_run = True

    for update in range(1, args.updates + 1):
        obs, actions, old_logp, ret, adv = synthetic_rollout(model, device)
        stats = ppo_update(
            model,
            opt,
            obs,
            actions,
            old_logp,
            ret,
            adv,
            clip_eps=args.clip_eps,
        )
        logger.info("update %d  %s", update, stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "from_bc": str(args.bc_checkpoint)}, args.out)
    logger.info("saved %s", args.out)


if __name__ == "__main__":
    main()
