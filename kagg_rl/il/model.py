"""Multi-head policy network shared by IL (BC) and PPO fine-tune."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .actions import FARMER_OPS, ITEMS, MARKET_OPS
from .features import feature_dim


class MultiHeadPolicy(nn.Module):
    """Actor (multi discrete heads + qty) with a scalar critic for PPO."""

    def __init__(self, obs_dim: int | None = None, hidden: int = 256, depth: int = 3):
        super().__init__()
        obs_dim = obs_dim or feature_dim()
        layers = []
        in_dim = obs_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)

        self.farmer_op = nn.Linear(hidden, len(FARMER_OPS))
        self.farmer_item = nn.Linear(hidden, len(ITEMS))
        self.farmer_qty = nn.Linear(hidden, 1)
        self.market_op = nn.Linear(hidden, len(MARKET_OPS))
        self.market_item = nn.Linear(hidden, len(ITEMS))
        self.market_qty = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "farmer_op": self.farmer_op(h),
            "farmer_item": self.farmer_item(h),
            "farmer_qty": torch.sigmoid(self.farmer_qty(h)).squeeze(-1),
            "market_op": self.market_op(h),
            "market_item": self.market_item(h),
            "market_qty": torch.sigmoid(self.market_qty(h)).squeeze(-1),
            "value": self.value(h).squeeze(-1),
        }

    def bc_loss(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(x)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        loss_fo = ce(out["farmer_op"], y["farmer_op"])
        loss_fi = ce(out["farmer_item"], y["farmer_item"])
        loss_fq = mse(out["farmer_qty"], y["farmer_qty"])
        loss_mo = ce(out["market_op"], y["market_op"])
        loss_mi = ce(out["market_item"], y["market_item"])
        loss_mq = mse(out["market_qty"], y["market_qty"])
        loss = loss_fo + 0.5 * loss_fi + 0.1 * loss_fq + loss_mo + 0.5 * loss_mi + 0.1 * loss_mq
        stats = {
            "loss": float(loss.detach()),
            "farmer_op": float(loss_fo.detach()),
            "market_op": float(loss_mo.detach()),
        }
        return loss, stats

    def accuracy(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            out = self.forward(x)
            fo = (out["farmer_op"].argmax(-1) == y["farmer_op"]).float().mean().item()
            mo = (out["market_op"].argmax(-1) == y["market_op"]).float().mean().item()
        return {"farmer_op_acc": fo, "market_op_acc": mo}

    def act_for_ppo(self, x: torch.Tensor):
        """Sample discrete heads for PPO; returns actions, logprob, value, entropy."""
        out = self.forward(x)
        dists = {
            "farmer_op": Categorical(logits=out["farmer_op"]),
            "farmer_item": Categorical(logits=out["farmer_item"]),
            "market_op": Categorical(logits=out["market_op"]),
            "market_item": Categorical(logits=out["market_item"]),
        }
        actions = {k: d.sample() for k, d in dists.items()}
        logprob = sum(d.log_prob(actions[k]) for k, d in dists.items())
        entropy = sum(d.entropy() for d in dists.values())
        # qty heads treated as deterministic mean for v0 PPO
        actions["farmer_qty"] = out["farmer_qty"]
        actions["market_qty"] = out["market_qty"]
        return actions, logprob, out["value"], entropy

    def evaluate_actions(self, x: torch.Tensor, actions: Dict[str, torch.Tensor]):
        out = self.forward(x)
        dists = {
            "farmer_op": Categorical(logits=out["farmer_op"]),
            "farmer_item": Categorical(logits=out["farmer_item"]),
            "market_op": Categorical(logits=out["market_op"]),
            "market_item": Categorical(logits=out["market_item"]),
        }
        logprob = sum(d.log_prob(actions[k]) for k, d in dists.items())
        entropy = sum(d.entropy() for d in dists.values())
        return logprob, out["value"], entropy
