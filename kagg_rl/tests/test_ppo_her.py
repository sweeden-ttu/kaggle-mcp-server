"""Unit tests for HER milestones and multi-discrete action framing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from kagg_rl.action_space import (
    ABLATION_STACK,
    IGNORE_FOR_ENV,
    PRIMARY_STACK,
    kaggriculture_action_spec,
)
from kagg_rl.her import (
    EpisodeMilestoneInfo,
    achieved_milestones,
    default_milestones,
    relabel_rewards_with_her,
    synthetic_milestone_info,
)
from kagg_rl.il.model import MultiHeadPolicy
from kagg_rl.ppo.train import ppo_update, synthetic_rollout


class TestActionSpace(unittest.TestCase):
    def test_multi_discrete_not_continuous(self):
        spec = kaggriculture_action_spec()
        self.assertGreater(spec.farmer_ops, 1)
        self.assertEqual(set(spec.discrete_heads), {"farmer_op", "farmer_item", "market_op", "market_item"})
        self.assertIn("PPO", PRIMARY_STACK)
        self.assertIn("HER", PRIMARY_STACK)
        self.assertIn("DQN", ABLATION_STACK)
        for name in IGNORE_FOR_ENV:
            self.assertNotIn(name, PRIMARY_STACK)


class TestHER(unittest.TestCase):
    def test_milestones_hit_on_rich_season(self):
        info = synthetic_milestone_info(48, final_cash=55_000.0)
        hit = achieved_milestones(info)
        names = {m.name for m in hit}
        self.assertIn("cash_10k", names)
        self.assertIn("cash_50k", names)
        self.assertIn("survive_feed", names)

    def test_relabel_adds_positive_bonus(self):
        horizon = 32
        rewards = torch.zeros(horizon)
        info = synthetic_milestone_info(horizon, final_cash=60_000.0)
        out = relabel_rewards_with_her(rewards, info, base_scale=1.0)
        self.assertGreater(int(out["n_milestones"]), 0)
        self.assertTrue(torch.all(out["her_bonus"] >= 0))
        self.assertGreater(float(out["rewards"].sum()), float(rewards.sum()))

    def test_no_bonus_when_broke(self):
        money = [100.0] * 16
        info = EpisodeMilestoneInfo(
            money=money,
            day=[0] * 16,
            herd_alive=[False] * 16,
            bank_delta=[-1.0] * 16,
        )
        hit = achieved_milestones(info, default_milestones())
        self.assertEqual(hit, [])


class TestPPOHER(unittest.TestCase):
    def test_synthetic_ppo_her_update(self):
        model = MultiHeadPolicy(obs_dim=40, hidden=32, depth=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        obs, actions, old_logp, ret, adv, her_meta = synthetic_rollout(
            model, torch.device("cpu"), horizon=32, use_her=True, final_cash=55_000.0
        )
        self.assertGreater(her_meta["n_milestones"], 0)
        stats = ppo_update(model, opt, obs, actions, old_logp, ret, adv, epochs=1, batch_size=16)
        self.assertTrue(all(k in stats for k in ("policy_loss", "value_loss", "entropy")))

    def test_roundtrip_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bc = td / "bc.pt"
            model = MultiHeadPolicy(obs_dim=40, hidden=32, depth=2)
            torch.save(
                {"model": model.state_dict(), "obs_dim": 40, "hidden": 32, "depth": 2},
                bc,
            )
            from kagg_rl.ppo.train import load_bc_policy

            loaded = load_bc_policy(bc, torch.device("cpu"))
            self.assertEqual(loaded.backbone[0].in_features, 40)


if __name__ == "__main__":
    unittest.main()
