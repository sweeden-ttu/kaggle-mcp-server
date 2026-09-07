"""Unit tests for IL feature/action encoding and a tiny BC smoke train."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from kagg_rl.il.actions import encode_step_action
from kagg_rl.il.dataset import build_arrays_from_seats, train_val_split
from kagg_rl.il.features import extract_features, feature_dim
from kagg_rl.il.ingest import SeatRef, seats_from_local_episodes
from kagg_rl.il.model import MultiHeadPolicy


SAMPLE = Path("/tmp/kagg-il-sample/datasets/il/episodes/00/100009300.json.gz")


class TestIL(unittest.TestCase):
    def test_feature_dim_stable(self):
        d = feature_dim()
        self.assertGreater(d, 20)
        self.assertEqual(d, feature_dim())

    def test_encode_actions(self):
        labels = encode_step_action(
            {"farmer": ["PICKUP", "WHEAT", 3], "market": [["SELL", "MILK", 2]], "hands": []}
        )
        self.assertIn("farmer_op", labels)
        self.assertGreaterEqual(labels["farmer_qty"], 0.0)

    @unittest.skipUnless(SAMPLE.exists(), "sample episode not downloaded")
    def test_ingest_and_bc_step(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dest = td / SAMPLE.name
            dest.write_bytes(SAMPLE.read_bytes())
            seats = seats_from_local_episodes(td, min_reward=0)
            self.assertGreaterEqual(len(seats), 1)
            X, Y = build_arrays_from_seats(seats[:1], max_transitions=64, stride=10)
            self.assertEqual(X.shape[1], feature_dim())
            train_ds, val_ds = train_val_split(X, Y, val_frac=0.25, seed=0)
            model = MultiHeadPolicy(obs_dim=X.shape[1], hidden=64, depth=2)
            xb, yb = train_ds[0]
            loss, stats = model.bc_loss(xb.unsqueeze(0), {k: v.unsqueeze(0) for k, v in yb.items()})
            self.assertTrue(torch.isfinite(loss))
            loss.backward()


if __name__ == "__main__":
    unittest.main()
