"""Torch dataset built from top-agent episode seats."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import encode_step_action
from .features import extract_features, feature_dim
from .ingest import SeatRef, iter_seat_transitions

logger = logging.getLogger(__name__)


class TransitionDataset(Dataset):
    """Stores (features, multi-head labels) for behavioral cloning."""

    def __init__(self, X: np.ndarray, Y: Dict[str, np.ndarray]):
        self.X = torch.from_numpy(X)
        self.Y = {k: torch.from_numpy(v) for k, v in Y.items()}
        assert len(self.X) == len(next(iter(self.Y.values())))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        labels = {k: v[idx] for k, v in self.Y.items()}
        return self.X[idx], labels


def build_arrays_from_seats(
    seats: Sequence[SeatRef],
    *,
    max_transitions: Optional[int] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    xs: List[np.ndarray] = []
    farmer_ops: List[int] = []
    farmer_items: List[int] = []
    farmer_qtys: List[float] = []
    market_ops: List[int] = []
    market_items: List[int] = []
    market_qtys: List[float] = []

    for seat in seats:
        if seat.path is None:
            continue
        try:
            transitions = list(iter_seat_transitions(seat))
        except Exception as e:
            logger.warning("failed reading %s: %s", seat.path, e)
            continue

        for i, (obs, action) in enumerate(transitions):
            if stride > 1 and (i % stride) != 0:
                continue
            feats = extract_features(obs, seat.seat)
            labels = encode_step_action(action)
            xs.append(feats)
            farmer_ops.append(labels["farmer_op"])
            farmer_items.append(labels["farmer_item"])
            farmer_qtys.append(labels["farmer_qty"])
            market_ops.append(labels["market_op"])
            market_items.append(labels["market_item"])
            market_qtys.append(labels["market_qty"])

            if max_transitions and len(xs) >= max_transitions:
                break
        if max_transitions and len(xs) >= max_transitions:
            break

    if not xs:
        raise RuntimeError(
            "No transitions extracted. Check --episodes-dir path and filters "
            "(min-reward / winners-only / top-agents)."
        )

    X = np.stack(xs).astype(np.float32)
    Y = {
        "farmer_op": np.asarray(farmer_ops, dtype=np.int64),
        "farmer_item": np.asarray(farmer_items, dtype=np.int64),
        "farmer_qty": np.asarray(farmer_qtys, dtype=np.float32),
        "market_op": np.asarray(market_ops, dtype=np.int64),
        "market_item": np.asarray(market_items, dtype=np.int64),
        "market_qty": np.asarray(market_qtys, dtype=np.float32),
    }
    logger.info(
        "Built IL arrays: %d transitions, feature_dim=%d",
        X.shape[0],
        X.shape[1],
    )
    return X, Y


def train_val_split(
    X: np.ndarray,
    Y: Dict[str, np.ndarray],
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[TransitionDataset, TransitionDataset]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    Y_train = {k: v[train_idx] for k, v in Y.items()}
    Y_val = {k: v[val_idx] for k, v in Y.items()}
    return TransitionDataset(X[train_idx], Y_train), TransitionDataset(X[val_idx], Y_val)


# Re-export for callers
FEATURE_DIM = None


def get_feature_dim() -> int:
    global FEATURE_DIM
    if FEATURE_DIM is None:
        FEATURE_DIM = feature_dim()
    return FEATURE_DIM
