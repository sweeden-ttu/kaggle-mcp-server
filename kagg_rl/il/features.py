"""Observation → compact numeric feature vector for IL / PPO."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .actions import ITEMS

MARKET_PRODUCTS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
]

SEED_TYPES = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]


def _tile_summary(tiles: List[List[Any]]) -> List[float]:
    """Summarize 10x10 farm tiles into coarse counts."""
    empty = 0
    locked = 0
    watered = 0
    cropish = 0
    animalish = 0
    other = 0
    for row in tiles or []:
        for cell in row:
            if cell is None:
                empty += 1
            elif cell == "LOCKED":
                locked += 1
            elif isinstance(cell, str):
                u = cell.upper()
                if "WATER" in u:
                    watered += 1
                elif any(c in u for c in ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAW", "PLANT")):
                    cropish += 1
                elif any(a in u for a in ("COW", "SHEEP", "GOOSE", "PASTURE")):
                    animalish += 1
                else:
                    other += 1
            elif isinstance(cell, dict):
                kind = str(cell.get("type") or cell.get("kind") or "").upper()
                if any(c in kind for c in ("WHEAT", "CARROT", "TOMATO", "MELON", "STRAW")):
                    cropish += 1
                elif any(a in kind for a in ("COW", "SHEEP", "GOOSE")):
                    animalish += 1
                elif cell.get("watered") or cell.get("water"):
                    watered += 1
                else:
                    other += 1
            else:
                other += 1
    total = max(empty + locked + watered + cropish + animalish + other, 1)
    return [
        empty / total,
        locked / total,
        watered / total,
        cropish / total,
        animalish / total,
        other / total,
    ]


def _dict_vec(d: Optional[Dict[str, Any]], keys: List[str], scale: float) -> List[float]:
    d = d or {}
    out = []
    for k in keys:
        try:
            out.append(float(d.get(k, 0)) / scale)
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def extract_features(obs: Dict[str, Any], seat: int) -> np.ndarray:
    """Build a fixed-length float32 feature vector from one seat's observation.

    Feature groups:
      - time (day/hour/step)
      - own money + farmer xy + hand count + unlocked quadrants
      - tile summary
      - market prices + inventories
      - private seeds + shed counts
    """
    feats: List[float] = []

    day = float(obs.get("day", 0))
    hour = float(obs.get("hour", 0))
    step = float(obs.get("step", day * 24 + hour))
    feats.extend([day / 30.0, hour / 24.0, step / 720.0])

    farms = obs.get("farms") or []
    me = farms[seat] if seat < len(farms) else {}
    money = float(me.get("money", 0.0))
    farmer = me.get("farmer") or [0, 0]
    fx = float(farmer[0]) / 9.0 if len(farmer) > 0 else 0.0
    fy = float(farmer[1]) / 9.0 if len(farmer) > 1 else 0.0
    hands = me.get("hands") or []
    unlocked = me.get("unlocked_quadrants") or []
    feats.extend([
        money / 200_000.0,
        fx,
        fy,
        float(len(hands)) / 10.0,
        float(len(unlocked)) / 4.0,
        float(me.get("hires_today", 0)) / 5.0,
    ])
    feats.extend(_tile_summary(me.get("tiles") or []))

    market = obs.get("market") or {}
    feats.extend(_dict_vec(market.get("prices"), MARKET_PRODUCTS, 300.0))
    feats.extend(_dict_vec(market.get("inventory"), MARKET_PRODUCTS, 10_000.0))

    private = obs.get("private") or {}
    feats.extend(_dict_vec(private.get("seeds"), SEED_TYPES, 20.0))
    shed_keys = SEED_TYPES + ["EGG", "MILK", "WOOL", "FERTILIZER", "COW", "SHEEP", "GOOSE"]
    feats.extend(_dict_vec(private.get("shed"), shed_keys, 50.0))

    # Opponent money (visible on shared farms list).
    opp_seat = 1 - seat
    if opp_seat < len(farms):
        opp_money = float(farms[opp_seat].get("money", 0.0))
    else:
        opp_money = 0.0
    feats.append(opp_money / 200_000.0)

    return np.asarray(feats, dtype=np.float32)


def feature_dim() -> int:
    # Compute once from a minimal stub obs.
    stub = {
        "day": 0,
        "hour": 0,
        "step": 0,
        "farms": [
            {
                "money": 0,
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": [],
                "tiles": [[None] * 10 for _ in range(10)],
                "hires_today": 0,
            },
            {
                "money": 0,
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": [],
                "tiles": [[None] * 10 for _ in range(10)],
                "hires_today": 0,
            },
        ],
        "market": {"prices": {}, "inventory": {}},
        "private": {"seeds": {}, "shed": {}},
    }
    return int(extract_features(stub, 0).shape[0])
