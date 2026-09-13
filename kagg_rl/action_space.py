"""Kaggriculture action space: multi-discrete (not continuous).

The competition farm is a branched discrete decision problem:

* farmer opcode (PASS / WATER / PLANT / HARVEST / …)
* farmer item / crop parameter
* market opcode (HIRE / SELL / BUY_SEED / BUY_ANIMAL / …)
* market item
* quantity heads (treated as soft [0, 1] regression, not continuous control)

Continuous-control algorithms (DDPG, TD3, SAC) are the wrong default for this
skeleton. Primary online learning uses **PPO** on the discrete heads; **HER**
supplies sparse end-of-season / milestone credit. Hierarchical DQN remains an
ablation / offline baseline only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .il.actions import FARMER_OPS, ITEMS, MARKET_OPS


@dataclass(frozen=True)
class MultiDiscreteSpec:
    """Explicit multi-discrete layout shared by BC and PPO."""

    farmer_ops: int
    farmer_items: int
    market_ops: int
    market_items: int
    qty_continuous_aux: bool = True  # qty heads are regression auxiliaries

    @property
    def discrete_heads(self) -> Tuple[str, ...]:
        return ("farmer_op", "farmer_item", "market_op", "market_item")

    def as_dict(self) -> Dict[str, int]:
        return {
            "farmer_op": self.farmer_ops,
            "farmer_item": self.farmer_items,
            "market_op": self.market_ops,
            "market_item": self.market_items,
        }


def kaggriculture_action_spec() -> MultiDiscreteSpec:
    return MultiDiscreteSpec(
        farmer_ops=len(FARMER_OPS),
        farmer_items=len(ITEMS),
        market_ops=len(MARKET_OPS),
        market_items=len(ITEMS),
        qty_continuous_aux=True,
    )


# Canonical framing string for docs / CLI banners.
PRIMARY_STACK = "BC → PPO (multi-discrete) + HER milestones"
ABLATION_STACK = "hierarchical Dueling Double DQN + PER (Path B ablation)"
IGNORE_FOR_ENV = ("A2C", "DDPG", "SAC", "TD3")
