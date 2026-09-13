"""Action vocabularies and encoding for Kaggriculture episodes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Farmer primary opcodes observed in top-agent tapes.
FARMER_OPS = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "WATER",
    "HARVEST",
    "FEED",
    "CARE",
    "COLLECT_FERTILIZER",
    "BUILD_PASTURE",
    "DROP",
    "PICKUP",
    "PLACE",
    "PLANT",
    "FERTILIZE",
    "UNLOCK",
    "OTHER",
]

# Market primary opcodes (first order only for BC v0).
MARKET_OPS = [
    "NONE",
    "HIRE",
    "SELL",
    "BUY_SEED",
    "BUY_ANIMAL",
    "BUY_QUADRANT",
    "BUY_FERTILIZER",
    "OTHER",
]

ITEMS = [
    "NONE",
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "COW",
    "SHEEP",
    "GOOSE",
    "OTHER",
]

FARMER_OP_TO_ID = {n: i for i, n in enumerate(FARMER_OPS)}
MARKET_OP_TO_ID = {n: i for i, n in enumerate(MARKET_OPS)}
ITEM_TO_ID = {n: i for i, n in enumerate(ITEMS)}


def _norm_token(tok: Any) -> str:
    if tok is None:
        return "NONE"
    return str(tok).upper()


def encode_farmer_action(farmer: Any) -> Tuple[int, int, float]:
    """Encode farmer command list → (op_id, item_id, qty_norm)."""
    if not farmer:
        return FARMER_OP_TO_ID["PASS"], ITEM_TO_ID["NONE"], 0.0

    if isinstance(farmer, str):
        toks = [farmer]
    elif isinstance(farmer, list):
        toks = farmer
    else:
        toks = [farmer]

    op = _norm_token(toks[0]) if toks else "PASS"
    if op not in FARMER_OP_TO_ID:
        op = "OTHER"

    item = "NONE"
    qty = 0.0
    if len(toks) >= 2 and not str(toks[1]).replace(".", "", 1).isdigit():
        item = _norm_token(toks[1])
        if item not in ITEM_TO_ID:
            item = "OTHER"
    if len(toks) >= 2:
        for t in toks[1:]:
            try:
                qty = float(t)
                break
            except (TypeError, ValueError):
                continue

    # Normalize quantity into a soft [0,1] for regression head (cap at 20).
    qty_norm = max(0.0, min(qty / 20.0, 1.0))
    return FARMER_OP_TO_ID[op], ITEM_TO_ID[item], qty_norm


def encode_market_action(market: Any) -> Tuple[int, int, float]:
    """Encode first market order → (op_id, item_id, qty_norm)."""
    if not market:
        return MARKET_OP_TO_ID["NONE"], ITEM_TO_ID["NONE"], 0.0

    order = market[0] if isinstance(market, list) else market
    if isinstance(order, str):
        toks = [order]
    elif isinstance(order, list):
        toks = order
    else:
        toks = [order]

    op = _norm_token(toks[0]) if toks else "NONE"
    if op not in MARKET_OP_TO_ID:
        op = "OTHER"

    item = "NONE"
    qty = 0.0
    if len(toks) >= 2 and not str(toks[1]).replace(".", "", 1).isdigit():
        item = _norm_token(toks[1])
        if item not in ITEM_TO_ID:
            item = "OTHER"
    if len(toks) >= 2:
        for t in toks[1:]:
            try:
                qty = float(t)
                break
            except (TypeError, ValueError):
                continue

    qty_norm = max(0.0, min(qty / 20.0, 1.0))
    return MARKET_OP_TO_ID[op], ITEM_TO_ID[item], qty_norm


def encode_step_action(action: Optional[Dict[str, Any]]) -> Dict[str, float]:
    action = action or {}
    f_op, f_item, f_qty = encode_farmer_action(action.get("farmer"))
    m_op, m_item, m_qty = encode_market_action(action.get("market"))
    return {
        "farmer_op": f_op,
        "farmer_item": f_item,
        "farmer_qty": f_qty,
        "market_op": m_op,
        "market_item": m_item,
        "market_qty": m_qty,
    }
