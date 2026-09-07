"""Behavioral cloning trainer: bootstrap a policy from top-agent gameplay history.

Example (your local dumps)::

    python -m kagg_rl.il.train_bc \\
      --episodes-dir /path/to/your/top_agent_episodes \\
      --min-reward 150000 \\
      --winners-only \\
      --epochs 5 \\
      --out checkpoints/bc.pt

With an HF-style store (index.csv + seats.csv + episodes/)::

    python -m kagg_rl.il.train_bc \\
      --store-dir /path/to/datasets/il \\
      --min-elo 2800 \\
      --min-reward 100000 \\
      --out checkpoints/bc.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Set

import torch
from torch.utils.data import DataLoader

from .dataset import build_arrays_from_seats, get_feature_dim, train_val_split
from .ingest import (
    attach_paths,
    load_index_elo,
    load_seat_table,
    seats_from_local_episodes,
    select_top_seats,
)
from .model import MultiHeadPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_bc")


def parse_args():
    p = argparse.ArgumentParser(description="Kaggriculture IL bootstrap (behavioral cloning)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--episodes-dir",
        type=Path,
        help="Directory of episode JSON/JSON.GZ files (your gameplay history)",
    )
    src.add_argument(
        "--store-dir",
        type=Path,
        help="HF-style IL store root containing index.csv, seats.csv, episodes/",
    )
    p.add_argument("--min-reward", type=float, default=0.0, help="Keep seats with final cash >= this")
    p.add_argument("--min-elo", type=float, default=0.0, help="Keep episodes with elo >= this (store-dir)")
    p.add_argument("--winners-only", action="store_true")
    p.add_argument("--top-k-seats", type=int, default=None, help="Keep only top-K seats by reward")
    p.add_argument(
        "--top-agents",
        type=str,
        default="",
        help="Comma-separated agent/team names to keep (exact match)",
    )
    p.add_argument("--max-transitions", type=int, default=200_000)
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth turn (speed/diversity)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints/bc.pt"))
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def resolve_seats(args):
    top_agents: Optional[Set[str]] = None
    if args.top_agents.strip():
        top_agents = {a.strip() for a in args.top_agents.split(",") if a.strip()}

    if args.episodes_dir is not None:
        logger.info("Scanning local episodes under %s", args.episodes_dir)
        seats = seats_from_local_episodes(
            args.episodes_dir,
            min_reward=args.min_reward,
            winners_only=args.winners_only,
        )
        if top_agents:
            seats = [s for s in seats if s.agent in top_agents]
        if args.top_k_seats:
            seats = seats[: args.top_k_seats]
        return seats

    store = args.store_dir
    seats_csv = store / "seats.csv"
    index_csv = store / "index.csv"
    episodes_root = store / "episodes" if (store / "episodes").exists() else store
    if not seats_csv.exists():
        raise FileNotFoundError(f"seats.csv not found in {store}")

    seats = load_seat_table(seats_csv)
    elo = load_index_elo(index_csv) if index_csv.exists() else {}
    seats = select_top_seats(
        seats,
        min_reward=args.min_reward,
        min_elo=args.min_elo,
        elo_by_episode=elo,
        winners_only=args.winners_only,
        top_agents=top_agents,
        top_k_by_reward=args.top_k_seats,
    )
    seats = attach_paths(seats, episodes_root)
    return seats


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    seats = resolve_seats(args)
    logger.info("Selected %d seats for IL", len(seats))
    if not seats:
        raise SystemExit("No seats selected — relax filters or check data path.")

    X, Y = build_arrays_from_seats(
        seats,
        max_transitions=args.max_transitions,
        stride=args.stride,
    )
    train_ds, val_ds = train_val_split(X, Y, val_frac=args.val_frac, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = MultiHeadPolicy(obs_dim=get_feature_dim(), hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            loss, _ = model.bc_loss(xb, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(n, 1)

        model.eval()
        vtotal = 0.0
        vn = 0
        acc_fo = 0.0
        acc_mo = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                loss, _ = model.bc_loss(xb, yb)
                acc = model.accuracy(xb, yb)
                vtotal += float(loss) * xb.size(0)
                vn += xb.size(0)
                acc_fo += acc["farmer_op_acc"] * xb.size(0)
                acc_mo += acc["market_op_acc"] * xb.size(0)
        val_loss = vtotal / max(vn, 1)
        fo = acc_fo / max(vn, 1)
        mo = acc_mo / max(vn, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "farmer_op_acc": fo,
            "market_op_acc": mo,
        }
        history.append(row)
        logger.info(
            "epoch %d  train=%.4f  val=%.4f  farmer_acc=%.3f  market_acc=%.3f",
            epoch,
            train_loss,
            val_loss,
            fo,
            mo,
        )

        if val_loss < best_val:
            best_val = val_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_dim": get_feature_dim(),
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "meta": {
                        "n_seats": len(seats),
                        "n_transitions": int(X.shape[0]),
                        "min_reward": args.min_reward,
                        "min_elo": args.min_elo,
                        "winners_only": args.winners_only,
                    },
                    "history": history,
                },
                args.out,
            )
            logger.info("saved checkpoint %s", args.out)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(history, indent=2))
    logger.info("done. best_val=%.4f  metrics=%s", best_val, metrics_path)


if __name__ == "__main__":
    main()
