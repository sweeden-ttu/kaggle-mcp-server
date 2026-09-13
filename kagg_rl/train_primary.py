"""Primary Kaggriculture RL entrypoint: BC → PPO + HER.

Demotes hierarchical DQN (Path B) to an ablation baseline. Continuous-control
algorithms (DDPG / SAC / TD3) and A2C are out of scope for this multi-discrete
farm action space.

Examples::

    # Dry-run plumbing (synthetic BC → PPO+HER)
    python -m kagg_rl.train_primary --dry-run --updates 5 \\
      --out checkpoints/ppo_her_primary.pt

    # Real BC from top-agent histories, then PPO+HER fine-tune
    python -m kagg_rl.train_primary \\
      --episodes-dir /path/to/top_agent_episodes \\
      --min-reward 100000 --winners-only \\
      --bc-epochs 3 --updates 20 \\
      --out checkpoints/ppo_her_primary.pt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch

from kagg_rl.action_space import (
    ABLATION_STACK,
    IGNORE_FOR_ENV,
    PRIMARY_STACK,
    kaggriculture_action_spec,
)
from kagg_rl.il.features import feature_dim
from kagg_rl.il.model import MultiHeadPolicy
from kagg_rl.ppo.train import load_bc_policy, ppo_update, synthetic_rollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_primary")


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Primary RL: {PRIMARY_STACK} (DQN Path B = ablation)"
    )
    p.add_argument("--bc-checkpoint", type=Path, default=None, help="Existing BC .pt")
    p.add_argument("--episodes-dir", type=Path, default=None)
    p.add_argument("--store-dir", type=Path, default=None)
    p.add_argument("--min-reward", type=float, default=0.0)
    p.add_argument("--winners-only", action="store_true")
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--bc-out", type=Path, default=Path("checkpoints/bc_primary.pt"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/ppo_her_primary.pt"))
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--her-scale", type=float, default=1.0)
    p.add_argument("--no-her", action="store_true")
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip episode BC; synthesize a BC checkpoint and run PPO+HER",
    )
    p.add_argument(
        "--note-ablation",
        action="store_true",
        help="Print Path B DQN ablation reminder and exit",
    )
    return p.parse_args()


def write_synthetic_bc(path: Path, device: torch.device) -> Path:
    """Create a randomly initialized BC-shaped checkpoint for dry-run PPO."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = feature_dim()
    model = MultiHeadPolicy(obs_dim=obs_dim, hidden=128, depth=2)
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "hidden": 128,
            "depth": 2,
            "meta": {"synthetic": True, "primary_stack": PRIMARY_STACK},
        },
        path,
    )
    logger.info("Wrote synthetic BC checkpoint %s (obs_dim=%d)", path, obs_dim)
    return path


def run_bc_cli(args) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "kagg_rl.il.train_bc",
        "--epochs",
        str(args.bc_epochs),
        "--out",
        str(args.bc_out),
        "--device",
        args.device,
        "--min-reward",
        str(args.min_reward),
    ]
    if args.winners_only:
        cmd.append("--winners-only")
    if args.episodes_dir is not None:
        cmd.extend(["--episodes-dir", str(args.episodes_dir)])
    elif args.store_dir is not None:
        cmd.extend(["--store-dir", str(args.store_dir)])
    else:
        raise SystemExit("Provide --episodes-dir, --store-dir, --bc-checkpoint, or --dry-run")
    logger.info("Running BC: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    return args.bc_out


def run_ppo_her(bc_path: Path, args) -> Path:
    device = torch.device(args.device)
    ckpt_meta = torch.load(bc_path, map_location=device, weights_only=False)
    model = load_bc_policy(bc_path, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_her = not args.no_her
    spec = kaggriculture_action_spec()

    for update in range(1, args.updates + 1):
        obs, actions, old_logp, ret, adv, her_meta = synthetic_rollout(
            model,
            device,
            horizon=args.horizon,
            use_her=use_her,
            her_scale=args.her_scale,
        )
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
        logger.info(
            "PPO+HER update %d  %s  milestones=%s",
            update,
            stats,
            her_meta.get("milestones_hit"),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "from_bc": str(bc_path),
            "primary_stack": PRIMARY_STACK,
            "ablation_stack": ABLATION_STACK,
            "ignore_algorithms": list(IGNORE_FOR_ENV),
            "use_her": use_her,
            "action_spec": spec.as_dict(),
            "obs_dim": ckpt_meta.get("obs_dim", model.backbone[0].in_features),
            "hidden": ckpt_meta.get("hidden", 256),
            "depth": ckpt_meta.get("depth", 3),
        },
        args.out,
    )
    logger.info("Primary checkpoint saved → %s", args.out)
    return args.out


def main():
    args = parse_args()
    spec = kaggriculture_action_spec()
    logger.info("PRIMARY: %s", PRIMARY_STACK)
    logger.info("ABLATION: %s", ABLATION_STACK)
    logger.info("Ignore for this env: %s", ", ".join(IGNORE_FOR_ENV))
    logger.info("Multi-discrete heads: %s", spec.as_dict())

    if args.note_ablation:
        logger.info(
            "Path B hierarchical DQN remains available via train_self_play / "
            "scripts/train_tier*_champion.py for ablations only — not the submission ceiling."
        )
        return

    if args.dry_run and args.bc_checkpoint is None:
        bc_path = write_synthetic_bc(args.bc_out, torch.device(args.device))
    elif args.bc_checkpoint is not None:
        bc_path = args.bc_checkpoint
    else:
        bc_path = run_bc_cli(args)

    run_ppo_her(bc_path, args)


if __name__ == "__main__":
    main()
