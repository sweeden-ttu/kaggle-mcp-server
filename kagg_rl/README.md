# Kaggriculture IL Bootstrap + PPO + HER (primary)

## Canonical stack

**Primary:** top-agent histories → **behavioral cloning** → **PPO** fine-tune / self-play + **HER** milestones  
**Ablation only:** hierarchical Dueling Double DQN + PER (Path B / `train_tier*_champion`)  
**Ignore for this env:** A2C, DDPG, SAC, TD3 (continuous / dominated)

Action space is **multi-discrete** (farmer op/item, market op/item) with soft qty regression heads — not continuous control.

```
Your top-agent episodes (until ~1 week before close)
        │
        ▼
  Filter (Elo / cash / winners / agent names)
        │
        ▼
  Behavioral cloning  →  checkpoints/bc.pt
        │
        ▼
  PPO fine-tune + HER milestones  →  checkpoints/ppo_her_primary.pt
        │
        ▼
  Ladder eval vs opponents/
```

## One-shot primary entrypoint

```bash
cd kaggle-mcp-server
# Dry-run (no episodes / no kaggle-environments)
python -m kagg_rl.train_primary --dry-run --updates 5 \
  --out checkpoints/ppo_her_primary.pt

# Or from repo root:
python scripts/train_ppo_her_primary.py --dry-run --updates 5
```

With real top-agent history:

```bash
python -m kagg_rl.train_primary \
  --episodes-dir /path/to/your/top_agent_episodes \
  --min-reward 149902 \
  --winners-only \
  --bc-epochs 5 \
  --updates 20 \
  --out checkpoints/ppo_her_primary.pt
```

## BC only

```bash
python -m kagg_rl.il.train_bc \
  --episodes-dir /path/to/your/top_agent_episodes \
  --min-reward 149902 \
  --winners-only \
  --top-k-seats 500 \
  --epochs 5 \
  --out checkpoints/bc.pt
```

HF-style store (`index.csv` + `seats.csv` + `episodes/{00-99}/`):

```bash
python -m kagg_rl.il.train_bc \
  --store-dir /path/to/datasets/il \
  --min-elo 2800 \
  --min-reward 100000 \
  --winners-only \
  --out checkpoints/bc.pt
```

## PPO + HER fine-tune (after BC)

```bash
python -m kagg_rl.ppo.train \
  --bc-checkpoint checkpoints/bc.pt \
  --dry-run \
  --updates 10 \
  --out checkpoints/ppo_ft.pt
```

`--no-her` disables milestone bonuses (ablation). Wire the real env in `kagg_rl/ppo/train.py` when ready.

## HER milestones

See `kagg_rl/her/milestones.py`. Default goals: survive/feed, cash 10k/50k/100k, late liquidation. Achieved goals densify sparse season-end money for GAE.

## Layout

| Path | Role |
|------|------|
| `kagg_rl/action_space.py` | Multi-discrete framing; primary vs ablation banners |
| `kagg_rl/her/` | Milestone HER relabeling |
| `kagg_rl/il/ingest.py` | Discover episodes, filter top seats |
| `kagg_rl/il/features.py` | Obs → float vector |
| `kagg_rl/il/actions.py` | Action → multi-head labels |
| `kagg_rl/il/dataset.py` | Build tensors from seats |
| `kagg_rl/il/model.py` | Shared multi-head policy + critic |
| `kagg_rl/il/train_bc.py` | BC trainer CLI |
| `kagg_rl/ppo/train.py` | PPO fine-tune from BC (+ HER) |
| `kagg_rl/train_primary.py` | **Primary** BC → PPO+HER entrypoint |

## Notes

- Farmer + first market order are modeled as classification heads (op/item) + qty regression.
- Prefer **high Elo + high final cash + winners** seats to avoid cloning weak play.
- After leaderboard histories stop (~1 week out), lean on self-play + ladder; do not expect pure offline DQN replay of an old meta to keep up.
- Path B DQN remains for controlled ablations only — not the submission brain.
