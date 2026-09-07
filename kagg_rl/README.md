# Kaggriculture IL Bootstrap + PPO Fine-tune

## What is imitation learning here?

**Behavioral cloning (BC):** treat top agents' gameplay as labeled demos  
`(observation → action)` and train a policy to imitate them with supervised learning.

That is the right bootstrap when you already have **all top-agent gameplay history**:
you skip cold-start PPO exploration and start near the meta.

**Then PPO** (optional): fine-tune the BC policy online so it can beat clones / adapt,
using a clipped policy-ratio update so training stays stable.

```
Your top-agent episodes
        │
        ▼
  Filter (Elo / cash / winners / agent names)
        │
        ▼
  Behavioral cloning  →  checkpoints/bc.pt
        │
        ▼
  PPO fine-tune (env) →  checkpoints/ppo_ft.pt
```

## Point the trainer at YOUR data

### Option A — flat folder of episode dumps

```bash
# Activate conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kagg-ppo

python -m kagg_rl.il.train_bc \
  --episodes-dir /path/to/your/top_agent_episodes \
  --min-reward 150000 \
  --winners-only \
  --top-k-seats 500 \
  --epochs 5 \
  --out checkpoints/bc.pt
```

Accepted files: `*.json` or `*.json.gz` Kaggle episode tapes with `steps[t][seat].observation/action`.

### Option B — HF-style store (`index.csv` + `seats.csv` + `episodes/{00-99}/`)

```bash
python -m kagg_rl.il.train_bc \
  --store-dir /path/to/datasets/il \
  --min-elo 2800 \
  --min-reward 100000 \
  --winners-only \
  --out checkpoints/bc.pt
```

### Filter to named top agents

```bash
python -m kagg_rl.il.train_bc \
  --episodes-dir /path/to/episodes \
  --top-agents "tetsuya,Crop Dusta,QQ Farming" \
  --min-reward 100000 \
  --out checkpoints/bc_top3.pt
```

## PPO fine-tune (after BC)

```bash
python -m kagg_rl.ppo.train \
  --bc-checkpoint checkpoints/bc.pt \
  --dry-run \
  --updates 10 \
  --out checkpoints/ppo_ft.pt
```

`--dry-run` exercises the clipped PPO objective without `kaggle-environments`.
Wire the real env in `kagg_rl/ppo/train.py` when ready.

## Layout

| Path | Role |
|------|------|
| `kagg_rl/il/ingest.py` | Discover episodes, filter top seats |
| `kagg_rl/il/features.py` | Obs → float vector |
| `kagg_rl/il/actions.py` | Action → multi-head labels |
| `kagg_rl/il/dataset.py` | Build tensors from seats |
| `kagg_rl/il/model.py` | Shared multi-head policy + critic |
| `kagg_rl/il/train_bc.py` | BC trainer CLI |
| `kagg_rl/ppo/train.py` | PPO fine-tune from BC |

## Notes

- Farmer + first market order are modeled as classification heads (op/item) + qty regression.
- Full multi-order market sequences / hand micro-actions are future work.
- Prefer **high Elo + high final cash + winners** seats to avoid cloning weak play.
- Deduplicate near-identical meta scripts (cluster/minhash) before training if your dump is clone-heavy — otherwise BC overfits one sell-order variant.
