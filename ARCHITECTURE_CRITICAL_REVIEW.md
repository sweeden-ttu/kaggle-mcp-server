# Critical Architecture Review & Sprint Specs

**Subject:** Kaggriculture knowledge pipeline (GBDT router + MoE convergence + expert YAML)  
**Verdict:** Useful as a *knowledge inventory and planning scaffold*; **not yet an architecture that can compete**. The current system routes text queries to strategy labels. Kaggriculture requires an `agent(obs) → action` loop that earns money over 720 turns against live opponents.

---

## 1. Critical Review by Component

### 1.1 Branch Feature Extractor (`extract_branch_features.py` + `branch_manifest.json`)

| Dimension | Assessment |
|-----------|------------|
| Intent | Vectorize 137 automation branches as training examples |
| What it actually measures | Diff size, module presence flags, maturity heuristic |
| Information content | **Low.** 137/137 branches have the full 7-module stack. Features barely discriminate. |
| Ground truth | **None.** No label tied to game win rate, bank balance, or Elo |
| Risk | Treating LOC/maturity as signal invents a phantom ranking |

**Critical failure:** Homogeneous branches cannot train a meaningful GBDT. You are boosting on near-constant features.

**Keep:** The manifest as an *audit of what the MoE automation produced*.  
**Discard for training:** Using branch maturity as a target proxy for agent quality.

---

### 1.2 GBDT Expert Router (`kaggriculture_router.py`)

| Dimension | Assessment |
|-----------|------------|
| Intent | Route NL queries → expert domains via boosted stumps |
| What it actually does | Keyword overlap (60%) + stump ensemble on synthetic residuals (40%) |
| Training | Stumps fit residuals of hand-crafted domain scores from branch maturity — **circular** |
| Prediction | Returns skill names and formula strings, not actions |
| Coupling to competition | **None.** No `obs` object, no action space, no environment |

**Critical failures:**
1. **Metaphorical GBDT.** Branches are not weak learners; residuals are not prediction errors on a real target.
2. **Keyword routing is sufficient** for the current task (and cheaper). The 50-stump ensemble adds complexity without validated lift.
3. **Wrong I/O contract.** Competition needs `obs → action`, not `query → strategy YAML`.
4. **Skill names are fictional** relative to this game (`kaggle-rl`, `kaggle-optimizer` are MoE labels, not callable game modules).

**Keep:** Domain taxonomy + keyword → expert mapping as a *dispatcher for human/agent planning*.  
**Replace for competition:** A policy router that selects among *implemented* agent strategies given game state features.

---

### 1.3 Expert Registry YAML (`kaggriculture_experts.yaml`)

| Dimension | Assessment |
|-----------|------------|
| Intent | Encode 8 game-agent expert definitions |
| Strength | Correct competition framing; reference tiers; honest meta notes |
| Weakness | Experts are **documentation**, not code. No callable implementations. |
| Formulas | Symbolic objectives, not trainable losses on episode returns |
| Loop configs | Copied MoE convergence params; unrelated to game reward |

**Critical failure:** Specs without implementations create a false sense of readiness. An expert that “does PPO” but has no trainer is a wish list item.

**Keep:** As product backlog / capability map.  
**Require:** Each expert must own a module with a testable interface (see Sprint specs).

---

### 1.4 Convergence Pipeline (`kaggriculture_pipeline.py`)

| Dimension | Assessment |
|-----------|------------|
| Intent | Iterate expert scores until L2 state converges |
| What converges | Synthetic scores with decaying noise — **not** bank balance or win rate |
| Demonstrated behavior | Monotonic L2 decay to ε ≈ 0.001 in ~20 iters |
| Validity | Convergence proves the loop algebra works; proves nothing about agent quality |

**Critical failure:** Optimizing `||sₙ − sₙ₋₁||₂` without a game-linked objective is **cargo-cult optimization**. You can converge forever on a bad strategy.

**Keep:** Loop controller pattern *if* state = real performance features (Elo, bank, win rate vs reference tiers).  
**Discard:** Hash-based noise / decay as stand-in for training feedback.

---

### 1.5 MLSysEng MoE Stack (branch `mlsyseng-moe-system-32ad`, not on main)

| Component | Fit for Kaggriculture |
|-----------|----------------------|
| PDF docling → chapter experts | **Poor.** Game needs episode replays, not textbook chapters |
| ChromaDB RAG over ML Principles | Useful only for literature/strategy notes; not for `agent()` |
| `build_entry` / `evolve` | Designed for notebook competitions; `evolve` uses **random scores** |
| Kaggle MCP tools | **Useful** — download kernels, datasets, competition files |
| Rust K-map / AC matcher | Orthogonal to farm simulation (unless used for strategy fingerprinting text) |
| FOL / Bayesian workbench on main | Orthogonal — formal logic UI, not game agent |

**Critical failure:** The MoE automation solved “extract ML textbook → register experts.” Kaggriculture needs “simulate farm → choose action → climb Elo.” Reusing MoE without rewiring I/O is architectural mismatch.

---

### 1.6 Data Analysis Gap (what is missing entirely)

| Needed data | Status |
|-------------|--------|
| Official/local game environment API | Not integrated |
| Episode replay JSON schema + parser | Documented only |
| Price curve reconstruction from market inventory | Spec only |
| Leaderboard / Elo telemetry | External datasets exist; not ingested |
| Reference agent ladder evaluation harness | Not built |
| Action space / observation space types | Not typed in code |

Without these, “training” and “prediction” are undefined.

---

### 1.7 Training & Prediction (as currently claimed)

| Claim | Reality |
|-------|---------|
| Training on 137 branches | Feature extraction + fitting synthetic residuals |
| Prediction | NL → domain scores → YAML strategy string |
| Evaluation | Manual visual inspection of printouts |
| Competition submission artifact | **None** |

**Bottom line:** There is no trained game policy and no prediction that interacts with the environment.

---

## 2. Architectural Verdict (one paragraph)

The pipeline correctly *discovered* the competition type and *catalogued* MoE automation output, then overfit a GBDT metaphor onto homogeneous git branches. Convergence and expert YAML look like a system but optimize the wrong objective. The salvageable core is: (1) competition knowledge + reference tiers, (2) Kaggle MCP for data/kernels, (3) a future **real** ensemble/router over *implemented* strategies scored by local simulation. Everything else must be rebuilt as an agent stack with measurable bank/Elo feedback.

---

## 3. Sprint Plan Overview

Sprints are ordered by dependency. Each sprint lists **specs** (acceptance criteria). Prefer shipping a weak but playable agent early over polishing the router.

```
Sprint 0  Critical freeze / truth audit
Sprint 1  Environment + observation/action contracts
Sprint 2  Replay ingestion + strategy fingerprinting
Sprint 3  Deterministic baseline agent (meta-tier target)
Sprint 4  Economic models (price, crop, livestock)
Sprint 5  Local evaluation harness vs reference tiers
Sprint 6  Real strategy router (replace metaphorical GBDT)
Sprint 7  Imitation learning from replays
Sprint 8  PPO / hybrid RL
Sprint 9  Opponent adaptation + submission pipeline
Sprint 10 MoE salvage (optional knowledge layer)
```

---

## Sprint 0 — Critical Freeze & Truth Audit

**Goal:** Stop treating planning docs as a trainable system. Establish a single source of truth for what is real vs aspirational.

### Spec 0.1 — Architecture status matrix
- Produce a living table: Component × {exists | stub | missing} × {used in agent | not used}.
- Mark `kaggriculture_router.fit()`, convergence loop, and branch maturity scores as **non-competitive** unless rewired to game metrics.
- **Done when:** Matrix committed; CI or README links to it.

### Spec 0.2 — Delete or quarantine false claims
- Update docs so “trained GBDT on 137 branches” is described as *prototype metaphor*, not production training.
- Gate any “CONVERGED” messaging behind a real metric (bank/Elo) or remove it from demos.
- **Done when:** No user-facing string implies game-ready training without evidence.

### Spec 0.3 — Success metrics definition
- Primary: Elo / Bradley-Terry rating on Kaggle ladder (or local proxy).
- Secondary: mean bank vs reference tiers 0–9.
- Tertiary: win rate vs byte-identical meta script.
- **Done when:** Metrics written with formulas and evaluation frequency.

---

## Sprint 1 — Environment & Contracts (Architecture: I/O)

**Goal:** Define the only interface that matters: observation → action.

### Spec 1.1 — Observation schema
- Typed model for `obs` (grid, player state, market inventories, farmhands, turn index, opponent-visible fields if any).
- Versioned JSON Schema + Python dataclasses/Pydantic.
- **Done when:** Round-trip parse of at least one real or recorded obs blob; unit tests.

### Spec 1.2 — Action schema
- Enumerate legal actions: move, buy, plant, water, fertilize, harvest, sell, hire, buy_quadrant, no-op.
- Validator: `is_legal(obs, action) → bool` with reason codes.
- **Done when:** Exhaustive tests for illegality (wrong tile, insufficient funds, wrong turn phase).

### Spec 1.3 — Agent protocol
- Interface: `class Agent: def act(self, obs) -> Action`.
- Deterministic seed support for replayability.
- **Done when:** Dummy agent returns legal actions for 720 turns in dry-run.

### Spec 1.4 — Local env adapter
- Wrap official or community simulator so `step(action) → (obs, reward, done, info)`.
- Reward default: Δbank (configurable).
- **Done when:** One episode completes offline without Kaggle network.

---

## Sprint 2 — Data Analysis: Replays & Telemetry

**Goal:** Turn public episodes into analyzable datasets.

### Spec 2.1 — Replay ingest
- Load Kaggriculture episode JSON (community datasets).
- Normalize to `(episode_id, turn, player, action, bank, market_state)`.
- **Done when:** ≥1k episodes ingested; schema validation report.

### Spec 2.2 — Strategy fingerprinting
- Features: crop mix timeline, hire turn, sell cadence, quadrant buy turn, action n-grams.
- Cluster / hash to detect meta clones.
- **Done when:** Reproduce “byte-identical meta” clustering on a labeled sample; precision/recall report.

### Spec 2.3 — Price curve reconstruction
- From replays: fit or table `price(product, inventory)`.
- Quantify sell-floor risk (units until $1).
- **Done when:** Offline plot + RMSE vs observed prices; documented assumptions.

### Spec 2.4 — Leaderboard telemetry snapshot
- Optional: ingest ladder telemetry for Elo trends.
- **Done when:** Time series of top scores stored; not required for agent v0.

---

## Sprint 3 — Deterministic Field Plan Agent (Architecture: Policy Core)

**Goal:** Ship a competitive *baseline* that can reach near-meta bank in self-play/local sim.

### Spec 3.1 — Script representation
- Data structure for a 720-turn (or 712-turn) plan: sequence of high-level ops + executor binding to tiles.
- Editable, not only hardcoded bytes.
- **Done when:** Load/save plan JSON; execute in env.

### Spec 3.2 — Priority executor
- Micro-policy: harvest > water > weed > plant; market free actions first.
- Pathfinding: BFS / Manhattan with farmhand assignment.
- **Done when:** Unit tests for path length and priority order.

### Spec 3.3 — Meta extraction
- From Sprint 2 fingerprints, extract one candidate meta plan.
- Replay plan in local env; measure bank.
- **Done when:** Mean bank ≥ tier-5 ($53k) minimum; stretch ≥ $150k toward meta.

### Spec 3.4 — Sell-order variants
- Parameterize sell layer (Broker/Ledger/Slotter/Closer differences).
- Grid search offline.
- **Done when:** Documented ranking of sell variants by bank; best variant selected as default.

---

## Sprint 4 — Economic Submodels (Data Analysis → Features)

**Goal:** Replace symbolic YAML formulas with computed features used by agents.

### Spec 4.1 — Crop NPV module
- Inputs: growth time, seed cost, expected sell price path, fertilizer bonus.
- Output: profit-per-tile ranking over remaining season.
- **Done when:** Matches qualitative meta (melons premium early; wheat for feed).

### Spec 4.2 — Market timing module
- Given inventory + concurrent sells model, recommend sell quantity this turn.
- Constraint: avoid price floor when possible.
- **Done when:** Backtest on replays improves revenue vs dump-all baseline.

### Spec 4.3 — Livestock ROI module
- Feed-chain accounting (wheat → products → fertilizer).
- **Done when:** Reproduces Rancher-Rita-class ROI ordering vs crop-only under fixed assumptions.

### Spec 4.4 — Feature vector for routing/RL
- Compact numeric state: bank, turn, market inventories, open tasks, crop ages, animal states, farmhand count.
- **Done when:** Stable feature dim documented; used by later sprints.

---

## Sprint 5 — Evaluation Harness (Training Infrastructure)

**Goal:** Make “better” measurable before any learning.

### Spec 5.1 — Reference tier suite
- Encode tiers 0–9 as opponent/benchmark agents (or stubs with recorded banks).
- **Done when:** `evaluate(agent, n_episodes)` returns bank mean/std and win rates.

### Spec 5.2 — Elo proxy
- Bradley-Terry / Elo among local agents.
- **Done when:** Ranking stable under re-seed; correlates with bank on held-out matchups.

### Spec 5.3 — Regression gates
- CI job: deterministic agent must not drop below bank threshold.
- **Done when:** Failing agent blocks merge.

### Spec 5.4 — Kill the fake convergence metric
- Replace pipeline demo with harness output (bank/Elo).
- Optional: keep L2 loop only if state includes real metrics.
- **Done when:** Demo scripts print competition metrics, not synthetic L2 alone.

---

## Sprint 6 — Real Strategy Router (Replace Metaphorical GBDT)

**Goal:** Dynamically pick the most useful *implemented* path — the original GBDT intent, grounded.

### Spec 6.1 — Strategy registry
- Register concrete agents: `MetaPlanV1`, `MelonFocus`, `LivestockScale`, `HybridCEO`, etc.
- Each exposes: `act`, `name`, `prerequisites`, `eval_stats`.
- **Done when:** Registry loads without YAML-only experts.

### Spec 6.2 — State-based router (not NL-keyword GBDT)
- Input: Sprint 4.4 feature vector (and/or turn phase).
- Output: strategy id + confidence.
- Models allowed: decision tree / GBDT / rules — **trained on labels from Sprint 5**.
- **Done when:** Cross-validated lift vs always-pick-MetaPlan on local Elo/bank.

### Spec 6.3 — Training labels
- Label = argmax strategy bank/Elo on matched seeds or self-play buckets.
- **Done when:** Dataset of (state, best_strategy) with leakage controls (no future market peek beyond obs).

### Spec 6.4 — Deprecate branch-maturity GBDT
- Move current router to `experiments/metaphor_gbdt/` or delete from critical path.
- **Done when:** Main pipeline imports real router only.

---

## Sprint 7 — Imitation Learning (Training)

**Goal:** Bootstrap policy from top replays before RL.

### Spec 7.1 — Dataset
- Map replay actions → supervised `(obs_features, action)` pairs.
- Filter to high-Elo episodes / meta cluster.
- **Done when:** Train/val split; class balance report for actions.

### Spec 7.2 — Behavioral cloning model
- Train classifier/policy head over discrete actions (or hierarchical: macro then micro).
- **Done when:** Val action accuracy above agreed baseline; legal-action mask applied.

### Spec 7.3 — Distillation into executor
- IL policy may propose goals; deterministic executor carries out movement.
- **Done when:** Hybrid agent bank ≥ IL-only and ≥ weak deterministic.

---

## Sprint 8 — PPO / Hybrid RL (Training)

**Goal:** Break or approach meta via learning with game reward.

### Spec 8.1 — Fast simulator path
- Target throughput (e.g. ≥1k–10k steps/sec) via vectorized or Rust/JAX env if available.
- **Done when:** Throughput benchmark logged.

### Spec 8.2 — Reward shaping
- Terminal bank + optional diversity / anti-floor penalties.
- Ablation table.
- **Done when:** At least 3 reward variants compared on Elo proxy.

### Spec 8.3 — Self-play curriculum
- Mix: self / active / banked opponents (e.g. 50/25/25).
- Include meta deterministic as banked opponent.
- **Done when:** Learning curves; no collapse to no-op.

### Spec 8.4 — Hybrid CEO/executor
- RL selects high-level intents (crop focus, hire, sell aggression); executor does pathfinding.
- **Done when:** Beats IL baseline; reports win rate vs meta script.

### Spec 8.5 — Prediction contract
- Online: `policy.predict(obs_features) → action` under latency budget for Kaggle runtime.
- **Done when:** Profiled p95 latency within competition limits.

---

## Sprint 9 — Opponent Adaptation & Submission

**Goal:** Use live observation to switch strategies; ship.

### Spec 9.1 — Opponent classifier
- Early-turn features → {meta_clone, livestock, melon, unknown}.
- **Done when:** Accuracy on held-out replays; confusion matrix.

### Spec 9.2 — Counter-policy table
- Map opponent class → strategy (e.g. exploit fixed sell schedule).
- **Done when:** Positive expected bank delta vs always-meta in matched sims.

### Spec 9.3 — Endgame search
- Last N turns: limited tree search / DP on remaining inventory sells.
- **Done when:** Measurable bank lift in endgame ablation.

### Spec 9.4 — Submission packaging
- Single `agent.py` (or required Kaggle entrypoint) with frozen weights/scripts.
- Smoke test: 5 local episodes + optional Kaggle submit.
- **Done when:** Valid submission accepted; daily quota plan documented.

---

## Sprint 10 — MoE Salvage (Optional Knowledge Layer)

**Goal:** Reuse automation work only where it helps competition engineering.

### Spec 10.1 — Kaggle MCP integration
- Automate download of kernels, episode datasets, competition files.
- **Done when:** One-command data refresh.

### Spec 10.2 — Strategy RAG (not ML Principles PDFs)
- Index *replay summaries, discussion notes, your own run logs*.
- Query: “why did bank stall at turn 400?”
- **Done when:** Retrieval helps humans; not required in `agent()`.

### Spec 10.3 — AC matcher for meta text
- Optional: scan discussions/kernels for strategy fingerprints.
- **Done when:** Useful alerts; no dependency in hot path.

### Spec 10.4 — Do not port PDF chapter experts into agent loop
- Explicit non-goal: textbook MoE experts selecting skills mid-game.
- **Done when:** Documented as out-of-scope for runtime.

---

## 4. Mapping: Current Components → Sprint Disposition

| Current artifact | Disposition |
|------------------|-------------|
| `branch_manifest.json` | Archive / audit (Sprint 0); not training data |
| Metaphor GBDT router | Quarantine (Sprint 6.4); replace with Spec 6.2 |
| Convergence on synthetic L2 | Replace with harness metrics (Sprint 5.4) |
| `kaggriculture_experts.yaml` | Backlog seed for Specs 3–9; each expert must become a module |
| `KAGGRICULTURE_PIPELINE.md` | Keep as context; rewrite “ready to compete” claims |
| MLSysEng MoE (`mlsyseng_mcp`) | Salvage MCP + optional RAG (Sprint 10); not agent core |
| FOL / Bayesian workbench | Out of critical path |

---

## 5. Recommended Sequencing for the Next Working Period

1. **Sprint 0 + 1** immediately — contracts and honesty.  
2. **Sprint 2 + 3** next — data + deterministic agent (highest Elo/$ ROI before deadline).  
3. **Sprint 5** in parallel with 3 — without harness you cannot trust progress.  
4. **Sprint 4** feeds 6–8.  
5. **Sprint 6** only after ≥2 real strategies exist to route between.  
6. **Sprint 7–9** only if meta ceiling is hit and time remains.  
7. **Sprint 10** anytime as tooling, never as blocker.

---

## 6. One-Sentence Spec for “GBDT over 137 branches”

**Reject as competition architecture:** boosting git-branch maturity to pick expert labels does not train or predict farm actions.  
**Accept as research note:** an interesting metaphor for routing among *future* implemented strategies, once those strategies have bank/Elo labels from a real simulator.
