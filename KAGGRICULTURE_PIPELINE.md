# Kaggriculture Challenge: ML Knowledge Pipeline

## What Kaggriculture Actually Is

**Kaggriculture is NOT a traditional ML/data-science competition.** It is a **turn-based farming simulation game** where two autonomous AI agents compete head-to-head on separate 10x10 tile farms over **720 turns** (30 in-game days × 24 turns/day). The agent with the most **money in the bank** at the end wins.

| Key Fact | Value |
|----------|-------|
| URL | [kaggle.com/competitions/kaggriculture](https://www.kaggle.com/competitions/kaggriculture) |
| Prize | $50,000 (top 10 × $5,000) |
| Teams | ~6,480+ |
| Deadline | September 30, 2026 |
| Evaluation | Bradley-Terry skill rating (ELO-like) |
| Current meta | Deterministic "field plan" scripts (~$186k bank) |
| RL ceiling | Emerging PPO/JAX approaches |

**Game mechanics**: Move on grid, buy seeds/animals, plant, water, fertilize, harvest, sell at dynamic market prices, hire farmhands, buy land quadrants. Market prices are supply/demand driven (start at 10k inventory, prices drop as inventory rises).

## What This Document Is

Over the past 3-4 months, approximately **150 automation branches** (`competition-mlsyseng-moe-system-*` and `sweeden-ttu/mlsyseng-moe-system-*`) ran on this repository, iteratively building an **MLSysEng Mixture of Experts** (MoE) system. Each run attempted to construct a pipeline for extracting ML knowledge from textbook PDFs, registering chapter-level experts, and using those experts to build Kaggle competition entries.

This document distills the accumulated knowledge, architecture patterns, and operational code from those 150+ runs into a single actionable pipeline for the **Kaggriculture** challenge. The MoE system's expert routing, convergence loops, and skill mapping have been retargeted from traditional ML domains to **game-agent AI domains** (deterministic planning, RL, market timing, pathfinding, opponent adaptation).

---

## 1. Architecture Overview: What the Automation Built

### 1.1 The MLSysEng MoE System

The core system is a **Mixture of Experts** architecture where each ML textbook chapter becomes a registered "expert" with:

- **Capabilities**: What the expert knows (inferred from chapter concepts)
- **Skills**: Which Kaggle tool-skills it activates (feature engineering, model training, etc.)
- **Strategy**: The expert's recommended workflow
- **Formula**: Mathematical objective (loss function, metrics)
- **Loop Config**: Convergence parameters for iterative refinement

**Components** (all implemented across the 150 branches):

| Component | File | Purpose |
|-----------|------|---------|
| MCP Server | `mlsyseng_mcp/server.py` | FastMCP server exposing 8 tools for knowledge extraction, expert querying, competition entry building |
| Database | `mlsyseng_mcp/database.py` | SQLite with WAL mode: chapters, experts, extraction status, convergence state |
| Docling Worker | `mlsyseng_mcp/docling_worker.py` | PDF extraction from ML Principles textbooks, concept identification via regex patterns |
| Embeddings | `mlsyseng_mcp/embeddings.py` | sentence-transformers (`all-MiniLM-L6-v2`) + ChromaDB for RAG retrieval |
| Expert Registry | `mlsyseng_mcp/expert_registry.py` | Registers experts from chapters, maps concepts to Kaggle skills |
| Loop Controller | `mlsyseng_mcp/loop_controller.py` | State convergence loop: `\|\|state[n] - state[n-1]\|\|_2 < epsilon` |
| Skill Generator | `skill_generator.py` | Generates platform configs for OpenClaw, Cursor, Claude Desktop, Gemini |
| Rust CLI | `kagg/rust/` | High-performance reimplementation: K-map simplification, Aho-Corasick pattern matching, expert registry |
| Ruby Lib | `kagg/ruby/` | Aho-Corasick matcher and K-map tests in Ruby |

### 1.2 The Knowledge Flow

```
ML Principles PDFs
        │
        ▼
  Docling Worker (extract text, identify concepts)
        │
        ▼
  SQLite Database (chapters table)
        │
        ▼
  Embedding Engine (chunk → embed → ChromaDB)
        │
        ▼
  Expert Registry (register chapter experts with skills/strategy/formula)
        │
        ▼
  Competition Entry Builder (RAG search → select experts → assemble skills)
        │
        ▼
  Convergence Loop (iterate until ||state[n]-state[n-1]||₂ < ε)
        │
        ▼
  Kaggle Notebook / Submission
```

---

## 2. ML Concept Vocabulary Extracted

The docling worker identifies these ML concepts from source material (used by the expert registry to map to Kaggle skills):

| Concept Domain | Patterns Matched | Kaggle Skills Activated |
|---------------|------------------|------------------------|
| Optimization | gradient descent, SGD, Adam, RMSProp, learning rate, momentum | `kaggle-optimizer`, `kaggle-model-trainer` |
| Deep Learning | neural network, deep learning, convolutional | `kaggle-deep-learning`, `kaggle-model-trainer` |
| Regularization | regularization, dropout, batch norm | `kaggle-model-trainer`, `kaggle-optimizer` |
| Feature Engineering | feature engineering, feature selection, dimensionality reduction | `kaggle-preprocessor`, `kaggle-feature-engineer` |
| Ensembles | ensemble, bagging, boosting, random forest | `kaggle-ensemble`, `kaggle-model-trainer` |
| Transformers | attention mechanism, transformer, self-attention | `kaggle-deep-learning`, `kaggle-nlp` |
| Validation | cross-validation, train-test split, overfitting, underfitting | `kaggle-validator`, `kaggle-preprocessor` |
| Unsupervised | clustering, k-means, hierarchical | `kaggle-unsupervised`, `kaggle-preprocessor` |
| Dimensionality | PCA, SVD, eigenvalue, eigenvector | `kaggle-preprocessor`, `kaggle-feature-engineer` |
| Bayesian | Bayesian, posterior, prior, likelihood | `kaggle-model-trainer`, `kaggle-optimizer` |
| Evaluation | precision, recall, F1-score, accuracy, AUC, ROC | (metrics for all skills) |
| Bias-Variance | bias-variance, model complexity, capacity | (diagnostic for all skills) |
| MoE | mixture of experts, gating network | `kaggle-moe`, `kaggle-model-trainer` |
| Distributed | distributed training, data parallel, model parallel | `kaggle-scaling`, `kaggle-model-trainer` |

---

## 3. Formula Templates for Competition Tasks

The expert registry infers which mathematical objective to use based on the task type:

### Classification Tasks
```
Objective: minimize_cross_entropy_loss
Function:  L = -Σ yᵢ * log(ŷᵢ)
Metrics:   accuracy, f1_score, precision, recall
```

### Regression Tasks
```
Objective: minimize_mse
Function:  L = (1/n) * Σ (yᵢ - ŷᵢ)²
Metrics:   rmse, mae, r2_score
```

### Unsupervised Tasks
```
Objective: minimize_inertia
Function:  J = Σ ||xᵢ - μₖ||²
Metrics:   silhouette_score, inertia
```

### General Optimization
```
Objective: minimize_validation_loss
Function:  L = f(X, θ, α)
Metrics:   accuracy, f1_score
```

---

## 4. The Convergence Loop Pattern

The most interesting pattern from the 150 automation runs is the **state convergence loop**, which iteratively refines a competition approach:

```python
# Convergence condition
exit_when: ||state[n] - state[n-1]||₂ < epsilon

# Parameters
epsilon = 0.001        # convergence threshold
max_iterations = 10    # hard stop
patience = 3           # consecutive converging iterations before exit
```

**How it works for a competition:**

1. **Initialize**: Select relevant experts for the competition via RAG
2. **Step**: Each expert scores itself against the current state, activating relevant skills
3. **Build state vector**: Combine expert scores and skill activations into a numeric vector
4. **Check convergence**: Compute L2 distance between consecutive state vectors
5. **Repeat**: Until converged or max iterations reached

This is directly applicable to iterative model refinement in the kaggriculture challenge.

---

## 5. The Kaggle MCP Server Tools

The base repository provides a Kaggle MCP server with these tools for competition interaction:

| Tool | Purpose |
|------|---------|
| `list_competitions` | Search/browse competitions (by category, search term) |
| `competition_details` | Get rules, evaluation metrics, prizes, timeline |
| `competition_leaderboard` | View current standings |
| `download_competition_files` | Get competition datasets |
| `list_datasets` | Search external datasets |
| `download_dataset` | Download datasets |
| `list_kernels` | Find relevant notebooks (filter by competition, language, votes) |
| `download_kernel` | Get top kernel source code |
| `diff_against_main` | Compare branch changes before evaluation |
| `parse_z3_proposed_model` | Visualize Z3 solver output as graph |

---

## 6. The Existing ML Strategy System

The FOL Workbench contains a rich ML strategy system already on main:

### Bayesian Feature Extractor
- Layered class architecture (Layer 1 constrains Layer 2, etc.)
- Attribute types: Numerical, Categorical, Boolean, Text, Logical
- Prior/posterior distributions for Bayesian inference
- Vocabulary universe tracking across all layers

### Decision Tree Designer
- Logical operators: AND, OR, NOT, XAND, NAND, FORALL (∀), EXISTS (∃), IMPLIES, IFF
- FOL formula export from visual decision trees
- Z3 solver validation of tree consistency

### Reverse Simulation System
- **Test-first workflow**: Define expected outputs, then find inputs that produce them
- **Hypothesis testing**: "Getting warmer" feedback loop for model discovery
- **Kaggle notebook generation**: Automated .ipynb creation with Z3 constraint solving

---

## 7. Applying This to the Kaggriculture Challenge

### 7.1 The Competition Landscape

The competition has a clear skill ladder (reference agents):

| Tier | Agent | Strategy | Bank |
|------|-------|----------|------|
| 0-3 | Basic agents | Simple farming | $3k-$17k |
| 4-5 | Melon Mateo / Rancher Rita | Premium crops / livestock | $44k-$53k |
| 6-9 | Meta field plans | Deterministic 712-turn scripts | **$186k** |

The **3.5× jump** from tier 5 to tier 6 is the meta barrier. Top ~150 players run byte-identical deterministic scripts.

### 7.2 Recommended Pipeline

```
Phase 1: Replay Intelligence
  ├── Download episode replays (kaggriculture-episodes dataset)
  ├── Parse action sequences from top-rated agents
  ├── Fingerprint strategies (detect meta clones)
  └── Extract the "meta field plan" action sequence

Phase 2: Economic Modeling
  ├── Model dynamic price curves (inventory → price mapping)
  ├── Compute per-crop ROI (wheat vs carrots vs melons)
  ├── Analyze fertilizer economics (+yield for 3 days)
  ├── Map livestock feed chains (wheat → animals → products)
  └── Identify optimal sell timing windows

Phase 3: Deterministic Baseline Agent
  ├── Implement the meta field plan (~$186k bank)
  ├── BFS pathfinding on tile grid
  ├── Priority-based task assignment (harvest > water > plant)
  ├── Farmhand route optimization
  └── Sell order optimization (the key differentiator at tier 6-9)

Phase 4: Reinforcement Learning (to break the meta)
  ├── Imitation learning from top replay action sequences
  ├── PPO training with JAX (~10k steps/sec)
  ├── Self-play: 50% self, 25% active opponent, 25% banked
  ├── Reward: terminal cash + production diversity bonuses
  └── Hybrid: RL "CEO" for high-level decisions + deterministic executor

Phase 5: Opponent Adaptation
  ├── Observe opponent's early turns
  ├── Classify opponent strategy (meta clone? RL? basic?)
  ├── Switch to counter-strategy
  ├── Endgame tree search for final-turn optimization
  └── Exploit deterministic opponents via market timing

Phase 6: Convergence & Submission
  ├── Run GBDT router + convergence loop
  ├── Each iteration refines expert scores and skill activations
  ├── Converge on optimal agent configuration
  └── Submit via Kaggle
```

### 7.3 Expert Definitions (Game-Agent Domains)

See `kaggriculture_experts.yaml` for full definitions. The 8 experts are:

| Expert | Domain | Objective |
|--------|--------|-----------|
| Deterministic Field Plan | Fixed 712-turn scripts | maximize_bank_balance |
| Market Timing | Price curve optimization | maximize_sell_revenue |
| Spatial Pathfinding | Grid movement, farmhand routing | minimize_wasted_turns |
| Crop Economics | Per-crop ROI, fertilizer, rotation | maximize_profit_per_tile |
| Livestock Management | Animals, feed chains, products | maximize_livestock_roi |
| Reinforcement Learning | PPO, self-play, hybrid architecture | maximize_expected_return |
| Replay Analysis | Episode parsing, strategy fingerprinting | maximize_replay_insight |
| Opponent Adaptation | Counter-strategies, game theory | maximize_win_probability |

### 7.4 Convergence Loop Configuration

```python
kaggriculture_loop_config = {
    "objective": "maximize_bank_balance",
    "exit_condition": "||state[n] - state[n-1]||_2 < epsilon",
    "epsilon": 0.001,
    "max_iterations": 25,
    "patience": 3,
    "state_dimensions": [
        "expert_domain_scores",      # 8 dimensions (one per domain)
        "skill_activation_levels",   # 7 dimensions (one per kaggle-skill)
        "expert_contribution_scores" # 8 dimensions (one per expert)
    ]
}
```

---

## 8. Code Assets Ready to Use

### From the MLSysEng MoE System (branch `sweeden-ttu/mlsyseng-moe-system-32ad`)

These modules are ready to integrate:

- **`mlsyseng_mcp/server.py`** — 8 MCP tools for the full knowledge → competition pipeline
- **`mlsyseng_mcp/database.py`** — SQLite schema for chapters, experts, convergence state
- **`mlsyseng_mcp/embeddings.py`** — RAG with sentence-transformers + ChromaDB
- **`mlsyseng_mcp/expert_registry.py`** — Expert creation with skill/formula inference
- **`mlsyseng_mcp/loop_controller.py`** — L2-norm convergence loop
- **`mlsyseng_mcp/docling_worker.py`** — PDF extraction with concept identification
- **`skill_generator.py`** — Multi-platform skill deployment
- **`skills.yaml`** — Complete MCP server and skill configuration

### From the Rust CLI (`kagg/rust/`)

- **`ac_matcher.rs`** — Aho-Corasick automaton for scanning text for ML conference names and top-100 ML datasets
- **`database.rs`** — Rust SQLite expert storage
- **`expert_registry.rs`** — Rust expert registry with same skill mapping as Python
- **`kmap.rs`** — K-map (Karnaugh map) simplification for boolean logic
- **`loop_controller.rs`** — Rust convergence loop

### From the FOL Workbench (on main)

- **`bayesian_feature_extractor.py`** — Layered Bayesian feature extraction
- **`kaggle_notebook_generator.py`** — Jupyter notebook generation with Z3 constraints
- **`ml_strategy_integration.py`** — Unified ML strategy system
- **`reverse_simulation_system.py`** — Test-first → hypothesis → notebook workflow
- **`hypothesis_tester.py`** — "Getting warmer" feedback loop

---

## 9. Quick Start: Using the Pipeline

### Step 1: Merge the MoE system into main

```bash
git merge origin/sweeden-ttu/mlsyseng-moe-system-32ad
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
# Additional for MoE:
pip install sentence-transformers chromadb docling pyyaml
```

### Step 3: Configure the MCP server

Add to your MCP client config:
```json
{
  "mcpServers": {
    "mlsyseng-moe": {
      "command": "python",
      "args": ["-m", "mlsyseng_mcp.server"],
      "env": {
        "ML_PRINCIPLES_PATH": "~/path/to/ml-reference-pdfs",
        "SQLITE_DB_PATH": "~/.mlsyseng/mlsyseng.db",
        "CHROMA_DB_PATH": "~/.mlsyseng/chroma_db"
      }
    }
  }
}
```

### Step 4: Extract knowledge and build entry

```
# Via MCP tools:
extract_knowledge()         # Index ML reference material
list_experts()              # See registered experts
build_entry("kaggriculture")  # Assemble competition entry
evolve("kaggriculture")     # Run convergence loop
```

### Step 5: Use rdagent integration

```
run_rdagent(
    competition="kaggriculture",
    description="agriculture prediction using satellite and tabular data"
)
```

---

## 10. Summary of What 150 Automation Runs Produced

| Metric | Count |
|--------|-------|
| Total branches created | ~150 |
| PRs opened (draft) | 11 |
| Unique Python modules created | 8 (server, database, embeddings, expert_registry, loop_controller, docling_worker, skill_generator, __init__) |
| Rust modules created | 6 (main, ac_matcher, database, expert_registry, kmap, loop_controller) |
| Ruby modules created | 3 (kagg.rb, ac_matcher.rb, tests) |
| MCP tools defined | 8 (extract_knowledge, evolve, search_concepts, list_experts, build_entry, run_rdagent, get_extraction_status, get_stats, ask_expert) |
| ML concept patterns | 30+ regex patterns covering all major ML domains |
| Skill mappings | 17 concept→skill mappings |
| Formula templates | 4 (classification, regression, unsupervised, optimization) |
| Conference/dataset patterns | AAAI, ICLR, NeurIPS, ICML, CVPR, ACL, + top-100 ML datasets |
| Platform targets | 5 (OpenClaw, Cursor, Claude Desktop, Gemini, Generic) |

The key insight across all these runs: the system converged on a **RAG-informed expert registry** pattern where ML knowledge is chunked, embedded, and retrieved contextually to assemble competition-specific strategies. The convergence loop with L2-norm exit condition provides a principled way to iterate until the approach stabilizes.

---

## 11. GBDT Expert Router: 137 Branches as Weak Learners

The file `kaggriculture_router.py` implements a **Gradient-Boosted Decision Tree router** that treats the 137 automation branches as an ensemble:

### The Data Structure

Each branch is vectorized into a feature vector:

```
[insertions/1000, commit_count, module_count, has_tests, has_rust,
 has_skills_yaml, has_kmap, has_ac_matcher, maturity_score/100]
```

These 137 vectors become the "training data" for a boosted ensemble of decision stumps.

### How It Works

1. **Branch Feature Extraction** (`extract_branch_features.py`): Scans all 137 branches, extracts diff stats, module presence, test coverage, and computes a maturity score
2. **Stump Fitting**: Each of the 50 decision stumps finds the best single split across 9 feature dimensions that minimizes residual prediction error for 9 agriculture domains
3. **Query Scoring**: A natural-language query is scored against agriculture concept vocabularies (remote sensing, crop science, soil, weather, geospatial, time series, tabular ML, deep learning, ensemble)
4. **Ensemble Prediction**: The 50 stumps boost the base prediction, adjusting domain scores based on which branch features correlate with which domains
5. **Routing**: The top-scoring domains activate specific Kaggle skills, select formulas, and configure convergence parameters

### Usage

```bash
# Route a specific query
python3 kaggriculture_router.py "predict crop yield from satellite and soil data"

# Default demo queries
python3 kaggriculture_router.py
```

### Why This Is a GBDT

The analogy is precise:
- **Weak learners**: Each decision stump splits on one branch feature dimension
- **Residual fitting**: Each round fits the error left by previous rounds
- **Boosting**: Predictions accumulate additively with a learning rate
- **Ensemble**: The final prediction is the sum of base score + all stump contributions

The difference from a standard GBDT: instead of predicting a scalar target, each stump predicts a **vector of domain scores** — one per agriculture concept domain. This makes it a multi-output GBDT where the routing decision is the argmax across domains.

### The Branch Manifest

The file `branch_manifest.json` contains the full feature extraction for all 137 branches, including:
- Diff statistics (files changed, insertions, deletions)
- Module presence flags (which of the 7 MoE modules each branch has)
- Test coverage, Rust port presence, skills.yaml presence
- Maturity score (composite of all features)
- Changed file lists

This manifest is the "training set" that the GBDT router learns from.
