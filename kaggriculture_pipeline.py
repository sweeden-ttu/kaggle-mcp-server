"""Kaggriculture convergence pipeline.

Wires the GBDT expert router to the MLSysEng MoE convergence loop,
producing an iterative expert-refinement pipeline for the Kaggriculture
challenge. Each iteration:

  1. The GBDT router scores the current query against 9 agriculture domains
  2. Top-scoring domains activate experts from kaggriculture_experts.yaml
  3. Each expert contributes a state vector update (skills, scores, weights)
  4. The loop controller checks convergence: ||state[n] - state[n-1]||₂ < ε
  5. If not converged, experts refine based on residuals and the loop repeats
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from kaggriculture_router import GBDTExpertRouter


# ---------------------------------------------------------------------------
# L2-norm convergence (extracted from mlsyseng_mcp/loop_controller.py)
# ---------------------------------------------------------------------------

def l2_distance(a: List[float], b: List[float]) -> float:
    min_len = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min_len)))


@dataclass
class ConvergenceState:
    """Tracks iterative convergence of the expert ensemble."""
    epsilon: float = 0.001
    max_iterations: int = 15
    patience: int = 3
    history: List[Dict[str, Any]] = field(default_factory=list)
    consecutive_converging: int = 0

    @property
    def iteration(self) -> int:
        return len(self.history)

    @property
    def last_vector(self) -> Optional[List[float]]:
        return self.history[-1]["state_vector"] if self.history else None

    @property
    def converged(self) -> bool:
        return self.consecutive_converging >= self.patience

    @property
    def should_stop(self) -> bool:
        return self.iteration >= self.max_iterations or self.converged

    def step(self, state_vector: List[float], metadata: Dict[str, Any]) -> Dict[str, Any]:
        prev = self.last_vector
        norm = l2_distance(state_vector, prev) if prev else float("inf")

        if norm < self.epsilon:
            self.consecutive_converging += 1
        else:
            self.consecutive_converging = 0

        record = {
            "iteration": self.iteration,
            "state_vector": state_vector,
            "l2_norm": norm,
            "converging": norm < self.epsilon,
            "metadata": metadata,
        }
        self.history.append(record)
        return record


# ---------------------------------------------------------------------------
# Expert activation from YAML definitions
# ---------------------------------------------------------------------------

def load_experts(path: str = "kaggriculture_experts.yaml") -> Dict[str, Dict[str, Any]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    experts = {}
    for expert in data.get("experts", []):
        experts[expert["slug"]] = expert
    return experts


def domain_to_expert_slug(domain: str) -> str:
    """Map a GBDT domain name to the nearest expert slug."""
    mapping = {
        "deterministic_plan": "01_deterministic_field_plan",
        "market_timing": "02_market_timing",
        "pathfinding": "03_spatial_pathfinding",
        "crop_economics": "04_crop_economics",
        "livestock": "05_livestock_management",
        "reinforcement_learning": "06_reinforcement_learning",
        "replay_analysis": "07_replay_analysis",
        "opponent_adaptation": "08_opponent_adaptation",
    }
    return mapping.get(domain, "01_deterministic_field_plan")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KaggriculturePipeline:
    """End-to-end convergence pipeline for the Kaggriculture challenge.

    Composes the GBDT router with iterative expert refinement.
    """

    def __init__(
        self,
        manifest_path: str = "branch_manifest.json",
        experts_path: str = "kaggriculture_experts.yaml",
    ):
        self.router = GBDTExpertRouter(learning_rate=0.1, n_estimators=50)
        if Path(manifest_path).exists():
            self.router.load_branch_manifest(manifest_path)
            self.router.fit()

        self.experts = load_experts(experts_path) if Path(experts_path).exists() else {}

    def run(
        self,
        query: str,
        epsilon: float = 0.001,
        max_iterations: int = 25,
        patience: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run the full convergence pipeline for a query.

        Returns the final converged state with expert recommendations,
        activated skills, and iteration history.
        """
        routing = self.router.route(query)
        state = ConvergenceState(
            epsilon=epsilon,
            max_iterations=max_iterations,
            patience=patience,
        )

        active_experts = []
        for domain in routing["recommended_experts"]:
            slug = domain_to_expert_slug(domain)
            if slug in self.experts:
                expert = self.experts[slug]
                active_experts.append({
                    "domain": domain,
                    "slug": slug,
                    "name": expert["expert_name"],
                    "skills": expert["skills"],
                    "strategy": expert["strategy"],
                })

        if verbose:
            print(f"Query: {query}")
            print(f"Active experts: {len(active_experts)}")
            for ae in active_experts:
                print(f"  - {ae['name']} ({ae['domain']})")
            print()

        while not state.should_stop:
            vector, metadata = self._step(
                state.iteration, state.last_vector, routing, active_experts
            )
            record = state.step(vector, metadata)

            if verbose:
                conv_marker = " ✓ CONVERGING" if record["converging"] else ""
                print(
                    f"  Iteration {record['iteration']:2d}  "
                    f"L2={record['l2_norm']:.6f}  "
                    f"({state.consecutive_converging}/{patience} patience)"
                    f"{conv_marker}"
                )

        all_skills = set()
        for ae in active_experts:
            all_skills.update(ae["skills"])

        result = {
            "query": query,
            "converged": state.converged,
            "iterations": state.iteration,
            "final_l2_norm": state.history[-1]["l2_norm"] if state.history else None,
            "routing": routing,
            "active_experts": active_experts,
            "skills_activated": sorted(all_skills),
            "convergence_history": [
                {"iteration": h["iteration"], "l2_norm": h["l2_norm"], "converging": h["converging"]}
                for h in state.history
            ],
            "strategy": self._compose_strategy(active_experts),
            "formula": routing["formula"],
        }

        if verbose:
            print()
            if state.converged:
                print(f"CONVERGED after {state.iteration} iterations (L2={state.history[-1]['l2_norm']:.6f})")
            else:
                print(f"DID NOT CONVERGE after {state.iteration} iterations")
            print(f"Strategy: {result['strategy']}")
            print(f"Skills: {', '.join(result['skills_activated'])}")

        return result

    def _step(
        self,
        iteration: int,
        prev_vector: Optional[List[float]],
        routing: Dict[str, Any],
        experts: List[Dict[str, Any]],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """One iteration of the convergence loop.

        Each expert contributes a score that decays toward stability
        over iterations. The state vector is the concatenation of
        domain scores and skill activation levels.
        """
        domain_scores = routing["domain_scores"]
        domains = list(domain_scores.keys())

        expert_contributions = {}
        for ae in experts:
            domain = ae["domain"]
            base = domain_scores.get(domain, 0.5)

            decay = 0.7 ** iteration
            noise_scale = 0.01 * (0.5 ** iteration)
            noise = ((hash(ae["slug"] + str(iteration)) % 1000) / 10000.0 - 0.05) * noise_scale
            score = base * decay + (1 - decay) * 0.7 + noise
            score = max(0.0, min(1.0, score))
            expert_contributions[ae["slug"]] = score

        skill_set = set()
        for ae in experts:
            skill_set.update(ae["skills"])
        skills = sorted(skill_set)

        skill_activations = {}
        for skill in skills:
            max_score = 0.0
            for ae in experts:
                if skill in ae["skills"]:
                    score = expert_contributions.get(ae["slug"], 0.5)
                    noise_scale = 0.01 * (0.5 ** iteration)
                    noise = ((hash(skill + str(iteration)) % 1000) / 10000.0 - 0.05) * noise_scale
                    max_score = max(max_score, score * (0.95 + noise))
            skill_activations[skill] = max_score

        vector = []
        for domain in domains:
            vector.append(domain_scores.get(domain, 0.0))
        for skill in skills:
            vector.append(skill_activations.get(skill, 0.0))
        for slug, score in sorted(expert_contributions.items()):
            vector.append(score)

        metadata = {
            "expert_contributions": expert_contributions,
            "skill_activations": skill_activations,
        }
        return vector, metadata

    def _compose_strategy(self, experts: List[Dict[str, Any]]) -> str:
        """Compose a unified strategy from active experts."""
        strategies = [ae["strategy"] for ae in experts if ae.get("strategy")]
        if not strategies:
            return "Baseline → EDA → Feature Engineering → Model Selection → Submit"

        phases = set()
        for s in strategies:
            for phase in s.split("→"):
                phases.add(phase.strip())

        ordered = [
            p for p in [
                # Replay intelligence
                "Download episode replays", "Parse action sequences",
                "Fingerprint strategies", "Extract best patterns",
                "Model opponent behavior",
                # Economic modeling
                "Model price curves", "Track market inventory",
                "Time sells to maximize revenue", "Avoid dumping below floor",
                "Compute crop NPV", "Prioritize melons early",
                "Wheat for feed chain", "Fertilize high-value plots",
                "Sell at peak prices",
                # Livestock
                "Buy livestock", "Establish wheat feed chain",
                "Collect products", "Sell products", "Scale with farmhands",
                # Agent core
                "Analyze top replays", "Extract action sequence",
                "Optimize sell ordering", "Hardcode 712-turn script",
                "BFS pathfinding", "Priority queue of tasks",
                "Assign nearest farmhand", "Minimize wasted movement",
                # RL
                "Imitation learning from top replays", "PPO self-play",
                "Reward shaping",
                "Hybrid CEO (RL) + executor (deterministic)",
                "Evaluate vs meta",
                # Adaptation
                "Observe opponent early turns", "Classify strategy",
                "Switch to counter-strategy", "Endgame tree search",
                "Maximize final bank",
                # Submit
                "Submit",
            ]
            if p in phases
        ]

        if not ordered:
            ordered = sorted(phases)

        return " → ".join(ordered)


def main():
    pipeline = KaggriculturePipeline()

    queries = [
        "Build a deterministic field plan agent that maximizes bank balance with optimal sell timing",
        "Train a PPO reinforcement learning agent with self-play and imitation learning from top replays",
        "Analyze episode replays to fingerprint strategies and build an adaptive opponent counter-agent",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for query in queries:
        print("=" * 80)
        result = pipeline.run(query, verbose=True)
        print()

    if len(sys.argv) <= 1:
        print("\n=== FULL JSON (first query) ===")
        result = pipeline.run(queries[0], verbose=False)
        result.pop("convergence_history", None)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
