"""GBDT-inspired Expert Router for the Kaggriculture Challenge.

Treats the 137 automation branches as an ensemble of weak learners.
Each branch contributed a slightly different implementation of the
MLSysEng MoE system. This router:

1. Extracts "split features" from each branch (what it implemented,
   how mature it is, what concepts it covers)
2. Builds a gradient-boosted routing tree that, given a query about
   an agriculture ML task, returns the optimal expert path — which
   modules to load, which skill mappings to use, and which convergence
   parameters to apply.

The key insight: instead of picking ONE branch, we boost across all
branches. Each branch's contribution is a residual improvement on the
previous ensemble prediction. The final prediction is the accumulated
expert configuration.
"""

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Domain: agriculture ML concept vocabulary
# ---------------------------------------------------------------------------

AGRICULTURE_CONCEPTS = {
    "remote_sensing": [
        "satellite", "sentinel", "landsat", "modis", "ndvi", "evi",
        "spectral", "band", "raster", "imagery", "pixel", "resolution",
    ],
    "crop_science": [
        "crop", "yield", "harvest", "planting", "germination", "phenology",
        "growing degree", "biomass", "canopy", "leaf area", "chlorophyll",
    ],
    "soil": [
        "soil", "ph", "nitrogen", "phosphorus", "potassium", "organic matter",
        "texture", "moisture", "drainage", "clay", "sand", "silt",
    ],
    "weather": [
        "temperature", "precipitation", "rainfall", "humidity", "wind",
        "solar radiation", "evapotranspiration", "frost", "drought",
    ],
    "geospatial": [
        "latitude", "longitude", "elevation", "slope", "aspect",
        "coordinate", "spatial", "geographic", "region", "zone",
    ],
    "time_series": [
        "temporal", "seasonal", "trend", "lag", "rolling", "window",
        "decomposition", "periodicity", "autoregressive",
    ],
    "tabular_ml": [
        "gradient boosting", "xgboost", "lightgbm", "catboost", "random forest",
        "feature importance", "shap", "permutation", "cross validation",
    ],
    "deep_learning": [
        "cnn", "convolutional", "resnet", "unet", "transformer", "attention",
        "lstm", "gru", "pretrained", "fine-tune", "transfer learning",
    ],
    "ensemble": [
        "ensemble", "stacking", "blending", "weighted average", "voting",
        "bagging", "boosting", "diversity", "out-of-fold",
    ],
}

# Maps agriculture concepts to the MoE skill paths (from expert_registry.py)
AGRICULTURE_SKILL_MAP = {
    "remote_sensing": ["kaggle-preprocessor", "kaggle-feature-engineer", "kaggle-deep-learning"],
    "crop_science": ["kaggle-model-trainer", "kaggle-feature-engineer"],
    "soil": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "weather": ["kaggle-feature-engineer", "kaggle-preprocessor"],
    "geospatial": ["kaggle-validator", "kaggle-preprocessor"],
    "time_series": ["kaggle-deep-learning", "kaggle-feature-engineer"],
    "tabular_ml": ["kaggle-ensemble", "kaggle-model-trainer", "kaggle-optimizer"],
    "deep_learning": ["kaggle-deep-learning", "kaggle-model-trainer"],
    "ensemble": ["kaggle-ensemble", "kaggle-optimizer"],
}


# ---------------------------------------------------------------------------
# Branch feature vector (the "training data" from 137 branches)
# ---------------------------------------------------------------------------

@dataclass
class BranchFeatures:
    """Feature vector extracted from one automation branch."""
    branch: str
    insertions: int = 0
    deletions: int = 0
    commit_count: int = 0
    module_count: int = 0
    has_tests: bool = False
    has_rust: bool = False
    has_skills_yaml: bool = False
    has_kmap: bool = False
    has_ac_matcher: bool = False
    maturity_score: float = 0.0
    modules_present: List[str] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Convert to numeric feature vector for GBDT splits."""
        return [
            self.insertions / 1000.0,
            self.commit_count,
            self.module_count,
            float(self.has_tests),
            float(self.has_rust),
            float(self.has_skills_yaml),
            float(self.has_kmap),
            float(self.has_ac_matcher),
            self.maturity_score / 100.0,
        ]


# ---------------------------------------------------------------------------
# Decision stump: one weak learner in the GBDT
# ---------------------------------------------------------------------------

@dataclass
class DecisionStump:
    """A single split in the gradient-boosted ensemble.

    Each stump says: "If feature[dim] <= threshold, go left; else right."
    The left/right values are residual expert-score adjustments.
    """
    feature_dim: int
    threshold: float
    left_value: Dict[str, float]   # domain → score adjustment if <= threshold
    right_value: Dict[str, float]  # domain → score adjustment if > threshold
    weight: float = 1.0

    def predict(self, x: List[float]) -> Dict[str, float]:
        if x[self.feature_dim] <= self.threshold:
            return {k: v * self.weight for k, v in self.left_value.items()}
        return {k: v * self.weight for k, v in self.right_value.items()}


# ---------------------------------------------------------------------------
# GBDT Expert Router
# ---------------------------------------------------------------------------

class GBDTExpertRouter:
    """Gradient-Boosted Decision Tree router for agriculture ML experts.

    The "training" phase builds stumps from the 137 branch feature vectors.
    The "inference" phase takes a natural-language query, scores it against
    agriculture concept domains, and routes through the boosted ensemble
    to produce a ranked expert recommendation.
    """

    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 50):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.stumps: List[DecisionStump] = []
        self.branch_data: List[BranchFeatures] = []
        self.base_scores: Dict[str, float] = {}

    def load_branch_manifest(self, manifest_path: str):
        """Load branch features from the extraction manifest."""
        with open(manifest_path) as f:
            data = json.load(f)

        self.branch_data = []
        for b in data["branches"]:
            bf = BranchFeatures(
                branch=b["branch"],
                insertions=b["insertions"],
                deletions=b["deletions"],
                commit_count=b["commit_count"],
                module_count=b["module_count"],
                has_tests=b["has_tests"],
                has_rust=b["has_rust"],
                has_skills_yaml=b["has_skills_yaml"],
                has_kmap=b["has_kmap"],
                has_ac_matcher=b["has_ac_matcher"],
                maturity_score=b["maturity_score"],
                modules_present=b["modules_present"],
            )
            self.branch_data.append(bf)

    def fit(self):
        """Build the boosted ensemble from branch features.

        Each branch's feature vector is a "training example". The target
        is the branch's maturity-weighted contribution to each agriculture
        domain. We fit stumps to predict domain scores from branch features.
        """
        if not self.branch_data:
            raise ValueError("No branch data loaded. Call load_branch_manifest first.")

        domains = list(AGRICULTURE_CONCEPTS.keys())

        targets: List[Dict[str, float]] = []
        for bf in self.branch_data:
            domain_scores = {}
            for domain in domains:
                base = bf.maturity_score / 100.0
                if bf.has_tests:
                    base += 0.1
                if bf.has_skills_yaml:
                    base += 0.08
                if bf.has_rust:
                    base += 0.05

                if domain == "tabular_ml" and "expert_registry" in bf.modules_present:
                    base += 0.15
                if domain == "ensemble" and "loop_controller" in bf.modules_present:
                    base += 0.12
                if domain == "deep_learning" and "embeddings" in bf.modules_present:
                    base += 0.1
                if domain == "remote_sensing" and bf.has_ac_matcher:
                    base += 0.08

                domain_scores[domain] = min(base, 1.0)
            targets.append(domain_scores)

        self.base_scores = {d: 0.5 for d in domains}

        residuals = []
        for t in targets:
            r = {d: t[d] - self.base_scores[d] for d in domains}
            residuals.append(r)

        X = [bf.to_vector() for bf in self.branch_data]
        n_features = len(X[0])

        for _round in range(self.n_estimators):
            best_stump = None
            best_loss = float("inf")

            for dim in range(n_features):
                values = sorted(set(x[dim] for x in X))
                thresholds = [
                    (values[i] + values[i + 1]) / 2
                    for i in range(len(values) - 1)
                ]
                if not thresholds:
                    continue

                for thr in thresholds[:10]:
                    left_res = defaultdict(float)
                    right_res = defaultdict(float)
                    left_n = 0
                    right_n = 0

                    for i, x in enumerate(X):
                        if x[dim] <= thr:
                            for d in domains:
                                left_res[d] += residuals[i][d]
                            left_n += 1
                        else:
                            for d in domains:
                                right_res[d] += residuals[i][d]
                            right_n += 1

                    if left_n == 0 or right_n == 0:
                        continue

                    left_val = {d: left_res[d] / left_n for d in domains}
                    right_val = {d: right_res[d] / right_n for d in domains}

                    loss = 0.0
                    for i, x in enumerate(X):
                        pred = left_val if x[dim] <= thr else right_val
                        for d in domains:
                            loss += (residuals[i][d] - pred[d]) ** 2

                    if loss < best_loss:
                        best_loss = loss
                        best_stump = DecisionStump(
                            feature_dim=dim,
                            threshold=thr,
                            left_value=left_val,
                            right_value=right_val,
                            weight=self.learning_rate,
                        )

            if best_stump is None:
                break

            self.stumps.append(best_stump)

            for i, x in enumerate(X):
                pred = best_stump.predict(x)
                for d in domains:
                    residuals[i][d] -= pred[d]

    def score_query(self, query: str) -> Dict[str, float]:
        """Score a natural-language query against agriculture domains."""
        query_lower = query.lower()
        domain_matches = {}

        for domain, keywords in AGRICULTURE_CONCEPTS.items():
            score = 0.0
            matched_keywords = []
            for kw in keywords:
                if kw in query_lower:
                    score += 1.0
                    matched_keywords.append(kw)
            if matched_keywords:
                score = score / len(keywords)
            domain_matches[domain] = score

        total = sum(domain_matches.values())
        if total > 0:
            domain_matches = {d: s / total for d, s in domain_matches.items()}
        else:
            n = len(domain_matches)
            domain_matches = {d: 1.0 / n for d in domain_matches}

        return domain_matches

    def route(self, query: str) -> Dict[str, Any]:
        """Route a query through the GBDT ensemble to get expert recommendations.

        Returns the top experts, their skills, and convergence parameters.
        """
        query_scores = self.score_query(query)

        avg_branch = BranchFeatures(
            branch="query",
            insertions=2500,
            commit_count=3,
            module_count=7,
            has_tests=True,
            has_skills_yaml=True,
            has_rust=False,
            maturity_score=80.0,
            modules_present=["server", "database", "embeddings",
                             "expert_registry", "loop_controller",
                             "docling_worker", "skill_generator"],
        )
        x = avg_branch.to_vector()

        ensemble_scores = dict(self.base_scores)
        for stump in self.stumps:
            pred = stump.predict(x)
            for d in pred:
                ensemble_scores[d] = ensemble_scores.get(d, 0) + pred[d]

        combined = {}
        for domain in ensemble_scores:
            combined[domain] = (
                0.6 * query_scores.get(domain, 0)
                + 0.4 * ensemble_scores.get(domain, 0)
            )

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

        skills_needed = set()
        for domain, score in ranked:
            if score > 0.01:
                for skill in AGRICULTURE_SKILL_MAP.get(domain, []):
                    skills_needed.add(skill)

        top_branches = sorted(
            self.branch_data,
            key=lambda bf: bf.maturity_score,
            reverse=True,
        )[:5]

        return {
            "query": query,
            "domain_scores": {d: round(s, 4) for d, s in ranked},
            "top_domains": [d for d, s in ranked[:3]],
            "skills_activated": sorted(skills_needed),
            "recommended_experts": [d for d, s in ranked if s > 0.05],
            "source_branches": [b.branch for b in top_branches],
            "convergence_config": {
                "epsilon": 0.001,
                "max_iterations": 15,
                "patience": 3,
                "objective": self._infer_objective(ranked[0][0] if ranked else "tabular_ml"),
            },
            "ensemble_size": len(self.stumps),
            "formula": self._infer_formula(ranked[0][0] if ranked else "tabular_ml"),
        }

    def _infer_objective(self, top_domain: str) -> str:
        obj_map = {
            "crop_science": "minimize_cross_entropy_loss",
            "remote_sensing": "minimize_rmse",
            "soil": "minimize_rmse",
            "weather": "minimize_mse",
            "time_series": "minimize_mse",
            "tabular_ml": "minimize_validation_loss",
            "deep_learning": "minimize_validation_loss",
            "ensemble": "minimize_validation_loss",
            "geospatial": "minimize_cv_variance",
        }
        return obj_map.get(top_domain, "minimize_validation_loss")

    def _infer_formula(self, top_domain: str) -> Dict[str, Any]:
        formulas = {
            "crop_science": {
                "function": "L = -Σ yᵢ * log(ŷᵢ)",
                "metrics": ["accuracy", "f1_score", "cohen_kappa"],
            },
            "remote_sensing": {
                "function": "NDVI = (NIR - Red) / (NIR + Red); L = RMSE(y, ŷ)",
                "metrics": ["rmse", "mae", "r2_score"],
            },
            "soil": {
                "function": "L = (1/n) * Σ (yᵢ - ŷᵢ)²",
                "metrics": ["rmse", "mae", "r2_score"],
            },
            "tabular_ml": {
                "function": "L = f(X, θ, α) via gradient boosting",
                "metrics": ["rmse", "mae", "competition_metric"],
            },
            "ensemble": {
                "function": "ŷ = Σ wᵢ * fᵢ(X), s.t. Σ wᵢ = 1",
                "metrics": ["competition_metric", "ensemble_diversity"],
            },
        }
        return formulas.get(top_domain, formulas["tabular_ml"])

    def explain(self, query: str) -> str:
        """Human-readable explanation of the routing decision."""
        result = self.route(query)
        lines = [
            f"Query: {result['query']}",
            f"",
            f"Top domains (from {result['ensemble_size']}-tree boosted ensemble):",
        ]
        for domain, score in result["domain_scores"].items():
            bar = "█" * int(score * 40)
            lines.append(f"  {domain:20s} {score:.3f} {bar}")

        lines.extend([
            f"",
            f"Recommended experts: {', '.join(result['recommended_experts'])}",
            f"Skills to activate:  {', '.join(result['skills_activated'])}",
            f"Objective:           {result['convergence_config']['objective']}",
            f"Formula:             {result['formula']['function']}",
            f"Metrics:             {', '.join(result['formula']['metrics'])}",
            f"Convergence:         ε={result['convergence_config']['epsilon']}, "
            f"max_iter={result['convergence_config']['max_iterations']}, "
            f"patience={result['convergence_config']['patience']}",
            f"",
            f"Best source branches:",
        ])
        for b in result["source_branches"]:
            lines.append(f"  - {b}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys

    manifest_path = "/tmp/branch_manifest.json"
    if not Path(manifest_path).exists():
        print("Run extract_branch_features.py first to create the manifest.", file=sys.stderr)
        sys.exit(1)

    router = GBDTExpertRouter(learning_rate=0.1, n_estimators=50)
    router.load_branch_manifest(manifest_path)
    router.fit()

    queries = [
        "Predict crop yield from satellite imagery and soil data for the kaggriculture challenge",
        "Classify crop types using Sentinel-2 time series with spatial cross-validation",
        "Build an ensemble of gradient boosted trees for tabular agriculture features",
        "Use NDVI temporal profiles and weather data to predict harvest date",
        "Feature engineering for soil properties combined with remote sensing indices",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for query in queries:
        print("=" * 80)
        print(router.explain(query))
        print()

    if len(sys.argv) <= 1:
        result = router.route(queries[0])
        print("\n=== JSON OUTPUT (for programmatic use) ===")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
