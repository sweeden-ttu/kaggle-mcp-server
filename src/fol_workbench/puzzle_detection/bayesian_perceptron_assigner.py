"""
Bayesian Perceptron Assigner Module

Integrates with BayesianFeatureExtractor to learn perceptron assignments
based on confidence and grid square features.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ..bayesian_feature_extractor import (
    BayesianFeatureExtractor,
    Attribute,
    AttributeType,
    Layer,
    ClassLayer
)
from .feature_extractor import GridSquareFeatures
from .perceptron_units import PerceptronUnit


@dataclass
class PerceptronAssignment:
    """Assignment of a perceptron to a grid square."""
    square_id: str
    perceptron_id: Optional[str]
    confidence: float
    assignment_confidence: float  # Confidence in the assignment itself
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BayesianPerceptronAssigner:
    """
    Uses BayesianFeatureExtractor to assign perceptrons to grid squares.
    
    Layer structure:
    - Layer 1: "Grid Features" - raw features from grid squares
    - Layer 2: "Perceptron Types" - different perceptron experts with confidence scores
    - Layer 3: "Assignment Layer" - learned assignments based on confidence
    """
    
    PERCEPTRON_TYPES: Tuple[str, ...] = (
        "TextPuzzlePerceptron",
        "ImagePuzzlePerceptron",
        "QuestionPuzzlePerceptron",
    )

    def __init__(self, bayesian_extractor: Optional[BayesianFeatureExtractor] = None):
        """Initialize Bayesian perceptron assigner."""
        self.bayesian_extractor = bayesian_extractor or BayesianFeatureExtractor()
        self._initialize_layers()
        
        # Track perceptron assignments
        self.assignments: Dict[str, PerceptronAssignment] = {}
    
    def _initialize_layers(self):
        """Initialize BayesianFeatureExtractor layers."""
        # Layer 1: Grid Features
        if 1 not in self.bayesian_extractor.layers:
            self.bayesian_extractor.create_layer(1, "Grid Features")
        
        # Layer 2: Perceptron Types
        if 2 not in self.bayesian_extractor.layers:
            self.bayesian_extractor.create_layer(2, "Perceptron Types")
        
        # Layer 3: Assignment Layer
        if 3 not in self.bayesian_extractor.layers:
            self.bayesian_extractor.create_layer(3, "Assignment Layer")

        # Add perceptron-type models (Layer 2) with meaningful priors.
        # Note: BayesianFeatureExtractor's update uses stringified evidence values,
        # so boolean priors are keyed by "True"/"False" and bins by strings.
        self._ensure_perceptron_type_models()

    def _ensure_perceptron_type_models(self):
        """Ensure Layer 2 has perceptron-type classes with priors over grid features."""
        def bool_attr(name: str, p_true: float) -> Attribute:
            p_true = float(np.clip(p_true, 0.0, 1.0))
            p_false = 1.0 - p_true
            return Attribute(
                name=name,
                attr_type=AttributeType.BOOLEAN,
                value=None,
                prior_distribution={"True": p_true, "False": p_false},
            )

        def cat_attr(name: str, priors: Dict[str, float]) -> Attribute:
            # Normalize
            total = float(sum(float(v) for v in priors.values()))
            if total <= 0:
                priors = {"unknown": 1.0}
                total = 1.0
            priors = {str(k): float(v) / total for k, v in priors.items()}
            return Attribute(
                name=name,
                attr_type=AttributeType.CATEGORICAL,
                value=None,
                prior_distribution=priors,
            )

        # Shared bins for numeric-ish features
        # (we pass these bins as categorical evidence rather than raw floats)
        text_bins = {"low": 0.25, "mid": 0.50, "high": 0.25}
        img_bins = {"low": 0.20, "mid": 0.40, "high": 0.40}

        # TextPuzzlePerceptron: expects text/handwriting, not necessarily images/questions.
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="TextPuzzlePerceptron",
            attributes=[
                bool_attr("has_text", 0.75),
                bool_attr("has_handwriting", 0.55),
                bool_attr("has_image", 0.20),
                bool_attr("has_question", 0.25),
                cat_attr("text_confidence_bin", dict(text_bins)),
                cat_attr("image_complexity_bin", dict(img_bins)),
                cat_attr("question_type", {"none": 0.60, "general": 0.25, "wh_question": 0.10, "selection": 0.05}),
                # Perceptron-quality attributes (used by selection; not provided during assignment)
                cat_attr("confidence_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
                cat_attr("accuracy_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
            ],
        )

        # ImagePuzzlePerceptron: expects images/high complexity, not necessarily text.
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="ImagePuzzlePerceptron",
            attributes=[
                bool_attr("has_text", 0.25),
                bool_attr("has_handwriting", 0.10),
                bool_attr("has_image", 0.85),
                bool_attr("has_question", 0.20),
                cat_attr("text_confidence_bin", dict(text_bins)),
                cat_attr("image_complexity_bin", {"low": 0.10, "mid": 0.30, "high": 0.60}),
                cat_attr("question_type", {"none": 0.70, "general": 0.20, "wh_question": 0.07, "selection": 0.03}),
                cat_attr("confidence_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
                cat_attr("accuracy_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
            ],
        )

        # QuestionPuzzlePerceptron: expects question patterns, especially selection/wh.
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="QuestionPuzzlePerceptron",
            attributes=[
                bool_attr("has_text", 0.55),
                bool_attr("has_handwriting", 0.10),
                bool_attr("has_image", 0.25),
                bool_attr("has_question", 0.90),
                cat_attr("text_confidence_bin", dict(text_bins)),
                cat_attr("image_complexity_bin", dict(img_bins)),
                cat_attr("question_type", {"selection": 0.55, "wh_question": 0.30, "general": 0.10, "none": 0.05}),
                cat_attr("confidence_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
                cat_attr("accuracy_bin", {"low": 0.40, "mid": 0.40, "high": 0.20}),
            ],
        )

    def _bin01(self, x: float, low: float = 0.33, high: float = 0.66) -> str:
        """Bin a [0,1] float into categorical buckets."""
        try:
            v = float(x)
        except Exception:
            return "unknown"
        if v < low:
            return "low"
        if v < high:
            return "mid"
        return "high"

    def _extract_attr_confidence(self, class_features: Dict[str, Any], attr_name: str) -> float:
        """Safely extract a scalar confidence for an attribute from extract_features() output."""
        v = class_features.get(attr_name)
        if isinstance(v, dict):
            c = v.get("confidence", 0.0)
            return float(c) if isinstance(c, (int, float)) else 0.0
        return 0.0

    def _infer_perceptron_type_from_id(self, perceptron_id: str) -> Optional[str]:
        """Heuristic mapping from perceptron IDs to type-class names."""
        pid = (perceptron_id or "").lower()
        if "question" in pid:
            return "QuestionPuzzlePerceptron"
        if "image" in pid:
            return "ImagePuzzlePerceptron"
        if "text" in pid or "hand" in pid:
            return "TextPuzzlePerceptron"
        return None
    
    def assign_perceptron(
        self,
        grid_features: GridSquareFeatures,
        available_perceptrons: Dict[str, PerceptronUnit]
    ) -> PerceptronAssignment:
        """
        Assign a perceptron to a grid square based on features.
        
        Args:
            grid_features: Features extracted from grid square
            available_perceptrons: Dictionary of available perceptrons by ID
            
        Returns:
            PerceptronAssignment
        """
        # Convert grid features to attributes
        attributes = grid_features.to_attributes()
        
        # Add grid square as a class in Layer 1
        self.bayesian_extractor.add_class_to_layer(
            layer_id=1,
            class_name=grid_features.square_id,
            attributes=attributes
        )

        # Discretize numeric-ish features into bins for the Bayesian extractor
        evidence = {
            "has_text": grid_features.has_text,
            "has_handwriting": grid_features.has_handwriting,
            "has_image": grid_features.has_image,
            "has_question": grid_features.has_question,
            "text_confidence_bin": self._bin01(grid_features.text_confidence),
            "image_complexity_bin": self._bin01(grid_features.image_complexity),
            "question_type": grid_features.question_type or "none",
        }

        layer2_features = self.bayesian_extractor.extract_features(evidence, layer_id=2)

        # Score each perceptron type by mean attribute confidence for observed evidence keys.
        type_scores: Dict[str, float] = {}
        for type_name in self.PERCEPTRON_TYPES:
            class_features = layer2_features.get(type_name, {}) if isinstance(layer2_features, dict) else {}
            if not isinstance(class_features, dict):
                type_scores[type_name] = 0.0
                continue

            confs = []
            for k in evidence.keys():
                confs.append(self._extract_attr_confidence(class_features, k))
            type_scores[type_name] = float(np.mean(confs)) if confs else 0.0

        # Pick the best type, fallback to deterministic heuristic if we have no signal.
        best_type = max(type_scores.items(), key=lambda kv: kv[1])[0] if type_scores else self._determine_perceptron_type(grid_features)
        best_type_score = type_scores.get(best_type, 0.0)
        if best_type_score <= 0.0:
            best_type = self._determine_perceptron_type(grid_features)
            best_type_score = type_scores.get(best_type, 0.0)

        # Choose a perceptron instance that matches the selected type (by ID heuristic).
        best_perceptron_id: Optional[str] = None
        best_perceptron_confidence = -1.0
        for pid, perceptron in available_perceptrons.items():
            pid_type = self._infer_perceptron_type_from_id(pid)
            if pid_type is not None and pid_type != best_type:
                continue
            if float(perceptron.confidence) > best_perceptron_confidence:
                best_perceptron_confidence = float(perceptron.confidence)
                best_perceptron_id = pid

        # Fallback: if we couldn't match by type, just pick the highest-confidence perceptron.
        if best_perceptron_id is None and available_perceptrons:
            best_perceptron_id, best_perceptron_confidence = max(
                ((pid, float(p.confidence)) for pid, p in available_perceptrons.items()),
                key=lambda kv: kv[1],
            )
        
        assignment = PerceptronAssignment(
            square_id=grid_features.square_id,
            perceptron_id=best_perceptron_id,
            confidence=float(best_perceptron_confidence) if best_perceptron_confidence >= 0 else 0.0,
            assignment_confidence=float(best_type_score),
            metadata={
                "perceptron_type": best_type,
                "type_scores": type_scores,
                "evidence": evidence,
            },
        )
        
        self.assignments[grid_features.square_id] = assignment
        return assignment
    
    def update_from_results(
        self,
        assignment: PerceptronAssignment,
        was_correct: bool,
        perceptron_confidence: float
    ):
        """
        Update Bayesian priors based on assignment results.
        
        Args:
            assignment: The assignment that was made
            was_correct: Whether the assignment was correct
            perceptron_confidence: Confidence of the perceptron prediction
        """
        # Update priors only for the perceptron type actually used for this assignment.
        perceptron_type = assignment.metadata.get("perceptron_type") if assignment.metadata else None
        if perceptron_type not in self.PERCEPTRON_TYPES:
            perceptron_type = self._infer_perceptron_type_from_id(assignment.perceptron_id or "") or "TextPuzzlePerceptron"

        evidence = (assignment.metadata or {}).get("evidence", {})
        if not isinstance(evidence, dict):
            evidence = {}

        layer = self.bayesian_extractor.layers.get(2)
        if not layer:
            return
        class_layer = layer.get_class(perceptron_type)
        if not class_layer:
            return

        # Simple evidence-based prior update:
        # - On correct: boost observed value probability.
        # - On incorrect: decay observed value probability slightly.
        boost = 1.10 if was_correct else 0.90
        floor = 1e-6
        for attr in class_layer.attributes:
            if not attr.prior_distribution:
                continue
            if attr.name not in evidence:
                continue
            observed_value = str(evidence[attr.name])

            # Support both keying styles.
            keys_to_try = [observed_value, f"{attr.name}_{observed_value}"]
            key = next((k for k in keys_to_try if k in attr.prior_distribution), None)
            if key is None:
                key = observed_value
                attr.prior_distribution[key] = 0.1

            attr.prior_distribution[key] = max(floor, float(attr.prior_distribution.get(key, 0.0)) * boost)
            # Renormalize
            total = float(sum(float(v) for v in attr.prior_distribution.values()))
            if total > 0:
                attr.prior_distribution = {k: float(v) / total for k, v in attr.prior_distribution.items()}

        # Track perceptron quality bins as additional evidence signals.
        confidence_bin = self._bin01(perceptron_confidence)
        quality_evidence = {"confidence_bin": confidence_bin}
        for attr in class_layer.attributes:
            if attr.name not in quality_evidence or not attr.prior_distribution:
                continue
            observed_value = str(quality_evidence[attr.name])
            if observed_value in attr.prior_distribution:
                attr.prior_distribution[observed_value] = max(floor, float(attr.prior_distribution[observed_value]) * boost)
                total = float(sum(float(v) for v in attr.prior_distribution.values()))
                if total > 0:
                    attr.prior_distribution = {k: float(v) / total for k, v in attr.prior_distribution.items()}

        self.bayesian_extractor.log_entry(
            f"Updated priors for {perceptron_type} (correct={was_correct}, p_conf={perceptron_confidence:.3f})"
        )
    
    def select_high_confidence_perceptrons(
        self,
        perceptrons: Dict[str, PerceptronUnit],
        threshold: float = 0.8
    ) -> List[Tuple[str, PerceptronUnit]]:
        """
        Select perceptrons with high confidence for persistence.
        
        Uses BayesianFeatureExtractor to evaluate and select based on
        posterior probabilities.
        
        Args:
            perceptrons: Dictionary of perceptrons
            threshold: Confidence threshold
            
        Returns:
            List of (perceptron_id, perceptron) tuples for high-confidence perceptrons
        """
        selected = []
        
        for perceptron_id, perceptron in perceptrons.items():
            # Evaluate perceptron using Bayesian inference over coarse quality bins.
            quality_evidence = {
                "confidence_bin": self._bin01(perceptron.confidence),
                "accuracy_bin": self._bin01(perceptron.accuracy),
            }

            layer2_features = self.bayesian_extractor.extract_features(quality_evidence, layer_id=2)

            max_score = 0.0
            for type_name, class_features in (layer2_features or {}).items():
                if not isinstance(class_features, dict):
                    continue
                confs = [
                    self._extract_attr_confidence(class_features, "confidence_bin"),
                    self._extract_attr_confidence(class_features, "accuracy_bin"),
                ]
                score = float(np.mean(confs)) if confs else 0.0
                max_score = max(max_score, score)

            combined_confidence = float(np.mean([float(perceptron.confidence), float(perceptron.accuracy), max_score]))
            if combined_confidence >= threshold:
                selected.append((perceptron_id, perceptron))
        
        # Sort by confidence (descending)
        selected.sort(key=lambda x: x[1].confidence, reverse=True)
        
        return selected
    
    def _determine_perceptron_type(self, features: GridSquareFeatures) -> str:
        """
        Determine which perceptron type is most appropriate.
        
        Args:
            features: Grid square features
            
        Returns:
            Perceptron type name
        """
        if features.has_question:
            return "QuestionPuzzlePerceptron"
        elif features.has_image and features.image_complexity > 0.3:
            return "ImagePuzzlePerceptron"
        elif features.has_text or features.has_handwriting:
            return "TextPuzzlePerceptron"
        else:
            return "TextPuzzlePerceptron"  # Default
    
    def get_assignments(self) -> Dict[str, PerceptronAssignment]:
        """Get all current assignments."""
        return self.assignments.copy()
    
    def get_bayesian_extractor(self) -> BayesianFeatureExtractor:
        """Get the underlying BayesianFeatureExtractor."""
        return self.bayesian_extractor
