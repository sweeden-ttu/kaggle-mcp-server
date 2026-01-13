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
    
    def __init__(self):
        """Initialize Bayesian perceptron assigner."""
        self.bayesian_extractor = BayesianFeatureExtractor()
        self._initialize_layers()
        
        # Track perceptron assignments
        self.assignments: Dict[str, PerceptronAssignment] = {}
    
    def _initialize_layers(self):
        """Initialize BayesianFeatureExtractor layers."""
        # Layer 1: Grid Features
        self.bayesian_extractor.create_layer(1, "Grid Features")
        
        # Layer 2: Perceptron Types
        self.bayesian_extractor.create_layer(2, "Perceptron Types")
        
        # Layer 3: Assignment Layer
        self.bayesian_extractor.create_layer(3, "Assignment Layer")
        
        # Add base perceptron types to Layer 2
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="TextPuzzlePerceptron",
            attributes=[
                Attribute("confidence", AttributeType.NUMERICAL, value=0.0),
                Attribute("accuracy", AttributeType.NUMERICAL, value=0.0)
            ]
        )
        
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="ImagePuzzlePerceptron",
            attributes=[
                Attribute("confidence", AttributeType.NUMERICAL, value=0.0),
                Attribute("accuracy", AttributeType.NUMERICAL, value=0.0)
            ]
        )
        
        self.bayesian_extractor.add_class_to_layer(
            layer_id=2,
            class_name="QuestionPuzzlePerceptron",
            attributes=[
                Attribute("confidence", AttributeType.NUMERICAL, value=0.0),
                Attribute("accuracy", AttributeType.NUMERICAL, value=0.0)
            ]
        )
    
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
        
        # Extract features using Bayesian inference
        features_data = {
            'has_text': grid_features.has_text,
            'has_handwriting': grid_features.has_handwriting,
            'has_image': grid_features.has_image,
            'has_question': grid_features.has_question,
            'text_confidence': grid_features.text_confidence,
            'image_complexity': grid_features.image_complexity
        }
        
        # Get features from Layer 2 (Perceptron Types)
        layer2_features = self.bayesian_extractor.extract_features(features_data, layer_id=2)
        
        # Select best perceptron based on confidence
        best_perceptron_id = None
        best_confidence = 0.0
        assignment_confidence = 0.0
        
        # Match grid features to perceptron types
        perceptron_type = self._determine_perceptron_type(grid_features)
        
        if perceptron_type in layer2_features:
            perceptron_data = layer2_features[perceptron_type]
            assignment_confidence = perceptron_data.get('confidence', 0.0)
            
            # Find perceptron with highest confidence for this type
            for pid, perceptron in available_perceptrons.items():
                if perceptron.confidence > best_confidence:
                    best_confidence = perceptron.confidence
                    best_perceptron_id = pid
        
        # If no perceptron found, use default based on type
        if best_perceptron_id is None and available_perceptrons:
            # Use first available perceptron
            best_perceptron_id = list(available_perceptrons.keys())[0]
            best_confidence = available_perceptrons[best_perceptron_id].confidence
        
        assignment = PerceptronAssignment(
            square_id=grid_features.square_id,
            perceptron_id=best_perceptron_id,
            confidence=best_confidence,
            assignment_confidence=assignment_confidence
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
        # Determine perceptron type from assignment
        # This would be stored in metadata or inferred
        
        # Update Bayesian prior based on success
        if was_correct:
            # Increase confidence for this type of assignment
            prior_update = {
                'confidence': perceptron_confidence,
                'accuracy': 1.0 if was_correct else 0.0
            }
        else:
            prior_update = {
                'confidence': perceptron_confidence * 0.9,  # Slight decrease
                'accuracy': 0.0
            }
        
        # Update Layer 2 perceptron types
        # This would need to know which perceptron type was used
        # For now, update all types proportionally
        for class_name in ["TextPuzzlePerceptron", "ImagePuzzlePerceptron", "QuestionPuzzlePerceptron"]:
            try:
                self.bayesian_extractor.update_bayesian_prior(
                    layer_id=2,
                    class_name=class_name,
                    prior=prior_update
                )
            except ValueError:
                # Class might not exist yet
                pass
    
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
            # Evaluate perceptron using Bayesian inference
            # Create a feature vector representing this perceptron
            perceptron_features = {
                'confidence': perceptron.confidence,
                'accuracy': perceptron.accuracy,
                'training_count': perceptron.training_count / 1000.0  # Normalize
            }
            
            # Extract features to get posterior confidence
            layer2_features = self.bayesian_extractor.extract_features(
                perceptron_features,
                layer_id=2
            )
            
            # Get maximum confidence from all perceptron types
            max_confidence = 0.0
            for class_name, class_features in layer2_features.items():
                conf = class_features.get('confidence', 0.0)
                max_confidence = max(max_confidence, conf)
            
            # Also consider the perceptron's own confidence
            combined_confidence = (max_confidence + perceptron.confidence) / 2.0
            
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
