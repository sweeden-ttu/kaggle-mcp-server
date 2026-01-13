"""
Perceptron Units Module

Binary perceptron classifier for puzzle detection with:
- Configurable learning rates
- Confidence scoring
- Online learning capability
- Accuracy tracking
- Weight persistence
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import json
from datetime import datetime


@dataclass
class PerceptronState:
    """State of a perceptron unit."""
    weights: np.ndarray
    bias: float
    learning_rate: float
    accuracy: float = 0.0
    confidence: float = 0.0
    training_count: int = 0
    correct_predictions: int = 0
    total_predictions: int = 0
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'accuracy': self.accuracy,
            'confidence': self.confidence,
            'training_count': self.training_count,
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'created': self.created,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerceptronState':
        """Deserialize from dictionary."""
        state = cls(
            weights=np.array(data['weights']),
            bias=data['bias'],
            learning_rate=data['learning_rate'],
            accuracy=data.get('accuracy', 0.0),
            confidence=data.get('confidence', 0.0),
            training_count=data.get('training_count', 0),
            correct_predictions=data.get('correct_predictions', 0),
            total_predictions=data.get('total_predictions', 0),
            created=data.get('created', datetime.now().isoformat()),
            last_updated=data.get('last_updated', datetime.now().isoformat())
        )
        return state


class PerceptronUnit:
    """
    Binary perceptron classifier for puzzle detection.
    
    Uses step function activation for binary classification.
    Supports online learning with configurable learning rates.
    """
    
    def __init__(
        self,
        input_size: int,
        learning_rate: float = 0.1,
        initial_weights: Optional[np.ndarray] = None,
        initial_bias: float = 0.0
    ):
        """
        Initialize perceptron unit.
        
        Args:
            input_size: Size of input feature vector
            learning_rate: Learning rate for weight updates
            initial_weights: Optional initial weights (random if None)
            initial_bias: Initial bias value
        """
        self.input_size = input_size
        self.learning_rate = learning_rate
        
        if initial_weights is not None:
            self.weights = initial_weights.copy()
        else:
            # Initialize with small random values
            self.weights = np.random.uniform(-0.1, 0.1, input_size)
        
        self.bias = initial_bias
        
        # Performance tracking
        self.accuracy = 0.0
        self.confidence = 0.0
        self.training_count = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Timestamps
        self.created = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict puzzle detection for given features.
        
        Args:
            features: Input feature vector (normalized)
            
        Returns:
            Tuple of (puzzle_detected: bool, confidence: float)
        """
        # Ensure features are normalized
        features = self._normalize_features(features)
        
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, features) + self.bias
        
        # Step function activation
        output = 1 if weighted_sum > 0 else 0
        
        # Confidence is based on distance from decision boundary
        # Normalize to 0-1 range
        confidence = min(1.0, abs(weighted_sum) / (np.linalg.norm(self.weights) + 1e-6))
        
        self.total_predictions += 1
        
        return (output == 1, confidence)
    
    def train(self, features: np.ndarray, target: int) -> bool:
        """
        Train perceptron on a single example (online learning).
        
        Args:
            features: Input feature vector
            target: Target output (0 or 1)
            
        Returns:
            True if prediction was correct, False otherwise
        """
        features = self._normalize_features(features)
        
        # Predict
        prediction, _ = self.predict(features)
        predicted_value = 1 if prediction else 0
        
        # Check if prediction is correct
        is_correct = (predicted_value == target)
        
        if is_correct:
            self.correct_predictions += 1
        else:
            # Update weights using perceptron learning rule
            error = target - predicted_value
            self.weights += self.learning_rate * error * features
            self.bias += self.learning_rate * error
        
        # Update statistics
        self.training_count += 1
        self.accuracy = self.correct_predictions / max(1, self.total_predictions)
        
        # Update confidence based on recent accuracy
        if self.training_count > 0:
            recent_accuracy = self.correct_predictions / max(1, self.total_predictions)
            self.confidence = recent_accuracy
        
        self.last_updated = datetime.now().isoformat()
        
        return is_correct
    
    def train_batch(self, features_list: List[np.ndarray], targets: List[int]) -> float:
        """
        Train on a batch of examples.
        
        Args:
            features_list: List of feature vectors
            targets: List of target outputs
            
        Returns:
            Accuracy on the batch
        """
        correct = 0
        for features, target in zip(features_list, targets):
            if self.train(features, target):
                correct += 1
        
        batch_accuracy = correct / len(features_list) if features_list else 0.0
        return batch_accuracy
    
    def update_learning_rate(self, new_learning_rate: float):
        """Update learning rate."""
        self.learning_rate = new_learning_rate
    
    def get_state(self) -> PerceptronState:
        """Get current state of perceptron."""
        return PerceptronState(
            weights=self.weights.copy(),
            bias=self.bias,
            learning_rate=self.learning_rate,
            accuracy=self.accuracy,
            confidence=self.confidence,
            training_count=self.training_count,
            correct_predictions=self.correct_predictions,
            total_predictions=self.total_predictions,
            created=self.created,
            last_updated=self.last_updated
        )
    
    def load_state(self, state: PerceptronState):
        """Load state into perceptron."""
        self.weights = state.weights.copy()
        self.bias = state.bias
        self.learning_rate = state.learning_rate
        self.accuracy = state.accuracy
        self.confidence = state.confidence
        self.training_count = state.training_count
        self.correct_predictions = state.correct_predictions
        self.total_predictions = state.total_predictions
        self.created = state.created
        self.last_updated = state.last_updated
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector.
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        features = np.array(features, dtype=float)
        
        # Ensure correct size
        if len(features) != self.input_size:
            raise ValueError(f"Feature size {len(features)} doesn't match input size {self.input_size}")
        
        # Normalize to unit length
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize perceptron to dictionary."""
        return self.get_state().to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], input_size: Optional[int] = None) -> 'PerceptronUnit':
        """Deserialize perceptron from dictionary."""
        state = PerceptronState.from_dict(data)
        
        if input_size is None:
            input_size = len(state.weights)
        
        perceptron = cls(
            input_size=input_size,
            learning_rate=state.learning_rate,
            initial_weights=state.weights,
            initial_bias=state.bias
        )
        perceptron.load_state(state)
        return perceptron
