"""
Main Puzzle Detector System

Orchestrates the entire pipeline:
screenshot → grid → features → perceptrons → assignment → learning
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .screenshot_capture import ScreenshotCapture
from .grid_segmentation import GridSegmentation, GridSquare
from .feature_extractor import FeatureExtractor, GridSquareFeatures
from .perceptron_units import PerceptronUnit
from .hyperparameter_tuner import HyperparameterTuner
from .confidence_evaluator import ConfidenceEvaluator
from .bayesian_perceptron_assigner import BayesianPerceptronAssigner
from .learning_pipeline import LearningPipeline
from ..database import Database


@dataclass
class DetectionResult:
    """Result from puzzle detection."""
    square_id: str
    puzzle_detected: bool
    confidence: float
    perceptron_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PuzzleDetector:
    """
    Main puzzle detection system.
    
    Orchestrates the complete pipeline:
    1. Capture screenshot from TorBrowser
    2. Generate adaptive grid
    3. Extract features from each grid square
    4. Assign perceptrons using Bayesian inference
    5. Detect puzzles
    6. Learn and improve
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        input_size: int = 7  # Number of features
    ):
        """
        Initialize puzzle detector.
        
        Args:
            database: Database instance for persistence
            input_size: Size of input feature vector
        """
        self.database = database or Database()
        self.input_size = input_size
        
        # Initialize components
        self.screenshot_capture = ScreenshotCapture()
        self.grid_segmentation = GridSegmentation()
        self.feature_extractor = FeatureExtractor()
        self.hyperparameter_tuner = HyperparameterTuner()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.bayesian_assigner = BayesianPerceptronAssigner()
        
        # Learning pipeline
        self.learning_pipeline = LearningPipeline(
            database=self.database,
            hyperparameter_tuner=self.hyperparameter_tuner,
            confidence_evaluator=self.confidence_evaluator,
            bayesian_assigner=self.bayesian_assigner
        )
        
        # Perceptron storage
        self.perceptrons: Dict[str, PerceptronUnit] = {}
        self._initialize_perceptrons()
    
    def _initialize_perceptrons(self):
        """Initialize perceptrons from database or create new ones."""
        # Try to load from database
        high_conf_perceptrons = self.database.get_high_confidence_perceptrons(threshold=0.7)
        
        if high_conf_perceptrons:
            # Load perceptrons from database
            for db_perceptron in high_conf_perceptrons:
                perceptron = PerceptronUnit(
                    input_size=self.input_size,
                    learning_rate=db_perceptron.learning_rate,
                    initial_weights=np.array(db_perceptron.weights),
                    initial_bias=0.0
                )
                # Restore state
                perceptron.accuracy = db_perceptron.accuracy
                perceptron.confidence = db_perceptron.confidence
                self.perceptrons[db_perceptron.perceptron_id] = perceptron
        else:
            # Create initial perceptrons
            self.perceptrons['text_puzzle'] = PerceptronUnit(
                input_size=self.input_size,
                learning_rate=0.1
            )
            self.perceptrons['image_puzzle'] = PerceptronUnit(
                input_size=self.input_size,
                learning_rate=0.1
            )
            self.perceptrons['question_puzzle'] = PerceptronUnit(
                input_size=self.input_size,
                learning_rate=0.1
            )
    
    def detect_from_screenshot(
        self,
        image: Optional[Image.Image] = None,
        capture_torbrowser: bool = True
    ) -> List[DetectionResult]:
        """
        Detect puzzles from a screenshot.
        
        Args:
            image: Optional PIL Image (captures if None)
            capture_torbrowser: Whether to capture TorBrowser window
            
        Returns:
            List of DetectionResult objects
        """
        # Step 1: Capture screenshot
        if image is None:
            if capture_torbrowser:
                image = self.screenshot_capture.capture_torbrowser()
            else:
                image = self.screenshot_capture.capture_full_screen()
        
        # Step 2: Generate adaptive grid
        grid_squares = self.grid_segmentation.segment_adaptive(image)
        
        # Step 3: Extract features
        features_list = self.feature_extractor.extract_batch(grid_squares)
        
        # Step 4: Assign perceptrons and detect
        results = []
        for features in features_list:
            # Assign perceptron using Bayesian inference
            assignment = self.bayesian_assigner.assign_perceptron(
                features,
                self.perceptrons
            )
            
            # Get perceptron
            if assignment.perceptron_id and assignment.perceptron_id in self.perceptrons:
                perceptron = self.perceptrons[assignment.perceptron_id]
            else:
                # Use default perceptron
                perceptron = list(self.perceptrons.values())[0]
            
            # Convert features to feature vector
            feature_vector = self._features_to_vector(features)
            
            # Predict
            puzzle_detected, confidence = perceptron.predict(feature_vector)
            
            result = DetectionResult(
                square_id=features.square_id,
                puzzle_detected=puzzle_detected,
                confidence=confidence,
                perceptron_id=assignment.perceptron_id,
                metadata={
                    'assignment_confidence': assignment.assignment_confidence,
                    'features': {
                        'has_text': features.has_text,
                        'has_image': features.has_image,
                        'has_question': features.has_question
                    }
                }
            )
            results.append(result)
        
        return results
    
    def train_on_examples(
        self,
        examples: List[Tuple[GridSquareFeatures, int]]
    ) -> Dict[str, Any]:
        """
        Train perceptrons on labeled examples.
        
        Args:
            examples: List of (features, target) tuples where target is 0 or 1
            
        Returns:
            Training results
        """
        # Provide both the feature vector and the original GridSquareFeatures so the
        # Bayesian assignment model can learn better square→perceptron routing.
        training_data: List[Tuple[np.ndarray, int, GridSquareFeatures]] = []
        for features, target in examples:
            feature_vector = self._features_to_vector(features)
            training_data.append((feature_vector, target, features))
        
        # Run learning pipeline episode
        results = self.learning_pipeline.run_episode(training_data, perceptrons=self.perceptrons)
        
        # Update in-memory perceptrons from pipeline state
        self.perceptrons.update(self.learning_pipeline.state.perceptrons)
        
        return results
    
    def run_learning_episode(
        self,
        training_data: List[Tuple[np.ndarray, int]],
        validation_data: Optional[List[Tuple[np.ndarray, int]]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete learning episode.
        
        Args:
            training_data: List of (features, target) tuples
            validation_data: Optional validation data
            
        Returns:
            Episode results
        """
        return self.learning_pipeline.run_episode(training_data, validation_data, perceptrons=self.perceptrons)
    
    def _features_to_vector(self, features: GridSquareFeatures) -> np.ndarray:
        """
        Convert GridSquareFeatures to feature vector.
        
        Args:
            features: GridSquareFeatures object
            
        Returns:
            Feature vector as numpy array
        """
        vector = np.array([
            1.0 if features.has_text else 0.0,
            1.0 if features.has_handwriting else 0.0,
            1.0 if features.has_image else 0.0,
            1.0 if features.has_question else 0.0,
            features.text_confidence,
            features.image_complexity,
            1.0 if features.question_type else 0.0
        ], dtype=float)
        
        return vector
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of system performance."""
        return {
            'perceptron_count': len(self.perceptrons),
            'average_accuracy': np.mean([p.accuracy for p in self.perceptrons.values()]) if self.perceptrons else 0.0,
            'average_confidence': np.mean([p.confidence for p in self.perceptrons.values()]) if self.perceptrons else 0.0,
            'confidence_evaluator': self.confidence_evaluator.get_performance_summary(),
            'database_perceptrons': len(self.database.get_high_confidence_perceptrons(threshold=0.0))
        }
    
    def save_state(self):
        """Save current state to database."""
        # High-confidence perceptrons are already saved by learning pipeline
        # This could save additional state if needed
        pass
    
    def load_state(self):
        """Load state from database."""
        self._initialize_perceptrons()
