"""
Puzzle Detection System for TorBrowser Screenshots

A system that captures TorBrowser screenshots, divides them into adaptive grid squares,
detects handwritten text/images/questions, and uses perceptron units with BayesianFeatureExtractor
to learn puzzle detection and perceptron assignment.
"""

try:
    from .screenshot_capture import ScreenshotCapture
    from .grid_segmentation import GridSegmentation
    from .feature_extractor import FeatureExtractor
    from .perceptron_units import PerceptronUnit
    from .hyperparameter_tuner import HyperparameterTuner
    from .confidence_evaluator import ConfidenceEvaluator
    from .bayesian_perceptron_assigner import BayesianPerceptronAssigner
    from .learning_pipeline import LearningPipeline
    from .puzzle_detector import PuzzleDetector
    
    __all__ = [
        'ScreenshotCapture',
        'GridSegmentation',
        'FeatureExtractor',
        'PerceptronUnit',
        'HyperparameterTuner',
        'ConfidenceEvaluator',
        'BayesianPerceptronAssigner',
        'LearningPipeline',
        'PuzzleDetector',
    ]
except ImportError as e:
    # Handle optional dependencies gracefully
    __all__ = []
    import warnings
    warnings.warn(f"Some puzzle detection modules could not be imported: {e}")
