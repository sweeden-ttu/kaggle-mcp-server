"""
Puzzle Detection System for TorBrowser Screenshots

A system that captures TorBrowser screenshots, divides them into adaptive grid squares,
detects handwritten text/images/questions, and uses perceptron units with BayesianFeatureExtractor
to learn puzzle detection and perceptron assignment.
"""

# Import modules - handle optional dependencies gracefully
from .screenshot_capture import ScreenshotCapture
from .grid_segmentation import GridSegmentation
from .feature_extractor import FeatureExtractor
from .perceptron_units import PerceptronUnit
from .hyperparameter_tuner import HyperparameterTuner

try:
    from .confidence_evaluator import ConfidenceEvaluator
except ImportError:
    # scipy is optional - create a stub if not available
    class ConfidenceEvaluator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ConfidenceEvaluator requires scipy. Install with: pip install scipy")

try:
    from .bayesian_perceptron_assigner import BayesianPerceptronAssigner
    from .learning_pipeline import LearningPipeline
    from .puzzle_detector import PuzzleDetector
except ImportError as e:
    import warnings
    warnings.warn(f"Some puzzle detection modules could not be imported: {e}")
    # Create stubs for missing modules
    class BayesianPerceptronAssigner:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"BayesianPerceptronAssigner could not be imported: {e}")
    class LearningPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"LearningPipeline could not be imported: {e}")
    class PuzzleDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PuzzleDetector could not be imported: {e}")

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
