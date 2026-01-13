"""
Puzzle Detection System for TorBrowser Screenshots

A system that captures TorBrowser screenshots, divides them into adaptive grid squares,
detects handwritten text/images/questions, and uses perceptron units with BayesianFeatureExtractor
to learn puzzle detection and perceptron assignment.
"""

# Import modules - handle optional dependencies gracefully.
#
# NOTE: Some modules depend on optional native/system packages (e.g., Pillow/PIL,
# OpenCV, Tesseract, SciPy). We keep imports lazy/guarded so non-UI / non-image
# parts of the codebase (e.g., logic pipelines) can still be imported in minimal
# environments.

try:
    from .screenshot_capture import ScreenshotCapture
except ImportError as e:
    class ScreenshotCapture:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"ScreenshotCapture could not be imported: {e}")

try:
    from .grid_segmentation import GridSegmentation
except ImportError as e:
    class GridSegmentation:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GridSegmentation could not be imported: {e}")

try:
    from .feature_extractor import FeatureExtractor
except ImportError as e:
    class FeatureExtractor:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"FeatureExtractor could not be imported: {e}")

try:
    from .perceptron_units import PerceptronUnit
except ImportError as e:
    class PerceptronUnit:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PerceptronUnit could not be imported: {e}")

try:
    from .hyperparameter_tuner import HyperparameterTuner
except ImportError as e:
    class HyperparameterTuner:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(f"HyperparameterTuner could not be imported: {e}")

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
