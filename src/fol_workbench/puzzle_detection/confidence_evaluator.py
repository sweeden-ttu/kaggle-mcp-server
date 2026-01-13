"""
Confidence Evaluator Module

Calculates confidence ratios based on measured improvements.
Provides evidence-based performance tracking and statistical significance testing.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from datetime import datetime, timedelta

try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Stub for ttest_ind if scipy not available
    def ttest_ind(*args, **kwargs):
        raise ImportError("scipy is required for statistical significance testing. Install with: pip install scipy")


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: str
    accuracy: float
    confidence: float
    sample_size: int = 1


@dataclass
class ImprovementEvidence:
    """Evidence of performance improvement."""
    confidence_ratio: float
    improvement_ratio: float
    baseline_confidence: float
    current_confidence: float
    baseline_accuracy: float
    current_accuracy: float
    statistical_significance: Optional[float] = None  # p-value
    is_significant: bool = False
    sample_size: int = 0
    meets_criteria: bool = False


class ConfidenceEvaluator:
    """
    Evaluates confidence and performance improvements.
    
    Calculates confidence ratios and provides evidence-based
    performance tracking with statistical significance testing.
    """
    
    def __init__(
        self,
        min_confidence_ratio: float = 0.1,
        min_sample_size: int = 100,
        significance_level: float = 0.05,
        baseline_window_size: int = 50
    ):
        """
        Initialize confidence evaluator.
        
        Args:
            min_confidence_ratio: Minimum confidence ratio for evidence (10% improvement)
            min_sample_size: Minimum sample size for statistical testing
            significance_level: P-value threshold for significance (default 0.05)
            baseline_window_size: Number of samples to use for baseline
        """
        self.min_confidence_ratio = min_confidence_ratio
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.baseline_window_size = baseline_window_size
        
        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        self.baseline_metrics: Optional[Tuple[float, float]] = None  # (accuracy, confidence)
        
        # Breadcrumb tracking for learning insights
        self.learning_breadcrumbs: List[Dict[str, Any]] = []
    
    def record_performance(
        self,
        accuracy: float,
        confidence: float,
        sample_size: int = 1
    ):
        """
        Record a performance measurement.
        
        Args:
            accuracy: Accuracy measurement
            confidence: Confidence measurement
            sample_size: Number of samples in this measurement
        """
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            confidence=confidence,
            sample_size=sample_size
        )
        self.performance_history.append(metric)
    
    def calculate_confidence_ratio(
        self,
        current_confidence: float,
        baseline_confidence: Optional[float] = None
    ) -> float:
        """
        Calculate confidence ratio: (current - baseline) / baseline.
        
        Args:
            current_confidence: Current confidence value
            baseline_confidence: Baseline confidence (uses stored baseline if None)
            
        Returns:
            Confidence ratio
        """
        if baseline_confidence is None:
            baseline_confidence = self.get_baseline_confidence()
        
        if baseline_confidence == 0:
            return 0.0
        
        ratio = (current_confidence - baseline_confidence) / baseline_confidence
        return ratio
    
    def calculate_improvement_ratio(
        self,
        current_accuracy: float,
        baseline_accuracy: Optional[float] = None
    ) -> float:
        """
        Calculate improvement ratio: (current - baseline) / baseline.
        
        Args:
            current_accuracy: Current accuracy value
            baseline_accuracy: Baseline accuracy (uses stored baseline if None)
            
        Returns:
            Improvement ratio
        """
        if baseline_accuracy is None:
            baseline_accuracy = self.get_baseline_accuracy()
        
        if baseline_accuracy == 0:
            return 0.0
        
        ratio = (current_accuracy - baseline_accuracy) / baseline_accuracy
        return ratio
    
    def evaluate_improvement(
        self,
        current_accuracy: Optional[float] = None,
        current_confidence: Optional[float] = None
    ) -> ImprovementEvidence:
        """
        Evaluate if there's evidence of improvement.
        
        Args:
            current_accuracy: Current accuracy (uses latest if None)
            current_confidence: Current confidence (uses latest if None)
            
        Returns:
            ImprovementEvidence object
        """
        if not self.performance_history:
            return ImprovementEvidence(
                confidence_ratio=0.0,
                improvement_ratio=0.0,
                baseline_confidence=0.0,
                current_confidence=0.0,
                baseline_accuracy=0.0,
                current_accuracy=0.0,
                meets_criteria=False
            )
        
        # Get current metrics
        if current_accuracy is None or current_confidence is None:
            latest = self.performance_history[-1]
            current_accuracy = latest.accuracy
            current_confidence = latest.confidence
        
        # Get baseline metrics
        baseline_accuracy = self.get_baseline_accuracy()
        baseline_confidence = self.get_baseline_confidence()
        
        # Calculate ratios
        confidence_ratio = self.calculate_confidence_ratio(current_confidence, baseline_confidence)
        improvement_ratio = self.calculate_improvement_ratio(current_accuracy, baseline_accuracy)
        
        # Statistical significance testing
        statistical_significance = None
        is_significant = False
        
        if len(self.performance_history) >= self.min_sample_size and SCIPY_AVAILABLE:
            # Split history into baseline and current periods
            split_point = len(self.performance_history) // 2
            
            baseline_accuracies = [
                m.accuracy for m in list(self.performance_history)[:split_point]
            ]
            current_accuracies = [
                m.accuracy for m in list(self.performance_history)[split_point:]
            ]
            
            if len(baseline_accuracies) > 1 and len(current_accuracies) > 1:
                try:
                    t_stat, p_value = ttest_ind(baseline_accuracies, current_accuracies)
                    statistical_significance = p_value
                    is_significant = p_value < self.significance_level
                except Exception:
                    pass
        
        # Check if meets criteria
        meets_criteria = (
            confidence_ratio > self.min_confidence_ratio and
            len(self.performance_history) >= self.min_sample_size and
            (is_significant or statistical_significance is None)
        )
        
        return ImprovementEvidence(
            confidence_ratio=confidence_ratio,
            improvement_ratio=improvement_ratio,
            baseline_confidence=baseline_confidence,
            current_confidence=current_confidence,
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            statistical_significance=statistical_significance,
            is_significant=is_significant,
            sample_size=len(self.performance_history),
            meets_criteria=meets_criteria
        )
    
    def get_baseline_confidence(self) -> float:
        """
        Get baseline confidence from history.
        
        Returns:
            Baseline confidence value
        """
        if not self.performance_history:
            return 0.0
        
        # Use first N samples as baseline
        baseline_samples = min(self.baseline_window_size, len(self.performance_history))
        baseline_metrics = list(self.performance_history)[:baseline_samples]
        
        if not baseline_metrics:
            return 0.0
        
        return np.mean([m.confidence for m in baseline_metrics])
    
    def get_baseline_accuracy(self) -> float:
        """
        Get baseline accuracy from history.
        
        Returns:
            Baseline accuracy value
        """
        if not self.performance_history:
            return 0.0
        
        # Use first N samples as baseline
        baseline_samples = min(self.baseline_window_size, len(self.performance_history))
        baseline_metrics = list(self.performance_history)[:baseline_samples]
        
        if not baseline_metrics:
            return 0.0
        
        return np.mean([m.accuracy for m in baseline_metrics])
    
    def get_current_confidence(self) -> float:
        """Get current confidence from latest measurement."""
        if not self.performance_history:
            return 0.0
        
        return self.performance_history[-1].confidence
    
    def get_current_accuracy(self) -> float:
        """Get current accuracy from latest measurement."""
        if not self.performance_history:
            return 0.0
        
        return self.performance_history[-1].accuracy
    
    def has_sufficient_evidence(self) -> bool:
        """
        Check if there's sufficient evidence for decision making.
        
        Returns:
            True if sufficient samples and evidence
        """
        if len(self.performance_history) < self.min_sample_size:
            return False
        
        evidence = self.evaluate_improvement()
        return evidence.meets_criteria
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.performance_history:
            return {
                'sample_count': 0,
                'baseline_accuracy': 0.0,
                'current_accuracy': 0.0,
                'baseline_confidence': 0.0,
                'current_confidence': 0.0,
                'confidence_ratio': 0.0,
                'improvement_ratio': 0.0
            }
        
        evidence = self.evaluate_improvement()
        
        return {
            'sample_count': len(self.performance_history),
            'baseline_accuracy': evidence.baseline_accuracy,
            'current_accuracy': evidence.current_accuracy,
            'baseline_confidence': evidence.baseline_confidence,
            'current_confidence': evidence.current_confidence,
            'confidence_ratio': evidence.confidence_ratio,
            'improvement_ratio': evidence.improvement_ratio,
            'statistical_significance': evidence.statistical_significance,
            'is_significant': evidence.is_significant,
            'meets_criteria': evidence.meets_criteria
        }
    
    def add_learning_breadcrumb(self, breadcrumb: Dict[str, Any]):
        """
        Add a learning breadcrumb from agents.
        
        Args:
            breadcrumb: Dictionary containing breadcrumb data with keys:
                - timestamp: ISO timestamp
                - event_type: Type of event
                - data: Event-specific data
                - source: Source of the breadcrumb
        """
        self.learning_breadcrumbs.append(breadcrumb)
        
        # Keep only last 1000 breadcrumbs
        if len(self.learning_breadcrumbs) > 1000:
            self.learning_breadcrumbs = self.learning_breadcrumbs[-1000:]
    
    def get_learning_breadcrumbs(self) -> List[Dict[str, Any]]:
        """Get all learning breadcrumbs."""
        return self.learning_breadcrumbs.copy()
    
    def get_breadcrumbs_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get breadcrumbs filtered by event type."""
        return [bc for bc in self.learning_breadcrumbs if bc.get("event_type") == event_type]