"""
Hyperparameter Tuner Module

Adjusts learning rates and hyperparameters based on performance.
Manages static parameter configuration for episodes.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime


class HyperparameterStrategy(Enum):
    """Strategies for hyperparameter tuning."""
    FIXED = "fixed"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    ADAPTIVE = "adaptive"


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameters."""
    learning_rate: float = 0.1
    batch_size: int = 1
    iterations: int = 100
    confidence_threshold_high: float = 0.8
    confidence_threshold_medium: float = 0.5
    confidence_threshold_low: float = 0.3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'iterations': self.iterations,
            'confidence_threshold_high': self.confidence_threshold_high,
            'confidence_threshold_medium': self.confidence_threshold_medium,
            'confidence_threshold_low': self.confidence_threshold_low,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HyperparameterConfig':
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class EpisodeResult:
    """Results from an episode."""
    episode_id: int
    config: HyperparameterConfig
    average_accuracy: float
    average_confidence: float
    performance_metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HyperparameterTuner:
    """
    Manages hyperparameter tuning and assignment.
    
    Supports multiple strategies:
    - Fixed: Use same parameters for all episodes
    - Grid search: Try all combinations
    - Random search: Random sampling
    - Adaptive: Adjust based on performance
    """
    
    def __init__(
        self,
        learning_rates: Optional[List[float]] = None,
        strategy: HyperparameterStrategy = HyperparameterStrategy.ADAPTIVE
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            learning_rates: List of learning rates to try
            strategy: Tuning strategy
        """
        if learning_rates is None:
            learning_rates = [0.01, 0.05, 0.1, 0.5]
        
        self.learning_rates = learning_rates
        self.strategy = strategy
        self.episode_history: List[EpisodeResult] = []
        self.current_episode = 0
        
        # Default confidence thresholds
        self.confidence_threshold_high = 0.8
        self.confidence_threshold_medium = 0.5
        self.confidence_threshold_low = 0.3
    
    def assign_hyperparameters(self, episode_id: Optional[int] = None) -> HyperparameterConfig:
        """
        Assign hyperparameters for an episode.
        
        Args:
            episode_id: Optional episode ID (auto-incremented if None)
            
        Returns:
            HyperparameterConfig for the episode
        """
        if episode_id is None:
            episode_id = self.current_episode
            self.current_episode += 1
        
        if self.strategy == HyperparameterStrategy.FIXED:
            config = self._get_fixed_config()
        elif self.strategy == HyperparameterStrategy.GRID_SEARCH:
            config = self._get_grid_search_config(episode_id)
        elif self.strategy == HyperparameterStrategy.RANDOM_SEARCH:
            config = self._get_random_search_config()
        elif self.strategy == HyperparameterStrategy.ADAPTIVE:
            config = self._get_adaptive_config()
        else:
            config = self._get_fixed_config()
        
        return config
    
    def record_episode_result(self, result: EpisodeResult):
        """
        Record results from an episode.
        
        Args:
            result: EpisodeResult to record
        """
        self.episode_history.append(result)
    
    def optimize(self) -> HyperparameterConfig:
        """
        Optimize hyperparameters based on history.
        
        Returns:
            Optimized HyperparameterConfig
        """
        if not self.episode_history:
            return self._get_default_config()
        
        # Find best performing episode
        best_result = max(
            self.episode_history,
            key=lambda r: r.average_accuracy * r.average_confidence
        )
        
        # Use best config as base, with slight variations
        config = best_result.config
        
        # Adaptive adjustment
        if self.strategy == HyperparameterStrategy.ADAPTIVE:
            # If performance is improving, keep similar learning rate
            # If performance is degrading, try different learning rate
            recent_results = self.episode_history[-5:] if len(self.episode_history) >= 5 else self.episode_history
            
            if len(recent_results) >= 2:
                recent_avg = np.mean([r.average_accuracy for r in recent_results])
                older_avg = np.mean([r.average_accuracy for r in self.episode_history[:-len(recent_results)]])
                
                if recent_avg < older_avg * 0.9:  # Performance degraded
                    # Try different learning rate
                    current_lr = config.learning_rate
                    available_lrs = [lr for lr in self.learning_rates if lr != current_lr]
                    if available_lrs:
                        config.learning_rate = np.random.choice(available_lrs)
        
        return config
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level category.
        
        Args:
            confidence: Confidence value (0.0-1.0)
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if confidence >= self.confidence_threshold_high:
            return 'high'
        elif confidence >= self.confidence_threshold_medium:
            return 'medium'
        else:
            return 'low'
    
    def _get_fixed_config(self) -> HyperparameterConfig:
        """Get fixed configuration."""
        return HyperparameterConfig(
            learning_rate=self.learning_rates[0],
            batch_size=1,
            iterations=100
        )
    
    def _get_grid_search_config(self, episode_id: int) -> HyperparameterConfig:
        """Get configuration from grid search."""
        # Simple grid search over learning rates
        lr_index = episode_id % len(self.learning_rates)
        learning_rate = self.learning_rates[lr_index]
        
        return HyperparameterConfig(
            learning_rate=learning_rate,
            batch_size=1,
            iterations=100
        )
    
    def _get_random_search_config(self) -> HyperparameterConfig:
        """Get random configuration."""
        learning_rate = np.random.choice(self.learning_rates)
        
        return HyperparameterConfig(
            learning_rate=learning_rate,
            batch_size=np.random.choice([1, 5, 10]),
            iterations=np.random.choice([50, 100, 200])
        )
    
    def _get_adaptive_config(self) -> HyperparameterConfig:
        """Get adaptive configuration based on history."""
        if not self.episode_history:
            return self._get_default_config()
        
        # Use best performing learning rate
        best_result = max(
            self.episode_history,
            key=lambda r: r.average_accuracy
        )
        
        return best_result.config
    
    def _get_default_config(self) -> HyperparameterConfig:
        """Get default configuration."""
        return HyperparameterConfig(
            learning_rate=0.1,
            batch_size=1,
            iterations=100
        )
