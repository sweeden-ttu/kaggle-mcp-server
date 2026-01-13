"""
Ultra-Large Language Model for Generating Pretrained Text Describing Experimental Observations

Generates descriptive text that explains observed experimental data and results
from the Bayesian feature extractor and decision tree systems.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
import re


@dataclass
class Observation:
    """Represents an experimental observation."""
    observation_id: str
    timestamp: str
    layer_id: int
    class_name: str
    attributes: Dict[str, Any]
    confidence_scores: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "layer_id": self.layer_id,
            "class_name": self.class_name,
            "attributes": self.attributes,
            "confidence_scores": self.confidence_scores,
            "context": self.context
        }


class UltraLargeLanguageModel:
    """
    Ultra-Large Language Model for generating pretrained text descriptions
    of experimental observations.
    
    This model uses pattern-based generation with vocabulary from the
    Bayesian feature extractor system to create descriptive text.
    """
    
    def __init__(self, vocabulary_universe: Optional[set] = None):
        self.vocabulary_universe = vocabulary_universe or set()
        self.observation_history: List[Observation] = []
        self.text_corpus: List[str] = []
        self.templates: List[str] = []
        
        # Initialize with base templates
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize text generation templates."""
        self.templates = [
            "The experimental data reveals that {class_name} exhibits {attribute_pattern} with {confidence_level} confidence.",
            "Observation {observation_id} demonstrates {attribute_pattern} in layer {layer_id}, indicating {interpretation}.",
            "Analysis of {class_name} shows {attribute_pattern}, suggesting {conclusion}.",
            "The Bayesian feature extraction for {class_name} yields {attribute_pattern}, with confidence scores of {confidence_detail}.",
            "In layer {layer_id}, the class {class_name} manifests {attribute_pattern}, revealing {insight}.",
            "Experimental results indicate that {class_name} possesses {attribute_pattern}, which aligns with {contextual_note}.",
            "The observed data for {class_name} demonstrates {attribute_pattern}, consistent with {confidence_level} probability.",
            "Feature extraction results show {attribute_pattern} for {class_name} in layer {layer_id}, implying {interpretation}.",
            "The Bayesian posterior distribution for {class_name} suggests {attribute_pattern}, with key indicators being {key_attributes}.",
            "Analysis reveals that {class_name} exhibits {attribute_pattern}, which corresponds to {confidence_level} statistical confidence."
        ]
    
    def add_vocabulary(self, vocabulary: set):
        """Add vocabulary to the universe."""
        self.vocabulary_universe.update(vocabulary)
    
    def record_observation(self, observation: Observation):
        """Record an observation for text generation."""
        self.observation_history.append(observation)
        
        # Generate text immediately
        text = self.generate_description(observation)
        self.text_corpus.append(text)
    
    def generate_description(self, observation: Observation) -> str:
        """Generate descriptive text for an observation."""
        import random
        
        template = random.choice(self.templates)
        
        # Extract attribute pattern
        attr_pattern = self._format_attributes(observation.attributes)
        
        # Format confidence level
        conf_level = self._format_confidence(observation.confidence_scores)
        
        # Generate interpretation
        interpretation = self._generate_interpretation(observation)
        
        # Generate conclusion/insight
        conclusion = self._generate_conclusion(observation)
        
        # Format confidence detail
        conf_detail = self._format_confidence_detail(observation.confidence_scores)
        
        # Key attributes
        key_attrs = self._get_key_attributes(observation.attributes, observation.confidence_scores)
        
        # Contextual note
        contextual_note = self._generate_contextual_note(observation)
        
        # Fill template
        text = template.format(
            class_name=observation.class_name,
            attribute_pattern=attr_pattern,
            confidence_level=conf_level,
            observation_id=observation.observation_id,
            layer_id=observation.layer_id,
            interpretation=interpretation,
            conclusion=conclusion,
            confidence_detail=conf_detail,
            key_attributes=key_attrs,
            contextual_note=contextual_note
        )
        
        return text
    
    def _format_attributes(self, attributes: Dict[str, Any]) -> str:
        """Format attributes into a readable pattern."""
        if not attributes:
            return "no observable attributes"
        
        parts = []
        for attr_name, attr_value in list(attributes.items())[:3]:  # Limit to first 3
            if isinstance(attr_value, dict):
                value = attr_value.get("value", "unknown")
            else:
                value = attr_value
            
            parts.append(f"{attr_name}={value}")
        
        if len(attributes) > 3:
            parts.append(f"and {len(attributes) - 3} more")
        
        return ", ".join(parts)
    
    def _format_confidence(self, confidence_scores: Dict[str, float]) -> str:
        """Format confidence scores into a level description."""
        if not confidence_scores:
            return "low"
        
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        
        if avg_confidence >= 0.8:
            return "high"
        elif avg_confidence >= 0.5:
            return "moderate"
        else:
            return "low"
    
    def _generate_interpretation(self, observation: Observation) -> str:
        """Generate an interpretation of the observation."""
        interpretations = [
            "strong statistical significance",
            "robust feature patterns",
            "consistent attribute relationships",
            "meaningful structural properties",
            "significant correlations",
            "distinctive characteristics",
            "notable patterns in the data",
            "promising experimental trends"
        ]
        
        import random
        return random.choice(interpretations)
    
    def _generate_conclusion(self, observation: Observation) -> str:
        """Generate a conclusion based on the observation."""
        conclusions = [
            "favorable experimental outcomes",
            "alignment with theoretical predictions",
            "potential for further investigation",
            "validation of initial hypotheses",
            "interesting experimental patterns",
            "significant data characteristics",
            "promising research directions",
            "important feature relationships"
        ]
        
        import random
        return random.choice(conclusions)
    
    def _format_confidence_detail(self, confidence_scores: Dict[str, float]) -> str:
        """Format confidence scores in detail."""
        if not confidence_scores:
            return "none available"
        
        parts = []
        for attr_name, score in list(confidence_scores.items())[:3]:
            parts.append(f"{attr_name}: {score:.2f}")
        
        return ", ".join(parts)
    
    def _get_key_attributes(self, attributes: Dict[str, Any], confidence_scores: Dict[str, float]) -> str:
        """Get key attributes with highest confidence."""
        if not confidence_scores:
            return "none identified"
        
        # Sort by confidence
        sorted_attrs = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        key_names = [name for name, _ in sorted_attrs[:3]]
        return ", ".join(key_names)
    
    def _generate_contextual_note(self, observation: Observation) -> str:
        """Generate a contextual note."""
        notes = [
            "established experimental protocols",
            "standard analytical frameworks",
            "previously observed patterns",
            "theoretical expectations",
            "baseline measurements",
            "control group comparisons",
            "historical data trends",
            "reference model predictions"
        ]
        
        import random
        return random.choice(notes)
    
    def generate_batch_descriptions(self, observations: List[Observation]) -> List[str]:
        """Generate descriptions for multiple observations."""
        descriptions = []
        for obs in observations:
            desc = self.generate_description(obs)
            descriptions.append(desc)
            self.text_corpus.append(desc)
        return descriptions
    
    def generate_pretrained_text(self, max_length: Optional[int] = None) -> str:
        """Generate a comprehensive pretrained text from all observations."""
        if not self.text_corpus:
            return "No experimental observations have been recorded yet."
        
        # Combine all text
        combined_text = "\n\n".join(self.text_corpus)
        
        if max_length and len(combined_text) > max_length:
            # Truncate but keep complete sentences
            truncated = combined_text[:max_length]
            # Find last sentence end
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > max_length * 0.8:  # Only truncate if we get reasonable amount
                combined_text = truncated[:cut_point + 1]
        
        return combined_text
    
    def save_pretrained_text(self, filepath: Path):
        """Save the generated pretrained text to a file."""
        text = self.generate_pretrained_text()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get statistics about the text corpus."""
        total_chars = sum(len(text) for text in self.text_corpus)
        total_words = sum(len(text.split()) for text in self.text_corpus)
        
        return {
            "num_observations": len(self.observation_history),
            "num_text_segments": len(self.text_corpus),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_segment": total_chars / len(self.text_corpus) if self.text_corpus else 0,
            "vocabulary_size": len(self.vocabulary_universe)
        }
