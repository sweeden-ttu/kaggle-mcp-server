"""
Bayesian Feature Extractor with Layered Class Collections

Implements a novel ML strategy using Bayesian feature extractors that organize
classes into layered collections with types and attributes. Each layer imposes
constraints on previous layers, creating a hierarchical feature extraction system.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class AttributeType(Enum):
    """Types of attributes that can be assigned to classes."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    LOGICAL = "logical"  # For FOL predicates


@dataclass
class Attribute:
    """Represents a single attribute of a class."""
    name: str
    attr_type: AttributeType
    value: Any = None
    prior_distribution: Optional[Dict[str, float]] = None  # Bayesian prior
    posterior_distribution: Optional[Dict[str, float]] = None  # Bayesian posterior
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "type": self.attr_type.value,
            "value": self.value,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Attribute':
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            attr_type=AttributeType(data["type"]),
            value=data.get("value"),
            confidence=data.get("confidence", 0.0)
        )


@dataclass
class ClassLayer:
    """Represents a class in a layer with attributes."""
    class_name: str
    layer_id: int
    attributes: List[Attribute] = field(default_factory=list)
    parent_classes: List[str] = field(default_factory=list)  # Classes from previous layers
    bayesian_prior: Dict[str, float] = field(default_factory=dict)
    
    def add_attribute(self, attr: Attribute):
        """Add an attribute to this class."""
        self.attributes.append(attr)
    
    def get_attribute(self, name: str) -> Optional[Attribute]:
        """Get attribute by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "class_name": self.class_name,
            "layer_id": self.layer_id,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "parent_classes": self.parent_classes,
            "bayesian_prior": self.bayesian_prior
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassLayer':
        """Deserialize from dictionary."""
        return cls(
            class_name=data["class_name"],
            layer_id=data["layer_id"],
            attributes=[Attribute.from_dict(attr_data) for attr_data in data.get("attributes", [])],
            parent_classes=data.get("parent_classes", []),
            bayesian_prior=data.get("bayesian_prior", {})
        )


@dataclass
class Layer:
    """Represents a layer of classes that imposes constraints on previous layers."""
    layer_id: int
    layer_name: str
    classes: Dict[str, ClassLayer] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)  # FOL constraints
    vocabulary: Set[str] = field(default_factory=set)  # Vocabulary for this layer
    
    def add_class(self, class_layer: ClassLayer):
        """Add a class to this layer."""
        self.classes[class_layer.class_name] = class_layer
        # Update vocabulary with class name and attribute names
        self.vocabulary.add(class_layer.class_name)
        for attr in class_layer.attributes:
            self.vocabulary.add(attr.name)
    
    def get_class(self, class_name: str) -> Optional[ClassLayer]:
        """Get class by name."""
        return self.classes.get(class_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "classes": {name: cls.to_dict() for name, cls in self.classes.items()},
            "constraints": self.constraints,
            "vocabulary": list(self.vocabulary)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Layer':
        """Deserialize from dictionary."""
        layer = cls(
            layer_id=data["layer_id"],
            layer_name=data["layer_name"],
            constraints=data.get("constraints", []),
            vocabulary=set(data.get("vocabulary", []))
        )
        for name, cls_data in data.get("classes", {}).items():
            class_layer = ClassLayer.from_dict(cls_data)
            layer.classes[name] = class_layer
        return layer


class BayesianFeatureExtractor:
    """
    Bayesian Feature Extractor with Layered Class Collections.
    
    Organizes classes into layers where each layer imposes constraints on
    previous layers. Uses Bayesian inference for feature extraction and
    attribute assignment.
    """
    
    def __init__(self):
        self.layers: Dict[int, Layer] = {}
        self.layer_order: List[int] = []  # Order of layers (lowest to highest)
        self.vocabulary_universe: Set[str] = set()  # Universe of all vocabularies
        self.log_history: List[str] = []
        
        # Initialize learning log
        self.log_entry("BEGIN S LEARNING")
    
    def log_entry(self, message: str):
        """Add entry to learning log."""
        timestamp = datetime.now().isoformat()
        entry = f"[{timestamp}] {message}"
        self.log_history.append(entry)
        print(entry)  # Also print for immediate visibility
    
    def create_layer(self, layer_id: int, layer_name: str) -> Layer:
        """Create a new layer."""
        if layer_id in self.layers:
            raise ValueError(f"Layer {layer_id} already exists")
        
        layer = Layer(layer_id=layer_id, layer_name=layer_name)
        self.layers[layer_id] = layer
        self.layer_order.append(layer_id)
        self.layer_order.sort()  # Keep ordered
        
        self.log_entry(f"Created layer {layer_id}: {layer_name}")
        return layer
    
    def add_class_to_layer(
        self,
        layer_id: int,
        class_name: str,
        attributes: Optional[List[Attribute]] = None,
        parent_classes: Optional[List[str]] = None
    ) -> ClassLayer:
        """Add a class to a layer."""
        if layer_id not in self.layers:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        layer = self.layers[layer_id]
        
        # Check if class already exists
        if class_name in layer.classes:
            return layer.classes[class_name]
        
        # Create class layer
        class_layer = ClassLayer(
            class_name=class_name,
            layer_id=layer_id,
            attributes=attributes or [],
            parent_classes=parent_classes or []
        )
        
        # Apply constraints from previous layers
        self._impose_layer_constraints(class_layer, layer_id)
        
        layer.add_class(class_layer)
        self.vocabulary_universe.update(layer.vocabulary)
        
        self.log_entry(f"Added class {class_name} to layer {layer_id}")
        return class_layer
    
    def _impose_layer_constraints(self, class_layer: ClassLayer, layer_id: int):
        """Impose constraints from previous layers on the new class."""
        # Find all previous layers
        previous_layers = [lid for lid in self.layer_order if lid < layer_id]
        
        for prev_layer_id in previous_layers:
            prev_layer = self.layers[prev_layer_id]
            
            # If this class has parent classes from previous layer, inherit constraints
            for parent_name in class_layer.parent_classes:
                if parent_name in prev_layer.classes:
                    parent_class = prev_layer.classes[parent_name]
                    # Inherit attributes from parent (with modifications)
                    for parent_attr in parent_class.attributes:
                        # Check if attribute already exists
                        if not class_layer.get_attribute(parent_attr.name):
                            # Create new attribute with inherited properties
                            new_attr = Attribute(
                                name=parent_attr.name,
                                attr_type=parent_attr.attr_type,
                                prior_distribution=parent_attr.posterior_distribution or parent_attr.prior_distribution
                            )
                            class_layer.add_attribute(new_attr)
    
    def assign_attribute(
        self,
        layer_id: int,
        class_name: str,
        attr_name: str,
        attr_type: AttributeType,
        value: Any = None
    ) -> Attribute:
        """Assign a single attribute to a class in a layer."""
        layer = self.layers.get(layer_id)
        if not layer:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        class_layer = layer.get_class(class_name)
        if not class_layer:
            raise ValueError(f"Class {class_name} not found in layer {layer_id}")
        
        # Check if attribute exists
        existing_attr = class_layer.get_attribute(attr_name)
        if existing_attr:
            # Update existing attribute
            existing_attr.value = value
            existing_attr.attr_type = attr_type
            return existing_attr
        
        # Create new attribute
        attr = Attribute(name=attr_name, attr_type=attr_type, value=value)
        class_layer.add_attribute(attr)
        layer.vocabulary.add(attr_name)
        self.vocabulary_universe.add(attr_name)
        
        self.log_entry(f"Assigned attribute {attr_name} to {class_name} in layer {layer_id}")
        return attr
    
    def update_bayesian_prior(
        self,
        layer_id: int,
        class_name: str,
        prior: Dict[str, float]
    ):
        """Update Bayesian prior distribution for a class."""
        layer = self.layers.get(layer_id)
        if not layer:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        class_layer = layer.get_class(class_name)
        if not class_layer:
            raise ValueError(f"Class {class_name} not found in layer {layer_id}")
        
        class_layer.bayesian_prior = prior
        
        # Update attribute priors
        for attr in class_layer.attributes:
            if attr.name in prior:
                if attr.prior_distribution is None:
                    attr.prior_distribution = {}
                attr.prior_distribution[attr.name] = prior[attr.name]
    
    def extract_features(self, data: Dict[str, Any], layer_id: int) -> Dict[str, Any]:
        """Extract features using Bayesian inference for a given layer."""
        layer = self.layers.get(layer_id)
        if not layer:
            raise ValueError(f"Layer {layer_id} does not exist")
        
        features = {}
        
        for class_name, class_layer in layer.classes.items():
            class_features = {}
            
            for attr in class_layer.attributes:
                # Use Bayesian inference to compute posterior
                if attr.prior_distribution:
                    # Simple Bayesian update (can be extended)
                    posterior = self._bayesian_update(attr, data)
                    attr.posterior_distribution = posterior
                    attr.confidence = max(posterior.values()) if posterior else 0.0
                    
                    class_features[attr.name] = {
                        "value": attr.value,
                        "confidence": attr.confidence,
                        "posterior": posterior
                    }
                else:
                    class_features[attr.name] = {
                        "value": attr.value,
                        "confidence": 0.0
                    }
            
            features[class_name] = class_features
        
        self.log_entry(f"Extracted features for layer {layer_id}")
        return features
    
    def _bayesian_update(self, attr: Attribute, data: Dict[str, Any]) -> Dict[str, float]:
        """Perform Bayesian update on attribute distribution."""
        # Simple Bayesian update implementation
        # In a real system, this would use proper Bayesian inference
        
        if not attr.prior_distribution:
            return {}
        
        posterior = attr.prior_distribution.copy()
        
        # Update based on observed data
        if attr.name in data:
            observed_value = data[attr.name]
            if observed_value in posterior:
                posterior[observed_value] *= 1.5  # Boost observed value
            else:
                posterior[observed_value] = 0.1  # Add new value with low probability
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        
        return posterior
    
    def get_all_layers(self) -> List[Layer]:
        """Get all layers in order."""
        return [self.layers[lid] for lid in self.layer_order]
    
    def get_vocabulary_universe(self) -> Set[str]:
        """Get the universe of all vocabularies across layers."""
        return self.vocabulary_universe.copy()
    
    def save(self, filepath: Path):
        """Save the feature extractor to a file."""
        data = {
            "layers": {lid: layer.to_dict() for lid, layer in self.layers.items()},
            "layer_order": self.layer_order,
            "vocabulary_universe": list(self.vocabulary_universe),
            "log_history": self.log_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: Path):
        """Load the feature extractor from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.layers = {}
        for lid_str, layer_data in data.get("layers", {}).items():
            layer = Layer.from_dict(layer_data)
            self.layers[layer.layer_id] = layer
        
        self.layer_order = data.get("layer_order", [])
        self.vocabulary_universe = set(data.get("vocabulary_universe", []))
        self.log_history = data.get("log_history", [])
    
    def get_log(self) -> str:
        """Get the learning log."""
        return "\n".join(self.log_history)
