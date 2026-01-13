"""
ML Strategy Integration Module

Integrates Bayesian Feature Extractors, Decision Tree Designer, and
Ultra-Large Language Model into a unified ML strategy system.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from .bayesian_feature_extractor import (
    BayesianFeatureExtractor, Layer, ClassLayer, Attribute, AttributeType
)
from .decision_tree_designer import DecisionTreeDesignerWidget, TreeNode, LogicalOperator
from .llm_observation_generator import UltraLargeLanguageModel, Observation
from .logic_layer import LogicEngine


class MLStrategySystem:
    """
    Unified ML Strategy System integrating:
    - Bayesian Feature Extractors with layered classes
    - Decision Tree Designer with drag-and-drop interface
    - Ultra-Large Language Model for observation descriptions
    """
    
    def __init__(self):
        self.feature_extractor = BayesianFeatureExtractor()
        self.llm = UltraLargeLanguageModel()
        self.logic_engine = LogicEngine()
        
        # Connect vocabulary systems
        self._sync_vocabularies()
        
        # Decision tree will be created via UI
    
    def _sync_vocabularies(self):
        """Synchronize vocabularies between systems."""
        vocab = self.feature_extractor.get_vocabulary_universe()
        self.llm.add_vocabulary(vocab)
    
    def create_experimental_setup(
        self,
        layer_configs: List[Dict[str, Any]],
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an experimental setup with layers and initial data.
        
        Args:
            layer_configs: List of layer configurations, each with:
                - layer_id: int
                - layer_name: str
                - classes: List of class configs with name, attributes, parent_classes
            initial_data: Initial data for feature extraction
        
        Returns:
            Dictionary with setup information
        """
        # Create layers
        for layer_config in layer_configs:
            layer_id = layer_config["layer_id"]
            layer_name = layer_config["layer_name"]
            
            layer = self.feature_extractor.create_layer(layer_id, layer_name)
            
            # Add classes to layer
            for class_config in layer_config.get("classes", []):
                class_name = class_config["name"]
                attributes = []
                
                for attr_config in class_config.get("attributes", []):
                    attr = Attribute(
                        name=attr_config["name"],
                        attr_type=AttributeType(attr_config["type"]),
                        value=attr_config.get("value")
                    )
                    attributes.append(attr)
                
                parent_classes = class_config.get("parent_classes", [])
                
                self.feature_extractor.add_class_to_layer(
                    layer_id, class_name, attributes, parent_classes
                )
        
        # Sync vocabularies
        self._sync_vocabularies()
        
        # Extract initial features if data provided
        observations = []
        if initial_data:
            for layer_id in self.feature_extractor.layer_order:
                features = self.feature_extractor.extract_features(initial_data, layer_id)
                
                # Create observations
                for class_name, class_features in features.items():
                    obs = Observation(
                        observation_id=str(uuid.uuid4()),
                        timestamp=datetime.now().isoformat(),
                        layer_id=layer_id,
                        class_name=class_name,
                        attributes=class_features,
                        confidence_scores={
                            attr_name: attr_data.get("confidence", 0.0)
                            for attr_name, attr_data in class_features.items()
                        }
                    )
                    observations.append(obs)
                    self.llm.record_observation(obs)
        
        return {
            "num_layers": len(self.feature_extractor.layers),
            "num_observations": len(observations),
            "vocabulary_size": len(self.feature_extractor.vocabulary_universe)
        }
    
    def extract_and_observe(self, data: Dict[str, Any], layer_id: int) -> Observation:
        """Extract features and create an observation."""
        features = self.feature_extractor.extract_features(data, layer_id)
        
        # Get first class (or combine all)
        if not features:
            raise ValueError(f"No features extracted for layer {layer_id}")
        
        class_name = list(features.keys())[0]
        class_features = features[class_name]
        
        obs = Observation(
            observation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            layer_id=layer_id,
            class_name=class_name,
            attributes=class_features,
            confidence_scores={
                attr_name: attr_data.get("confidence", 0.0)
                for attr_name, attr_data in class_features.items()
            }
        )
        
        self.llm.record_observation(obs)
        return obs
    
    def generate_observation_report(self) -> str:
        """Generate a comprehensive report of all observations."""
        return self.llm.generate_pretrained_text()
    
    def export_decision_tree_to_fol(self, tree_widget: DecisionTreeDesignerWidget) -> Optional[str]:
        """Export decision tree to FOL formula."""
        return tree_widget.scene.to_fol_formula()
    
    def validate_decision_tree(self, tree_widget: DecisionTreeDesignerWidget) -> Dict[str, Any]:
        """Validate decision tree using the logic engine."""
        formula = self.export_decision_tree_to_fol(tree_widget)
        if not formula:
            return {"valid": False, "error": "No decision tree found"}
        
        try:
            # Parse and validate formula
            self.logic_engine.reset()
            success, error = self.logic_engine.add_formula(formula)
            
            if not success:
                return {"valid": False, "error": error}
            
            # Check satisfiability
            result = self.logic_engine.check_satisfiability()
            
            return {
                "valid": True,
                "formula": formula,
                "satisfiable": result.result.value if result.result else "unknown",
                "model": result.model.interpretation if result.model else None
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_learning_log(self) -> str:
        """Get the learning log from feature extractor."""
        return self.feature_extractor.get_log()
    
    def save_system_state(self, directory: Path):
        """Save the entire system state."""
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save feature extractor
        self.feature_extractor.save(directory / "feature_extractor.json")
        
        # Save LLM pretrained text
        self.llm.save_pretrained_text(directory / "pretrained_text.txt")
        
        # Save learning log
        log_file = directory / "learning_log.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(self.get_learning_log())
    
    def load_system_state(self, directory: Path):
        """Load the entire system state."""
        # Load feature extractor
        extractor_file = directory / "feature_extractor.json"
        if extractor_file.exists():
            self.feature_extractor.load(extractor_file)
            self._sync_vocabularies()
        
        # Note: LLM observations are not saved/loaded in this version
        # In a full implementation, you'd save observation_history as JSON
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        llm_stats = self.llm.get_corpus_statistics()
        
        return {
            "feature_extractor": {
                "num_layers": len(self.feature_extractor.layers),
                "layer_order": self.feature_extractor.layer_order,
                "vocabulary_size": len(self.feature_extractor.vocabulary_universe),
                "log_entries": len(self.feature_extractor.log_history)
            },
            "llm": llm_stats,
            "logic_engine": {
                "variables": len(self.logic_engine.variables),
                "functions": len(self.logic_engine.functions),
                "constraints": len(self.logic_engine.constraints)
            }
        }
