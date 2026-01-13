"""
PROLOG Brain Core Module

A comprehensive PROLOG-based reasoning system with attention tensors and perceptron connectives
optimized for Kaggle competitions. This module integrates first-order logic with neural
network concepts using PROLOG syntax and semantics.

Key Features:
- Attention tensor mechanisms for competition-specific feature weighting
- Perceptron connectives implemented as PROLOG predicates
- Kaggle competition optimization strategies
- Integration with existing learning pipeline
- Tensor operations for attention weights
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import subprocess
import tempfile
import re


@dataclass
class AttentionTensor:
    """Attention tensor for competition-specific feature weighting."""
    tensor_id: str
    weights: np.ndarray
    competition_type: str
    feature_names: List[str]
    attention_heads: int = 8
    dropout_rate: float = 0.1
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tensor_id': self.tensor_id,
            'weights': self.weights.tolist(),
            'competition_type': self.competition_type,
            'feature_names': self.feature_names,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
            'created': self.created
        }


@dataclass
class PrologPerceptron:
    """PROLOG-based perceptron with connectives."""
    perceptron_id: str
    input_size: int
    weights: np.ndarray
    bias: float
    connective_rules: List[str]
    activation_predicate: str = "sigmoid_activation"
    confidence: float = 0.0
    accuracy: float = 0.0
    
    def to_prolog_facts(self) -> List[str]:
        """Convert perceptron to PROLOG facts."""
        facts = []
        facts.append(f"perceptron('{self.perceptron_id}', {self.input_size}, {self.confidence}, {self.accuracy}).")
        
        for i, weight in enumerate(self.weights):
            facts.append(f"weight('{self.perceptron_id}', {i}, {weight}).")
        
        facts.append(f"bias('{self.perceptron_id}', {self.bias}).")
        
        for rule in self.connective_rules:
            facts.append(f"connective_rule('{self.perceptron_id}', '{rule}').")
        
        facts.append(f"activation_predicate('{self.perceptron_id}', '{self.activation_predicate}').")
        
        return facts
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict using PROLOG-based perceptron.
        
        Args:
            features: Input feature vector
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Calculate weighted sum
        weighted_sum = np.dot(self.weights, features) + self.bias
        
        # Apply activation function based on predicate
        if self.activation_predicate == "sigmoid_activation":
            output = 1 / (1 + np.exp(-weighted_sum))
            prediction = output > 0.5
        elif self.activation_predicate == "relu_activation":
            output = max(0, weighted_sum)
            prediction = output > 0
        elif self.activation_predicate == "tanh_activation":
            output = np.tanh(weighted_sum)
            prediction = output > 0
        else:
            # Default to step function
            output = 1 if weighted_sum > 0 else 0
            prediction = output == 1
        
        # Confidence based on distance from decision boundary
        confidence = min(1.0, abs(weighted_sum) / (np.linalg.norm(self.weights) + 1e-6))
        
        return prediction, confidence


class PrologBrain:
    """
    PROLOG-based brain system with attention tensors for Kaggle competitions.
    
    This system combines first-order logic reasoning with neural network concepts
    using PROLOG syntax, optimized for Kaggle competition-specific features.
    """
    
    def __init__(self, swipl_path: str = "swipl"):
        """
        Initialize PROLOG brain.
        
        Args:
            swipl_path: Path to SWI-Prolog executable
        """
        self.swipl_path = swipl_path
        self.attention_tensors: Dict[str, AttentionTensor] = {}
        self.prolog_perceptrons: Dict[str, PrologPerceptron] = {}
        self.competition_knowledge: Dict[str, Any] = {}
        self.reasoning_engine = PrologReasoningEngine(swipl_path)
        self.tensor_operations = TensorOperations()
        
        # Initialize core PROLOG knowledge base
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize core PROLOG knowledge base with competition rules."""
        core_rules = [
            # Basic perceptron activation rules
            "activation_output(PerceptronId, Input, Output) :-",
            "    perceptron(PerceptronId, Size, Confidence, Accuracy),",
            "    weighted_sum(PerceptronId, Input, Sum),",
            "    bias(PerceptronId, Bias),",
            "    Total is Sum + Bias,",
            "    sigmoid_activation(Total, Output).",
            "",
            "weighted_sum(PerceptronId, Input, Sum) :-",
            "    findall(Weight * Value,",
            "           (weight(PerceptronId, Index, Weight),",
            "            nth0(Index, Input, Value)),",
            "           Products),",
            "    sum_list(Products, Sum).",
            "",
            "sigmoid_activation(X, Output) :-",
            "    Output is 1 / (1 + exp(-X)).",
            "",
            # Attention tensor rules
            "attention_weight(TensorId, FeatureName, Weight) :-",
            "    attention_tensor(TensorId, Weights, CompetitionType, FeatureNames),",
            "    nth0(Index, FeatureNames, FeatureName),",
            "    nth0(Index, Weights, Weight).",
            "",
            # Competition optimization rules
            "optimize_for_competition(CompetitionType, Strategy) :-",
            "    competition_type(CompetitionType, Features, Algorithms),",
            "    best_strategy(Features, Algorithms, Strategy).",
            "",
            "best_strategy(Featureures, Algorithms, Strategy) :-",
            "    find_feature_weights(Features, Weights),",
            "    select_algorithms(Algorithms, Weights, Strategy)."
        ]
        
        self.reasoning_engine.load_rules(core_rules)
    
    def create_attention_tensor(
        self,
        tensor_id: str,
        competition_type: str,
        feature_names: List[str],
        initial_weights: Optional[np.ndarray] = None
    ) -> AttentionTensor:
        """
        Create attention tensor for specific competition type.
        
        Args:
            tensor_id: Unique identifier for the tensor
            competition_type: Type of Kaggle competition
            feature_names: Names of features to attend to
            initial_weights: Optional initial weights
            
        Returns:
            AttentionTensor instance
        """
        if initial_weights is None:
            weights = np.random.normal(0, 0.1, len(feature_names))
        else:
            weights = initial_weights.copy()
        
        tensor = AttentionTensor(
            tensor_id=tensor_id,
            weights=weights,
            competition_type=competition_type,
            feature_names=feature_names
        )
        
        self.attention_tensors[tensor_id] = tensor
        
        # Add to PROLOG knowledge base
        prolog_facts = [
            f"attention_tensor('{tensor_id}', {weights.tolist()}, '{competition_type}', {feature_names}).",
            f"attention_heads('{tensor_id}', {tensor.attention_heads}).",
            f"dropout_rate('{tensor_id}', {tensor.dropout_rate})."
        ]
        
        self.reasoning_engine.load_facts(prolog_facts)
        return tensor
    
    def create_prolog_perceptron(
        self,
        perceptron_id: str,
        input_size: int,
        connective_rules: List[str],
        initial_weights: Optional[np.ndarray] = None,
        initial_bias: float = 0.0,
        activation_predicate: str = "sigmoid_activation"
    ) -> PrologPerceptron:
        """
        Create PROLOG-based perceptron with connectives.
        
        Args:
            perceptron_id: Unique identifier
            input_size: Size of input vector
            connective_rules: PROLOG rules for connectives
            initial_weights: Optional initial weights
            initial_bias: Initial bias value
            activation_predicate: Activation function name (default: sigmoid_activation)
            
        Returns:
            PrologPerceptron instance
        """
        if initial_weights is None:
            weights = np.random.uniform(-0.1, 0.1, input_size)
        else:
            weights = initial_weights.copy()
        
        perceptron = PrologPerceptron(
            perceptron_id=perceptron_id,
            input_size=input_size,
            weights=weights,
            bias=initial_bias,
            connective_rules=connective_rules,
            activation_predicate=activation_predicate
        )
        
        self.prolog_perceptrons[perceptron_id] = perceptron
        
        # Add to PROLOG knowledge base
        facts = perceptron.to_prolog_facts()
        self.reasoning_engine.load_facts(facts)
        
        return perceptron
    
    def predict_with_attention(
        self,
        features: np.ndarray,
        competition_type: str,
        perceptron_id: str
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Make prediction using attention tensor for feature weighting.
        
        Args:
            features: Input feature vector
            competition_type: Type of competition
            perceptron_id: ID of perceptron to use
            
        Returns:
            Tuple of (prediction, confidence, attention_weights)
        """
        if perceptron_id not in self.prolog_perceptrons:
            raise ValueError(f"Perceptron {perceptron_id} not found")
        
        # Find appropriate attention tensor
        tensor = None
        for t in self.attention_tensors.values():
            if t.competition_type == competition_type:
                tensor = t
                break
        
        if tensor is None:
            # No attention tensor for this competition type, use default prediction
            perceptron = self.prolog_perceptrons[perceptron_id]
            prediction, confidence = perceptron.predict(features)
            attention_weights = {f"feature_{i}": 1.0 for i in range(len(features))}
            return float(prediction), confidence, attention_weights
        
        # Apply attention weights to features
        weighted_features = features * tensor.weights
        
        # Use PROLOG reasoning for prediction
        query = f"activation_output('{perceptron_id}', {weighted_features.tolist()}, Output)."
        result = self.reasoning_engine.query(query)
        
        if result and result[0].get('Output') is not None:
            prediction = float(result[0]['Output'])
            
            # Calculate confidence based on distance from decision boundary
            perceptron = self.prolog_perceptrons[perceptron_id]
            weighted_sum = np.dot(perceptron.weights, weighted_features) + perceptron.bias
            confidence = min(1.0, abs(weighted_sum) / (np.linalg.norm(perceptron.weights) + 1e-6))
            
            attention_weights = dict(zip(tensor.feature_names, tensor.weights))
            
            return prediction, confidence, attention_weights
        else:
            # Fallback to regular perceptron prediction
            perceptron = self.prolog_perceptrons[perceptron_id]
            prediction, confidence = perceptron.predict(features)
            attention_weights = dict(zip(tensor.feature_names, tensor.weights))
            return float(prediction), confidence, attention_weights
    
    def optimize_for_kaggle_competition(
        self,
        competition_data: Dict[str, Any],
        competition_type: str
    ) -> Dict[str, Any]:
        """
        Optimize brain configuration for specific Kaggle competition.
        
        Args:
            competition_data: Competition dataset and metadata
            competition_type: Type of competition (tabular, nlp, cv, etc.)
            
        Returns:
            Optimization results with recommended configuration
        """
        # Extract competition features
        features = self._extract_competition_features(competition_data)
        
        # Create or update attention tensor
        tensor_id = f"attention_{competition_type}"
        if tensor_id not in self.attention_tensors:
            self.create_attention_tensor(
                tensor_id=tensor_id,
                competition_type=competition_type,
                feature_names=features['feature_names']
            )
        
        # Query PROLOG for optimization strategy
        query = f"optimize_for_competition('{competition_type}', Strategy)."
        strategy_result = self.reasoning_engine.query(query)
        
        # Generate competition-specific perceptrons
        perceptron_configs = self._generate_competition_perceptrons(competition_type, features)
        
        optimization_results = {
            'competition_type': competition_type,
            'features_extracted': features,
            'strategy': strategy_result[0] if strategy_result else None,
            'perceptron_configs': perceptron_configs,
            'attention_tensor_id': tensor_id,
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        return optimization_results
    
    def _extract_competition_features(self, competition_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from competition data."""
        features = {
            'feature_names': [],
            'data_types': [],
            'target_type': None,
            'sample_count': 0,
            'feature_count': 0,
            'missing_values': 0
        }
        
        if 'data' in competition_data:
            data = competition_data['data']
            if hasattr(data, 'columns'):
                features['feature_names'] = list(data.columns)
                features['feature_count'] = len(data.columns)
                features['sample_count'] = len(data)
                
                # Data type analysis
                for col in data.columns:
                    dtype = str(data[col].dtype)
                    if 'int' in dtype or 'float' in dtype:
                        features['data_types'].append('numeric')
                    elif 'object' in dtype or 'str' in dtype:
                        features['data_types'].append('categorical')
                    else:
                        features['data_types'].append('other')
                
                # Missing value analysis
                features['missing_values'] = int(data.isnull().sum().sum())
        
        if 'target' in competition_data:
            target = competition_data['target']
            if hasattr(target, 'dtype'):
                dtype = str(target.dtype)
                if 'int' in dtype or 'float' in dtype:
                    features['target_type'] = 'regression'
                else:
                    features['target_type'] = 'classification'
        
        return features
    
    def _generate_competition_perceptrons(
        self,
        competition_type: str,
        features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate perceptron configurations for specific competition type."""
        configs = []
        
        # Competition-specific configurations
        if competition_type == "tabular":
            configs.append({
                'perceptron_id': f"tabular_main_{len(features['feature_names'])}",
                'input_size': len(features['feature_names']),
                'connective_rules': [
                    "tabular_feature_selection(Features, Selected) :-",
                    "    select_numeric_features(Features, Numeric),",
                    "    handle_missing_values(Numeric, Clean),",
                    "    normalize_features(Clean, Selected)."
                ],
                'activation_predicate': 'relu_activation'
            })
        
        elif competition_type == "nlp":
            configs.append({
                'perceptron_id': f"nlp_text_processor",
                'input_size': 300,  # Word embeddings size
                'connective_rules': [
                    "text_feature_extraction(Text, Features) :-",
                    "    tokenize_text(Text, Tokens),",
                    "    extract_embeddings(Tokens, Embeddings),",
                    "    aggregate_embeddings(Embeddings, Features)."
                ],
                'activation_predicate': 'tanh_activation'
            })
        
        elif competition_type == "computer_vision":
            configs.append({
                'perceptron_id': f"cv_feature_extractor",
                'input_size': 512,  # CNN feature size
                'connective_rules': [
                    "image_feature_extraction(Image, Features) :-",
                    "    extract_cnn_features(Image, CNN_Features),",
                    "    apply_spatial_attention(CNN_Features, Attended),",
                    "    flatten_features(Attended, Features)."
                ],
                'activation_predicate': 'leaky_relu_activation'
            })
        
        return configs
    
    def train_with_prolog_feedback(
        self,
        training_data: List[Tuple[np.ndarray, float]],
        competition_type: str,
        perceptron_id: str,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Train perceptron with PROLOG-based feedback and reasoning.
        
        Args:
            training_data: List of (features, target) pairs
            competition_type: Type of competition
            perceptron_id: ID of perceptron to train
            epochs: Number of training epochs
            
        Returns:
            Training results with performance metrics
        """
        if perceptron_id not in self.prolog_perceptrons:
            raise ValueError(f"Perceptron {perceptron_id} not found")
        
        perceptron = self.prolog_perceptrons[perceptron_id]
        # Find attention tensor by competition type
        tensor = None
        for t in self.attention_tensors.values():
            if t.competition_type == competition_type:
                tensor = t
                break
        
        training_history = []
        total_loss = 0.0
        correct_predictions = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            
            for features, target in training_data:
                # Apply attention weighting if available
                if tensor is not None and len(tensor.weights) == len(features):
                    weighted_features = features * tensor.weights
                else:
                    weighted_features = features
                
                # Make prediction
                prediction, confidence = perceptron.predict(weighted_features)
                prediction_float = float(prediction)
                
                # Calculate error
                error = target - prediction_float
                epoch_loss += error ** 2
                
                # Check if prediction was correct (using threshold for classification)
                if abs(error) < 0.5:
                    epoch_correct += 1
                
                # Update weights with attention-weighted learning
                if tensor is not None and len(tensor.weights) == len(features):
                    learning_rate = 0.1 * np.mean(tensor.weights)  # Default learning rate
                else:
                    learning_rate = 0.1  # Default learning rate
                
                # Weight update with PROLOG reasoning feedback
                weight_update = learning_rate * error * weighted_features
                perceptron.weights += weight_update
                perceptron.bias += learning_rate * error
                
                # Update PROLOG knowledge base with training feedback
                feedback_query = f"training_feedback('{perceptron_id}', {prediction_float}, {target}, {abs(error) < 0.5})."
                self.reasoning_engine.load_facts([feedback_query])
            
            avg_epoch_loss = epoch_loss / len(training_data)
            epoch_accuracy = epoch_correct / len(training_data)
            
            training_history.append({
                'epoch': epoch,
                'loss': avg_epoch_loss,
                'accuracy': epoch_accuracy
            })
            
            total_loss += avg_epoch_loss
            correct_predictions += epoch_correct
        
        # Update perceptron performance metrics
        perceptron.accuracy = correct_predictions / (len(training_data) * epochs)
        perceptron.confidence = 1.0 - (total_loss / epochs)
        
        # Update PROLOG knowledge base with final performance
        performance_facts = [
            f"perceptron_performance('{perceptron_id}', {perceptron.accuracy}, {perceptron.confidence}).",
            f"training_completed('{perceptron_id}', '{datetime.now().isoformat()}')."
        ]
        self.reasoning_engine.load_facts(performance_facts)
        
        return {
            'perceptron_id': perceptron_id,
            'competition_type': competition_type,
            'epochs_trained': epochs,
            'final_accuracy': perceptron.accuracy,
            'final_confidence': perceptron.confidence,
            'training_history': training_history,
            'attention_tensor_used': tensor is not None
        }
    
    def save_brain_state(self, filepath: str) -> bool:
        """Save complete brain state to file."""
        try:
            brain_state = {
                'attention_tensors': {k: v.to_dict() for k, v in self.attention_tensors.items()},
                'prolog_perceptrons': {
                    k: {
                        'perceptron_id': v.perceptron_id,
                        'input_size': v.input_size,
                        'weights': v.weights.tolist(),
                        'bias': v.bias,
                        'connective_rules': v.connective_rules,
                        'activation_predicate': v.activation_predicate,
                        'confidence': v.confidence,
                        'accuracy': v.accuracy
                    } for k, v in self.prolog_perceptrons.items()
                },
                'competition_knowledge': self.competition_knowledge,
                'saved_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(brain_state, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving brain state: {e}")
            return False
    
    def load_brain_state(self, filepath: str) -> bool:
        """Load brain state from file."""
        try:
            with open(filepath, 'r') as f:
                brain_state = json.load(f)
            
            # Load attention tensors
            for tensor_id, tensor_data in brain_state['attention_tensors'].items():
                tensor = AttentionTensor(
                    tensor_id=tensor_id,
                    weights=np.array(tensor_data['weights']),
                    competition_type=tensor_data['competition_type'],
                    feature_names=tensor_data['feature_names'],
                    attention_heads=tensor_data['attention_heads'],
                    dropout_rate=tensor_data['dropout_rate'],
                    created=tensor_data['created']
                )
                self.attention_tensors[tensor_id] = tensor
            
            # Load perceptrons
            for perceptron_id, perceptron_data in brain_state['prolog_perceptrons'].items():
                perceptron = PrologPerceptron(
                    perceptron_id=perceptron_id,
                    input_size=perceptron_data['input_size'],
                    weights=np.array(perceptron_data['weights']),
                    bias=perceptron_data['bias'],
                    connective_rules=perceptron_data['connective_rules'],
                    activation_predicate=perceptron_data['activation_predicate'],
                    confidence=perceptron_data['confidence'],
                    accuracy=perceptron_data['accuracy']
                )
                self.prolog_perceptrons[perceptron_id] = perceptron
            
            # Load competition knowledge
            self.competition_knowledge = brain_state.get('competition_knowledge', {})
            
            return True
        except Exception as e:
            print(f"Error loading brain state: {e}")
            return False


class PrologReasoningEngine:
    """PROLOG reasoning engine for rule-based inference."""
    
    def __init__(self, swipl_path: str = "swipl"):
        """Initialize PROLOG reasoning engine."""
        self.swipl_path = swipl_path
        self.knowledge_base = []
        self.rules = []
        
        # Test if SWI-Prolog is available
        try:
            subprocess.run([swipl_path, "--version"], capture_output=True, check=True)
            self.swipl_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.swipl_available = False
            print("Warning: SWI-Prolog not found. Using fallback reasoning.")
    
    def load_rules(self, rules: List[str]):
        """Load PROLOG rules into knowledge base."""
        self.rules.extend(rules)
        self.knowledge_base.extend(rules)
    
    def load_facts(self, facts: List[str]):
        """Load PROLOG facts into knowledge base."""
        self.knowledge_base.extend(facts)
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute PROLOG query and return results.
        
        Args:
            query: PROLOG query string
            
        Returns:
            List of result dictionaries
        """
        if not self.swipl_available:
            return self._fallback_query(query)
        
        try:
            # Create temporary file with knowledge base
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
                # Write knowledge base
                for item in self.knowledge_base:
                    f.write(item + "\n")
                
                # Write query
                f.write(f"\n?- {query}.\n")
                temp_file = f.name
            
            # Execute PROLOG query
            result = subprocess.run(
                [self.swipl_path, "-q", "-s", temp_file, "-t", "halt"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse results
            return self._parse_prolog_output(result.stdout)
            
        except Exception as e:
            print(f"PROLOG query error: {e}")
            return self._fallback_query(query)
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_file)
            except:
                pass
    
    def _fallback_query(self, query: str) -> List[Dict[str, Any]]:
        """Fallback reasoning when PROLOG is not available."""
        # Simple pattern matching for common queries
        if "activation_output" in query:
            return [{'Output': 0.5}]  # Default activation
        elif "attention_weight" in query:
            return [{'Weight': 1.0}]  # Default attention
        elif "optimize_for_competition" in query:
            return [{'Strategy': 'default_strategy'}]
        else:
            return []
    
    def _parse_prolog_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse SWI-Prolog output into structured results."""
        results = []
        
        lines = output.strip().split('\n')
        for line in lines:
            if line.startswith('Output = ') or '=' in line:
                # Simple parsing for variable bindings
                parts = line.split('=')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = parts[1].strip().rstrip('.')
                    try:
                        # Try to convert to number
                        if '.' in var_value:
                            var_value = float(var_value)
                        else:
                            var_value = int(var_value)
                    except ValueError:
                        # Keep as string
                        pass
                    results.append({var_name: var_value})
        
        return results


class TensorOperations:
    """Tensor operations for attention mechanisms."""
    
    @staticmethod
    def softmax_attention(weights: np.ndarray) -> np.ndarray:
        """Apply softmax to attention weights."""
        exp_weights = np.exp(weights - np.max(weights))
        return exp_weights / np.sum(exp_weights)
    
    @staticmethod
    def multi_head_attention(
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        num_heads: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-head attention mechanism."""
        head_dim = query.shape[-1] // num_heads
        
        # Split into heads
        query_heads = query.reshape(-1, num_heads, head_dim)
        key_heads = key.reshape(-1, num_heads, head_dim)
        value_heads = value.reshape(-1, num_heads, head_dim)
        
        # Compute attention scores
        scores = np.matmul(query_heads, key_heads.transpose(0, 2, 1))
        scores = scores / np.sqrt(head_dim)
        
        # Apply softmax
        attention_weights = TensorOperations.softmax_attention(scores)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, value_heads)
        
        # Concatenate heads
        output = attended.reshape(-1, num_heads * head_dim)
        
        return output, attention_weights
    
    @staticmethod
    def attention_dropout(attention_weights: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """Apply dropout to attention weights."""
        if dropout_rate > 0:
            mask = np.random.binomial(1, 1 - dropout_rate, attention_weights.shape)
            return attention_weights * mask / (1 - dropout_rate)
        return attention_weights
    
    @staticmethod
    def compute_attention_scores(query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Compute scaled dot-product attention scores."""
        return np.dot(query, key.T) / np.sqrt(key.shape[-1])



# Integration with existing perceptron units
class PrologPerceptronAdapter:
    """Adapter to integrate PrologPerceptron with existing PerceptronUnit."""
    
    @staticmethod
    def adapt_to_existing_perceptron(prolog_perceptron: PrologPerceptron) -> 'PerceptronUnit':
        """Convert PrologPerceptron to existing PerceptronUnit format."""
        from .puzzle_detection.perceptron_units import PerceptronUnit
        
        unit = PerceptronUnit(
            input_size=prolog_perceptron.input_size,
            learning_rate=0.1,
            initial_weights=prolog_perceptron.weights,
            initial_bias=prolog_perceptron.bias
        )
        
        unit.accuracy = prolog_perceptron.accuracy
        unit.confidence = prolog_perceptron.confidence
        
        return unit
    
    @staticmethod
    def adapt_from_existing_perceptron(
        perceptron_unit: 'PerceptronUnit',
        perceptron_id: str,
        connective_rules: List[str]
    ) -> PrologPerceptron:
        """Convert existing PerceptronUnit to PrologPerceptron format."""
        state = perceptron_unit.get_state()
        
        return PrologPerceptron(
            perceptron_id=perceptron_id,
            input_size=perceptron_unit.input_size,
            weights=state.weights,
            bias=state.bias,
            connective_rules=connective_rules,
            confidence=perceptron_unit.confidence,
            accuracy=perceptron_unit.accuracy
        )

