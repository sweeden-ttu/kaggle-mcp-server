#!/usr/bin/env python3
"""
Kaggle PROLOG Brain Demo

Demonstrates the PROLOG-based brain system with attention tensors and perceptron connectives
optimized for Kaggle competitions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from prolog_brain import PrologBrain, AttentionTensor, PrologPerceptron
    from puzzle_detection.learning_pipeline import LearningPipeline
    PROLOG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PROLOG brain modules: {e}")
    PROLOG_AVAILABLE = False


def create_sample_kaggle_data():
    """Create sample Kaggle-style competition data."""
    np.random.seed(42)
    
    # Sample tabular competition data
    n_samples = 1000
    n_features = 20
    
    # Generate features
    features = np.random.randn(n_samples, n_features)
    
    # Generate target (binary classification)
    weights = np.random.randn(n_features)
    logits = features @ weights + np.random.randn(n_samples) * 0.1
    target = (logits > 0).astype(int)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_names)
    df['target'] = target
    
    return {
        'data': df,
        'features': feature_names,
        'target': 'target',
        'task_type': 'binary_classification',
        'competition_name': 'Sample_Tabular_Competition'
    }


def demo_prolog_brain():
    """Demonstrate PROLOG brain functionality."""
    if not PROLOG_AVAILABLE:
        print("PROLOG brain not available. Skipping demo.")
        return
    
    print("=== Kaggle PROLOG Brain Demo ===")
    
    # Create sample competition data
    competition_data = create_sample_kaggle_data()
    print(f"Created sample competition data with {len(competition_data['data'])} samples")
    print(f"Features: {len(competition_data['features'])} variables")
    
    # Initialize PROLOG brain
    brain = PrologBrain(swipl_path="swipl")
    print("PROLOG brain initialized")
    
    # Create attention tensor for tabular competition
    tensor = brain.create_attention_tensor(
        tensor_id="tabular_attention_tensor",
        competition_type="tabular",
        feature_names=competition_data['features']
    )
    print(f"Created attention tensor with shape {tensor.weights.shape}")
    
    # Create PROLOG-based perceptron
    perceptron = brain.create_prolog_perceptron(
        perceptron_id="tabular_perceptron_1",
        input_size=len(competition_data['features']),
        connective_rules=[
            "tabular_feature_selection(Features, Selected) :-",
            "    select_numeric_features(Features, Numeric),",
            "    normalize_features(Numeric, Selected)."
        ],
        activation_predicate="sigmoid_activation"
    )
    print(f"Created PROLOG perceptron with {perceptron.input_size} inputs")
    
    # Optimize for competition
    optimization_results = brain.optimize_for_kaggle_competition(
        competition_data=competition_data,
        competition_type="tabular"
    )
    print(f"Optimization completed: {len(optimization_results['perceptron_configs'])} perceptron configs generated")
    
    # Prepare training data
    df = competition_data['data']
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    training_data = [(X[i], float(y[i])) for i in range(min(100, len(X)))]
    
    # Train with PROLOG feedback
    training_results = brain.train_with_prolog_feedback(
        training_data=training_data,
        competition_type="tabular",
        perceptron_id="tabular_perceptron_1",
        epochs=10
    )
    
    print(f"Training completed:")
    print(f"  Final accuracy: {training_results['final_accuracy']:.3f}")
    print(f"  Final confidence: {training_results['final_confidence']:.3f}")
    print(f"  Attention tensor used: {training_results['attention_tensor_used']}")
    
    # Test prediction with attention
    test_features = X[0]
    prediction, confidence, attention_weights = brain.predict_with_attention(
        features=test_features,
        competition_type="tabular",
        perceptron_id="tabular_perceptron_1"
    )
    
    print(f"Prediction: {prediction:.3f}, Confidence: {confidence:.3f}")
    print(f"Top 3 attention weights: {sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Save brain state
    brain.save_brain_state('/tmp/kaggle_prolog_brain_state.json')
    print("Brain state saved to /tmp/kaggle_prolog_brain_state.json")
    
    print("=== Demo completed successfully! ===")


def demo_integration_with_learning_pipeline():
    """Demonstrate integration with existing learning pipeline."""
    if not PROLOG_AVAILABLE:
        print("PROLOG brain not available. Skipping integration demo.")
        return
    
    print("
=== Integration with Learning Pipeline Demo ===")
    
    # Create sample data
    competition_data = create_sample_kaggle_data()
    
    # Initialize learning pipeline with PROLOG brain
    pipeline = LearningPipeline(
        enable_prolog_integration=True
    )
    print("Learning pipeline with PROLOG brain integration initialized")
    
    # Optimize for Kaggle competition
    optimization_results = pipeline.optimize_for_kaggle_competition(
        competition_data=competition_data,
        competition_type="tabular"
    )
    
    print(f"Pipeline optimization results:")
    print(f"  Perceptrons created: {len(optimization_results['created_perceptrons'])}")
    print(f"  Attention tensors: {len(optimization_results['attention_tensors_created'])}")
    print(f"  Knowledge base size: {optimization_results['prolog_knowledge_base_size']} rules")
    
    # Prepare training data for pipeline
    df = competition_data['data']
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Convert to pipeline format (limited size for demo)
    from puzzle_detection.learning_pipeline import GridSquareFeatures
    training_data = [(X[i], y[i], None) for i in range(min(50, len(X)))]
    
    # Train with PROLOG attention
    training_results = pipeline.train_with_prolog_attention(
        training_data=training_data,
        competition_type="tabular",
        epochs=5
    )
    
    print(f"Pipeline training results:")
    print(f"  Perceptrons trained: {training_results['total_perceptrons_trained']}")
    print(f"  Attention mechanism: {training_results['attention_mechanism_used']}")
    
    print("=== Integration demo completed successfully! ===")


if __name__ == "__main__":
    print("Starting Kaggle PROLOG Brain Demo...")
    
    # Run basic PROLOG brain demo
    demo_prolog_brain()
    
    # Run integration demo
    demo_integration_with_learning_pipeline()
    
    print("
All demos completed! Check the generated files and results above.")
