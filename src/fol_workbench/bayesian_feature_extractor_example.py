"""
Example: Feeding Data into BayesianFeatureExtractor

This script demonstrates how to feed data into the BayesianFeatureExtractor
and use it for layered, constraint-aware feature extraction with Bayesian priors/posteriors.
"""

from pathlib import Path
from .bayesian_feature_extractor import (
    BayesianFeatureExtractor,
    Attribute,
    AttributeType
)


def main():
    """Main example demonstrating BayesianFeatureExtractor usage."""
    
    # Initialize the feature extractor
    print("=" * 60)
    print("BayesianFeatureExtractor - Data Feeding Example")
    print("=" * 60)
    
    extractor = BayesianFeatureExtractor()
    
    # ============================================================
    # Step 1: Create Layers
    # ============================================================
    print("\n[Step 1] Creating layers...")
    
    layer1 = extractor.create_layer(1, "Base Layer")
    layer2 = extractor.create_layer(2, "Feature Layer")
    layer3 = extractor.create_layer(3, "Advanced Layer")
    
    # ============================================================
    # Step 2: Add Classes to Layers with Attributes
    # ============================================================
    print("\n[Step 2] Adding classes to layers...")
    
    # Layer 1: Base classes
    extractor.add_class_to_layer(
        layer_id=1,
        class_name="Animal",
        attributes=[
            Attribute("species", AttributeType.CATEGORICAL, value="unknown"),
            Attribute("age", AttributeType.NUMERICAL, value=0),
            Attribute("is_mammal", AttributeType.BOOLEAN, value=False)
        ]
    )
    
    extractor.add_class_to_layer(
        layer_id=1,
        class_name="Plant",
        attributes=[
            Attribute("genus", AttributeType.CATEGORICAL, value="unknown"),
            Attribute("height", AttributeType.NUMERICAL, value=0.0),
            Attribute("is_flowering", AttributeType.BOOLEAN, value=False)
        ]
    )
    
    # Layer 2: Feature classes (inherit from Layer 1)
    extractor.add_class_to_layer(
        layer_id=2,
        class_name="Mammal",
        parent_classes=["Animal"],
        attributes=[
            Attribute("fur_color", AttributeType.CATEGORICAL, value="unknown"),
            Attribute("weight", AttributeType.NUMERICAL, value=0.0)
        ]
    )
    
    extractor.add_class_to_layer(
        layer_id=2,
        class_name="Tree",
        parent_classes=["Plant"],
        attributes=[
            Attribute("trunk_diameter", AttributeType.NUMERICAL, value=0.0),
            Attribute("leaf_type", AttributeType.CATEGORICAL, value="unknown")
        ]
    )
    
    # Layer 3: Advanced classes (inherit from Layer 2)
    extractor.add_class_to_layer(
        layer_id=3,
        class_name="Dog",
        parent_classes=["Mammal"],
        attributes=[
            Attribute("breed", AttributeType.CATEGORICAL, value="unknown"),
            Attribute("is_trained", AttributeType.BOOLEAN, value=False)
        ]
    )
    
    # ============================================================
    # Step 3: Assign Additional Attributes
    # ============================================================
    print("\n[Step 3] Assigning additional attributes...")
    
    extractor.assign_attribute(1, "Animal", "habitat", AttributeType.TEXT, value="unknown")
    extractor.assign_attribute(2, "Mammal", "diet", AttributeType.CATEGORICAL, value="omnivore")
    extractor.assign_attribute(3, "Dog", "owner_name", AttributeType.TEXT, value="unknown")
    
    # ============================================================
    # Step 4: Update Bayesian Priors
    # ============================================================
    print("\n[Step 4] Setting Bayesian priors...")
    
    # Prior for Animal species
    extractor.update_bayesian_prior(
        layer_id=1,
        class_name="Animal",
        prior={
            "species": 0.4,  # Probability of having a species attribute
            "age": 0.6,      # Probability of having age
            "is_mammal": 0.3  # Probability of being a mammal
        }
    )
    
    # Prior for Mammal fur colors
    extractor.update_bayesian_prior(
        layer_id=2,
        class_name="Mammal",
        prior={
            "fur_color_brown": 0.3,
            "fur_color_black": 0.2,
            "fur_color_white": 0.2,
            "fur_color_gray": 0.15,
            "fur_color_other": 0.15
        }
    )
    
    # Prior for Dog breeds
    extractor.update_bayesian_prior(
        layer_id=3,
        class_name="Dog",
        prior={
            "breed_labrador": 0.25,
            "breed_golden_retriever": 0.20,
            "breed_german_shepherd": 0.15,
            "breed_bulldog": 0.10,
            "breed_other": 0.30
        }
    )
    
    # ============================================================
    # Step 5: Feed Data and Extract Features
    # ============================================================
    print("\n[Step 5] Feeding data and extracting features...")
    
    # Feed data sample 1
    data1 = {
        "species": "Canis lupus",
        "age": 3,
        "is_mammal": True,
        "fur_color": "brown",
        "weight": 25.5,
        "breed": "labrador",
        "is_trained": True,
        "owner_name": "John"
    }
    
    print("\n--- Data Sample 1 ---")
    print(f"Input data: {data1}")
    
    features1 = extractor.extract_features(data1, layer_id=3)
    print("\nExtracted features (Layer 3):")
    for class_name, class_features in features1.items():
        print(f"  {class_name}:")
        for attr_name, attr_data in class_features.items():
            print(f"    {attr_name}: value={attr_data.get('value')}, "
                  f"confidence={attr_data.get('confidence', 0):.3f}")
            if 'posterior' in attr_data:
                print(f"      posterior: {attr_data['posterior']}")
    
    # Feed data sample 2
    data2 = {
        "species": "Felis catus",
        "age": 2,
        "is_mammal": True,
        "fur_color": "gray",
        "weight": 4.2
    }
    
    print("\n--- Data Sample 2 ---")
    print(f"Input data: {data2}")
    
    features2 = extractor.extract_features(data2, layer_id=2)
    print("\nExtracted features (Layer 2):")
    for class_name, class_features in features2.items():
        print(f"  {class_name}:")
        for attr_name, attr_data in class_features.items():
            print(f"    {attr_name}: value={attr_data.get('value')}, "
                  f"confidence={attr_data.get('confidence', 0):.3f}")
    
    # Feed data sample 3 (Plant data)
    data3 = {
        "genus": "Quercus",
        "height": 15.0,
        "is_flowering": False,
        "trunk_diameter": 0.8,
        "leaf_type": "deciduous"
    }
    
    print("\n--- Data Sample 3 (Plant) ---")
    print(f"Input data: {data3}")
    
    features3 = extractor.extract_features(data3, layer_id=2)
    print("\nExtracted features (Layer 2):")
    for class_name, class_features in features3.items():
        print(f"  {class_name}:")
        for attr_name, attr_data in class_features.items():
            print(f"    {attr_name}: value={attr_data.get('value')}, "
                  f"confidence={attr_data.get('confidence', 0):.3f}")
    
    # ============================================================
    # Step 6: Inspect Layers and Vocabulary
    # ============================================================
    print("\n[Step 6] Inspecting layers and vocabulary...")
    
    all_layers = extractor.get_all_layers()
    print(f"\nTotal layers: {len(all_layers)}")
    for layer in all_layers:
        print(f"  Layer {layer.layer_id}: {layer.layer_name} "
              f"({len(layer.classes)} classes, {len(layer.vocabulary)} vocabulary items)")
    
    vocab = extractor.get_vocabulary_universe()
    print(f"\nVocabulary universe size: {len(vocab)}")
    print(f"Sample vocabulary: {sorted(list(vocab))[:10]}")
    
    # ============================================================
    # Step 7: Persistence (Save and Load)
    # ============================================================
    print("\n[Step 7] Testing persistence...")
    
    save_path = Path("bayesian_extractor_example.json")
    extractor.save(save_path)
    print(f"Saved to: {save_path}")
    
    # Create a new extractor and load
    extractor2 = BayesianFeatureExtractor()
    extractor2.load(save_path)
    print(f"Loaded from: {save_path}")
    print(f"Loaded layers: {len(extractor2.get_all_layers())}")
    print(f"Loaded vocabulary size: {len(extractor2.get_vocabulary_universe())}")
    
    # ============================================================
    # Step 8: Learning Log
    # ============================================================
    print("\n[Step 8] Learning log:")
    print("-" * 60)
    log = extractor.get_log()
    print(log)
    print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
