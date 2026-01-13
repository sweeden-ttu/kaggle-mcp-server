#!/usr/bin/env python3
"""Test script for K-map simplifier functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fol_workbench.data_layer import DataLayer
from fol_workbench.herbrand_converter import HerbrandConverter
from fol_workbench.kmap_simplifier import KMapSimplifier

def test_basic_conversion():
    """Test basic FOL to propositional conversion."""
    print("Testing basic conversion...")
    
    data_layer = DataLayer()
    converter = HerbrandConverter(data_layer)
    
    # Test implication parsing
    formula = "Implies(And(x, y), x)"
    impl = converter._parse_implication(formula)
    
    assert impl is not None, "Should parse implication"
    # Note: parsing removes spaces, so "And(x, y)" becomes "And(x,y)"
    assert "And" in impl.premise and "x" in impl.premise and "y" in impl.premise, "Premise should contain And(x,y)"
    assert impl.conclusion.strip() == "x", "Conclusion should be parsed correctly"
    print("✓ Implication parsing works")
    
    # Test Boolean variable conversion
    prop_formula = converter.convert_to_boolean_vars(formula)
    assert len(prop_formula.variables) > 0, "Should have variables"
    print(f"✓ Boolean variable conversion: {len(prop_formula.variables)} variables")
    
    # Test ground instances conversion
    prop_formula2 = converter.convert_to_ground_instances(formula, constants=["a", "b"])
    assert len(prop_formula2.variables) > 0, "Should have variables"
    print(f"✓ Ground instances conversion: {len(prop_formula2.variables)} variables")

def test_kmap_generation():
    """Test K-map generation."""
    print("\nTesting K-map generation...")
    
    simplifier = KMapSimplifier()
    
    # Test 2-variable K-map
    minterms = [0, 1, 3]  # A'B' + A'B + AB
    kmap = simplifier.create_kmap(minterms, 2, ["A", "B"])
    
    assert kmap is not None, "Should create 2-variable K-map"
    assert len(kmap.grid) == 2, "Should have 2 rows"
    assert len(kmap.grid[0]) == 2, "Should have 2 columns"
    print("✓ 2-variable K-map created")
    
    # Test 3-variable K-map
    minterms3 = [0, 1, 2, 5, 6, 7]
    kmap3 = simplifier.create_kmap(minterms3, 3, ["A", "B", "C"])
    
    assert kmap3 is not None, "Should create 3-variable K-map"
    assert len(kmap3.grid) == 2, "Should have 2 rows"
    assert len(kmap3.grid[0]) == 4, "Should have 4 columns"
    print("✓ 3-variable K-map created")
    
    # Test simplification
    simplified = simplifier.simplify(kmap)
    assert simplified is not None, "Should simplify K-map"
    assert simplified.sop, "Should have simplified expression"
    print(f"✓ K-map simplified: {simplified.sop}")

def test_checkpoint_integration():
    """Test integration with checkpoints."""
    print("\nTesting checkpoint integration...")
    
    data_layer = DataLayer()
    converter = HerbrandConverter(data_layer)
    
    # Create a test checkpoint with an implication
    checkpoint = data_layer.save_checkpoint(
        formula="Implies(And(x, y), x)",
        constraints=[],
        metadata={"test": True}
    )
    
    # Extract implications
    implications = converter.extract_implications_from_checkpoint(checkpoint.id)
    
    assert len(implications) > 0, "Should find implications"
    print(f"✓ Found {len(implications)} implication(s) in checkpoint")
    
    # Clean up
    data_layer.delete_checkpoint(checkpoint.id)
    print("✓ Test checkpoint cleaned up")

def main():
    """Run all tests."""
    print("=" * 50)
    print("K-map Simplifier Test Suite")
    print("=" * 50)
    
    try:
        test_basic_conversion()
        test_kmap_generation()
        test_checkpoint_integration()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
