"""Example FOL formulas for testing the workbench."""

from .conversation_semantic_model import ConversationSemanticModel

# Create semantic model to get assigned values
_semantic_model = ConversationSemanticModel()
_semantic_formula = _semantic_model.to_formula()
_semantic_negated = _semantic_model.get_negated_formula()
_semantic_export = _semantic_model.export_model()

# Example formulas that can be used in the FOL Workbench
# Updated with assigned values from conversation_semantic_model

EXAMPLES = {
    "Simple Conjunction": "And(x, y)",
    
    "Disjunction": "Or(x, Not(y))",
    
    "Implication": "Implies(x, y)",
    
    "Biconditional": "Iff(x, y)",
    
    "Complex Formula": "And(Or(x, y), Implies(x, z))",
    
    "Negation": "Not(And(x, y))",
    
    "Contradiction": "And(x, Not(x))",
    
    "Tautology": "Or(x, Not(x))",
    
    "Three Variables": "And(x, Or(y, z))",
    
    "Nested Logic": "Implies(And(x, y), Or(z, Not(x)))",
    
    # Examples from conversation semantic model
    "Semantic Model Formula": _semantic_formula,
    
    "Semantic Model Negated": _semantic_negated,
    
    "Semantic Implication": "Implies(implements(reverse_simulation_system), has_test_first_simulator(reverse_simulation_system))",
    
    "Semantic Dependency": "Implies(depends_on(design_feedback, test_first_simulator), And(design_feedback, test_first_simulator))",
}

# Example constraints
EXAMPLE_CONSTRAINTS = {
    "Boolean Constraints": [
        "And(x, y)",
        "Or(Not(x), z)"
    ],
    
    "Simple": [
        "x",
        "y"
    ],
}

# SMT-LIB examples
SMT_LIB_EXAMPLES = {
    "Simple Boolean": """(set-logic QF_UF)
(declare-fun x () Bool)
(declare-fun y () Bool)
(assert (and x y))
(check-sat)
(get-model)
""",
    
    "Implication": """(set-logic QF_UF)
(declare-fun x () Bool)
(declare-fun y () Bool)
(assert (=> x y))
(check-sat)
(get-model)
""",
}
