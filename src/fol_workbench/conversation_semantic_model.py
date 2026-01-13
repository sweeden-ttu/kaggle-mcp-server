"""
Semantic Model of Conversation using Hilbert Base Predicates

This module creates a semantic model of the conversation using only
Hilbert's fundamental logical predicates:
- Not (negation)
- And (conjunction)
- Or (disjunction)
- Implies (implication)

These are the minimal set needed to express all logical formulas.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from .logic_layer import LogicEngine, ValidationResult


@dataclass
class ConversationEntity:
    """Represents an entity in the conversation."""
    name: str
    type: str  # "component", "action", "feature", "system"
    properties: Dict[str, Any]


@dataclass
class SemanticRelation:
    """Represents a relation between entities using base predicates."""
    relation_type: str  # "implements", "uses", "creates", "depends_on"
    subject: str
    object: str


class ConversationSemanticModel:
    """
    Semantic model of the conversation using only Hilbert base predicates.
    
    Models the conversation as logical formulas using:
    - Not: Negation
    - And: Conjunction
    - Or: Disjunction
    - Implies: Implication
    """
    
    def __init__(self):
        """Initialize the semantic model."""
        self.entities: Dict[str, ConversationEntity] = {}
        self.relations: List[SemanticRelation] = []
        self.logic_engine = LogicEngine()
        
        # Define base predicates (Hilbert's minimal set)
        self.base_predicates = {
            "Not": lambda x: f"Not({x})",
            "And": lambda x, y: f"And({x}, {y})",
            "Or": lambda x, y: f"Or({x}, {y})",
            "Implies": lambda x, y: f"Implies({x}, {y})"
        }
        
        # Build the semantic model from conversation
        self._build_model()
    
    def _build_model(self):
        """Build semantic model from conversation content."""
        # Entities from conversation
        entities_data = [
            ("test_first_simulator", "component", {"purpose": "test-first development"}),
            ("design_feedback", "component", {"purpose": "intelligent feedback"}),
            ("hypothesis_tester", "component", {"purpose": "hypothesis testing"}),
            ("kaggle_notebook_generator", "component", {"purpose": "notebook generation"}),
            ("reverse_simulation_system", "system", {"purpose": "integration"}),
            ("mcp_server", "system", {"purpose": "API interface"}),
            ("getting_warmer_loop", "feature", {"purpose": "interactive feedback"}),
            ("backtracking", "feature", {"purpose": "hypothesis refinement"}),
            ("reverse_simulation", "feature", {"purpose": "input guessing"}),
            ("test_first_approach", "methodology", {"purpose": "TDD"}),
        ]
        
        for name, entity_type, properties in entities_data:
            self.entities[name] = ConversationEntity(name, entity_type, properties)
        
        # Relations using base predicates
        relations_data = [
            ("implements", "reverse_simulation_system", "test_first_simulator"),
            ("implements", "reverse_simulation_system", "design_feedback"),
            ("implements", "reverse_simulation_system", "hypothesis_tester"),
            ("implements", "reverse_simulation_system", "kaggle_notebook_generator"),
            ("uses", "hypothesis_tester", "getting_warmer_loop"),
            ("uses", "hypothesis_tester", "backtracking"),
            ("uses", "kaggle_notebook_generator", "reverse_simulation"),
            ("depends_on", "design_feedback", "test_first_simulator"),
            ("depends_on", "hypothesis_tester", "design_feedback"),
            ("depends_on", "mcp_server", "reverse_simulation_system"),
        ]
        
        for rel_type, subject, obj in relations_data:
            self.relations.append(SemanticRelation(rel_type, subject, obj))
    
    def to_formula(self, negate: bool = False) -> str:
        """
        Convert semantic model to FOL formula using only base predicates.
        
        Args:
            negate: If True, negate the entire formula
        
        Returns:
            FOL formula string using only Not, And, Or, Implies
        """
        # Build formula from relations
        # Each relation becomes an implication: relation(subject, object) -> property
        formulas = []
        
        for relation in self.relations:
            # Encode relation as: Implies(relation_type(subject), has_object(object))
            rel_pred = f"{relation.relation_type}({relation.subject})"
            obj_pred = f"has_{relation.object}({relation.subject})"
            formula = self.base_predicates["Implies"](rel_pred, obj_pred)
            formulas.append(formula)
        
        # Combine all formulas with And
        if not formulas:
            main_formula = "True"
        elif len(formulas) == 1:
            main_formula = formulas[0]
        else:
            # Build nested And: And(formula1, And(formula2, And(...)))
            main_formula = formulas[0]
            for f in formulas[1:]:
                main_formula = self.base_predicates["And"](main_formula, f)
        
        # Negate if requested
        if negate:
            main_formula = self.base_predicates["Not"](main_formula)
        
        return main_formula
    
    def get_negated_formula(self) -> str:
        """Get the negated version of the main formula."""
        return self.to_formula(negate=True)
    
    def validate_model(self) -> ValidationResult:
        """Validate the semantic model using the logic engine."""
        formula = self.to_formula()
        self.logic_engine.reset()
        self.logic_engine.add_formula_with_tracking(formula)
        result = self.logic_engine.check_satisfiability()
        return result.result
    
    def export_model(self) -> Dict[str, Any]:
        """Export the semantic model as a dictionary."""
        return {
            "entities": {
                name: {
                    "type": entity.type,
                    "properties": entity.properties
                }
                for name, entity in self.entities.items()
            },
            "relations": [
                {
                    "type": rel.relation_type,
                    "subject": rel.subject,
                    "object": rel.object
                }
                for rel in self.relations
            ],
            "formula": self.to_formula(),
            "negated_formula": self.get_negated_formula()
        }


def create_conversation_semantic_model() -> ConversationSemanticModel:
    """Create and return a semantic model of the conversation."""
    return ConversationSemanticModel()


if __name__ == "__main__":
    # Create model
    model = create_conversation_semantic_model()
    
    # Get formulas
    formula = model.to_formula()
    negated = model.get_negated_formula()
    
    print("Semantic Model Formula (using only Hilbert base predicates):")
    print(f"  {formula}")
    print("\nNegated Formula:")
    print(f"  {negated}")
    
    # Export
    export = model.export_model()
    print("\nExported Model:")
    import json
    print(json.dumps(export, indent=2))
