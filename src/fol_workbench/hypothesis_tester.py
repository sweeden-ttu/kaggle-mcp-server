"""
Hypothesis Testing System with Backtracking

Implements an interactive hypothesis testing system with:
- "Getting warmer" feedback loop
- Backtracking when hypotheses fail
- Alternative hypothesis generation
- Step-by-step refinement
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import copy
from collections import deque

from .logic_layer import LogicEngine, ValidationResult
from .test_first_simulator import UnitModel, TestFirstSimulator
from .design_feedback import DesignFeedbackAnalyzer, FeedbackItem


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    PENDING = "pending"
    TESTING = "testing"
    WARMER = "warmer"  # Getting closer
    COLDER = "colder"  # Getting further
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BACKTRACKED = "backtracked"


@dataclass
class Hypothesis:
    """A hypothesis to test."""
    id: str
    description: str
    model: UnitModel
    assumptions: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None  # For backtracking
    step_number: int = 0
    logical_evaluations: Dict[str, Any] = field(default_factory=dict)  # Store logical relation evaluations


@dataclass
class HypothesisStep:
    """A step in the hypothesis testing process."""
    step_number: int
    hypothesis: Hypothesis
    action: str
    result: Any
    feedback: Optional[str] = None
    user_response: Optional[str] = None  # "yes", "no", "warmer", "colder"


class HypothesisTester:
    """
    Interactive hypothesis testing system with backtracking.
    
    Features:
    - Test hypotheses step by step
    - Ask user "Are we getting warmer?"
    - Backtrack when user says "no"
    - Generate alternative hypotheses
    - Track hypothesis tree
    """
    
    def __init__(
        self,
        logic_engine: Optional[LogicEngine] = None,
        simulator: Optional[TestFirstSimulator] = None,
        feedback_analyzer: Optional[DesignFeedbackAnalyzer] = None
    ):
        """Initialize the hypothesis tester."""
        self.logic_engine = logic_engine or LogicEngine()
        self.simulator = simulator or TestFirstSimulator(self.logic_engine)
        self.feedback_analyzer = feedback_analyzer or DesignFeedbackAnalyzer(self.logic_engine)
        
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.hypothesis_history: List[HypothesisStep] = []
        self.current_hypothesis_id: Optional[str] = None
        self.hypothesis_tree: Dict[str, List[str]] = {}  # parent -> children
        
        # Callback for user interaction
        self.user_feedback_callback: Optional[Callable[[str], str]] = None
    
    def set_user_feedback_callback(self, callback: Callable[[str], str]):
        """Set callback for getting user feedback."""
        self.user_feedback_callback = callback
    
    def create_hypothesis(
        self,
        description: str,
        model: UnitModel,
        assumptions: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None
    ) -> Hypothesis:
        """Create a new hypothesis."""
        hypothesis_id = f"hyp_{len(self.hypotheses) + 1}"
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            description=description,
            model=model,
            assumptions=assumptions or {},
            parent_id=parent_id,
            step_number=len(self.hypothesis_history)
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        
        # Update tree
        if parent_id:
            if parent_id not in self.hypothesis_tree:
                self.hypothesis_tree[parent_id] = []
            self.hypothesis_tree[parent_id].append(hypothesis_id)
        
        return hypothesis
    
    def test_hypothesis(
        self,
        hypothesis_id: str,
        auto_continue: bool = False
    ) -> Tuple[HypothesisStatus, Optional[str]]:
        """
        Test a hypothesis and interact with user.
        
        Returns:
            (status, message) tuple
        """
        if hypothesis_id not in self.hypotheses:
            return HypothesisStatus.REJECTED, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        self.current_hypothesis_id = hypothesis_id
        hypothesis.status = HypothesisStatus.TESTING
        
        # Step 1: Run tests
        step = HypothesisStep(
            step_number=len(self.hypothesis_history) + 1,
            hypothesis=hypothesis,
            action="run_tests",
            result=None
        )
        
        test_results = self.simulator.run_tests(hypothesis.model.name)
        step.result = {
            "total": len(test_results),
            "passed": sum(1 for r in test_results if r.status.value == "passed"),
            "failed": sum(1 for r in test_results if r.status.value == "failed")
        }
        
        self.hypothesis_history.append(step)
        
        # Step 2: Get design feedback
        feedback = self.feedback_analyzer.analyze_model(hypothesis.model, self.simulator)
        step.feedback = f"Generated {len(feedback)} feedback items"
        
        # Step 3: Ask user if we're getting warmer
        if not auto_continue:
            user_response = self._ask_user_feedback(hypothesis, test_results, feedback)
            step.user_response = user_response
            
            if user_response.lower() in ["yes", "warmer", "y"]:
                hypothesis.status = HypothesisStatus.WARMER
                hypothesis.confidence += 0.1
                return HypothesisStatus.WARMER, "Hypothesis is getting warmer! Continuing..."
            elif user_response.lower() in ["no", "colder", "n"]:
                hypothesis.status = HypothesisStatus.COLDER
                return self._backtrack_and_propose_alternative(hypothesis)
            else:
                # Ambiguous response, ask again
                return HypothesisStatus.TESTING, "Please respond with 'yes' or 'no'"
        else:
            # Auto-continue mode
            if step.result["passed"] > step.result["failed"]:
                hypothesis.status = HypothesisStatus.WARMER
                hypothesis.confidence += 0.1
                return HypothesisStatus.WARMER, "Tests mostly passed, continuing..."
            else:
                return self._backtrack_and_propose_alternative(hypothesis)
    
    def _ask_user_feedback(
        self,
        hypothesis: Hypothesis,
        test_results: List,
        feedback: List[FeedbackItem]
    ) -> str:
        """Ask user if we're getting warmer."""
        if self.user_feedback_callback:
            message = self._format_feedback_message(hypothesis, test_results, feedback)
            return self.user_feedback_callback(message)
        else:
            # Default: print and return (for testing)
            message = self._format_feedback_message(hypothesis, test_results, feedback)
            print(message)
            return input("Are we getting warmer? (yes/no): ").strip()
    
    def _format_feedback_message(
        self,
        hypothesis: Hypothesis,
        test_results: List,
        feedback: List[FeedbackItem]
    ) -> str:
        """Format feedback message for user."""
        passed = sum(1 for r in test_results if r.status.value == "passed")
        total = len(test_results)
        
        lines = [
            f"\n{'='*70}",
            f"Hypothesis: {hypothesis.description}",
            f"Step: {hypothesis.step_number}",
            f"Confidence: {hypothesis.confidence:.1%}",
            f"\nTest Results: {passed}/{total} passed",
            f"Feedback Items: {len(feedback)}",
        ]
        
        # Show key feedback
        errors = [f for f in feedback if f.level.value == "error"]
        warnings = [f for f in feedback if f.level.value == "warning"]
        
        if errors:
            lines.append(f"\nErrors: {len(errors)}")
            for err in errors[:3]:  # Show first 3
                lines.append(f"  - {err.message}")
        
        if warnings:
            lines.append(f"\nWarnings: {len(warnings)}")
            for warn in warnings[:3]:  # Show first 3
                lines.append(f"  - {warn.message}")
        
        lines.append(f"\n{'='*70}")
        lines.append("Are we getting warmer? (yes/no): ")
        
        return "\n".join(lines)
    
    def _backtrack_and_propose_alternative(
        self,
        hypothesis: Hypothesis
    ) -> Tuple[HypothesisStatus, str]:
        """Backtrack and propose an alternative hypothesis."""
        hypothesis.status = HypothesisStatus.BACKTRACKED
        
        # Step backwards
        if hypothesis.parent_id:
            parent = self.hypotheses[hypothesis.parent_id]
            parent.status = HypothesisStatus.PENDING
            message = f"Backtracked to parent hypothesis: {parent.description}"
        else:
            message = "Backtracked to beginning - no parent hypothesis"
        
        # Generate alternative
        alternative = self._generate_alternative_hypothesis(hypothesis)
        if alternative:
            message += f"\nProposed alternative: {alternative.description}"
            alternative.status = HypothesisStatus.PENDING
        
        return HypothesisStatus.BACKTRACKED, message
    
    def _generate_alternative_hypothesis(
        self,
        original: Hypothesis
    ) -> Optional[Hypothesis]:
        """Generate an alternative hypothesis based on feedback."""
        # Get feedback for original
        feedback = self.feedback_analyzer.analyze_model(original.model, self.simulator)
        
        # Create modified model
        new_model = copy.deepcopy(original.model)
        
        # Apply simplifications based on feedback
        for item in feedback:
            if item.principle.value == "simplicity" and item.suggestion:
                # Try to simplify
                if "decompose" in item.suggestion.lower():
                    # This would require more sophisticated logic
                    pass
        
        # Create new hypothesis
        alternative = self.create_hypothesis(
            description=f"Alternative to: {original.description}",
            model=new_model,
            assumptions=original.assumptions.copy(),
            parent_id=original.id
        )
        
        return alternative
    
    def run_hypothesis_loop(
        self,
        initial_hypothesis: Hypothesis,
        max_steps: int = 10,
        target_confidence: float = 0.8
    ) -> Tuple[Hypothesis, List[HypothesisStep]]:
        """
        Run the complete hypothesis testing loop.
        
        Returns:
            (final_hypothesis, steps) tuple
        """
        steps = []
        current = initial_hypothesis
        
        for step_num in range(max_steps):
            status, message = self.test_hypothesis(current.id)
            steps.append(self.hypothesis_history[-1])
            
            print(f"\nStep {step_num + 1}: {message}")
            
            if status == HypothesisStatus.ACCEPTED:
                break
            
            if status == HypothesisStatus.BACKTRACKED:
                # Find next hypothesis to test
                if current.parent_id:
                    current = self.hypotheses[current.parent_id]
                else:
                    # Generate new root hypothesis
                    current = self._generate_alternative_hypothesis(current)
                    if not current:
                        break
            
            if current.confidence >= target_confidence:
                current.status = HypothesisStatus.ACCEPTED
                break
        
        return current, steps
    
    def export_hypothesis_tree(self, path: Path) -> Path:
        """Export hypothesis tree to JSON."""
        data = {
            "hypotheses": [
                {
                    "id": h.id,
                    "description": h.description,
                    "status": h.status.value,
                    "confidence": h.confidence,
                    "parent_id": h.parent_id,
                    "step_number": h.step_number
                }
                for h in self.hypotheses.values()
            ],
            "history": [
                {
                    "step_number": s.step_number,
                    "hypothesis_id": s.hypothesis.id,
                    "action": s.action,
                    "user_response": s.user_response
                }
                for s in self.hypothesis_history
            ],
            "tree": self.hypothesis_tree
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def visualize_hypothesis_tree(self) -> str:
        """Generate a text visualization of the hypothesis tree."""
        lines = ["Hypothesis Tree:\n"]
        
        # Find root hypotheses
        roots = [h for h in self.hypotheses.values() if h.parent_id is None]
        
        def print_node(hypothesis: Hypothesis, indent: int = 0):
            prefix = "  " * indent
            status_symbol = {
                "warmer": "↑",
                "colder": "↓",
                "accepted": "✓",
                "rejected": "✗",
                "backtracked": "←"
            }.get(hypothesis.status.value, "○")
            
            lines.append(
                f"{prefix}{status_symbol} [{hypothesis.id}] {hypothesis.description} "
                f"(confidence: {hypothesis.confidence:.1%})"
            )
            
            # Print children
            if hypothesis.id in self.hypothesis_tree:
                for child_id in self.hypothesis_tree[hypothesis.id]:
                    if child_id in self.hypotheses:
                        print_node(self.hypotheses[child_id], indent + 1)
        
        for root in roots:
            print_node(root)
        
        return "\n".join(lines)
    
    def get_premises_for_hypothesis(self, hypothesis_id: str) -> List[str]:
        """
        Get all premises (formulas) for a hypothesis from the predicate tree.
        
        Collects formulas from:
        1. Parent hypotheses (recursively)
        2. Assumptions of the hypothesis
        3. Model invariants and properties
        
        Args:
            hypothesis_id: ID of the hypothesis
            
        Returns:
            List of premise formula strings
        """
        if hypothesis_id not in self.hypotheses:
            return []
        
        hypothesis = self.hypotheses[hypothesis_id]
        premises = []
        
        # Add parent hypotheses' formulas recursively
        if hypothesis.parent_id:
            parent_premises = self.get_premises_for_hypothesis(hypothesis.parent_id)
            premises.extend(parent_premises)
        
        # Add parent's model formula if it exists
        if hypothesis.parent_id and hypothesis.parent_id in self.hypotheses:
            parent_hyp = self.hypotheses[hypothesis.parent_id]
            if parent_hyp.model.formula:
                premises.append(parent_hyp.model.formula)
        
        # Add assumptions as formulas
        for key, value in hypothesis.assumptions.items():
            if isinstance(value, str):
                premises.append(value)
            elif isinstance(value, dict) and "formula" in value:
                premises.append(value["formula"])
        
        # Add model invariants and properties
        premises.extend(hypothesis.model.invariants)
        premises.extend(hypothesis.model.properties)
        
        return premises
    
    def evaluate_turnstile(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊢ (turnstile): Syntactic derivability Γ ⊢ φ
        
        Checks if the hypothesis formula can be syntactically derived from its premises.
        This is done by checking if (premises AND NOT hypothesis) is unsatisfiable.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_derivable, proof_or_error_message)
        """
        if hypothesis_id not in self.hypotheses:
            return False, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        premises = self.get_premises_for_hypothesis(hypothesis_id)
        conclusion = hypothesis.model.formula
        
        if not conclusion:
            return False, "Hypothesis has no formula"
        
        # Create a temporary logic engine for this evaluation
        temp_engine = LogicEngine()
        
        # Add all premises
        for premise in premises:
            if premise:
                success, error = temp_engine.add_formula(premise)
                if not success:
                    return False, f"Failed to parse premise: {error}"
        
        # Check if (premises AND NOT conclusion) is unsatisfiable
        # If unsatisfiable, then premises ⊢ conclusion (syntactically derivable)
        conclusion_expr = temp_engine.parse_formula(conclusion)
        if conclusion_expr is None:
            return False, f"Failed to parse conclusion: {conclusion}"
        
        from z3 import Not
        temp_engine.add_constraint(Not(conclusion_expr))
        
        result = temp_engine.check_satisfiability()
        
        if result.result == ValidationResult.UNSATISFIABLE:
            return True, "Syntactically derivable: Γ ⊢ φ"
        elif result.result == ValidationResult.SATISFIABLE:
            return False, "Not syntactically derivable: counterexample exists"
        else:
            return False, f"Unknown: {result.error_message or 'Could not determine'}"
    
    def evaluate_not_derivable(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊬ (not derivable): No proof exists
        
        Checks if the hypothesis formula cannot be syntactically derived from premises.
        This is the negation of ⊢.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_not_derivable, message)
        """
        is_derivable, msg = self.evaluate_turnstile(hypothesis_id)
        return not is_derivable, f"⊬: {msg}" if not is_derivable else "Derivable (not ⊬)"
    
    def evaluate_reverse_turnstile(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊣ (reverse turnstile): Definitions
        
        Checks if the hypothesis represents a definition (bidirectional equivalence).
        A definition typically has the form: φ ⊣ ψ (φ is defined as ψ)
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_definition, message)
        """
        if hypothesis_id not in self.hypotheses:
            return False, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        formula = hypothesis.model.formula
        
        if not formula:
            return False, "Hypothesis has no formula"
        
        # Check if formula contains biconditional (Iff) or equality
        # Definitions are often biconditional statements
        if "Iff" in formula or "==" in formula or "↔" in formula:
            return True, "Appears to be a definition (contains biconditional)"
        
        # Check if hypothesis has a parent and they are equivalent
        if hypothesis.parent_id and hypothesis.parent_id in self.hypotheses:
            parent_hyp = self.hypotheses[hypothesis.parent_id]
            if parent_hyp.model.formula:
                # Check if they are equivalent
                temp_engine = LogicEngine()
                temp_engine.add_formula(f"Iff({formula}, {parent_hyp.model.formula})")
                result = temp_engine.check_satisfiability()
                if result.result == ValidationResult.SATISFIABLE:
                    return True, "Definitional equivalence with parent"
        
        return False, "Not identified as a definition"
    
    def evaluate_interderivable(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊣⊢ (interderivable): Syntactic equivalence
        
        Checks if the hypothesis is syntactically equivalent to its parent hypothesis.
        Two formulas are interderivable if each can be derived from the other.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_interderivable, message)
        """
        if hypothesis_id not in self.hypotheses:
            return False, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        if not hypothesis.parent_id or hypothesis.parent_id not in self.hypotheses:
            return False, "No parent hypothesis for comparison"
        
        parent_hyp = self.hypotheses[hypothesis.parent_id]
        
        if not hypothesis.model.formula or not parent_hyp.model.formula:
            return False, "Missing formula in hypothesis or parent"
        
        formula1 = hypothesis.model.formula
        formula2 = parent_hyp.model.formula
        
        # Check bidirectional derivability: (formula1 ⊢ formula2) AND (formula2 ⊢ formula1)
        temp_engine1 = LogicEngine()
        temp_engine1.add_formula(formula1)
        from z3 import Not
        temp_engine1.add_constraint(Not(temp_engine1.parse_formula(formula2)))
        result1 = temp_engine1.check_satisfiability()
        
        temp_engine2 = LogicEngine()
        temp_engine2.add_formula(formula2)
        temp_engine2.add_constraint(Not(temp_engine2.parse_formula(formula1)))
        result2 = temp_engine2.check_satisfiability()
        
        # Both directions must be unsatisfiable (both derivable)
        if (result1.result == ValidationResult.UNSATISFIABLE and 
            result2.result == ValidationResult.UNSATISFIABLE):
            return True, "Interderivable: ⊣⊢ (syntactically equivalent)"
        else:
            return False, "Not interderivable"
    
    def evaluate_semantic_entailment(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊨ (double turnstile): Semantic entailment Γ ⊨ φ
        
        Checks if the premises semantically entail the hypothesis formula.
        This is done using prove_implication which checks semantic validity.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_entailed, message)
        """
        if hypothesis_id not in self.hypotheses:
            return False, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        premises = self.get_premises_for_hypothesis(hypothesis_id)
        conclusion = hypothesis.model.formula
        
        if not conclusion:
            return False, "Hypothesis has no formula"
        
        if not premises:
            # No premises, check if conclusion is valid (always true)
            temp_engine = LogicEngine()
            from z3 import Not
            temp_engine.add_constraint(Not(temp_engine.parse_formula(conclusion)))
            result = temp_engine.check_satisfiability()
            if result.result == ValidationResult.UNSATISFIABLE:
                return True, "Semantically valid (no premises needed)"
            else:
                return False, "Not semantically valid"
        
        # Combine premises with And
        if len(premises) == 1:
            premise_formula = premises[0]
        else:
            premise_formula = f"And({', '.join(premises)})"
        
        # Use prove_implication to check semantic entailment
        temp_engine = LogicEngine()
        result = temp_engine.prove_implication(premise_formula, conclusion)
        
        if result.result == ValidationResult.UNSATISFIABLE:
            return True, "Semantically entailed: Γ ⊨ φ"
        elif result.result == ValidationResult.SATISFIABLE:
            return False, f"Not semantically entailed: countermodel exists"
        else:
            return False, f"Unknown: {result.error_message or 'Could not determine'}"
    
    def evaluate_does_not_entail(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊭ (does not entail): Countermodel exists
        
        Checks if the premises do NOT semantically entail the hypothesis.
        This is the negation of ⊨.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (does_not_entail, message)
        """
        is_entailed, msg = self.evaluate_semantic_entailment(hypothesis_id)
        return not is_entailed, f"⊭: {msg}" if not is_entailed else "Entailed (not ⊭)"
    
    def evaluate_forces(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊩ (forces): Forcing relation
        
        Forcing is stronger than semantic entailment. It means that in all models
        where the premises hold, the conclusion must hold, and this is provable.
        This combines both syntactic derivability and semantic validity.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_forced, message)
        """
        # Check both syntactic derivability and semantic entailment
        syntactically_derivable, syn_msg = self.evaluate_turnstile(hypothesis_id)
        semantically_entailed, sem_msg = self.evaluate_semantic_entailment(hypothesis_id)
        
        if syntactically_derivable and semantically_entailed:
            return True, "Forces: ⊩ (both syntactically derivable and semantically entailed)"
        else:
            reasons = []
            if not syntactically_derivable:
                reasons.append("not syntactically derivable")
            if not semantically_entailed:
                reasons.append("not semantically entailed")
            return False, f"Does not force: {', '.join(reasons)}"
    
    def evaluate_strong_validity(self, hypothesis_id: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate ⊪ (triple forces): Strong validity
        
        Strong validity means the formula is valid in all possible models and contexts,
        and this validity can be proven. This is stronger than forcing.
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Tuple of (is_strongly_valid, message)
        """
        if hypothesis_id not in self.hypotheses:
            return False, "Hypothesis not found"
        
        hypothesis = self.hypotheses[hypothesis_id]
        formula = hypothesis.model.formula
        
        if not formula:
            return False, "Hypothesis has no formula"
        
        # Check if formula is valid (true in all models)
        temp_engine = LogicEngine()
        from z3 import Not
        negated = Not(temp_engine.parse_formula(formula))
        temp_engine.add_constraint(negated)
        result = temp_engine.check_satisfiability()
        
        if result.result == ValidationResult.UNSATISFIABLE:
            # Formula is valid - check if it's also provable (syntactically derivable)
            syntactically_derivable, syn_msg = self.evaluate_turnstile(hypothesis_id)
            if syntactically_derivable:
                return True, "Strongly valid: ⊪ (valid and provable)"
            else:
                return False, "Valid but not provable (not strongly valid)"
        else:
            return False, "Not valid in all models"
    
    def evaluate_all_logical_relations(self, hypothesis_id: str) -> Dict[str, Any]:
        """
        Evaluate all logical relations for a hypothesis.
        
        Evaluates:
        - ⊢ (turnstile): Syntactic derivability
        - ⊬ (not derivable): No proof exists
        - ⊣ (reverse turnstile): Definitions
        - ⊣⊢ (interderivable): Syntactic equivalence
        - ⊨ (double turnstile): Semantic entailment
        - ⊭ (does not entail): Countermodel exists
        - ⊩ (forces): Forcing relation
        - ⊪ (triple forces): Strong validity
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            
        Returns:
            Dictionary mapping relation symbols to evaluation results
        """
        if hypothesis_id not in self.hypotheses:
            return {"error": "Hypothesis not found"}
        
        results = {}
        
        # Evaluate each relation
        results["⊢"] = self.evaluate_turnstile(hypothesis_id)
        results["⊬"] = self.evaluate_not_derivable(hypothesis_id)
        results["⊣"] = self.evaluate_reverse_turnstile(hypothesis_id)
        results["⊣⊢"] = self.evaluate_interderivable(hypothesis_id)
        results["⊨"] = self.evaluate_semantic_entailment(hypothesis_id)
        results["⊭"] = self.evaluate_does_not_entail(hypothesis_id)
        results["⊩"] = self.evaluate_forces(hypothesis_id)
        results["⊪"] = self.evaluate_strong_validity(hypothesis_id)
        
        # Store results in hypothesis
        self.hypotheses[hypothesis_id].logical_evaluations = results
        
        return results
    
    def evaluate_all_hypotheses(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all logical relations for all hypotheses in the tree.
        
        Returns:
            Dictionary mapping hypothesis IDs to their evaluation results
        """
        all_results = {}
        for hypothesis_id in self.hypotheses.keys():
            all_results[hypothesis_id] = self.evaluate_all_logical_relations(hypothesis_id)
        return all_results
    
    def get_evaluation_summary(self) -> str:
        """
        Get a formatted summary of all logical relation evaluations.
        
        Returns:
            Formatted string summary
        """
        lines = ["Logical Relation Evaluations for All Hypotheses:\n"]
        lines.append("=" * 80)
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            lines.append(f"\nHypothesis: {hypothesis.id} - {hypothesis.description}")
            lines.append(f"Formula: {hypothesis.model.formula}")
            lines.append("-" * 80)
            
            if hypothesis.logical_evaluations:
                for symbol, (result, message) in hypothesis.logical_evaluations.items():
                    status = "✓" if result else "✗"
                    lines.append(f"  {status} {symbol}: {message}")
            else:
                # Evaluate if not already done
                evaluations = self.evaluate_all_logical_relations(hypothesis_id)
                for symbol, (result, message) in evaluations.items():
                    status = "✓" if result else "✗"
                    lines.append(f"  {status} {symbol}: {message}")
            
            lines.append("")
        
        return "\n".join(lines)