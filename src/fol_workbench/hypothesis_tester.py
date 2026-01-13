"""
Hypothesis Testing System with Backtracking

Implements an interactive hypothesis testing system with:
- "Getting warmer" feedback loop
- Backtracking when hypotheses fail
- Alternative hypothesis generation
- Step-by-step refinement
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
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
