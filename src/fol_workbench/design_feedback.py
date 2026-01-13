"""
Intelligent Design Feedback System

This module provides intelligent feedback on model designs, analyzing:
- Key data topics and patterns
- Design principles and best practices
- Performance characteristics
- First-principles analysis
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import re
from collections import Counter, defaultdict

from .logic_layer import LogicEngine, ValidationResult
from .test_first_simulator import UnitModel, AutomataTheory, TestFirstSimulator


class FeedbackLevel(Enum):
    """Severity level of feedback."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUGGESTION = "suggestion"


class DesignPrinciple(Enum):
    """Design principles to check against."""
    SIMPLICITY = "simplicity"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


@dataclass
class FeedbackItem:
    """A single piece of design feedback."""
    level: FeedbackLevel
    principle: DesignPrinciple
    category: str
    message: str
    suggestion: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class TopicAnalysis:
    """Analysis of key data topics."""
    topics: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (topic1, relation, topic2)
    importance_scores: Dict[str, float] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)


class DesignFeedbackAnalyzer:
    """
    Analyzes models and provides intelligent design feedback.
    
    Uses first-principles analysis to identify:
    - Key data topics and their relationships
    - Design issues and improvements
    - Performance bottlenecks
    - Best practice violations
    """
    
    def __init__(self, logic_engine: Optional[LogicEngine] = None):
        """Initialize the feedback analyzer."""
        self.logic_engine = logic_engine or LogicEngine()
        self.feedback_history: List[FeedbackItem] = []
    
    def analyze_model(
        self,
        model: UnitModel,
        simulator: Optional[TestFirstSimulator] = None
    ) -> List[FeedbackItem]:
        """Analyze a unit model and provide feedback."""
        feedback = []
        
        # Analyze formula complexity
        feedback.extend(self._analyze_complexity(model))
        
        # Analyze variable usage
        feedback.extend(self._analyze_variables(model))
        
        # Analyze test coverage
        if simulator:
            feedback.extend(self._analyze_test_coverage(model, simulator))
        
        # Analyze properties and invariants
        feedback.extend(self._analyze_properties(model))
        
        # First-principles analysis
        feedback.extend(self._first_principles_analysis(model))
        
        self.feedback_history.extend(feedback)
        return feedback
    
    def analyze_automata(
        self,
        theory: AutomataTheory,
        simulator: Optional[TestFirstSimulator] = None
    ) -> List[FeedbackItem]:
        """Analyze an automata theory and provide feedback."""
        feedback = []
        
        # Analyze state coverage
        feedback.extend(self._analyze_states(theory))
        
        # Analyze transitions
        feedback.extend(self._analyze_transitions(theory))
        
        # Analyze completeness
        feedback.extend(self._analyze_completeness(theory))
        
        # Analyze test coverage
        if simulator:
            feedback.extend(self._analyze_automata_tests(theory, simulator))
        
        self.feedback_history.extend(feedback)
        return feedback
    
    def extract_topics(self, model: UnitModel) -> TopicAnalysis:
        """Extract key data topics from a model."""
        topics = []
        relationships = []
        importance = {}
        
        # Extract variables as topics
        for var_name, var_type in model.variables.items():
            topics.append(var_name)
            importance[var_name] = 1.0
        
        # Extract relationships from formula
        formula = model.formula
        relationships.extend(self._extract_relationships(formula))
        
        # Extract patterns
        patterns = self._extract_patterns(formula)
        
        return TopicAnalysis(
            topics=topics,
            relationships=relationships,
            importance_scores=importance,
            patterns=patterns
        )
    
    def propose_changes(
        self,
        model: UnitModel,
        feedback: List[FeedbackItem]
    ) -> List[Dict[str, Any]]:
        """Propose specific changes based on feedback."""
        proposals = []
        
        # Group feedback by principle
        by_principle = defaultdict(list)
        for item in feedback:
            by_principle[item.principle].append(item)
        
        # Generate proposals
        for principle, items in by_principle.items():
            if principle == DesignPrinciple.SIMPLICITY:
                proposals.extend(self._propose_simplifications(model, items))
            elif principle == DesignPrinciple.COMPLETENESS:
                proposals.extend(self._propose_completions(model, items))
            elif principle == DesignPrinciple.PERFORMANCE:
                proposals.extend(self._propose_performance_improvements(model, items))
        
        return proposals
    
    def _analyze_complexity(self, model: UnitModel) -> List[FeedbackItem]:
        """Analyze formula complexity."""
        feedback = []
        formula = model.formula
        
        # Count nesting depth
        depth = self._calculate_nesting_depth(formula)
        if depth > 5:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.WARNING,
                principle=DesignPrinciple.SIMPLICITY,
                category="complexity",
                message=f"High nesting depth ({depth}) may reduce readability",
                suggestion="Consider breaking into smaller sub-formulas",
                confidence=0.8
            ))
        
        # Count operators
        operator_count = len(re.findall(r'\b(And|Or|Not|Implies|Iff)\b', formula))
        if operator_count > 10:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.INFO,
                principle=DesignPrinciple.SIMPLICITY,
                category="complexity",
                message=f"Formula contains {operator_count} logical operators",
                suggestion="Consider decomposition if formula is hard to understand",
                confidence=0.6
            ))
        
        return feedback
    
    def _analyze_variables(self, model: UnitModel) -> List[FeedbackItem]:
        """Analyze variable usage."""
        feedback = []
        
        # Check for unused variables
        formula_vars = set(re.findall(r'\b([a-z][a-zA-Z0-9]*)\b', model.formula))
        declared_vars = set(model.variables.keys())
        unused = declared_vars - formula_vars
        
        if unused:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.WARNING,
                principle=DesignPrinciple.CONSISTENCY,
                category="variables",
                message=f"Unused variables: {', '.join(unused)}",
                suggestion="Remove unused variable declarations",
                evidence=list(unused),
                confidence=1.0
            ))
        
        # Check for undeclared variables
        undeclared = formula_vars - declared_vars
        if undeclared:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.ERROR,
                principle=DesignPrinciple.CORRECTNESS,
                category="variables",
                message=f"Undeclared variables in formula: {', '.join(undeclared)}",
                suggestion="Declare all variables before use",
                evidence=list(undeclared),
                confidence=1.0
            ))
        
        return feedback
    
    def _analyze_test_coverage(
        self,
        model: UnitModel,
        simulator: TestFirstSimulator
    ) -> List[FeedbackItem]:
        """Analyze test coverage."""
        feedback = []
        
        if not model.test_cases:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.WARNING,
                principle=DesignPrinciple.COMPLETENESS,
                category="testing",
                message="No test cases defined",
                suggestion="Add test cases to validate model behavior",
                confidence=1.0
            ))
            return feedback
        
        # Run tests and analyze results
        results = simulator.run_tests(model.name)
        passed = sum(1 for r in results if r.status.value == "passed")
        total = len(results)
        
        coverage = passed / total if total > 0 else 0.0
        
        if coverage < 0.8:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.WARNING,
                principle=DesignPrinciple.COMPLETENESS,
                category="testing",
                message=f"Low test pass rate: {coverage:.1%}",
                suggestion="Review failing tests and fix model or test cases",
                confidence=0.9
            ))
        
        return feedback
    
    def _analyze_properties(self, model: UnitModel) -> List[FeedbackItem]:
        """Analyze properties and invariants."""
        feedback = []
        
        if not model.properties:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.INFO,
                principle=DesignPrinciple.COMPLETENESS,
                category="properties",
                message="No properties defined",
                suggestion="Consider adding properties to specify expected behavior",
                confidence=0.7
            ))
        
        if not model.invariants:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.INFO,
                principle=DesignPrinciple.CORRECTNESS,
                category="invariants",
                message="No invariants defined",
                suggestion="Add invariants to ensure model correctness",
                confidence=0.7
            ))
        
        return feedback
    
    def _analyze_states(self, theory: AutomataTheory) -> List[FeedbackItem]:
        """Analyze automata states."""
        feedback = []
        
        if not theory.states:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.ERROR,
                principle=DesignPrinciple.COMPLETENESS,
                category="states",
                message="No states defined",
                suggestion="Define at least one state",
                confidence=1.0
            ))
        
        if not theory.initial_state:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.ERROR,
                principle=DesignPrinciple.COMPLETENESS,
                category="states",
                message="No initial state defined",
                suggestion="Define an initial state",
                confidence=1.0
            ))
        
        if theory.initial_state and theory.initial_state not in theory.states:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.ERROR,
                principle=DesignPrinciple.CORRECTNESS,
                category="states",
                message=f"Initial state '{theory.initial_state}' not in states set",
                suggestion="Add initial state to states set",
                confidence=1.0
            ))
        
        return feedback
    
    def _analyze_transitions(self, theory: AutomataTheory) -> List[FeedbackItem]:
        """Analyze automata transitions."""
        feedback = []
        
        # Check for unreachable states
        reachable = {theory.initial_state} if theory.initial_state else set()
        for from_state, _, to_state in theory.transitions:
            if from_state in reachable:
                reachable.add(to_state)
        
        unreachable = theory.states - reachable
        if unreachable:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.WARNING,
                principle=DesignPrinciple.COMPLETENESS,
                category="transitions",
                message=f"Unreachable states: {', '.join(unreachable)}",
                suggestion="Add transitions to reach these states or remove them",
                evidence=list(unreachable),
                confidence=0.9
            ))
        
        return feedback
    
    def _analyze_completeness(self, theory: AutomataTheory) -> List[FeedbackItem]:
        """Analyze automata completeness."""
        feedback = []
        
        # Check for missing transitions
        transition_map = defaultdict(set)
        for from_state, label, to_state in theory.transitions:
            transition_map[from_state].add(label)
        
        missing = []
        for state in theory.states:
            for symbol in theory.alphabet:
                if symbol not in transition_map[state]:
                    missing.append((state, symbol))
        
        if missing and len(missing) < len(theory.states) * len(theory.alphabet):
            feedback.append(FeedbackItem(
                level=FeedbackLevel.INFO,
                principle=DesignPrinciple.COMPLETENESS,
                category="transitions",
                message=f"{len(missing)} possible transitions not defined",
                suggestion="Consider if missing transitions are intentional (partial automaton)",
                confidence=0.7
            ))
        
        return feedback
    
    def _analyze_automata_tests(
        self,
        theory: AutomataTheory,
        simulator: TestFirstSimulator
    ) -> List[FeedbackItem]:
        """Analyze automata test coverage."""
        return self._analyze_test_coverage(
            UnitModel(name=theory.name, model_type=None, formula=""),  # Dummy model
            simulator
        )
    
    def _first_principles_analysis(self, model: UnitModel) -> List[FeedbackItem]:
        """Perform first-principles analysis."""
        feedback = []
        
        # Check for contradictions
        if "And(x, Not(x))" in model.formula or "And(Not(x), x)" in model.formula:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.ERROR,
                principle=DesignPrinciple.CORRECTNESS,
                category="logic",
                message="Formula contains explicit contradiction",
                suggestion="Remove contradictory terms",
                confidence=1.0
            ))
        
        # Check for tautologies
        if "Or(x, Not(x))" in model.formula:
            feedback.append(FeedbackItem(
                level=FeedbackLevel.INFO,
                principle=DesignPrinciple.SIMPLICITY,
                category="logic",
                message="Formula contains tautology (always true)",
                suggestion="Consider if tautology is necessary",
                confidence=0.8
            ))
        
        return feedback
    
    def _extract_relationships(self, formula: str) -> List[Tuple[str, str, str]]:
        """Extract relationships from formula."""
        relationships = []
        
        # Look for binary predicates
        pattern = r'(\w+)\((\w+),\s*(\w+)\)'
        matches = re.findall(pattern, formula)
        for pred, arg1, arg2 in matches:
            relationships.append((arg1, pred, arg2))
        
        return relationships
    
    def _extract_patterns(self, formula: str) -> List[str]:
        """Extract patterns from formula."""
        patterns = []
        
        # Check for common patterns
        if re.search(r'And\([^)]*,\s*And\(', formula):
            patterns.append("nested_conjunctions")
        if re.search(r'Or\([^)]*,\s*Or\(', formula):
            patterns.append("nested_disjunctions")
        if re.search(r'Implies\([^)]*,\s*Implies\(', formula):
            patterns.append("nested_implications")
        
        return patterns
    
    def _calculate_nesting_depth(self, formula: str) -> int:
        """Calculate maximum nesting depth."""
        depth = 0
        max_depth = 0
        
        for char in formula:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        
        return max_depth
    
    def _propose_simplifications(
        self,
        model: UnitModel,
        feedback_items: List[FeedbackItem]
    ) -> List[Dict[str, Any]]:
        """Propose simplifications."""
        proposals = []
        
        for item in feedback_items:
            if "complexity" in item.category:
                proposals.append({
                    "type": "simplification",
                    "description": "Break complex formula into smaller parts",
                    "action": "decompose_formula",
                    "priority": "medium",
                    "estimated_effort": "low"
                })
        
        return proposals
    
    def _propose_completions(
        self,
        model: UnitModel,
        feedback_items: List[FeedbackItem]
    ) -> List[Dict[str, Any]]:
        """Propose completions."""
        proposals = []
        
        for item in feedback_items:
            if "testing" in item.category:
                proposals.append({
                    "type": "completion",
                    "description": "Add test cases to improve coverage",
                    "action": "add_test_cases",
                    "priority": "high",
                    "estimated_effort": "medium"
                })
        
        return proposals
    
    def _propose_performance_improvements(
        self,
        model: UnitModel,
        feedback_items: List[FeedbackItem]
    ) -> List[Dict[str, Any]]:
        """Propose performance improvements."""
        proposals = []
        
        # Analyze formula for performance issues
        if len(model.formula) > 1000:
            proposals.append({
                "type": "performance",
                "description": "Large formula may impact solver performance",
                "action": "optimize_formula",
                "priority": "low",
                "estimated_effort": "high"
            })
        
        return proposals
    
    def generate_feedback_report(
        self,
        feedback: List[FeedbackItem],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a formatted feedback report."""
        report_lines = ["# Design Feedback Report\n"]
        
        # Group by level
        by_level = defaultdict(list)
        for item in feedback:
            by_level[item.level].append(item)
        
        for level in [FeedbackLevel.ERROR, FeedbackLevel.WARNING, FeedbackLevel.INFO, FeedbackLevel.SUGGESTION]:
            if level in by_level:
                report_lines.append(f"\n## {level.value.upper()}\n")
                for item in by_level[level]:
                    report_lines.append(f"### {item.category} ({item.principle.value})")
                    report_lines.append(f"- {item.message}")
                    if item.suggestion:
                        report_lines.append(f"  - Suggestion: {item.suggestion}")
                    if item.evidence:
                        report_lines.append(f"  - Evidence: {', '.join(item.evidence)}")
                    report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path.write_text(report)
        
        return report
