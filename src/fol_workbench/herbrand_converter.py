"""
Herbrand Converter: Convert FOL formulas to propositional form.

This module converts First-Order Logic formulas to propositional logic
by generating Herbrand base (ground instances) or treating predicates as
Boolean variables. Used for K-map simplification of implications.
"""

import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .data_layer import DataLayer


@dataclass
class Implication:
    """Represents a logical implication."""
    premise: str
    conclusion: str
    checkpoint_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PropositionalFormula:
    """Represents a propositional formula with variable mapping."""
    formula: str  # Propositional formula string
    variables: List[str]  # List of Boolean variable names
    variable_map: Dict[str, str]  # Maps original predicates to Boolean vars


@dataclass
class TruthTable:
    """Represents a truth table."""
    variables: List[str]
    rows: List[Dict[str, bool]]  # Each row is a variable assignment
    output: List[bool]  # Output value for each row
    minterms: List[int]  # Indices of rows where output is True


class HerbrandConverter:
    """
    Convert FOL formulas to propositional form using Herbrand base.
    
    Supports two conversion methods:
    1. Ground instances: Generate all ground instances of predicates
    2. Boolean variables: Treat each predicate atom as a Boolean variable
    """
    
    def __init__(self, data_layer: DataLayer):
        """
        Initialize the converter.
        
        Args:
            data_layer: DataLayer instance for loading checkpoints
        """
        self.data_layer = data_layer
        self.predicate_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(')
        self.variable_pattern = re.compile(r'\b([a-z][a-z0-9_]*)\b')
    
    def extract_implications_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> List[Implication]:
        """
        Extract implications from a checkpoint.
        
        Looks for formulas of the form Implies(premise, conclusion)
        in the checkpoint's formulas and constraints.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            List of Implication objects found in the checkpoint
        """
        checkpoint_data = self.data_layer.load_checkpoint(checkpoint_id)
        if not checkpoint_data:
            return []
        
        implications = []
        
        # Check main formula
        formula = checkpoint_data.get("formula", "")
        if formula:
            impl = self._parse_implication(formula)
            if impl:
                impl.checkpoint_id = checkpoint_id
                impl.metadata = checkpoint_data.get("metadata", {})
                implications.append(impl)
        
        # Check constraints
        for constraint in checkpoint_data.get("constraints", []):
            impl = self._parse_implication(constraint)
            if impl:
                impl.checkpoint_id = checkpoint_id
                impl.metadata = checkpoint_data.get("metadata", {})
                implications.append(impl)
        
        # Check all_formulas
        for formula_str in checkpoint_data.get("all_formulas", []):
            impl = self._parse_implication(formula_str)
            if impl:
                impl.checkpoint_id = checkpoint_id
                impl.metadata = checkpoint_data.get("metadata", {})
                implications.append(impl)
        
        return implications
    
    def _parse_implication(self, formula: str) -> Optional[Implication]:
        """
        Parse an implication from a formula string.
        
        Looks for patterns like: Implies(premise, conclusion)
        
        Args:
            formula: Formula string to parse
            
        Returns:
            Implication object if found, None otherwise
        """
        # Remove whitespace for easier parsing
        formula_clean = formula.replace(" ", "")
        
        # Look for Implies(..., ...) - handle nested parentheses
        if formula_clean.startswith("Implies(") and formula_clean.endswith(")"):
            # Find the comma that separates premise and conclusion
            # by tracking parentheses depth
            depth = 0
            comma_pos = -1
            for i, char in enumerate(formula_clean[8:], start=8):  # Start after "Implies("
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth < 0:
                        break
                elif char == ',' and depth == 0:
                    comma_pos = i
                    break
            
            if comma_pos > 0:
                premise = formula_clean[8:comma_pos].strip()
                conclusion = formula_clean[comma_pos+1:-1].strip()  # -1 to remove closing )
                return Implication(premise=premise, conclusion=conclusion)
        
        # Also check for → or -> notation
        if '→' in formula or '->' in formula:
            parts = re.split(r'→|->', formula, 1)
            if len(parts) == 2:
                return Implication(
                    premise=parts[0].strip(),
                    conclusion=parts[1].strip()
                )
        
        return None
    
    def convert_to_ground_instances(
        self,
        formula: str,
        constants: Optional[List[str]] = None
    ) -> PropositionalFormula:
        """
        Convert FOL formula to propositional using ground instances (Herbrand base).
        
        Generates all ground instances by replacing variables with constants.
        For example: P(x) with constants [a, b] becomes P(a), P(b).
        
        Args:
            formula: FOL formula string
            constants: List of constants to use for grounding. If None, extracts from formula.
            
        Returns:
            PropositionalFormula with ground instances as Boolean variables
        """
        if constants is None:
            constants = self._extract_constants(formula)
        
        # Extract predicates and their variables
        predicates = self._extract_predicates(formula)
        
        # Generate all ground instances
        ground_instances = []
        variable_map = {}
        var_counter = 0
        
        for pred_name, args in predicates:
            if not args:  # Predicate with no arguments (propositional)
                var_name = f"P_{pred_name}"
                if var_name not in variable_map:
                    variable_map[var_name] = f"P_{pred_name}_{var_counter}"
                    var_counter += 1
                ground_instances.append((pred_name, [], variable_map[var_name]))
            else:
                # Generate ground instances for each constant combination
                for const in constants:
                    # Simple case: single variable replaced with constant
                    if len(args) == 1:
                        var_name = f"P_{pred_name}_{const}"
                        if var_name not in variable_map:
                            variable_map[var_name] = f"P_{var_counter}"
                            var_counter += 1
                        ground_instances.append((pred_name, [const], variable_map[var_name]))
        
        # Replace in formula
        prop_formula = formula
        for pred_name, args, var_name in ground_instances:
            if args:
                old_pattern = f"{pred_name}({','.join(args)})"
            else:
                old_pattern = pred_name
            prop_formula = prop_formula.replace(old_pattern, var_name)
        
        # Extract unique variables
        variables = sorted(set(variable_map.values()))
        
        return PropositionalFormula(
            formula=prop_formula,
            variables=variables,
            variable_map={v: k for k, v in variable_map.items()}
        )
    
    def convert_to_boolean_vars(self, formula: str) -> PropositionalFormula:
        """
        Convert FOL formula to propositional by treating predicates as Boolean variables.
        
        Each predicate atom (e.g., P(x), Q(a, b)) becomes a single Boolean variable.
        This is simpler but loses some structure.
        
        Args:
            formula: FOL formula string
            
        Returns:
            PropositionalFormula with predicates mapped to Boolean variables
        """
        predicates = self._extract_predicates(formula)
        
        variable_map = {}
        var_counter = 0
        prop_formula = formula
        
        for pred_name, args in predicates:
            # Create a unique variable name for this predicate instance
            if args:
                pred_key = f"{pred_name}({','.join(args)})"
            else:
                pred_key = pred_name
            
            if pred_key not in variable_map:
                var_name = f"B_{var_counter}"
                variable_map[pred_key] = var_name
                var_counter += 1
            
            # Replace in formula
            prop_formula = prop_formula.replace(pred_key, variable_map[pred_key])
        
        variables = sorted(set(variable_map.values()))
        
        return PropositionalFormula(
            formula=prop_formula,
            variables=variables,
            variable_map={v: k for k, v in variable_map.items()}
        )
    
    def _extract_constants(self, formula: str) -> List[str]:
        """
        Extract constants from a formula.
        
        Constants are typically lowercase identifiers that appear as arguments
        to predicates but are not variables (which are also lowercase).
        We use a heuristic: if it appears in a predicate argument position,
        it's likely a constant.
        
        Args:
            formula: Formula string
            
        Returns:
            List of constant names found
        """
        constants = set()
        
        # Find all predicate arguments
        for match in re.finditer(r'\(([^)]+)\)', formula):
            args_str = match.group(1)
            # Split by comma and check each argument
            for arg in args_str.split(','):
                arg = arg.strip()
                # If it's a simple identifier (not a complex expression), treat as constant
                if re.match(r'^[a-z][a-z0-9_]*$', arg):
                    constants.add(arg)
        
        return sorted(list(constants))
    
    def _extract_predicates(self, formula: str) -> List[Tuple[str, List[str]]]:
        """
        Extract predicates and their arguments from a formula.
        
        Args:
            formula: Formula string
            
        Returns:
            List of (predicate_name, arguments) tuples
        """
        predicates = []
        
        # Find all predicate patterns: P(...) or P
        for match in self.predicate_pattern.finditer(formula):
            pred_name = match.group(1)
            start_pos = match.end()
            
            # Find the arguments
            if start_pos < len(formula) and formula[start_pos] == '(':
                # Find matching closing parenthesis
                depth = 1
                end_pos = start_pos + 1
                while end_pos < len(formula) and depth > 0:
                    if formula[end_pos] == '(':
                        depth += 1
                    elif formula[end_pos] == ')':
                        depth -= 1
                    end_pos += 1
                
                args_str = formula[start_pos + 1:end_pos - 1]
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
            else:
                args = []
            
            predicates.append((pred_name, args))
        
        return predicates
    
    def generate_truth_table(
        self,
        propositional_formula: PropositionalFormula,
        max_vars: int = 4
    ) -> Optional[TruthTable]:
        """
        Generate a truth table for a propositional formula.
        
        Args:
            propositional_formula: PropositionalFormula to evaluate
            max_vars: Maximum number of variables to support (default 4 for K-map)
            
        Returns:
            TruthTable if successful, None if too many variables
        """
        variables = propositional_formula.variables
        
        if len(variables) > max_vars:
            return None  # Too many variables for K-map
        
        if len(variables) == 0:
            # Constant formula
            return TruthTable(
                variables=[],
                rows=[{}],
                output=[self._evaluate_propositional(propositional_formula.formula, {})],
                minterms=[0] if self._evaluate_propositional(propositional_formula.formula, {}) else []
            )
        
        # Generate all combinations of variable assignments
        num_rows = 2 ** len(variables)
        rows = []
        output = []
        minterms = []
        
        for i in range(num_rows):
            assignment = {}
            for j, var in enumerate(variables):
                assignment[var] = bool((i >> j) & 1)
            
            rows.append(assignment)
            result = self._evaluate_propositional(propositional_formula.formula, assignment)
            output.append(result)
            
            if result:
                minterms.append(i)
        
        return TruthTable(
            variables=variables,
            rows=rows,
            output=output,
            minterms=minterms
        )
    
    def _evaluate_propositional(self, formula: str, assignment: Dict[str, bool]) -> bool:
        """
        Evaluate a propositional formula with given variable assignments.
        
        Supports: And, Or, Not, Implies, Iff, True, False
        
        Args:
            formula: Propositional formula string
            assignment: Dictionary mapping variable names to boolean values
            
        Returns:
            Boolean result of evaluation
        """
        # Replace variables with their values
        eval_formula = formula
        for var, value in assignment.items():
            eval_formula = re.sub(rf'\b{var}\b', 'True' if value else 'False', eval_formula)
        
        # Replace logical operators with Python equivalents
        eval_formula = eval_formula.replace('And', 'and')
        eval_formula = eval_formula.replace('Or', 'or')
        eval_formula = eval_formula.replace('Not', 'not')
        eval_formula = eval_formula.replace('Implies', 'lambda a, b: not a or b')
        eval_formula = eval_formula.replace('Iff', 'lambda a, b: a == b')
        
        # Handle Implies and Iff (they need special handling)
        # For now, use a simple approach: replace with Python equivalents
        # This is a simplified evaluator - for production, use a proper parser
        
        try:
            # Use eval with safe context
            result = eval(eval_formula, {"__builtins__": {}}, {"True": True, "False": False})
            return bool(result)
        except:
            # Fallback: try to parse manually
            return self._evaluate_manual(formula, assignment)
    
    def _evaluate_manual(self, formula: str, assignment: Dict[str, bool]) -> bool:
        """
        Manual evaluation of propositional formula (fallback).
        
        Args:
            formula: Formula string
            assignment: Variable assignments
            
        Returns:
            Boolean result
        """
        # Simple recursive descent parser
        formula = formula.strip()
        
        # Handle parentheses
        if formula.startswith('(') and formula.endswith(')'):
            return self._evaluate_manual(formula[1:-1], assignment)
        
        # Handle Not
        if formula.startswith('Not('):
            inner = formula[4:-1]
            return not self._evaluate_manual(inner, assignment)
        
        # Handle And
        if 'And(' in formula:
            # Find matching And(...)
            # Simplified: assume two arguments
            parts = self._split_operator(formula, 'And')
            if len(parts) == 2:
                return (self._evaluate_manual(parts[0], assignment) and
                        self._evaluate_manual(parts[1], assignment))
        
        # Handle Or
        if 'Or(' in formula:
            parts = self._split_operator(formula, 'Or')
            if len(parts) == 2:
                return (self._evaluate_manual(parts[0], assignment) or
                        self._evaluate_manual(parts[1], assignment))
        
        # Handle Implies
        if 'Implies(' in formula:
            parts = self._split_operator(formula, 'Implies')
            if len(parts) == 2:
                return (not self._evaluate_manual(parts[0], assignment) or
                        self._evaluate_manual(parts[1], assignment))
        
        # Handle variable or constant
        if formula in assignment:
            return assignment[formula]
        if formula == 'True':
            return True
        if formula == 'False':
            return False
        
        return False
    
    def _split_operator(self, formula: str, op: str) -> List[str]:
        """
        Split formula by operator, handling nested parentheses.
        
        Args:
            formula: Formula string
            op: Operator name (e.g., 'And', 'Or')
            
        Returns:
            List of argument strings
        """
        # Find operator position
        op_pattern = f"{op}("
        if op_pattern not in formula:
            return [formula]
        
        start = formula.find(op_pattern) + len(op_pattern)
        
        # Find matching closing parenthesis
        depth = 1
        end = start
        while end < len(formula) and depth > 0:
            if formula[end] == '(':
                depth += 1
            elif formula[end] == ')':
                depth -= 1
            end += 1
        
        inner = formula[start:end-1]
        
        # Split by comma at top level
        parts = []
        depth = 0
        current = ""
        for char in inner:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def extract_minterms(self, truth_table: TruthTable) -> List[int]:
        """
        Extract minterms from a truth table.
        
        Args:
            truth_table: TruthTable to extract from
            
        Returns:
            List of minterm indices (rows where output is True)
        """
        return truth_table.minterms
