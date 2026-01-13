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
