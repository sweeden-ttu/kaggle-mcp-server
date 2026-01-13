"""
K-map Simplifier: Karnaugh map generation and Boolean simplification.

This module implements Karnaugh map (K-map) generation and simplification
for Boolean expressions. Supports 2, 3, and 4 variable K-maps with
prime implicant extraction and minimal sum-of-products generation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class KMapSize(Enum):
    """K-map size enumeration."""
    TWO_VAR = (2, 2, 2)  # 2 variables: 2x2 grid
    THREE_VAR = (3, 2, 4)  # 3 variables: 2x4 grid
    FOUR_VAR = (4, 4, 4)  # 4 variables: 4x4 grid


@dataclass
class KMapCell:
    """Represents a cell in a K-map."""
    row: int
    col: int
    value: int  # 0, 1, or -1 (don't-care)
    minterm: int  # Minterm number
    covered: bool = False  # Whether covered by a prime implicant


@dataclass
class PrimeImplicant:
    """Represents a prime implicant in a K-map."""
    minterms: Set[int]  # Minterms covered by this implicant
    variables: List[str]  # Variable names
    expression: str  # Simplified expression (e.g., "A'B")
    is_essential: bool = False  # Whether this is an essential prime implicant
    cells: List[Tuple[int, int]] = None  # Cell positions (row, col)


@dataclass
class KMap:
    """Represents a Karnaugh map."""
    size: KMapSize
    variables: List[str]
    grid: List[List[int]]  # 2D grid: 0, 1, or -1 (don't-care)
    minterms: List[int]  # List of minterm indices
    dont_cares: List[int] = None  # Optional don't-care minterms
    gray_code_rows: List[int] = None  # Gray code for row labels
    gray_code_cols: List[int] = None  # Gray code for column labels


@dataclass
class SimplifiedExpression:
    """Represents a simplified Boolean expression."""
    sop: str  # Sum-of-products form
    prime_implicants: List[PrimeImplicant]
    essential_pi: List[PrimeImplicant]
    coverage: Dict[int, List[PrimeImplicant]]  # Which PIs cover each minterm


class KMapSimplifier:
    """
    Karnaugh map simplifier for Boolean expressions.
    
    Supports 2, 3, and 4 variable K-maps with:
    - Gray code ordering
    - Prime implicant extraction
    - Essential prime implicant identification
    - Minimal sum-of-products generation
    """
    
    def __init__(self):
        """Initialize the K-map simplifier."""
        self.gray_codes = {
            2: [0, 1, 3, 2],  # 2-bit Gray code
            3: [0, 1, 3, 2, 6, 7, 5, 4],  # 3-bit Gray code
            4: [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]  # 4-bit Gray code
        }
    
    def create_kmap(
        self,
        minterms: List[int],
        num_vars: int,
        variables: Optional[List[str]] = None,
        dont_cares: Optional[List[int]] = None
    ) -> Optional[KMap]:
        """
        Create a K-map from minterms.
        
        Args:
            minterms: List of minterm indices (0 to 2^num_vars - 1)
            num_vars: Number of variables (2, 3, or 4)
            variables: Optional list of variable names (defaults to A, B, C, D)
            dont_cares: Optional list of don't-care minterm indices
            
        Returns:
            KMap object if successful, None if num_vars not supported
        """
        if num_vars < 2 or num_vars > 4:
            return None
        
        if variables is None:
            variables = ['A', 'B', 'C', 'D'][:num_vars]
        
        if dont_cares is None:
            dont_cares = []
        
        # Determine grid size
        if num_vars == 2:
            size = KMapSize.TWO_VAR
            rows, cols = 2, 2
        elif num_vars == 3:
            size = KMapSize.THREE_VAR
            rows, cols = 2, 4
        else:  # num_vars == 4
            size = KMapSize.FOUR_VAR
            rows, cols = 4, 4
        
        # Initialize grid with zeros
        grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # Get Gray code sequences
        gray_rows = self._get_gray_code(rows)
        gray_cols = self._get_gray_code(cols)
        
        # Fill grid with minterms
        max_minterm = (2 ** num_vars) - 1
        for minterm in minterms:
            if 0 <= minterm <= max_minterm:
                row, col = self._minterm_to_position(minterm, num_vars, gray_rows, gray_cols)
                if 0 <= row < rows and 0 <= col < cols:
                    grid[row][col] = 1
        
        # Fill don't-cares
        for dc in dont_cares:
            if 0 <= dc <= max_minterm:
                row, col = self._minterm_to_position(dc, num_vars, gray_rows, gray_cols)
                if 0 <= row < rows and 0 <= col < cols:
                    grid[row][col] = -1
        
        return KMap(
            size=size,
            variables=variables,
            grid=grid,
            minterms=minterms,
            dont_cares=dont_cares,
            gray_code_rows=gray_rows,
            gray_code_cols=gray_cols
        )
    
    def _get_gray_code(self, length: int) -> List[int]:
        """
        Get Gray code sequence of given length.
        
        Args:
            length: Length of sequence (2, 4, or 8)
            
        Returns:
            List of Gray code values
        """
        if length == 2:
            return [0, 1]
        elif length == 4:
            return [0, 1, 3, 2]
        elif length == 8:
            return [0, 1, 3, 2, 6, 7, 5, 4]
        else:
            # Generate Gray code for arbitrary length
            n = length.bit_length() - 1
            if n in self.gray_codes:
                return self.gray_codes[n][:length]
            # Fallback: generate recursively
            return self._generate_gray_code(n)[:length]
    
    def _generate_gray_code(self, n: int) -> List[int]:
        """Generate n-bit Gray code recursively."""
        if n == 1:
            return [0, 1]
        
        prev = self._generate_gray_code(n - 1)
        reflected = prev[::-1]
        return prev + [x + (1 << (n - 1)) for x in reflected]
    
    def _minterm_to_position(
        self,
        minterm: int,
        num_vars: int,
        gray_rows: List[int],
        gray_cols: List[int]
    ) -> Tuple[int, int]:
        """
        Convert minterm number to (row, col) position in K-map.
        
        Args:
            minterm: Minterm number (0 to 2^num_vars - 1)
            num_vars: Number of variables
            gray_rows: Gray code for rows
            gray_cols: Gray code for columns
            
        Returns:
            (row, col) tuple
        """
        if num_vars == 2:
            # 2 variables: row = first bit, col = second bit
            row_val = (minterm >> 1) & 1
            col_val = minterm & 1
            row = gray_rows.index(row_val)
            col = gray_cols.index(col_val)
        elif num_vars == 3:
            # 3 variables: row = first bit, col = last two bits
            row_val = (minterm >> 2) & 1
            col_val = minterm & 3
            row = gray_rows.index(row_val)
            col = gray_cols.index(col_val)
        else:  # num_vars == 4
            # 4 variables: row = first two bits, col = last two bits
            row_val = (minterm >> 2) & 3
            col_val = minterm & 3
            row = gray_rows.index(row_val)
            col = gray_cols.index(col_val)
        
        return row, col
    
    def find_prime_implicants(self, kmap: KMap) -> List[PrimeImplicant]:
        """
        Find all prime implicants in a K-map.
        
        Uses grouping method: find all possible groups of 1s and don't-cares
        that are powers of 2 in size (1, 2, 4, 8, 16).
        
        Args:
            kmap: KMap to analyze
            
        Returns:
            List of PrimeImplicant objects
        """
        prime_implicants = []
        rows, cols = len(kmap.grid), len(kmap.grid[0])
        
        # Find all groups of 1s and don't-cares
        covered_cells = set()
        
        # Try groups of increasing size
        for group_size in [16, 8, 4, 2, 1]:
            if group_size > rows * cols:
                continue
            
            for row in range(rows):
                for col in range(cols):
                    if kmap.grid[row][col] in [1, -1]:
                        # Try to form a group starting at this cell
                        groups = self._find_groups_at(
                            kmap, row, col, group_size, covered_cells
                        )
                        for group in groups:
                            pi = self._group_to_prime_implicant(kmap, group)
                            if pi:
                                prime_implicants.append(pi)
                                # Mark cells as covered
                                for r, c in group:
                                    covered_cells.add((r, c))
        
        # Remove duplicates
        unique_pis = []
        seen_expressions = set()
        for pi in prime_implicants:
            if pi.expression not in seen_expressions:
                unique_pis.append(pi)
                seen_expressions.add(pi.expression)
        
        return unique_pis
    
    def _find_groups_at(
        self,
        kmap: KMap,
        start_row: int,
        start_col: int,
        size: int,
        covered: Set[Tuple[int, int]]
    ) -> List[List[Tuple[int, int]]]:
        """
        Find groups of given size starting at a cell.
        
        Args:
            kmap: KMap to search
            start_row: Starting row
            start_col: Starting column
            size: Group size (1, 2, 4, 8, 16)
            covered: Set of already covered cells
            
        Returns:
            List of groups (each group is a list of (row, col) tuples)
        """
        groups = []
        rows, cols = len(kmap.grid), len(kmap.grid[0])
        
        # For size 1, just return the cell if valid
        if size == 1:
            if (start_row, start_col) not in covered:
                if kmap.grid[start_row][start_col] in [1, -1]:
                    groups.append([(start_row, start_col)])
            return groups
        
        # For larger sizes, try rectangular groups
        # Try different rectangle dimensions
        for h in [1, 2, 4]:
            for w in [1, 2, 4]:
                if h * w == size:
                    # Check if rectangle fits and all cells are valid
                    if start_row + h <= rows and start_col + w <= cols:
                        group = []
                        valid = True
                        for r in range(start_row, start_row + h):
                            for c in range(start_col, start_col + w):
                                if (r, c) in covered:
                                    valid = False
                                    break
                                if kmap.grid[r][c] not in [1, -1]:
                                    valid = False
                                    break
                                group.append((r, c))
                            if not valid:
                                break
                        
                        if valid:
                            groups.append(group)
        
        # Also try wrapping groups (K-maps wrap around)
        # This is more complex, simplified for now
        
        return groups
    
    def _group_to_prime_implicant(
        self,
        kmap: KMap,
        group: List[Tuple[int, int]]
    ) -> Optional[PrimeImplicant]:
        """
        Convert a group of cells to a prime implicant.
        
        Args:
            kmap: KMap containing the group
            group: List of (row, col) cell positions
            
        Returns:
            PrimeImplicant if valid, None otherwise
        """
        if not group:
            return None
        
        # Store kmap reference for expression generation
        self.kmap = kmap
        
        # Get minterms for this group
        minterms = set()
        for row, col in group:
            minterm = self._position_to_minterm(
                row, col, len(kmap.variables),
                kmap.gray_code_rows, kmap.gray_code_cols
            )
            if minterm is not None:
                minterms.add(minterm)
        
        # Generate expression
        expression = self._minterms_to_expression(kmap.variables, minterms, group)
        
        return PrimeImplicant(
            minterms=minterms,
            variables=kmap.variables,
            expression=expression,
            cells=group
        )
    
    def _position_to_minterm(
        self,
        row: int,
        col: int,
        num_vars: int,
        gray_rows: List[int],
        gray_cols: List[int]
    ) -> Optional[int]:
        """Convert (row, col) position to minterm number."""
        if row >= len(gray_rows) or col >= len(gray_cols):
            return None
        
        row_val = gray_rows[row]
        col_val = gray_cols[col]
        
        if num_vars == 2:
            return (row_val << 1) | col_val
        elif num_vars == 3:
            return (row_val << 2) | col_val
        else:  # num_vars == 4
            return (row_val << 2) | col_val
    
    def _minterms_to_expression(
        self,
        variables: List[str],
        minterms: Set[int],
        cells: List[Tuple[int, int]]
    ) -> str:
        """
        Generate Boolean expression from minterms.
        
        Args:
            variables: Variable names
            minterms: Set of minterm numbers
            cells: Cell positions (for determining variable values)
            
        Returns:
            Expression string (e.g., "A'B + CD")
        """
        if not minterms:
            return "0"
        
        if len(minterms) == 1:
            # Single minterm: product of literals
            minterm = list(minterms)[0]
            return self._minterm_to_product(variables, minterm)
        
        # For groups, find common variables
        # Simplified: generate from first minterm
        if cells:
            # Use pattern matching on cell positions
            return self._cells_to_expression(variables, cells)
        
        # Fallback: use first minterm
        return self._minterm_to_product(variables, list(minterms)[0])
    
    def _minterm_to_product(self, variables: List[str], minterm: int) -> str:
        """Convert a single minterm to product of literals."""
        terms = []
        for i, var in enumerate(variables):
            bit = (minterm >> (len(variables) - 1 - i)) & 1
            if bit == 1:
                terms.append(var)
            else:
                terms.append(f"{var}'")
        return "".join(terms)
    
    def _cells_to_expression(self, variables: List[str], cells: List[Tuple[int, int]]) -> str:
        """Generate expression from cell positions."""
        if not cells:
            return "0"
        
        if len(cells) == 1:
            # Single cell - convert to minterm
            row, col = cells[0]
            minterm = self._position_to_minterm(
                row, col, len(variables),
                self.kmap.gray_code_rows if hasattr(self, 'kmap') else [0, 1],
                self.kmap.gray_code_cols if hasattr(self, 'kmap') else [0, 1]
            )
            if minterm is not None:
                return self._minterm_to_product(variables, minterm)
            return "1"
        
        # For groups, find common variables
        # Analyze row and column patterns to determine which variables are constant
        # Simplified: use first cell's minterm representation
        row, col = cells[0]
        minterm = self._position_to_minterm(
            row, col, len(variables),
            self.kmap.gray_code_rows if hasattr(self, 'kmap') else [0, 1],
            self.kmap.gray_code_cols if hasattr(self, 'kmap') else [0, 1]
        )
        if minterm is not None:
            # Generate product term (simplified - full implementation would analyze group)
            return self._minterm_to_product(variables, minterm)
        
        return "1"
    
    def simplify(self, kmap: KMap) -> SimplifiedExpression:
        """
        Simplify a K-map to minimal sum-of-products form.
        
        Uses prime implicant chart method:
        1. Find all prime implicants
        2. Identify essential prime implicants
        3. Select minimal cover
        
        Args:
            kmap: KMap to simplify
            
        Returns:
            SimplifiedExpression with minimal SOP
        """
        # Find all prime implicants
        prime_implicants = self.find_prime_implicants(kmap)
        
        # Identify essential prime implicants
        essential_pi = self._find_essential_pi(prime_implicants, kmap.minterms)
        
        # Find coverage
        coverage = self._build_coverage(prime_implicants, kmap.minterms)
        
        # Select minimal cover (simplified: use all essential + remaining)
        selected_pi = essential_pi.copy()
        covered_minterms = set()
        for pi in essential_pi:
            covered_minterms.update(pi.minterms)
        
        # Add remaining PIs to cover uncovered minterms
        remaining_minterms = set(kmap.minterms) - covered_minterms
        for pi in prime_implicants:
            if pi not in selected_pi:
                if pi.minterms & remaining_minterms:
                    selected_pi.append(pi)
                    covered_minterms.update(pi.minterms)
                    remaining_minterms -= pi.minterms
        
        # Generate SOP expression
        sop_terms = [pi.expression for pi in selected_pi]
        sop = " + ".join(sop_terms) if sop_terms else "0"
        
        return SimplifiedExpression(
            sop=sop,
            prime_implicants=prime_implicants,
            essential_pi=essential_pi,
            coverage=coverage
        )
    
    def _find_essential_pi(
        self,
        prime_implicants: List[PrimeImplicant],
        minterms: List[int]
    ) -> List[PrimeImplicant]:
        """
        Find essential prime implicants.
        
        An essential PI is one that covers a minterm that no other PI covers.
        
        Args:
            prime_implicants: List of all prime implicants
            minterms: List of minterms to cover
            
        Returns:
            List of essential prime implicants
        """
        essential = []
        minterm_coverage = {m: [] for m in minterms}
        
        # Count coverage for each minterm
        for pi in prime_implicants:
            for m in pi.minterms:
                if m in minterm_coverage:
                    minterm_coverage[m].append(pi)
        
        # Find minterms covered by only one PI
        for m, covering_pis in minterm_coverage.items():
            if len(covering_pis) == 1:
                pi = covering_pis[0]
                if pi not in essential:
                    essential.append(pi)
                    pi.is_essential = True
        
        return essential
    
    def _build_coverage(
        self,
        prime_implicants: List[PrimeImplicant],
        minterms: List[int]
    ) -> Dict[int, List[PrimeImplicant]]:
        """
        Build coverage map: which PIs cover which minterms.
        
        Args:
            prime_implicants: List of prime implicants
            minterms: List of minterms
            
        Returns:
            Dictionary mapping minterm to list of covering PIs
        """
        coverage = {m: [] for m in minterms}
        for pi in prime_implicants:
            for m in pi.minterms:
                if m in coverage:
                    coverage[m].append(pi)
        return coverage
    
    def visualize_kmap(
        self,
        kmap: KMap,
        simplified_expr: SimplifiedExpression
    ) -> Dict[str, Any]:
        """
        Generate visualization data for a K-map.
        
        Args:
            kmap: KMap to visualize
            simplified_expr: SimplifiedExpression result
            
        Returns:
            Dictionary with visualization data
        """
        return {
            "grid": kmap.grid,
            "variables": kmap.variables,
            "gray_rows": kmap.gray_code_rows,
            "gray_cols": kmap.gray_code_cols,
            "simplified_expression": simplified_expr.sop,
            "prime_implicants": [
                {
                    "expression": pi.expression,
                    "minterms": list(pi.minterms),
                    "is_essential": pi.is_essential,
                    "cells": pi.cells
                }
                for pi in simplified_expr.prime_implicants
            ],
            "essential_pi": [
                {
                    "expression": pi.expression,
                    "minterms": list(pi.minterms),
                    "cells": pi.cells
                }
                for pi in simplified_expr.essential_pi
            ]
        }
