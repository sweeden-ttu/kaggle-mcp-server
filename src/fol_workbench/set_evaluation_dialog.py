"""
Set Evaluation Dialog - Interface for creating Prolog applications with set operations.

Provides a dialog window for:
- Building set expressions using mathematical symbols
- Generating Prolog code from set expressions
- Visual symbol palette for set operations
"""

from typing import Optional, Dict, List, Any
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QPushButton,
    QTextEdit, QLabel, QGroupBox, QFormLayout, QLineEdit, QMessageBox,
    QFileDialog, QGridLayout, QScrollArea, QSplitter, QPlainTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCharFormat, QColor


class SetSymbolButton(QPushButton):
    """Button widget for set operation symbols."""
    
    def __init__(self, symbol: str, code: str, name: str, notes: str, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.code = code
        self.name = name
        self.notes = notes
        
        # Display symbol and name
        self.setText(f"{symbol}\n{name}")
        self.setToolTip(f"{name}\nCode: {code}\nNotes: {notes}")
        self.setMinimumSize(80, 60)
        self.setMaximumSize(100, 70)
        
        # Style the button
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.setFont(font)


class SetEvaluationDialog(QDialog):
    """Main dialog for Set Evaluation System with Prolog generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Evaluation - Prolog Generator")
        self.setMinimumSize(1200, 800)
        
        # Store expression being built
        self.current_expression: List[str] = []
        
        # Prolog code generator
        self.prolog_code: str = ""
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Expression Builder
        tabs.addTab(self._create_expression_builder_tab(), "Expression Builder")
        
        # Tab 2: Prolog Code Generator
        tabs.addTab(self._create_prolog_generator_tab(), "Prolog Generator")
        
        # Tab 3: Help & Documentation
        tabs.addTab(self._create_help_tab(), "Help & Documentation")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Expression")
        clear_btn.clicked.connect(self._clear_expression)
        button_layout.addWidget(clear_btn)
        
        generate_btn = QPushButton("Generate Prolog")
        generate_btn.clicked.connect(self._generate_prolog)
        button_layout.addWidget(generate_btn)
        
        save_btn = QPushButton("Save Prolog File")
        save_btn.clicked.connect(self._save_prolog_file)
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_expression_builder_tab(self) -> QWidget:
        """Create the expression builder tab with symbol palette."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Splitter for symbol palette and expression editor
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Symbol palette
        palette_widget = QWidget()
        palette_layout = QVBoxLayout(palette_widget)
        
        palette_label = QLabel("Set Operation Symbols")
        palette_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        palette_layout.addWidget(palette_label)
        
        # Scroll area for symbols
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Group symbols by category
        self._add_symbol_group(scroll_layout, "Membership", [
            ("∈", "\\in", "Element of", "x ∈ S"),
            ("∉", "\\notin", "Not element", "Non-membership"),
            ("∋", "\\ni", "Contains", "S ∋ x"),
        ])
        
        self._add_symbol_group(scroll_layout, "Set Relations", [
            ("⊂", "\\subset", "Proper subset", "Strictly contained"),
            ("⊆", "\\subseteq", "Subset or equal", "Contained or equal"),
            ("⊃", "\\supset", "Proper superset", "Also implication"),
            ("⊇", "\\supseteq", "Superset or equal", "Contains or equal"),
        ])
        
        self._add_symbol_group(scroll_layout, "Set Operations", [
            ("∪", "\\cup", "Union", "A ∪ B"),
            ("∩", "\\cap", "Intersection", "A ∩ B"),
            ("∖", "\\setminus", "Set difference", "A ∖ B"),
        ])
        
        self._add_symbol_group(scroll_layout, "Special Sets", [
            ("∅", "\\emptyset", "Empty set", "No elements"),
            ("℘", "\\wp", "Power set", "℘(S)"),
        ])
        
        self._add_symbol_group(scroll_layout, "Equivalence & Identity", [
            ("=", "=", "Equality", "Standard"),
            ("≠", "\\neq", "Not equal", "Inequality"),
            ("≡", "\\equiv", "Equivalent", "Logical equivalence"),
            ("≅", "\\cong", "Congruent", "Isomorphic"),
            (":=", ":=", "Definition", "Defined as"),
            ("≜", "\\triangleq", "Defined (triangle)", "Alt definition"),
        ])
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        palette_layout.addWidget(scroll)
        
        splitter.addWidget(palette_widget)
        
        # Right: Expression editor
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)
        
        editor_label = QLabel("Set Expression")
        editor_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        editor_layout.addWidget(editor_label)
        
        # Expression display
        self.expression_display = QTextEdit()
        self.expression_display.setReadOnly(True)
        self.expression_display.setFont(QFont("Courier New", 14))
        self.expression_display.setPlaceholderText("Click symbols to build your expression...")
        self.expression_display.setMinimumHeight(150)
        editor_layout.addWidget(self.expression_display)
        
        # Manual input
        manual_label = QLabel("Manual Input (or edit expression directly):")
        editor_layout.addWidget(manual_label)
        
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("Type set expression manually...")
        self.manual_input.returnPressed.connect(self._add_manual_input)
        editor_layout.addWidget(self.manual_input)
        
        # Variable/Set input
        var_group = QGroupBox("Variables and Sets")
        var_layout = QFormLayout()
        
        self.set_a_input = QLineEdit()
        self.set_a_input.setPlaceholderText("e.g., A, Set1, {1,2,3}")
        var_layout.addRow("Set A:", self.set_a_input)
        
        self.set_b_input = QLineEdit()
        self.set_b_input.setPlaceholderText("e.g., B, Set2, {4,5,6}")
        var_layout.addRow("Set B:", self.set_b_input)
        
        self.element_input = QLineEdit()
        self.element_input.setPlaceholderText("e.g., x, 5, element")
        var_layout.addRow("Element:", self.element_input)
        
        var_group.setLayout(var_layout)
        editor_layout.addWidget(var_group)
        
        # Expression history
        history_label = QLabel("Expression History:")
        editor_layout.addWidget(history_label)
        
        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setMaximumHeight(100)
        self.history_list.setFont(QFont("Courier New", 10))
        editor_layout.addWidget(self.history_list)
        
        editor_layout.addStretch()
        splitter.addWidget(editor_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        return widget
    
    def _add_symbol_group(self, layout: QVBoxLayout, group_name: str, symbols: List[tuple]):
        """Add a group of symbol buttons."""
        group_label = QLabel(group_name)
        group_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(group_label)
        
        grid = QGridLayout()
        grid.setSpacing(5)
        
        row = 0
        col = 0
        for symbol, code, name, notes in symbols:
            btn = SetSymbolButton(symbol, code, name, notes)
            btn.clicked.connect(lambda checked, s=symbol, c=code, n=name: self._on_symbol_clicked(s, c, n))
            grid.addWidget(btn, row, col)
            col += 1
            if col >= 4:  # 4 columns per row
                col = 0
                row += 1
        
        layout.addLayout(grid)
        layout.addSpacing(10)
    
    def _on_symbol_clicked(self, symbol: str, code: str, name: str):
        """Handle symbol button click."""
        # Get current variables
        set_a = self.set_a_input.text().strip() or "A"
        set_b = self.set_b_input.text().strip() or "B"
        element = self.element_input.text().strip() or "x"
        
        # Build expression based on symbol
        expr = self._build_expression_from_symbol(symbol, code, name, set_a, set_b, element)
        
        if expr:
            self.current_expression.append(expr)
            self._update_expression_display()
    
    def _build_expression_from_symbol(self, symbol: str, code: str, name: str, 
                                      set_a: str, set_b: str, element: str) -> Optional[str]:
        """Build expression string from symbol and variables."""
        # Membership operations
        if symbol == "∈":
            return f"{element} ∈ {set_a}"
        elif symbol == "∉":
            return f"{element} ∉ {set_a}"
        elif symbol == "∋":
            return f"{set_a} ∋ {element}"
        
        # Set relations
        elif symbol == "⊂":
            return f"{set_a} ⊂ {set_b}"
        elif symbol == "⊆":
            return f"{set_a} ⊆ {set_b}"
        elif symbol == "⊃":
            return f"{set_a} ⊃ {set_b}"
        elif symbol == "⊇":
            return f"{set_a} ⊇ {set_b}"
        
        # Set operations
        elif symbol == "∪":
            return f"{set_a} ∪ {set_b}"
        elif symbol == "∩":
            return f"{set_a} ∩ {set_b}"
        elif symbol == "∖":
            return f"{set_a} ∖ {set_b}"
        
        # Special sets
        elif symbol == "∅":
            return "∅"
        elif symbol == "℘":
            return f"℘({set_a})"
        
        # Equivalence
        elif symbol == "=":
            return f"{set_a} = {set_b}"
        elif symbol == "≠":
            return f"{set_a} ≠ {set_b}"
        elif symbol == "≡":
            return f"{set_a} ≡ {set_b}"
        elif symbol == "≅":
            return f"{set_a} ≅ {set_b}"
        elif symbol == ":=":
            return f"{set_a} := {set_b}"
        elif symbol == "≜":
            return f"{set_a} ≜ {set_b}"
        
        return None
    
    def _add_manual_input(self):
        """Add manual input to expression."""
        text = self.manual_input.text().strip()
        if text:
            self.current_expression.append(text)
            self._update_expression_display()
            self.manual_input.clear()
    
    def _update_expression_display(self):
        """Update the expression display."""
        if not self.current_expression:
            self.expression_display.clear()
            return
        
        # Display current expression
        expr_text = " ".join(self.current_expression)
        self.expression_display.setPlainText(expr_text)
        
        # Add to history
        history_text = self.history_list.toPlainText()
        if history_text:
            history_text += "\n" + expr_text
        else:
            history_text = expr_text
        
        self.history_list.setPlainText(history_text)
    
    def _clear_expression(self):
        """Clear the current expression."""
        self.current_expression.clear()
        self.expression_display.clear()
        self.prolog_code = ""
        QMessageBox.information(self, "Cleared", "Expression cleared.")
    
    def _create_prolog_generator_tab(self) -> QWidget:
        """Create the Prolog code generator tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel(
            "Build your set expression in the Expression Builder tab, then click "
            "'Generate Prolog' to create Prolog code."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Prolog code display
        code_label = QLabel("Generated Prolog Code:")
        code_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(code_label)
        
        self.prolog_display = QPlainTextEdit()
        self.prolog_display.setFont(QFont("Courier New", 11))
        self.prolog_display.setPlaceholderText("Prolog code will appear here after generation...")
        layout.addWidget(self.prolog_display)
        
        # Preview section
        preview_label = QLabel("Expression Preview:")
        preview_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(preview_label)
        
        self.preview_display = QTextEdit()
        self.preview_display.setReadOnly(True)
        self.preview_display.setFont(QFont("Courier New", 10))
        self.preview_display.setMaximumHeight(150)
        layout.addWidget(self.preview_display)
        
        return widget
    
    def _create_help_tab(self) -> QWidget:
        """Create help and documentation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setFont(QFont("Arial", 10))
        
        help_content = """
<h2>Set Evaluation - Prolog Generator</h2>

<h3>Overview</h3>
<p>This tool allows you to build set expressions using mathematical symbols and generate Prolog code automatically.</p>

<h3>Symbol Reference</h3>

<h4>Membership Operations</h4>
<ul>
<li><b>∈</b> (\\in) - Element of: x ∈ S</li>
<li><b>∉</b> (\\notin) - Not element: Non-membership</li>
<li><b>∋</b> (\\ni) - Contains: S ∋ x</li>
</ul>

<h4>Set Relations</h4>
<ul>
<li><b>⊂</b> (\\subset) - Proper subset: Strictly contained</li>
<li><b>⊆</b> (\\subseteq) - Subset or equal: Contained or equal</li>
<li><b>⊃</b> (\\supset) - Proper superset: Also implication</li>
<li><b>⊇</b> (\\supseteq) - Superset or equal: Contains or equal</li>
</ul>

<h4>Set Operations</h4>
<ul>
<li><b>∪</b> (\\cup) - Union: A ∪ B</li>
<li><b>∩</b> (\\cap) - Intersection: A ∩ B</li>
<li><b>∖</b> (\\setminus) - Set difference: A ∖ B</li>
</ul>

<h4>Special Sets</h4>
<ul>
<li><b>∅</b> (\\emptyset) - Empty set: No elements</li>
<li><b>℘</b> (\\wp) - Power set: ℘(S)</li>
</ul>

<h4>Equivalence & Identity</h4>
<ul>
<li><b>=</b> - Equality: Standard</li>
<li><b>≠</b> (\\neq) - Not equal: Inequality</li>
<li><b>≡</b> (\\equiv) - Equivalent: Logical equivalence</li>
<li><b>≅</b> (\\cong) - Congruent: Isomorphic</li>
<li><b>:=</b> - Definition: Defined as</li>
<li><b>≜</b> (\\triangleq) - Defined (triangle): Alt definition</li>
</ul>

<h3>Usage</h3>
<ol>
<li>Enter set names or elements in the variable fields (Set A, Set B, Element)</li>
<li>Click symbol buttons to build your expression</li>
<li>Or type expressions manually in the manual input field</li>
<li>Click "Generate Prolog" to create Prolog code</li>
<li>Save the Prolog file using "Save Prolog File"</li>
</ol>

<h3>Prolog Code Generation</h3>
<p>The generator creates Prolog predicates for set operations. Each symbol is mapped to appropriate Prolog predicates:</p>
<ul>
<li>Membership → member/2 predicate</li>
<li>Set operations → union/3, intersection/3, difference/3 predicates</li>
<li>Set relations → subset/2, proper_subset/2 predicates</li>
<li>Equality → equality checks</li>
</p>
        """
        
        help_text.setHtml(help_content)
        layout.addWidget(help_text)
        
        return widget
    
    def _generate_prolog(self):
        """Generate Prolog code from current expression."""
        if not self.current_expression:
            QMessageBox.warning(self, "No Expression", "Please build an expression first.")
            return
        
        # Get variables
        set_a = self.set_a_input.text().strip() or "A"
        set_b = self.set_b_input.text().strip() or "B"
        element = self.element_input.text().strip() or "x"
        
        # Generate Prolog code
        prolog_lines = [
            "% Set Evaluation Prolog Program",
            "% Generated from set expression",
            "",
            "% Basic set operations",
            "",
            "% Membership predicate",
            "member(X, [X|_]).",
            "member(X, [_|T]) :- member(X, T).",
            "",
            "% Not a member",
            "not_member(_, []).",
            "not_member(X, [H|T]) :- X \\= H, not_member(X, T).",
            "",
            "% Union",
            "union([], L, L).",
            "union([H|T], L2, [H|Result]) :- not_member(H, L2), union(T, L2, Result).",
            "union([H|T], L2, Result) :- member(H, L2), union(T, L2, Result).",
            "",
            "% Intersection",
            "intersection([], _, []).",
            "intersection([H|T], L2, [H|Result]) :- member(H, L2), intersection(T, L2, Result).",
            "intersection([H|T], L2, Result) :- not_member(H, L2), intersection(T, L2, Result).",
            "",
            "% Set difference",
            "difference([], _, []).",
            "difference([H|T], L2, [H|Result]) :- not_member(H, L2), difference(T, L2, Result).",
            "difference([H|T], L2, Result) :- member(H, L2), difference(T, L2, Result).",
            "",
            "% Subset",
            "subset([], _).",
            "subset([H|T], L) :- member(H, L), subset(T, L).",
            "",
            "% Proper subset",
            "proper_subset(A, B) :- subset(A, B), not(subset(B, A)).",
            "",
            "% Set equality",
            "set_equal(A, B) :- subset(A, B), subset(B, A).",
            "",
            "% Empty set check",
            "empty_set([]).",
            "",
            "% Power set (simplified - generates all subsets)",
            "power_set([], [[]]).",
            "power_set([H|T], PowerSet) :-",
            "    power_set(T, SubPowerSet),",
            "    add_to_all(H, SubPowerSet, WithH),",
            "    append(SubPowerSet, WithH, PowerSet).",
            "",
            "add_to_all(_, [], []).",
            "add_to_all(X, [L|Ls], [[X|L]|Rest]) :- add_to_all(X, Ls, Rest).",
            "",
            "% Generated queries from expression:",
            ""
        ]
        
        # Add queries based on expression
        expr_text = " ".join(self.current_expression)
        prolog_lines.append(f"% Expression: {expr_text}")
        prolog_lines.append("")
        
        # Parse expression and generate queries
        queries = self._parse_expression_to_prolog(expr_text, set_a, set_b, element)
        prolog_lines.extend(queries)
        
        self.prolog_code = "\n".join(prolog_lines)
        self.prolog_display.setPlainText(self.prolog_code)
        
        # Update preview
        preview_text = f"<b>Expression:</b> {expr_text}<br><br>"
        preview_text += f"<b>Variables:</b><br>"
        preview_text += f"Set A: {set_a}<br>"
        preview_text += f"Set B: {set_b}<br>"
        preview_text += f"Element: {element}<br><br>"
        preview_text += f"<b>Generated {len(queries)} Prolog queries</b>"
        self.preview_display.setHtml(preview_text)
        
        QMessageBox.information(self, "Generated", "Prolog code generated successfully!")
    
    def _parse_expression_to_prolog(self, expr: str, set_a: str, set_b: str, element: str) -> List[str]:
        """Parse expression and generate Prolog queries."""
        queries = []
        
        # Parse each expression component
        expr_parts = expr.split()
        
        for i, part in enumerate(expr_parts):
            # Membership operations
            if "∈" in part:
                queries.append(f"% Membership query")
                queries.append(f"% Query: {part}")
                # Extract element and set from expression
                if "∈" in part:
                    parts = part.split("∈")
                    if len(parts) == 2:
                        elem, set_name = parts[0].strip(), parts[1].strip()
                        queries.append(f"% member({elem}, {set_name}).")
                queries.append("")
            
            elif "∉" in part:
                queries.append(f"% Non-membership query")
                queries.append(f"% Query: {part}")
                if "∉" in part:
                    parts = part.split("∉")
                    if len(parts) == 2:
                        elem, set_name = parts[0].strip(), parts[1].strip()
                        queries.append(f"% not_member({elem}, {set_name}).")
                queries.append("")
            
            elif "∋" in part:
                queries.append(f"% Contains query")
                queries.append(f"% Query: {part}")
                if "∋" in part:
                    parts = part.split("∋")
                    if len(parts) == 2:
                        set_name, elem = parts[0].strip(), parts[1].strip()
                        queries.append(f"% member({elem}, {set_name}).")
                queries.append("")
            
            # Set relations
            elif "⊆" in part:
                queries.append(f"% Subset query")
                queries.append(f"% Query: {part}")
                if "⊆" in part:
                    parts = part.split("⊆")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% subset({set1}, {set2}).")
                queries.append("")
            
            elif "⊂" in part:
                queries.append(f"% Proper subset query")
                queries.append(f"% Query: {part}")
                if "⊂" in part:
                    parts = part.split("⊂")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% proper_subset({set1}, {set2}).")
                queries.append("")
            
            elif "⊇" in part:
                queries.append(f"% Superset query")
                queries.append(f"% Query: {part}")
                if "⊇" in part:
                    parts = part.split("⊇")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% subset({set2}, {set1}).")
                queries.append("")
            
            elif "⊃" in part:
                queries.append(f"% Proper superset query")
                queries.append(f"% Query: {part}")
                if "⊃" in part:
                    parts = part.split("⊃")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% proper_subset({set2}, {set1}).")
                queries.append("")
            
            # Set operations
            elif "∪" in part:
                queries.append(f"% Union query")
                queries.append(f"% Query: {part}")
                if "∪" in part:
                    parts = part.split("∪")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% union({set1}, {set2}, Result).")
                queries.append("")
            
            elif "∩" in part:
                queries.append(f"% Intersection query")
                queries.append(f"% Query: {part}")
                if "∩" in part:
                    parts = part.split("∩")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% intersection({set1}, {set2}, Result).")
                queries.append("")
            
            elif "∖" in part:
                queries.append(f"% Set difference query")
                queries.append(f"% Query: {part}")
                if "∖" in part:
                    parts = part.split("∖")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% difference({set1}, {set2}, Result).")
                queries.append("")
            
            # Special sets
            elif part == "∅":
                queries.append(f"% Empty set check")
                queries.append(f"% Query: {part}")
                queries.append(f"% empty_set(Set).")
                queries.append("")
            
            elif "℘" in part:
                queries.append(f"% Power set query")
                queries.append(f"% Query: {part}")
                # Extract set name from ℘(Set)
                if "℘(" in part and ")" in part:
                    start = part.find("℘(") + 2
                    end = part.find(")", start)
                    if end > start:
                        set_name = part[start:end].strip()
                        queries.append(f"% power_set({set_name}, PowerSet).")
                queries.append("")
            
            # Equivalence
            elif "=" in part and "≠" not in part and ":=" not in part and "≜" not in part:
                # Check if it's a set equality (not just any =)
                if i > 0 and i < len(expr_parts) - 1:
                    queries.append(f"% Equality query")
                    queries.append(f"% Query: {part}")
                    if "=" in part:
                        parts = part.split("=")
                        if len(parts) == 2:
                            set1, set2 = parts[0].strip(), parts[1].strip()
                            queries.append(f"% set_equal({set1}, {set2}).")
                    queries.append("")
            
            elif "≠" in part:
                queries.append(f"% Inequality query")
                queries.append(f"% Query: {part}")
                if "≠" in part:
                    parts = part.split("≠")
                    if len(parts) == 2:
                        set1, set2 = parts[0].strip(), parts[1].strip()
                        queries.append(f"% not(set_equal({set1}, {set2})).")
                queries.append("")
        
        if not queries:
            queries.append("% No specific queries generated from expression")
            queries.append("% You can manually add queries using the predicates defined above")
            queries.append("")
            queries.append("% Example queries:")
            queries.append(f"% member({element}, {set_a}).")
            queries.append(f"% union({set_a}, {set_b}, Result).")
            queries.append(f"% subset({set_a}, {set_b}).")
        
        return queries
    
    def _save_prolog_file(self):
        """Save the generated Prolog code to a file."""
        if not self.prolog_code:
            QMessageBox.warning(self, "No Code", "Please generate Prolog code first.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Prolog File",
            "",
            "Prolog Files (*.pl);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.prolog_code)
                QMessageBox.information(self, "Saved", f"Prolog file saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")
