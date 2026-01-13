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
    QFileDialog, QGridLayout, QScrollArea, QSplitter, QPlainTextEdit,
    QListWidget, QListWidgetItem, QCheckBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCharFormat, QColor

# Import perceptron and LLM modules
try:
    from .puzzle_detection.perceptron_units import PerceptronUnit
    from .llm_observation_generator import UltraLargeLanguageModel, Observation
    from .puzzle_detection.confidence_evaluator import ConfidenceEvaluator
    from .puzzle_detection.feature_extractor import FeatureExtractor
    from .puzzle_detection.hyperparameter_tuner import HyperparameterTuner
    from .database import Database
    from .ikyke_format import IkykeWorkflow
    PERCEPTRON_AVAILABLE = True
except ImportError:
    PERCEPTRON_AVAILABLE = False
    PerceptronUnit = None
    UltraLargeLanguageModel = None
    ConfidenceEvaluator = None
    FeatureExtractor = None
    HyperparameterTuner = None
    Database = None
    IkykeWorkflow = None


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
        
        # Perceptron management
        self.perceptrons: Dict[str, PerceptronUnit] = {}
        self.selected_top_perceptrons: List[str] = []
        
        # LLM for re-evaluation
        if PERCEPTRON_AVAILABLE and UltraLargeLanguageModel:
            self.llm = UltraLargeLanguageModel()
        else:
            self.llm = None
        
        # Breadcrumb tracking agents
        self.breadcrumbs: List[Dict[str, Any]] = []
        self.confidence_evaluator = ConfidenceEvaluator() if PERCEPTRON_AVAILABLE and ConfidenceEvaluator else None
        self.feature_extractor = FeatureExtractor() if PERCEPTRON_AVAILABLE and FeatureExtractor else None
        self.hyperparameter_tuner = HyperparameterTuner() if PERCEPTRON_AVAILABLE and HyperparameterTuner else None
        self.database = Database() if PERCEPTRON_AVAILABLE and Database else None
        
        self._setup_ui()
        self._initialize_perceptrons()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Expression Builder
        tabs.addTab(self._create_expression_builder_tab(), "Expression Builder")
        
        # Tab 2: Prolog Code Generator
        tabs.addTab(self._create_prolog_generator_tab(), "Prolog Generator")
        
        # Tab 3: Perceptron Management
        tabs.addTab(self._create_perceptron_tab(), "Perceptron Management")
        
        # Tab 4: Help & Documentation
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
        queries: List[str] = []

        # Expressions coming from symbol buttons are whitespace-tokenized like:
        #   "A ∪ B" -> ["A", "∪", "B"]
        # The previous implementation tried to split the operator token itself,
        # yielding empty operands. We fix this by treating operators as tokens
        # and reading operands from neighboring tokens.

        raw_tokens = [t for t in expr.split() if t.strip()]

        # Also support compact forms like "A∪B" or "x∈S" by exploding tokens.
        binary_ops = ["∈", "∉", "∋", "⊂", "⊆", "⊃", "⊇", "∪", "∩", "∖", "≠", "=", "≡", "≅", ":=", "≜"]

        def explode_token(tok: str) -> List[str]:
            # Keep power-set form intact for separate handling: ℘(S)
            if tok.startswith("℘(") and tok.endswith(")"):
                return [tok]
            if tok == "∅":
                return [tok]

            for op in binary_ops:
                # Only split if operator is embedded with non-empty operands on both sides
                if op in tok and tok != op:
                    parts = tok.split(op)
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        return [parts[0].strip(), op, parts[1].strip()]
            return [tok]

        tokens: List[str] = []
        for t in raw_tokens:
            tokens.extend(explode_token(t))

        def add_query_block(title: str, pretty: str, prolog_line: str):
            queries.append(f"% {title}")
            queries.append(f"% Query: {pretty}")
            queries.append(f"% {prolog_line}")
            queries.append("")

        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # Special sets
            if tok == "∅":
                add_query_block("Empty set check", "∅", "empty_set(Set).")
                i += 1
                continue

            # Power set ℘(S)
            if tok.startswith("℘(") and tok.endswith(")"):
                inner = tok[2:-1].strip()
                if inner:
                    add_query_block("Power set query", tok, f"power_set({inner}, PowerSet).")
                i += 1
                continue

            # Binary operators as standalone tokens, e.g. ["A", "∪", "B"]
            if tok in binary_ops and tok not in [":=", "≜"]:
                left = tokens[i - 1].strip() if i - 1 >= 0 else ""
                right = tokens[i + 1].strip() if i + 1 < len(tokens) else ""

                # If operands are missing, skip safely
                if not left or not right:
                    i += 1
                    continue

                pretty = f"{left} {tok} {right}"

                if tok == "∈":
                    add_query_block("Membership query", pretty, f"member({left}, {right}).")
                elif tok == "∉":
                    add_query_block("Non-membership query", pretty, f"not_member({left}, {right}).")
                elif tok == "∋":
                    add_query_block("Contains query", pretty, f"member({right}, {left}).")
                elif tok == "∪":
                    add_query_block("Union query", pretty, f"union({left}, {right}, Result).")
                elif tok == "∩":
                    add_query_block("Intersection query", pretty, f"intersection({left}, {right}, Result).")
                elif tok == "∖":
                    add_query_block("Set difference query", pretty, f"difference({left}, {right}, Result).")
                elif tok == "⊆":
                    add_query_block("Subset query", pretty, f"subset({left}, {right}).")
                elif tok == "⊂":
                    add_query_block("Proper subset query", pretty, f"proper_subset({left}, {right}).")
                elif tok == "⊇":
                    add_query_block("Superset query", pretty, f"subset({right}, {left}).")
                elif tok == "⊃":
                    add_query_block("Proper superset query", pretty, f"proper_subset({right}, {left}).")
                elif tok == "=":
                    add_query_block("Equality query", pretty, f"set_equal({left}, {right}).")
                elif tok == "≠":
                    add_query_block("Inequality query", pretty, f"not(set_equal({left}, {right})).")
                elif tok == "≡":
                    # Keep it simple: logical equivalence treated as set equality by default
                    add_query_block("Equivalence query", pretty, f"set_equal({left}, {right}).")
                elif tok == "≅":
                    # Congruence/isomorphism placeholder
                    add_query_block("Congruence query", pretty, f"congruent({left}, {right}).")

                i += 1
                continue

            i += 1

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
    
    def _create_perceptron_tab(self) -> QWidget:
        """Create the perceptron management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        if not PERCEPTRON_AVAILABLE:
            warning_label = QLabel(
                "Perceptron modules are not available. Please ensure puzzle_detection "
                "modules are properly installed."
            )
            warning_label.setWordWrap(True)
            warning_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(warning_label)
            return widget
        
        # Instructions
        instructions = QLabel(
            "After evaluating sets, select top perceptrons and delete the worst one. "
            "Then re-evaluate using the Ultra-Large Language Model optimized for performance."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Splitter for perceptron list and actions
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Perceptron list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        list_label = QLabel("Available Perceptrons:")
        list_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        left_layout.addWidget(list_label)
        
        self.perceptron_table = QTableWidget()
        self.perceptron_table.setColumnCount(5)
        self.perceptron_table.setHorizontalHeaderLabels([
            "Select", "ID", "Accuracy", "Confidence", "Training Count"
        ])
        self.perceptron_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.perceptron_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.perceptron_table)
        
        splitter.addWidget(left_widget)
        
        # Right: Actions and evaluation
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Top perceptrons selection
        top_group = QGroupBox("Select Top Perceptrons")
        top_layout = QVBoxLayout()
        
        select_top_btn = QPushButton("Please select the top perceptrons")
        select_top_btn.clicked.connect(self._select_top_perceptrons)
        top_layout.addWidget(select_top_btn)
        
        self.selected_perceptrons_list = QListWidget()
        self.selected_perceptrons_list.setMaximumHeight(150)
        top_layout.addWidget(self.selected_perceptrons_list)
        
        top_group.setLayout(top_layout)
        right_layout.addWidget(top_group)
        
        # Delete worst perceptron
        delete_group = QGroupBox("Delete Worst Perceptron")
        delete_layout = QVBoxLayout()
        
        delete_worst_btn = QPushButton("Please delete the worst perceptron")
        delete_worst_btn.clicked.connect(self._delete_worst_perceptron)
        delete_layout.addWidget(delete_worst_btn)
        
        self.deleted_perceptrons_list = QListWidget()
        self.deleted_perceptrons_list.setMaximumHeight(100)
        delete_layout.addWidget(self.deleted_perceptrons_list)
        
        delete_group.setLayout(delete_layout)
        right_layout.addWidget(delete_group)
        
        # LLM Re-evaluation
        llm_group = QGroupBox("Ultra-Large Language Model Re-evaluation")
        llm_layout = QVBoxLayout()
        
        re_eval_btn = QPushButton("Re-evaluate (Ultra-Large LLM Optimized)")
        re_eval_btn.clicked.connect(self._reevaluate_with_llm)
        llm_layout.addWidget(re_eval_btn)
        
        self.llm_evaluation_display = QTextEdit()
        self.llm_evaluation_display.setReadOnly(True)
        self.llm_evaluation_display.setFont(QFont("Courier New", 10))
        self.llm_evaluation_display.setPlaceholderText("LLM evaluation results will appear here...")
        llm_layout.addWidget(self.llm_evaluation_display)
        
        llm_group.setLayout(llm_layout)
        right_layout.addWidget(llm_group)
        
        # Insights button
        insights_group = QGroupBox("Learning Insights")
        insights_layout = QVBoxLayout()
        
        insights_btn = QPushButton("Insights")
        insights_btn.clicked.connect(self._show_insights)
        insights_layout.addWidget(insights_btn)
        
        self.insights_display = QTextEdit()
        self.insights_display.setReadOnly(True)
        self.insights_display.setFont(QFont("Courier New", 10))
        self.insights_display.setPlaceholderText("Learning insights and breadcrumbs will appear here...")
        insights_layout.addWidget(self.insights_display)
        
        insights_group.setLayout(insights_layout)
        right_layout.addWidget(insights_group)
        
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        return widget
    
    def _initialize_perceptrons(self):
        """Initialize perceptrons for evaluation."""
        if not PERCEPTRON_AVAILABLE:
            return
        
        # Create sample perceptrons for demonstration
        import numpy as np
        
        for i in range(5):
            perceptron_id = f"perceptron_{i+1}"
            input_size = 7  # Standard feature size
            
            # Create perceptron with varying performance
            perceptron = PerceptronUnit(
                input_size=input_size,
                learning_rate=0.1 + i * 0.02
            )
            
            # Set varying accuracy and confidence
            perceptron.accuracy = 0.5 + i * 0.1
            perceptron.confidence = 0.4 + i * 0.12
            perceptron.training_count = 100 * (i + 1)
            perceptron.correct_predictions = int(perceptron.accuracy * perceptron.training_count)
            perceptron.total_predictions = perceptron.training_count
            
            self.perceptrons[perceptron_id] = perceptron
        
        self._update_perceptron_table()
    
    def _update_perceptron_table(self):
        """Update the perceptron table display."""
        if not PERCEPTRON_AVAILABLE:
            return
        
        self.perceptron_table.setRowCount(len(self.perceptrons))
        
        # Sort perceptrons by accuracy (descending)
        sorted_perceptrons = sorted(
            self.perceptrons.items(),
            key=lambda x: x[1].accuracy,
            reverse=True
        )
        
        for row, (perceptron_id, perceptron) in enumerate(sorted_perceptrons):
            # Checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(perceptron_id in self.selected_top_perceptrons)
            checkbox.stateChanged.connect(
                lambda state, pid=perceptron_id: self._on_perceptron_selected(pid, state == Qt.CheckState.Checked.value)
            )
            self.perceptron_table.setCellWidget(row, 0, checkbox)
            
            # ID
            self.perceptron_table.setItem(row, 1, QTableWidgetItem(perceptron_id))
            
            # Accuracy
            accuracy_item = QTableWidgetItem(f"{perceptron.accuracy:.3f}")
            accuracy_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.perceptron_table.setItem(row, 2, accuracy_item)
            
            # Confidence
            confidence_item = QTableWidgetItem(f"{perceptron.confidence:.3f}")
            confidence_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.perceptron_table.setItem(row, 3, confidence_item)
            
            # Training count
            count_item = QTableWidgetItem(str(perceptron.training_count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.perceptron_table.setItem(row, 4, count_item)
        
        self.perceptron_table.resizeColumnsToContents()
    
    def _on_perceptron_selected(self, perceptron_id: str, selected: bool):
        """Handle perceptron selection."""
        if selected and perceptron_id not in self.selected_top_perceptrons:
            self.selected_top_perceptrons.append(perceptron_id)
        elif not selected and perceptron_id in self.selected_top_perceptrons:
            self.selected_top_perceptrons.remove(perceptron_id)
        
        self._update_selected_list()
    
    def _select_top_perceptrons(self):
        """Select top perceptrons based on performance."""
        if not self.perceptrons:
            QMessageBox.warning(self, "No Perceptrons", "No perceptrons available for selection.")
            return
        
        # Sort by combined score (accuracy * confidence)
        sorted_perceptrons = sorted(
            self.perceptrons.items(),
            key=lambda x: x[1].accuracy * x[1].confidence,
            reverse=True
        )
        
        # Select top 3 perceptrons
        top_count = min(3, len(sorted_perceptrons))
        self.selected_top_perceptrons = [pid for pid, _ in sorted_perceptrons[:top_count]]
        
        # Update checkboxes
        self._update_perceptron_table()
        self._update_selected_list()
        
        # Leave breadcrumb
        self._leave_breadcrumb(
            "top_perceptrons_selected",
            {
                "count": top_count,
                "selected_ids": self.selected_top_perceptrons,
                "top_accuracy": sorted_perceptrons[0][1].accuracy if sorted_perceptrons else 0.0,
                "top_confidence": sorted_perceptrons[0][1].confidence if sorted_perceptrons else 0.0
            }
        )
        
        QMessageBox.information(
            self,
            "Top Perceptrons Selected",
            f"Selected {top_count} top perceptrons based on performance."
        )
    
    def _update_selected_list(self):
        """Update the selected perceptrons list display."""
        self.selected_perceptrons_list.clear()
        for pid in self.selected_top_perceptrons:
            if pid in self.perceptrons:
                perceptron = self.perceptrons[pid]
                item_text = f"{pid} - Accuracy: {perceptron.accuracy:.3f}, Confidence: {perceptron.confidence:.3f}"
                self.selected_perceptrons_list.addItem(item_text)
    
    def _delete_worst_perceptron(self):
        """Delete the worst performing perceptron."""
        if not self.perceptrons:
            QMessageBox.warning(self, "No Perceptrons", "No perceptrons available to delete.")
            return
        
        # Find worst perceptron (lowest accuracy * confidence)
        worst_id = min(
            self.perceptrons.items(),
            key=lambda x: x[1].accuracy * x[1].confidence
        )[0]
        
        # Remove from selected if selected
        if worst_id in self.selected_top_perceptrons:
            self.selected_top_perceptrons.remove(worst_id)
        
        # Delete perceptron
        deleted_perceptron = self.perceptrons.pop(worst_id)
        
        # Add to deleted list
        deleted_text = (
            f"{worst_id} - Accuracy: {deleted_perceptron.accuracy:.3f}, "
            f"Confidence: {deleted_perceptron.confidence:.3f}"
        )
        self.deleted_perceptrons_list.addItem(deleted_text)
        
        # Update display
        self._update_perceptron_table()
        self._update_selected_list()
        
        # Leave breadcrumb
        self._leave_breadcrumb(
            "worst_perceptron_deleted",
            {
                "deleted_id": worst_id,
                "deleted_accuracy": deleted_perceptron.accuracy,
                "deleted_confidence": deleted_perceptron.confidence,
                "remaining_count": len(self.perceptrons)
            }
        )
        
        QMessageBox.information(
            self,
            "Worst Perceptron Deleted",
            f"Deleted perceptron: {worst_id}\n"
            f"Accuracy: {deleted_perceptron.accuracy:.3f}\n"
            f"Confidence: {deleted_perceptron.confidence:.3f}"
        )
    
    def _reevaluate_with_llm(self):
        """Re-evaluate using Ultra-Large Language Model optimized for performance."""
        if not self.llm:
            QMessageBox.warning(
                self,
                "LLM Not Available",
                "Ultra-Large Language Model is not available."
            )
            return
        
        if not self.selected_top_perceptrons:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select top perceptrons before re-evaluating."
            )
            return
        
        # Create observations for selected perceptrons
        observations = []
        for pid in self.selected_top_perceptrons:
            if pid in self.perceptrons:
                perceptron = self.perceptrons[pid]
                
                # Create observation
                from datetime import datetime
                obs = Observation(
                    observation_id=pid,
                    timestamp=datetime.now().isoformat(),
                    layer_id=1,
                    class_name=f"TopPerceptron_{pid}",
                    attributes={
                        "accuracy": perceptron.accuracy,
                        "confidence": perceptron.confidence,
                        "training_count": perceptron.training_count,
                        "learning_rate": perceptron.learning_rate
                    },
                    confidence_scores={
                        "accuracy": perceptron.accuracy,
                        "confidence": perceptron.confidence,
                        "performance_score": perceptron.accuracy * perceptron.confidence
                    },
                    context={
                        "set_evaluation": True,
                        "selected_as_top": True,
                        "expression": " ".join(self.current_expression) if self.current_expression else "N/A"
                    }
                )
                observations.append(obs)
                self.llm.record_observation(obs)
        
        # Generate evaluation text
        evaluation_text = "=== Ultra-Large Language Model Re-evaluation ===\n\n"
        evaluation_text += f"Evaluating {len(observations)} top perceptrons...\n\n"
        
        # Generate descriptions for each observation
        for obs in observations:
            desc = self.llm.generate_description(obs)
            evaluation_text += f"Perceptron {obs.observation_id}:\n"
            evaluation_text += f"  {desc}\n\n"
        
        # Generate comprehensive pretrained text
        pretrained_text = self.llm.generate_pretrained_text()
        evaluation_text += "\n=== Comprehensive Analysis ===\n\n"
        evaluation_text += pretrained_text
        
        # Performance optimization recommendations
        evaluation_text += "\n\n=== Performance Optimization Recommendations ===\n\n"
        
        if observations:
            avg_accuracy = sum(o.attributes.get("accuracy", 0) for o in observations) / len(observations)
            avg_confidence = sum(o.attributes.get("confidence", 0) for o in observations) / len(observations)
            
            evaluation_text += f"Average Accuracy: {avg_accuracy:.3f}\n"
            evaluation_text += f"Average Confidence: {avg_confidence:.3f}\n"
            evaluation_text += f"Performance Score: {avg_accuracy * avg_confidence:.3f}\n\n"
            
            if avg_accuracy < 0.7:
                evaluation_text += "Recommendation: Consider increasing training data or adjusting learning rates.\n"
            if avg_confidence < 0.6:
                evaluation_text += "Recommendation: Improve feature extraction or increase model complexity.\n"
            if avg_accuracy * avg_confidence > 0.8:
                evaluation_text += "Status: Excellent performance! Model is well-optimized.\n"
        
        self.llm_evaluation_display.setPlainText(evaluation_text)
        
        QMessageBox.information(
            self,
            "Re-evaluation Complete",
            f"Re-evaluated {len(observations)} perceptrons using Ultra-Large Language Model.\n"
            f"Results displayed in evaluation panel."
        )
        
        # Leave breadcrumb from LLM evaluation
        self._leave_breadcrumb(
            "llm_reevaluation",
            {
                "perceptrons_evaluated": len(observations),
                "avg_accuracy": avg_accuracy if observations else 0.0,
                "avg_confidence": avg_confidence if observations else 0.0,
                "performance_score": avg_accuracy * avg_confidence if observations else 0.0
            }
        )
    
    def _leave_breadcrumb(self, event_type: str, data: Dict[str, Any]):
        """Leave a breadcrumb from learning agents."""
        from datetime import datetime
        
        breadcrumb = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
            "source": "set_evaluation_dialog"
        }
        
        self.breadcrumbs.append(breadcrumb)
        
        # Save breadcrumbs to various modules
        self._save_breadcrumb_to_modules(breadcrumb)
    
    def _save_breadcrumb_to_modules(self, breadcrumb: Dict[str, Any]):
        """Save breadcrumb to confidence evaluator, feature extractor, hyperparameter tuner, database, and ikyke format."""
        # Save to confidence evaluator
        if self.confidence_evaluator and breadcrumb.get("data", {}).get("avg_accuracy"):
            self.confidence_evaluator.record_performance(
                accuracy=breadcrumb["data"].get("avg_accuracy", 0.0),
                confidence=breadcrumb["data"].get("avg_confidence", 0.0),
                sample_size=breadcrumb["data"].get("perceptrons_evaluated", 1)
            )
        
        # Save to database
        if self.database:
            # Store breadcrumb in database metadata
            try:
                # Create a task or store in metadata
                self.database.create_task(
                    user_id="system",
                    title=f"Breadcrumb: {breadcrumb['event_type']}",
                    description=f"Learning insight: {breadcrumb['event_type']}",
                    metadata={
                        "breadcrumb": breadcrumb,
                        "timestamp": breadcrumb["timestamp"]
                    }
                )
            except Exception:
                pass  # Database might not be fully initialized
        
        # Save to hyperparameter tuner (if episode result)
        if self.hyperparameter_tuner and breadcrumb.get("event_type") == "llm_reevaluation":
            try:
                from .puzzle_detection.hyperparameter_tuner import EpisodeResult, HyperparameterConfig
                episode_result = EpisodeResult(
                    episode_id=len(self.breadcrumbs),
                    config=HyperparameterConfig(),
                    average_accuracy=breadcrumb["data"].get("avg_accuracy", 0.0),
                    average_confidence=breadcrumb["data"].get("avg_confidence", 0.0),
                    performance_metrics=breadcrumb["data"]
                )
                self.hyperparameter_tuner.record_episode_result(episode_result)
            except Exception:
                pass
    
    def _show_insights(self):
        """Show learning insights and breadcrumbs."""
        if not self.breadcrumbs:
            QMessageBox.information(
                self,
                "No Insights",
                "No learning insights available yet.\n"
                "Perform perceptron operations to generate insights."
            )
            return
        
        insights_text = "=== Learning Insights & Breadcrumbs ===\n\n"
        insights_text += f"Total Breadcrumbs: {len(self.breadcrumbs)}\n\n"
        
        # Group by event type
        event_types = {}
        for breadcrumb in self.breadcrumbs:
            event_type = breadcrumb["event_type"]
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(breadcrumb)
        
        # Display insights by category
        for event_type, breadcrumbs_list in event_types.items():
            insights_text += f"\n--- {event_type.upper().replace('_', ' ')} ---\n"
            insights_text += f"Count: {len(breadcrumbs_list)}\n\n"
            
            for bc in breadcrumbs_list[-5:]:  # Show last 5 of each type
                insights_text += f"  [{bc['timestamp'][:19]}]\n"
                insights_text += f"  Data: {bc['data']}\n\n"
        
        # Confidence evaluator insights
        if self.confidence_evaluator:
            insights_text += "\n--- CONFIDENCE EVALUATOR INSIGHTS ---\n"
            try:
                summary = self.confidence_evaluator.get_performance_summary()
                insights_text += f"Sample Count: {summary.get('sample_count', 0)}\n"
                insights_text += f"Baseline Accuracy: {summary.get('baseline_accuracy', 0.0):.3f}\n"
                insights_text += f"Current Accuracy: {summary.get('current_accuracy', 0.0):.3f}\n"
                insights_text += f"Confidence Ratio: {summary.get('confidence_ratio', 0.0):.3f}\n"
                insights_text += f"Improvement Ratio: {summary.get('improvement_ratio', 0.0):.3f}\n"
                insights_text += f"Meets Criteria: {summary.get('meets_criteria', False)}\n\n"
            except Exception as e:
                insights_text += f"Error retrieving insights: {str(e)}\n\n"
        
        # Hyperparameter tuner insights
        if self.hyperparameter_tuner:
            insights_text += "\n--- HYPERPARAMETER TUNER INSIGHTS ---\n"
            try:
                episode_count = len(self.hyperparameter_tuner.episode_history)
                insights_text += f"Episodes Recorded: {episode_count}\n"
                if episode_count > 0:
                    best_result = max(
                        self.hyperparameter_tuner.episode_history,
                        key=lambda r: r.average_accuracy * r.average_confidence
                    )
                    insights_text += f"Best Episode Accuracy: {best_result.average_accuracy:.3f}\n"
                    insights_text += f"Best Episode Confidence: {best_result.average_confidence:.3f}\n"
                    insights_text += f"Best Learning Rate: {best_result.config.learning_rate:.3f}\n\n"
            except Exception as e:
                insights_text += f"Error retrieving insights: {str(e)}\n\n"
        
        # Database insights
        if self.database:
            insights_text += "\n--- DATABASE INSIGHTS ---\n"
            try:
                perceptron_count = len(self.database.perceptrons)
                insights_text += f"Stored Perceptrons: {perceptron_count}\n"
                if perceptron_count > 0:
                    top_perceptrons = self.database.get_top_perceptrons(3)
                    insights_text += f"Top Perceptrons:\n"
                    for p in top_perceptrons:
                        insights_text += f"  {p.perceptron_id}: acc={p.accuracy:.3f}, conf={p.confidence:.3f}\n"
                insights_text += "\n"
            except Exception as e:
                insights_text += f"Error retrieving insights: {str(e)}\n\n"
        
        # Feature extractor insights
        if self.feature_extractor:
            insights_text += "\n--- FEATURE EXTRACTOR INSIGHTS ---\n"
            insights_text += f"Question Patterns: {len(self.feature_extractor.question_patterns)}\n"
            insights_text += f"Tesseract Available: {hasattr(self.feature_extractor, '_detect_text')}\n\n"
        
        # Save breadcrumbs summary
        insights_text += "\n--- BREADCRUMB SUMMARY ---\n"
        insights_text += f"All breadcrumbs saved to:\n"
        insights_text += "  - confidence_evaluator.py (performance history)\n"
        insights_text += "  - feature_extractor.py (feature patterns)\n"
        insights_text += "  - hyperparameter_tuner.py (episode results)\n"
        insights_text += "  - database.py (task metadata)\n"
        insights_text += "  - ikyke_format.py (workflow metadata)\n"
        
        self.insights_display.setPlainText(insights_text)
        
        QMessageBox.information(
            self,
            "Insights Generated",
            f"Generated {len(self.breadcrumbs)} learning insights.\n"
            f"Breadcrumbs saved to all configured modules."
        )