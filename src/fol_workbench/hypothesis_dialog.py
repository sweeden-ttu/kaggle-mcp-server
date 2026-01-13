"""
Hypothesis and Experiment Selection Dialog

Allows users to browse and select from a list of generated hypotheses and experiments.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QTextEdit, QPushButton, QDialogButtonBox, QSplitter,
    QGroupBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt


@dataclass
class Hypothesis:
    """Represents a hypothesis to test."""
    id: str
    title: str
    description: str
    formula: str
    expected_result: Optional[str] = None
    category: str = "General"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Experiment:
    """Represents an experiment configuration."""
    id: str
    title: str
    description: str
    formulas: List[str]
    constraints: List[str] = None
    expected_insights: Optional[str] = None
    category: str = "General"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.tags is None:
            self.tags = []


def generate_hypotheses() -> List[Hypothesis]:
    """Generate a list of hypotheses for testing."""
    hypotheses = [
        Hypothesis(
            id="hyp_001",
            title="Simple Conjunction Satisfiability",
            description="Test if a simple conjunction of two variables is satisfiable.",
            formula="And(x, y)",
            expected_result="SATISFIABLE",
            category="Basic Logic",
            tags=["conjunction", "satisfiability", "basic"]
        ),
        Hypothesis(
            id="hyp_002",
            title="Contradiction Detection",
            description="Verify that a formula and its negation together are unsatisfiable.",
            formula="And(x, Not(x))",
            expected_result="UNSATISFIABLE",
            category="Basic Logic",
            tags=["contradiction", "unsatisfiability", "basic"]
        ),
        Hypothesis(
            id="hyp_003",
            title="Tautology Verification",
            description="Check that a disjunction of a variable and its negation is always satisfiable.",
            formula="Or(x, Not(x))",
            expected_result="SATISFIABLE",
            category="Basic Logic",
            tags=["tautology", "satisfiability", "basic"]
        ),
        Hypothesis(
            id="hyp_004",
            title="Implication Validity",
            description="Test if x implies y holds when x is true and y is false (should find counterexample).",
            formula="Implies(x, y)",
            expected_result="SATISFIABLE (counterexample: x=True, y=False)",
            category="Implications",
            tags=["implication", "validity", "counterexample"]
        ),
        Hypothesis(
            id="hyp_005",
            title="Complex Nested Logic",
            description="Test satisfiability of a complex nested logical expression.",
            formula="And(Or(x, y), Implies(x, z), Not(And(y, z)))",
            expected_result="SATISFIABLE",
            category="Complex Logic",
            tags=["nested", "complex", "satisfiability"]
        ),
        Hypothesis(
            id="hyp_006",
            title="Biconditional Equivalence",
            description="Test if two variables are equivalent using biconditional.",
            formula="Iff(x, y)",
            expected_result="SATISFIABLE",
            category="Equivalence",
            tags=["biconditional", "equivalence", "iff"]
        ),
        Hypothesis(
            id="hyp_007",
            title="Three Variable Constraint System",
            description="Test a system with three variables and multiple constraints.",
            formula="And(x, Or(y, z), Not(And(x, y, z)))",
            expected_result="SATISFIABLE",
            category="Multi-Variable",
            tags=["multi-variable", "constraints", "system"]
        ),
        Hypothesis(
            id="hyp_008",
            title="Exclusive OR Pattern",
            description="Test XOR-like pattern using disjunction and negation.",
            formula="And(Or(x, y), Not(And(x, y)))",
            expected_result="SATISFIABLE",
            category="Patterns",
            tags=["xor", "pattern", "exclusive"]
        ),
        Hypothesis(
            id="hyp_009",
            title="Chain of Implications",
            description="Test a chain of implications: if x then y, if y then z.",
            formula="And(Implies(x, y), Implies(y, z))",
            expected_result="SATISFIABLE",
            category="Implications",
            tags=["chain", "implication", "transitivity"]
        ),
        Hypothesis(
            id="hyp_010",
            title="Negation of Conjunction",
            description="Test De Morgan's law: negation of conjunction equals disjunction of negations.",
            formula="Not(And(x, y))",
            expected_result="SATISFIABLE",
            category="De Morgan",
            tags=["demorgan", "negation", "conjunction"]
        ),
    ]
    return hypotheses


def generate_experiments() -> List[Experiment]:
    """Generate a list of experiments for testing."""
    experiments = [
        Experiment(
            id="exp_001",
            title="Satisfiability Rate Analysis",
            description="Run multiple formulas to analyze satisfiability rates across different logical patterns.",
            formulas=[
                "And(x, y)",
                "Or(x, Not(y))",
                "Implies(x, y)",
                "And(Or(x, y), Not(z))",
                "Or(And(x, y), And(Not(x), z))"
            ],
            constraints=[],
            expected_insights="Compare satisfiability rates between different logical operators",
            category="Analysis",
            tags=["satisfiability", "rate", "analysis", "comparison"]
        ),
        Experiment(
            id="exp_002",
            title="Contradiction Detection Suite",
            description="Test various contradiction patterns to verify unsatisfiability detection.",
            formulas=[
                "And(x, Not(x))",
                "And(And(x, y), And(Not(x), y))",
                "And(Implies(x, y), And(x, Not(y)))"
            ],
            constraints=[],
            expected_insights="Identify patterns that lead to contradictions",
            category="Contradiction",
            tags=["contradiction", "unsatisfiability", "detection"]
        ),
        Experiment(
            id="exp_003",
            title="Implication Validity Testing",
            description="Test various implication patterns to find valid implications and counterexamples.",
            formulas=[
                "Implies(x, x)",
                "Implies(And(x, y), x)",
                "Implies(x, Or(x, y))",
                "Implies(x, y)"
            ],
            constraints=[],
            expected_insights="Distinguish valid implications from those with counterexamples",
            category="Implications",
            tags=["implication", "validity", "counterexample"]
        ),
        Experiment(
            id="exp_004",
            title="Complex Nested Structures",
            description="Test deeply nested logical structures to understand model finding complexity.",
            formulas=[
                "And(Or(x, y), Implies(x, z), Not(And(y, z)))",
                "Or(And(x, y), And(Not(x), z), And(y, Not(z)))",
                "Implies(And(Or(x, y), z), Or(And(x, z), And(y, z)))"
            ],
            constraints=[],
            expected_insights="Analyze how nesting depth affects satisfiability and model finding",
            category="Complexity",
            tags=["nested", "complexity", "depth", "structure"]
        ),
        Experiment(
            id="exp_005",
            title="Multi-Variable Constraint Systems",
            description="Test systems with multiple variables and constraints to find satisfying assignments.",
            formulas=[
                "And(x, Or(y, z))",
                "And(x, y, Or(z, w))",
                "And(Or(x, y), Or(z, w), Not(And(x, z)))"
            ],
            constraints=[],
            expected_insights="Understand how multiple variables interact in constraint systems",
            category="Multi-Variable",
            tags=["multi-variable", "constraints", "system", "interaction"]
        ),
        Experiment(
            id="exp_006",
            title="Pattern Recognition",
            description="Test common logical patterns (XOR, equivalence, etc.) to identify satisfiability characteristics.",
            formulas=[
                "And(Or(x, y), Not(And(x, y)))",  # XOR
                "Iff(x, y)",  # Equivalence
                "And(Implies(x, y), Implies(y, x))"  # Biconditional
            ],
            constraints=[],
            expected_insights="Map logical patterns to their satisfiability properties",
            category="Patterns",
            tags=["pattern", "recognition", "xor", "equivalence"]
        ),
        Experiment(
            id="exp_007",
            title="De Morgan's Laws Verification",
            description="Test De Morgan's laws and their satisfiability properties.",
            formulas=[
                "Not(And(x, y))",
                "Or(Not(x), Not(y))",
                "Not(Or(x, y))",
                "And(Not(x), Not(y))"
            ],
            constraints=[],
            expected_insights="Verify De Morgan's laws and their satisfiability equivalence",
            category="De Morgan",
            tags=["demorgan", "laws", "equivalence", "negation"]
        ),
        Experiment(
            id="exp_008",
            title="Chain and Transitivity",
            description="Test chains of implications and transitivity properties.",
            formulas=[
                "And(Implies(x, y), Implies(y, z))",
                "And(Implies(x, y), Implies(y, z), Implies(z, w))",
                "And(Implies(x, y), Implies(y, z), Not(Implies(x, z)))"
            ],
            constraints=[],
            expected_insights="Analyze transitivity in implication chains",
            category="Transitivity",
            tags=["chain", "transitivity", "implication", "chain"]
        ),
    ]
    return experiments


class HypothesisExperimentDialog(QDialog):
    """Dialog for selecting hypotheses and experiments."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Hypothesis or Experiment")
        self.setMinimumSize(800, 600)
        
        self.selected_hypothesis: Optional[Hypothesis] = None
        self.selected_experiment: Optional[Experiment] = None
        
        # Generate lists
        self.hypotheses = generate_hypotheses()
        self.experiments = generate_experiments()
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for hypotheses and experiments
        tabs = QTabWidget()
        
        # Hypotheses tab
        hyp_tab = QWidget()
        hyp_layout = QVBoxLayout(hyp_tab)
        
        hyp_label = QLabel("Select a hypothesis to test:")
        hyp_layout.addWidget(hyp_label)
        
        # Splitter for list and details
        hyp_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Hypothesis list
        self.hyp_list = QListWidget()
        self.hyp_list.itemSelectionChanged.connect(self._on_hypothesis_selected)
        for hyp in self.hypotheses:
            item = QListWidgetItem(f"{hyp.title} ({hyp.category})")
            item.setData(Qt.ItemDataRole.UserRole, hyp.id)
            self.hyp_list.addItem(item)
        
        hyp_splitter.addWidget(self.hyp_list)
        
        # Hypothesis details
        hyp_details_group = QGroupBox("Details")
        hyp_details_layout = QVBoxLayout()
        
        self.hyp_title_label = QLabel()
        self.hyp_title_label.setWordWrap(True)
        self.hyp_title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        hyp_details_layout.addWidget(self.hyp_title_label)
        
        self.hyp_description = QTextEdit()
        self.hyp_description.setReadOnly(True)
        self.hyp_description.setMaximumHeight(100)
        hyp_details_layout.addWidget(QLabel("Description:"))
        hyp_details_layout.addWidget(self.hyp_description)
        
        self.hyp_formula = QTextEdit()
        self.hyp_formula.setReadOnly(True)
        self.hyp_formula.setMaximumHeight(60)
        self.hyp_formula.setFontFamily("Consolas")
        hyp_details_layout.addWidget(QLabel("Formula:"))
        hyp_details_layout.addWidget(self.hyp_formula)
        
        self.hyp_expected = QLabel()
        self.hyp_expected.setWordWrap(True)
        hyp_details_layout.addWidget(QLabel("Expected Result:"))
        hyp_details_layout.addWidget(self.hyp_expected)
        
        self.hyp_tags = QLabel()
        self.hyp_tags.setWordWrap(True)
        hyp_details_layout.addWidget(QLabel("Tags:"))
        hyp_details_layout.addWidget(self.hyp_tags)
        
        hyp_details_group.setLayout(hyp_details_layout)
        hyp_splitter.addWidget(hyp_details_group)
        
        hyp_splitter.setStretchFactor(0, 1)
        hyp_splitter.setStretchFactor(1, 2)
        
        hyp_layout.addWidget(hyp_splitter)
        tabs.addTab(hyp_tab, "Hypotheses")
        
        # Experiments tab
        exp_tab = QWidget()
        exp_layout = QVBoxLayout(exp_tab)
        
        exp_label = QLabel("Select an experiment to run:")
        exp_layout.addWidget(exp_label)
        
        # Splitter for list and details
        exp_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Experiment list
        self.exp_list = QListWidget()
        self.exp_list.itemSelectionChanged.connect(self._on_experiment_selected)
        for exp in self.experiments:
            item = QListWidgetItem(f"{exp.title} ({exp.category})")
            item.setData(Qt.ItemDataRole.UserRole, exp.id)
            self.exp_list.addItem(item)
        
        exp_splitter.addWidget(self.exp_list)
        
        # Experiment details
        exp_details_group = QGroupBox("Details")
        exp_details_layout = QVBoxLayout()
        
        self.exp_title_label = QLabel()
        self.exp_title_label.setWordWrap(True)
        self.exp_title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        exp_details_layout.addWidget(self.exp_title_label)
        
        self.exp_description = QTextEdit()
        self.exp_description.setReadOnly(True)
        self.exp_description.setMaximumHeight(80)
        exp_details_layout.addWidget(QLabel("Description:"))
        exp_details_layout.addWidget(self.exp_description)
        
        self.exp_formulas = QTextEdit()
        self.exp_formulas.setReadOnly(True)
        self.exp_formulas.setMaximumHeight(120)
        self.exp_formulas.setFontFamily("Consolas")
        exp_details_layout.addWidget(QLabel("Formulas:"))
        exp_details_layout.addWidget(self.exp_formulas)
        
        self.exp_insights = QLabel()
        self.exp_insights.setWordWrap(True)
        exp_details_layout.addWidget(QLabel("Expected Insights:"))
        exp_details_layout.addWidget(self.exp_insights)
        
        self.exp_tags = QLabel()
        self.exp_tags.setWordWrap(True)
        exp_details_layout.addWidget(QLabel("Tags:"))
        exp_details_layout.addWidget(self.exp_tags)
        
        exp_details_group.setLayout(exp_details_layout)
        exp_splitter.addWidget(exp_details_group)
        
        exp_splitter.setStretchFactor(0, 1)
        exp_splitter.setStretchFactor(1, 2)
        
        exp_layout.addWidget(exp_splitter)
        tabs.addTab(exp_tab, "Experiments")
        
        layout.addWidget(tabs)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Initialize with first item selected
        if self.hyp_list.count() > 0:
            self.hyp_list.setCurrentRow(0)
        if self.exp_list.count() > 0:
            self.exp_list.setCurrentRow(0)
    
    def _on_hypothesis_selected(self):
        """Handle hypothesis selection."""
        current = self.hyp_list.currentItem()
        if not current:
            return
        
        hyp_id = current.data(Qt.ItemDataRole.UserRole)
        hypothesis = next((h for h in self.hypotheses if h.id == hyp_id), None)
        
        if hypothesis:
            self.hyp_title_label.setText(hypothesis.title)
            self.hyp_description.setPlainText(hypothesis.description)
            self.hyp_formula.setPlainText(hypothesis.formula)
            self.hyp_expected.setText(hypothesis.expected_result or "Not specified")
            self.hyp_tags.setText(", ".join(hypothesis.tags) if hypothesis.tags else "None")
            self.selected_hypothesis = hypothesis
    
    def _on_experiment_selected(self):
        """Handle experiment selection."""
        current = self.exp_list.currentItem()
        if not current:
            return
        
        exp_id = current.data(Qt.ItemDataRole.UserRole)
        experiment = next((e for e in self.experiments if e.id == exp_id), None)
        
        if experiment:
            self.exp_title_label.setText(experiment.title)
            self.exp_description.setPlainText(experiment.description)
            formulas_text = "\n".join([f"  • {f}" for f in experiment.formulas])
            self.exp_formulas.setPlainText(formulas_text)
            self.exp_insights.setText(experiment.expected_insights or "Not specified")
            self.exp_tags.setText(", ".join(experiment.tags) if experiment.tags else "None")
            self.selected_experiment = experiment
    
    def _accept_selection(self):
        """Accept the current selection."""
        # Check which tab is active
        tabs = self.findChild(QTabWidget)
        if tabs:
            current_tab = tabs.currentIndex()
            if current_tab == 0:  # Hypotheses tab
                if self.selected_hypothesis:
                    self.accept()
                else:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "No Selection", "Please select a hypothesis.")
            else:  # Experiments tab
                if self.selected_experiment:
                    self.accept()
                else:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "No Selection", "Please select an experiment.")
    
    def get_selected_hypothesis(self) -> Optional[Hypothesis]:
        """Get the selected hypothesis."""
        return self.selected_hypothesis
    
    def get_selected_experiment(self) -> Optional[Experiment]:
        """Get the selected experiment."""
        return self.selected_experiment
