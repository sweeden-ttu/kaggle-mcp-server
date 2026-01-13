"""UI Layer: PyQt6-based graphical interface for FOL workbench."""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QTreeWidget, QTreeWidgetItem, QDockWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QDialog,
    QLineEdit, QComboBox, QSpinBox, QFormLayout, QDialogButtonBox,
    QSplitter, QGroupBox, QPlainTextEdit, QStatusBar, QMenuBar, QMenu,
    QToolBar, QAction, QHeaderView, QInputDialog, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter, QTextCursor

from .logic_layer import LogicEngine, ValidationResult, ValidationInfo
from .data_layer import DataLayer, Checkpoint, ProjectMetadata
from .predicate_dialog import PredicateDialog
from .ikyke_format import IkykeFileFormat, IkykeWorkflow
from .ikyke_protocol import IkykeProtocol
from .hypothesis_dialog import HypothesisExperimentDialog
from .autocomplete import SuggestionEngine
from .autocomplete_widget import AutocompleteManager
from .ml_strategy_dialog import MLStrategyDialog


class FormulaHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for FOL formulas."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []
        
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))
        keyword_format.setFontWeight(700)
        keywords = ['And', 'Or', 'Not', 'Implies', 'Iff', 'ForAll', 'Exists', 'True', 'False']
        for keyword in keywords:
            pattern = f'\\b{keyword}\\b'
            self.highlighting_rules.append((pattern, keyword_format))
        
        # Operators
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor(220, 220, 170))
        operators = ['&', '|', '!', '->', '<->', '∀', '∃']
        for op in operators:
            self.highlighting_rules.append((op, operator_format))
        
        # Variables (simple pattern)
        variable_format = QTextCharFormat()
        variable_format.setForeground(QColor(156, 220, 254))
        self.highlighting_rules.append((r'\b[a-z][a-zA-Z0-9_]*\b', variable_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(181, 206, 168))
        self.highlighting_rules.append((r'\b\d+\b', number_format))
    
    def highlightBlock(self, text):
        """Apply highlighting rules."""
        for pattern, format in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class FormulaEditor(QPlainTextEdit):
    """Enhanced text editor for FOL formulas with autocomplete."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 12))
        self.highlighter = FormulaHighlighter(self.document())
        self.setPlaceholderText("Enter FOL formula here...\nExample: And(x, Or(y, Not(z)))")
        
        # Autocomplete system
        self.suggestion_engine = SuggestionEngine()
        self.autocomplete_manager = None  # Will be initialized by parent
    
    def get_formula(self) -> str:
        """Get the current formula text."""
        return self.toPlainText().strip()
    
    def set_formula(self, formula: str):
        """Set the formula text."""
        self.setPlainText(formula)
    
    def keyPressEvent(self, event):
        """Handle key press events for autocomplete navigation."""
        # Let autocomplete manager handle navigation keys
        if self.autocomplete_manager:
            if self.autocomplete_manager.handle_key_event(event):
                return
        
        # Call parent implementation
        super().keyPressEvent(event)
    
    def focusOutEvent(self, event):
        """Hide autocomplete when editor loses focus."""
        if self.autocomplete_manager:
            self.autocomplete_manager.hide_suggestions()
        super().focusOutEvent(event)


class ModelViewer(QTableWidget):
    """Widget to display Z3 model information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Variable", "Value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    
    def display_model(self, model_info):
        """Display model information."""
        if model_info is None:
            self.setRowCount(0)
            return
        
        self.setRowCount(len(model_info.interpretation))
        row = 0
        for var_name, value in model_info.interpretation.items():
            self.setItem(row, 0, QTableWidgetItem(str(var_name)))
            self.setItem(row, 1, QTableWidgetItem(str(value)))
            row += 1


class CheckpointManager(QTreeWidget):
    """Widget to manage checkpoints."""
    
    checkpoint_selected = pyqtSignal(str)  # checkpoint_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Checkpoints")
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.checkpoints: Dict[str, Checkpoint] = {}
    
    def load_checkpoints(self, checkpoints: List[Checkpoint]):
        """Load and display checkpoints."""
        self.clear()
        self.checkpoints.clear()
        
        for checkpoint in checkpoints:
            item = QTreeWidgetItem(self)
            item.setText(0, checkpoint.id)
            item.setData(0, Qt.ItemDataRole.UserRole, checkpoint.id)
            item.setText(1, checkpoint.timestamp[:19])  # Truncate microseconds
            self.checkpoints[checkpoint.id] = checkpoint
        
        self.setColumnCount(2)
        self.setHeaderLabels(["ID", "Timestamp"])
    
    def _on_item_double_clicked(self, item, column):
        """Handle double-click on checkpoint."""
        checkpoint_id = item.data(0, Qt.ItemDataRole.UserRole)
        if checkpoint_id:
            self.checkpoint_selected.emit(checkpoint_id)


class ProjectDialog(QDialog):
    """Dialog for creating/editing projects."""
    
    def __init__(self, parent=None, project: Optional[ProjectMetadata] = None):
        super().__init__(parent)
        self.setWindowTitle("Project" if project is None else "Edit Project")
        self.project = project
        
        layout = QFormLayout(self)
        
        self.name_edit = QLineEdit()
        self.name_edit.setText(project.name if project else "")
        if project:
            self.name_edit.setEnabled(False)  # Can't rename
        layout.addRow("Name:", self.name_edit)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        self.desc_edit.setText(project.description if project else "")
        layout.addRow("Description:", self.desc_edit)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_project_data(self) -> Dict[str, Any]:
        """Get project data from dialog."""
        return {
            "name": self.name_edit.text().strip(),
            "description": self.desc_edit.toPlainText().strip()
        }


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOL Workbench - First-Order Logic Validation & Model Proposal")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize layers
        self.data_layer = DataLayer()
        self.logic_engine = LogicEngine()
        self.current_project: Optional[ProjectMetadata] = None
        
        # Track all formulas for checkpoint restoration
        self.tracked_formulas: List[str] = []
        
        # IKYKE workflow
        self.current_ikyke_workflow: Optional[IkykeWorkflow] = None
        self.ikyke_protocol: Optional[IkykeProtocol] = None
        
        # Analytics and preferences
        self.click_analytics: Dict[str, int] = {}
        self.touch_events: List[Dict[str, Any]] = []
        
        # Create UI
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_widget()
        self._create_dock_widgets()
        
        # Load recent checkpoints
        self._refresh_checkpoints()
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)
        
        file_menu.addSeparator()
        
        save_checkpoint_action = QAction("&Save Checkpoint", self)
        save_checkpoint_action.setShortcut("Ctrl+S")
        save_checkpoint_action.triggered.connect(self._save_checkpoint)
        file_menu.addAction(save_checkpoint_action)
        
        file_menu.addSeparator()
        
        import_smt_action = QAction("&Import SMT-LIB", self)
        import_smt_action.triggered.connect(self._import_smt_lib)
        file_menu.addAction(import_smt_action)
        
        export_smt_action = QAction("&Export SMT-LIB", self)
        export_smt_action.triggered.connect(self._export_smt_lib)
        file_menu.addAction(export_smt_action)
        
        file_menu.addSeparator()
        
        # Dataset management
        create_dataset_action = QAction("&Create Dataset", self)
        create_dataset_action.triggered.connect(self._create_dataset)
        file_menu.addAction(create_dataset_action)
        
        import_dataset_action = QAction("&Import Dataset Folder", self)
        import_dataset_action.triggered.connect(self._import_dataset_folder)
        file_menu.addAction(import_dataset_action)
        
        file_menu.addSeparator()
        
        # IKYKE workflow
        new_ikyke_action = QAction("&New IKYKE Workflow", self)
        new_ikyke_action.triggered.connect(self._new_ikyke_workflow)
        file_menu.addAction(new_ikyke_action)
        
        open_ikyke_action = QAction("&Open IKYKE Workflow", self)
        open_ikyke_action.triggered.connect(self._open_ikyke_workflow)
        file_menu.addAction(open_ikyke_action)
        
        run_ikyke_action = QAction("&Run IKYKE Workflow", self)
        run_ikyke_action.triggered.connect(self._run_ikyke_workflow)
        file_menu.addAction(run_ikyke_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        validate_action = QAction("&Validate Formula", self)
        validate_action.setShortcut("F5")
        validate_action.triggered.connect(self._validate_formula)
        tools_menu.addAction(validate_action)
        
        find_model_action = QAction("&Find Model", self)
        find_model_action.setShortcut("F6")
        find_model_action.triggered.connect(self._find_model)
        tools_menu.addAction(find_model_action)
        
        prove_action = QAction("&Prove Implication", self)
        prove_action.setShortcut("F7")
        prove_action.triggered.connect(self._prove_implication)
        tools_menu.addAction(prove_action)
        
        tools_menu.addSeparator()
        
        define_predicate_action = QAction("&Define Predicate/Function", self)
        define_predicate_action.triggered.connect(self._define_predicate)
        tools_menu.addAction(define_predicate_action)
        
        tools_menu.addSeparator()
        
        select_hypothesis_action = QAction("&Select Hypothesis/Experiment", self)
        select_hypothesis_action.setShortcut("Ctrl+H")
        select_hypothesis_action.triggered.connect(self._select_hypothesis_experiment)
        tools_menu.addAction(select_hypothesis_action)
        
        tools_menu.addSeparator()
        
        # ML Strategy System
        ml_strategy_action = QAction("&ML Strategy System", self)
        ml_strategy_action.setShortcut("Ctrl+M")
        ml_strategy_action.triggered.connect(self._open_ml_strategy)
        tools_menu.addAction(ml_strategy_action)
        
        tools_menu.addSeparator()
        
        preferences_action = QAction("&Preferences & Analytics", self)
        preferences_action.triggered.connect(self._show_preferences_analytics)
        tools_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        toggle_checkpoints = QAction("&Checkpoints", self)
        toggle_checkpoints.setCheckable(True)
        toggle_checkpoints.setChecked(True)
        toggle_checkpoints.triggered.connect(self._toggle_checkpoints_dock)
        view_menu.addAction(toggle_checkpoints)
        
        toggle_model = QAction("&Model Viewer", self)
        toggle_model.setCheckable(True)
        toggle_model.setChecked(True)
        toggle_model.triggered.connect(self._toggle_model_dock)
        view_menu.addAction(toggle_model)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self._validate_formula)
        toolbar.addWidget(validate_btn)
        
        find_model_btn = QPushButton("Find Model")
        find_model_btn.clicked.connect(self._find_model)
        toolbar.addWidget(find_model_btn)
        
        toolbar.addSeparator()
        
        save_btn = QPushButton("Save Checkpoint")
        save_btn.clicked.connect(self._save_checkpoint)
        toolbar.addWidget(save_btn)
    
    def _create_status_bar(self):
        """Create status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_central_widget(self):
        """Create central widget."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        
        # Formula editor
        formula_group = QGroupBox("Formula Editor")
        formula_layout = QVBoxLayout()
        
        self.formula_editor = FormulaEditor()
        
        # Initialize autocomplete
        self.autocomplete_manager = AutocompleteManager(
            self.formula_editor,
            self.formula_editor.suggestion_engine,
            on_suggestion_selected=self._on_suggestion_selected
        )
        self.formula_editor.autocomplete_manager = self.autocomplete_manager
        
        formula_layout.addWidget(self.formula_editor)
        
        # Constraints editor
        constraints_label = QLabel("Constraints (one per line):")
        formula_layout.addWidget(constraints_label)
        
        self.constraints_editor = QPlainTextEdit()
        self.constraints_editor.setFont(QFont("Consolas", 11))
        self.constraints_editor.setPlaceholderText("Enter additional constraints here...")
        self.constraints_editor.setMaximumHeight(150)
        formula_layout.addWidget(self.constraints_editor)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.validate_btn = QPushButton("Validate (F5)")
        self.validate_btn.clicked.connect(self._validate_formula)
        button_layout.addWidget(self.validate_btn)
        
        self.find_model_btn = QPushButton("Find Model (F6)")
        self.find_model_btn.clicked.connect(self._find_model)
        button_layout.addWidget(self.find_model_btn)
        
        self.prove_btn = QPushButton("Prove Implication (F7)")
        self.prove_btn.clicked.connect(self._prove_implication)
        button_layout.addWidget(self.prove_btn)
        
        button_layout.addStretch()
        formula_layout.addLayout(button_layout)
        
        formula_group.setLayout(formula_layout)
        layout.addWidget(formula_group)
        
        # Results area
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setFont(QFont("Consolas", 10))
        results_layout.addWidget(self.results_display)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
    
    def _create_dock_widgets(self):
        """Create dockable widgets."""
        # Checkpoints dock
        self.checkpoints_dock = QDockWidget("Checkpoints", self)
        self.checkpoints_widget = CheckpointManager()
        self.checkpoints_widget.checkpoint_selected.connect(self._load_checkpoint)
        self.checkpoints_dock.setWidget(self.checkpoints_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.checkpoints_dock)
        
        # Model viewer dock
        self.model_dock = QDockWidget("Model Viewer", self)
        self.model_viewer = ModelViewer()
        self.model_dock.setWidget(self.model_viewer)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.model_dock)
    
    def _toggle_checkpoints_dock(self, checked):
        """Toggle checkpoints dock visibility."""
        self.checkpoints_dock.setVisible(checked)
    
    def _toggle_model_dock(self, checked):
        """Toggle model dock visibility."""
        self.model_dock.setVisible(checked)
    
    def _new_project(self):
        """Create a new project."""
        dialog = ProjectDialog(self)
        if dialog.exec():
            data = dialog.get_project_data()
            if data["name"]:
                project = self.data_layer.save_project(
                    name=data["name"],
                    description=data["description"]
                )
                self.current_project = project
                self.status_bar.showMessage(f"Created project: {project.name}")
    
    def _open_project(self):
        """Open an existing project."""
        projects = self.data_layer.list_projects()
        if not projects:
            QMessageBox.information(self, "No Projects", "No projects found.")
            return
        
        # Simple selection dialog
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Open Project")
        layout = QVBoxLayout(dialog)
        
        list_widget = QListWidget()
        for project in projects:
            item = QListWidgetItem(f"{project.name} - {project.description[:50]}")
            item.setData(Qt.ItemDataRole.UserRole, project.name)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec():
            selected = list_widget.currentItem()
            if selected:
                project_name = selected.data(Qt.ItemDataRole.UserRole)
                self.current_project = self.data_layer.load_project(project_name)
                self.status_bar.showMessage(f"Opened project: {project_name}")
    
    def _save_checkpoint(self):
        """
        Save current state as checkpoint with complete formula tracking.
        
        This method saves:
        - Main formula
        - Constraint formulas
        - All tracked formulas (for complete restoration)
        
        To restore, we load the checkpoint and re-apply all formulas using
        logic_engine.restore_from_formulas(all_formulas).
        """
        formula = self.formula_editor.get_formula()
        constraints = [
            line.strip() for line in self.constraints_editor.toPlainText().split('\n')
            if line.strip()
        ]
        
        if not formula and not constraints:
            QMessageBox.warning(self, "No Content", "Please enter a formula or constraints.")
            return
        
        # Collect all formulas for restoration
        all_formulas = []
        if formula:
            all_formulas.append(formula)
        all_formulas.extend(constraints)
        
        # Also include tracked formulas from the engine
        tracked = self.logic_engine.get_tracked_formulas()
        for f in tracked:
            if f not in all_formulas:
                all_formulas.append(f)
        
        checkpoint = self.data_layer.save_checkpoint(
            formula=formula,
            constraints=constraints,
            metadata={"project": self.current_project.name if self.current_project else None},
            all_formulas=all_formulas
        )
        
        self._refresh_checkpoints()
        self.status_bar.showMessage(f"Saved checkpoint: {checkpoint.id}")
    
    def _load_checkpoint(self, checkpoint_id: str):
        """
        Load a checkpoint and restore all formulas.
        
        This method:
        1. Loads checkpoint data (including all_formulas)
        2. Resets the logic engine
        3. Re-applies all formulas using restore_from_formulas()
        4. Updates the UI with the main formula and constraints
        """
        checkpoint_data = self.data_layer.load_checkpoint(checkpoint_id)
        if checkpoint_data:
            # Restore all formulas to the engine
            all_formulas = checkpoint_data.get("all_formulas", [])
            if all_formulas:
                success, error = self.logic_engine.restore_from_formulas(all_formulas)
                if not success:
                    QMessageBox.warning(
                        self,
                        "Restore Warning",
                        f"Some formulas could not be restored: {error}"
                    )
            
            # Update UI
            self.formula_editor.set_formula(checkpoint_data.get("formula", ""))
            self.constraints_editor.setPlainText('\n'.join(checkpoint_data.get("constraints", [])))
            self.status_bar.showMessage(f"Loaded checkpoint: {checkpoint_id}")
    
    def _refresh_checkpoints(self):
        """Refresh checkpoint list."""
        checkpoints = self.data_layer.list_checkpoints()
        self.checkpoints_widget.load_checkpoints(checkpoints)
    
    def _validate_formula(self):
        """Validate the current formula."""
        formula = self.formula_editor.get_formula()
        if not formula:
            QMessageBox.warning(self, "No Formula", "Please enter a formula to validate.")
            return
        
        self.status_bar.showMessage("Validating formula...")
        QTimer.singleShot(0, lambda: self._do_validate(formula))
    
    def _do_validate(self, formula: str):
        """Perform validation (called asynchronously)."""
        try:
            self.logic_engine.reset()
            # Use tracking version to save formulas for checkpoints
            success, error = self.logic_engine.add_formula_with_tracking(formula)
            
            if not success:
                self._show_result(f"Error: {error}", None)
                return
            
            # Add constraints
            constraints = [
                line.strip() for line in self.constraints_editor.toPlainText().split('\n')
                if line.strip()
            ]
            for constraint in constraints:
                success, error = self.logic_engine.add_formula_with_tracking(constraint)
                if not success:
                    self._show_result(f"Error in constraint: {error}", None)
                    return
            
            result = self.logic_engine.check_satisfiability()
            self._show_result(result, result.model)
            self.status_bar.showMessage("Validation complete")
        except Exception as e:
            self._show_result(f"Error: {str(e)}", None)
            self.status_bar.showMessage("Validation failed")
    
    def _find_model(self):
        """Find a model for the current formula."""
        formula = self.formula_editor.get_formula()
        if not formula:
            QMessageBox.warning(self, "No Formula", "Please enter a formula.")
            return
        
        self.status_bar.showMessage("Finding model...")
        QTimer.singleShot(0, lambda: self._do_find_model(formula))
    
    def _do_find_model(self, formula: str):
        """Perform model finding."""
        try:
            result = self.logic_engine.find_model(formula)
            
            # Add constraints if any
            constraints = [
                line.strip() for line in self.constraints_editor.toPlainText().split('\n')
                if line.strip()
            ]
            if constraints:
                self.logic_engine.reset()
                self.logic_engine.add_formula(formula)
                for constraint in constraints:
                    self.logic_engine.add_formula(constraint)
                result = self.logic_engine.check_satisfiability()
            
            self._show_result(result, result.model)
            self.status_bar.showMessage("Model search complete")
        except Exception as e:
            self._show_result(f"Error: {str(e)}", None)
            self.status_bar.showMessage("Model search failed")
    
    def _prove_implication(self):
        """Prove an implication."""
        from PyQt6.QtWidgets import QInputDialog
        
        premise, ok1 = QInputDialog.getText(self, "Premise", "Enter premise formula:")
        if not ok1:
            return
        
        conclusion, ok2 = QInputDialog.getText(self, "Conclusion", "Enter conclusion formula:")
        if not ok2:
            return
        
        self.status_bar.showMessage("Proving implication...")
        QTimer.singleShot(0, lambda: self._do_prove_implication(premise, conclusion))
    
    def _do_prove_implication(self, premise: str, conclusion: str):
        """Perform implication proof."""
        try:
            result = self.logic_engine.prove_implication(premise, conclusion)
            self._show_result(result, result.model)
            self.status_bar.showMessage("Proof complete")
        except Exception as e:
            self._show_result(f"Error: {str(e)}", None)
            self.status_bar.showMessage("Proof failed")
    
    def _show_result(self, validation_info: ValidationInfo, model_info):
        """Display validation result."""
        if isinstance(validation_info, str):
            # Error message string
            self.results_display.setPlainText(validation_info)
            self.model_viewer.display_model(None)
            return
        
        result_text = f"Result: {validation_info.result.value}\n\n"
        
        if validation_info.error_message:
            result_text += f"Error: {validation_info.error_message}\n\n"
        
        if model_info:
            result_text += "Model Found:\n"
            result_text += "=" * 50 + "\n"
            for var, value in model_info.interpretation.items():
                result_text += f"  {var} = {value}\n"
            result_text += "\n"
            
            self.model_viewer.display_model(model_info)
        else:
            self.model_viewer.display_model(None)
        
        if validation_info.statistics:
            result_text += "Statistics:\n"
            result_text += "=" * 50 + "\n"
            for key, value in validation_info.statistics.items():
                result_text += f"  {key}: {value}\n"
        
        self.results_display.setPlainText(result_text)
    
    def _import_smt_lib(self):
        """Import SMT-LIB file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import SMT-LIB", "", "SMT-LIB Files (*.smt2);;All Files (*)"
        )
        if filename:
            # Read file directly
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                success, error = self.logic_engine.from_smt_lib(content)
                if success:
                    self.formula_editor.set_formula("")  # Clear formula editor
                    self.results_display.setPlainText("SMT-LIB file imported successfully.")
                    self.status_bar.showMessage(f"Imported: {filename}")
                else:
                    QMessageBox.warning(self, "Import Error", f"Failed to parse SMT-LIB: {error}")
            except Exception as e:
                QMessageBox.warning(self, "Import Error", f"Failed to load file: {str(e)}")
    
    def _export_smt_lib(self):
        """Export to SMT-LIB format."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export SMT-LIB", "", "SMT-LIB Files (*.smt2);;All Files (*)"
        )
        if filename:
            smt_content = self.logic_engine.to_smt_lib()
            path = self.data_layer.save_smt_lib(smt_content, Path(filename).stem)
            self.status_bar.showMessage(f"Exported to: {path}")
    
    def _define_predicate(self):
        """Open dialog to define a new predicate or function."""
        dialog = PredicateDialog(self)
        result = dialog.get_predicate_info()
        
        if result:
            name, arity, symbol_type, domain_types, codomain = result
            try:
                self.logic_engine.define_predicate(
                    name=name,
                    arity=arity,
                    symbol_type=symbol_type,
                    domain_types=domain_types,
                    codomain=codomain
                )
                self.status_bar.showMessage(
                    f"Defined {symbol_type.lower()}: {name} (arity={arity})"
                )
                QMessageBox.information(
                    self,
                    "Predicate Defined",
                    f"Successfully defined {symbol_type.lower()} '{name}' with arity {arity}.\n"
                    f"You can now use it in formulas."
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to define predicate: {str(e)}")
    
    def _create_dataset(self):
        """Create a new dataset with auto-generated name."""
        name, ok = QInputDialog.getText(
            self,
            "Create Dataset",
            "Dataset name (leave empty for auto-generated):"
        )
        
        if ok:
            if not name.strip():
                name = None  # Auto-generate
            
            dataset_path = self.data_layer.create_dataset(name)
            self.status_bar.showMessage(f"Created dataset: {dataset_path.name}")
            QMessageBox.information(
                self,
                "Dataset Created",
                f"Dataset created: {dataset_path.name}\n\n"
                f"Location: {dataset_path}"
            )
    
    def _import_dataset_folder(self):
        """Import a folder as a dataset."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder",
            str(self.data_layer.datasets_dir)
        )
        
        if folder_path:
            folder = Path(folder_path)
            dataset_path = self.data_layer.import_dataset_folder(folder)
            
            if dataset_path:
                self.status_bar.showMessage(f"Imported dataset: {dataset_path.name}")
                QMessageBox.information(
                    self,
                    "Dataset Imported",
                    f"Successfully imported dataset: {dataset_path.name}\n\n"
                    f"Location: {dataset_path}"
                )
            else:
                QMessageBox.warning(self, "Import Error", "Failed to import dataset folder.")
    
    def _new_ikyke_workflow(self):
        """Create a new IKYKE workflow."""
        name, ok = QInputDialog.getText(
            self,
            "New IKYKE Workflow",
            "Workflow name:"
        )
        
        if ok and name:
            workflow = IkykeFileFormat.create_default(name)
            workflow.formulas = [self.formula_editor.get_formula()] if self.formula_editor.get_formula() else []
            workflow.constraints = [
                line.strip() for line in self.constraints_editor.toPlainText().split('\n')
                if line.strip()
            ]
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save IKYKE Workflow",
                "",
                "IKYKE Files (*.ikyke);;All Files (*)"
            )
            
            if filename:
                IkykeFileFormat.save(workflow, filename)
                self.current_ikyke_workflow = workflow
                self.status_bar.showMessage(f"Created IKYKE workflow: {name}")
    
    def _open_ikyke_workflow(self):
        """Open an existing IKYKE workflow."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open IKYKE Workflow",
            "",
            "IKYKE Files (*.ikyke);;All Files (*)"
        )
        
        if filename:
            try:
                workflow = IkykeFileFormat.load(filename)
                self.current_ikyke_workflow = workflow
                self.status_bar.showMessage(f"Opened IKYKE workflow: {workflow.name}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load workflow: {str(e)}")
    
    def _run_ikyke_workflow(self):
        """Run the current IKYKE workflow."""
        if not self.current_ikyke_workflow:
            QMessageBox.warning(
                self,
                "No Workflow",
                "Please create or open an IKYKE workflow first."
            )
            return
        
        # Create protocol engine
        self.ikyke_protocol = IkykeProtocol(
            workflow=self.current_ikyke_workflow,
            logic_engine=self.logic_engine,
            data_layer=self.data_layer
        )
        
        # Set up callbacks
        self.ikyke_protocol.on_phase_change = lambda phase: self.status_bar.showMessage(
            f"IKYKE Phase: {phase.value}"
        )
        self.ikyke_protocol.on_formula_result = lambda result: self._update_ikyke_results(result)
        self.ikyke_protocol.on_save = lambda: self.status_bar.showMessage("IKYKE: Auto-saved")
        
        # Start workflow
        self.ikyke_protocol.start()
        self.status_bar.showMessage("IKYKE workflow started...")
        
        QMessageBox.information(
            self,
            "IKYKE Started",
            f"IKYKE workflow '{self.current_ikyke_workflow.name}' has started.\n\n"
            f"It will run for {self.current_ikyke_workflow.run.duration_min}-"
            f"{self.current_ikyke_workflow.run.duration_max} minutes, then "
            f"automatically evaluate, query, and analyze results."
        )
    
    def _update_ikyke_results(self, result):
        """Update UI with IKYKE formula result."""
        # This could update a results panel or log
        pass
    
    def _select_hypothesis_experiment(self):
        """Open dialog to select a hypothesis or experiment."""
        dialog = HypothesisExperimentDialog(self)
        
        if dialog.exec():
            # Check if hypothesis was selected
            hypothesis = dialog.get_selected_hypothesis()
            if hypothesis:
                # Load hypothesis into formula editor
                self.formula_editor.set_formula(hypothesis.formula)
                self.status_bar.showMessage(
                    f"Loaded hypothesis: {hypothesis.title}"
                )
                QMessageBox.information(
                    self,
                    "Hypothesis Loaded",
                    f"Hypothesis '{hypothesis.title}' has been loaded.\n\n"
                    f"Formula: {hypothesis.formula}\n\n"
                    f"Expected: {hypothesis.expected_result or 'Not specified'}\n\n"
                    f"You can now validate or find a model for this hypothesis."
                )
                return
            
            # Check if experiment was selected
            experiment = dialog.get_selected_experiment()
            if experiment:
                # Create an IKYKE workflow from the experiment
                workflow = IkykeFileFormat.create_default(experiment.title)
                workflow.formulas = experiment.formulas
                workflow.constraints = experiment.constraints
                workflow.header.description = experiment.description
                
                self.current_ikyke_workflow = workflow
                
                # Optionally load first formula into editor
                if experiment.formulas:
                    self.formula_editor.set_formula(experiment.formulas[0])
                
                self.status_bar.showMessage(
                    f"Loaded experiment: {experiment.title}"
                )
                
                response = QMessageBox.question(
                    self,
                    "Experiment Loaded",
                    f"Experiment '{experiment.title}' has been loaded.\n\n"
                    f"Formulas: {len(experiment.formulas)}\n"
                    f"Expected Insights: {experiment.expected_insights or 'Not specified'}\n\n"
                    f"Would you like to run this experiment as an IKYKE workflow?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if response == QMessageBox.StandardButton.Yes:
                    self._run_ikyke_workflow()
    
    def _open_ml_strategy(self):
        """Open ML Strategy System dialog."""
        dialog = MLStrategyDialog(self)
        dialog.exec()
    
    def _on_suggestion_selected(self, text: str):
        """Handle when user selects an autocomplete suggestion."""
        # Track usage
        self.formula_editor.suggestion_engine.track_usage(text)
        self.status_bar.showMessage(f"Applied suggestion: {text[:50]}...")
    
    def _track_click(self, widget_name: str, event_type: str = "click"):
        """Track click/usage analytics."""
        key = f"{widget_name}:{event_type}"
        self.click_analytics[key] = self.click_analytics.get(key, 0) + 1
        
        # Update font preferences based on clicks (if applicable)
        if hasattr(self.formula_editor, 'font'):
            font = self.formula_editor.font()
            self.formula_editor.suggestion_engine.update_font_preference(
                font.family(),
                font.pointSize()
            )
    
    def _track_touch_event(self, event_type: str, data: Dict[str, Any]):
        """Track touch/click events for analytics."""
        event_data = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.touch_events.append(event_data)
        
        # Keep only last 1000 events
        if len(self.touch_events) > 1000:
            self.touch_events = self.touch_events[-1000:]
    
    def _show_preferences_analytics(self):
        """Show preferences and analytics dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QTextEdit, QLabel
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Preferences & Analytics")
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        tabs = QTabWidget()
        
        # Analytics tab
        analytics_tab = QTextEdit()
        analytics_tab.setReadOnly(True)
        analytics_tab.setFont(QFont("Consolas", 10))
        
        analytics_data = self.formula_editor.suggestion_engine.get_analytics()
        analytics_text = "=== Usage Analytics ===\n\n"
        analytics_text += f"Total Suggestions Used: {analytics_data['total_suggestions_used']}\n\n"
        analytics_text += "Most Used Formulas:\n"
        for formula, count in analytics_data['most_used'].items():
            analytics_text += f"  {formula}: {count} times\n"
        
        analytics_text += f"\nPreferred Suggestions: {analytics_data['preferred_count']}\n"
        analytics_text += f"Ignored Suggestions: {analytics_data['ignored_count']}\n\n"
        
        analytics_text += "Font Preferences:\n"
        font_prefs = analytics_data['font_preferences']
        analytics_text += f"  Family: {font_prefs['family']}\n"
        analytics_text += f"  Size: {font_prefs['size']}\n\n"
        
        analytics_text += "=== Click Analytics ===\n\n"
        for key, count in self.click_analytics.items():
            analytics_text += f"{key}: {count} clicks\n"
        
        analytics_tab.setPlainText(analytics_text)
        tabs.addTab(analytics_tab, "Analytics")
        
        # Preferences tab
        prefs_tab = QTextEdit()
        prefs_tab.setReadOnly(True)
        prefs_tab.setFont(QFont("Consolas", 10))
        
        prefs = self.formula_editor.suggestion_engine.preferences
        prefs_text = "=== User Preferences ===\n\n"
        prefs_text += f"Font Family: {prefs.font_family}\n"
        prefs_text += f"Font Size: {prefs.font_size}\n\n"
        prefs_text += f"Preferred Suggestions ({len(prefs.preferred_suggestions)}):\n"
        for suggestion in list(prefs.preferred_suggestions)[:20]:
            prefs_text += f"  • {suggestion}\n"
        prefs_text += f"\nIgnored Suggestions ({len(prefs.ignored_suggestions)}):\n"
        for suggestion in list(prefs.ignored_suggestions)[:20]:
            prefs_text += f"  • {suggestion}\n"
        
        prefs_tab.setPlainText(prefs_text)
        tabs.addTab(prefs_tab, "Preferences")
        
        layout.addWidget(tabs)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        
        dialog.exec()
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About FOL Workbench",
            "FOL Workbench v1.0.0\n\n"
            "A professional tool for First-Order Logic validation and model proposal.\n\n"
            "Built with PyQt6 and Z3 Solver.\n\n"
            "Includes IKYKE protocol for automated workflows.\n\n"
            "Features intelligent autocomplete with learning capabilities."
        )
