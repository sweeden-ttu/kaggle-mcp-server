"""
ML Strategy Dialog - Main interface for the ML Strategy System

Provides a dialog window for:
- Bayesian Feature Extractor with layered classes
- Decision Tree Designer with drag-and-drop
- Ultra-Large Language Model observation generator
"""

from typing import Optional, Dict, List, Any
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QPushButton,
    QTextEdit, QLabel, QListWidget, QListWidgetItem, QGroupBox, QFormLayout,
    QLineEdit, QSpinBox, QComboBox, QMessageBox, QFileDialog, QTreeWidget,
    QTreeWidgetItem, QSplitter, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .ml_strategy_integration import MLStrategySystem
from .decision_tree_designer import DecisionTreeDesignerWidget
from .bayesian_feature_extractor import AttributeType


class MLStrategyDialog(QDialog):
    """Main dialog for ML Strategy System."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ML Strategy System - Bayesian Feature Extractors & Decision Trees")
        self.setMinimumSize(1200, 800)
        
        # Initialize ML Strategy System
        self.ml_system = MLStrategySystem()
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Bayesian Feature Extractor
        tabs.addTab(self._create_feature_extractor_tab(), "Bayesian Feature Extractor")
        
        # Tab 2: Decision Tree Designer
        tabs.addTab(self._create_decision_tree_tab(), "Decision Tree Designer")
        
        # Tab 3: Observation Generator & Reports
        tabs.addTab(self._create_observation_tab(), "Observations & Reports")
        
        # Tab 4: System Status & Logs
        tabs.addTab(self._create_status_tab(), "System Status & Logs")
        
        layout.addWidget(tabs)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save System State")
        save_btn.clicked.connect(self._save_system_state)
        button_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load System State")
        load_btn.clicked.connect(self._load_system_state)
        button_layout.addWidget(load_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_feature_extractor_tab(self) -> QWidget:
        """Create the Bayesian Feature Extractor tab."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left: Layer management
        left_panel = QGroupBox("Layers & Classes")
        left_layout = QVBoxLayout()
        
        # Layer list
        self.layer_list = QTreeWidget()
        self.layer_list.setHeaderLabels(["Layer ID", "Layer Name", "Classes"])
        left_layout.addWidget(self.layer_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self._add_layer)
        btn_layout.addWidget(add_layer_btn)
        
        add_class_btn = QPushButton("Add Class")
        add_class_btn.clicked.connect(self._add_class)
        btn_layout.addWidget(add_class_btn)
        
        add_attr_btn = QPushButton("Add Attribute")
        add_attr_btn.clicked.connect(self._add_attribute)
        btn_layout.addWidget(add_attr_btn)
        
        left_layout.addLayout(btn_layout)
        
        # Extract features button
        extract_btn = QPushButton("Extract Features")
        extract_btn.clicked.connect(self._extract_features)
        left_layout.addWidget(extract_btn)
        
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)
        
        # Right: Details and vocabulary
        right_panel = QGroupBox("Details & Vocabulary")
        right_layout = QVBoxLayout()
        
        # Vocabulary display
        vocab_label = QLabel("Vocabulary Universe:")
        right_layout.addWidget(vocab_label)
        
        self.vocab_display = QTextEdit()
        self.vocab_display.setReadOnly(True)
        self.vocab_display.setFont(QFont("Courier", 10))
        right_layout.addWidget(self.vocab_display)
        
        # Update vocabulary display
        self._update_vocabulary_display()
        
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)
        
        return widget
    
    def _create_decision_tree_tab(self) -> QWidget:
        """Create the Decision Tree Designer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Decision tree designer
        self.tree_designer = DecisionTreeDesignerWidget()
        layout.addWidget(self.tree_designer)
        
        # Bottom: Formula display and validation
        bottom_layout = QHBoxLayout()
        
        formula_label = QLabel("FOL Formula:")
        bottom_layout.addWidget(formula_label)
        
        self.formula_display = QLineEdit()
        self.formula_display.setReadOnly(True)
        bottom_layout.addWidget(self.formula_display)
        
        validate_btn = QPushButton("Validate Tree")
        validate_btn.clicked.connect(self._validate_tree)
        bottom_layout.addWidget(validate_btn)
        
        layout.addLayout(bottom_layout)
        
        # Connect tree changes to formula display
        self.tree_designer.tree_changed.connect(self.formula_display.setText)
        
        return widget
    
    def _create_observation_tab(self) -> QWidget:
        """Create the Observations & Reports tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Top: Generate observation button
        top_layout = QHBoxLayout()
        
        gen_obs_btn = QPushButton("Generate Observation")
        gen_obs_btn.clicked.connect(self._generate_observation)
        top_layout.addWidget(gen_obs_btn)
        
        gen_report_btn = QPushButton("Generate Report")
        gen_report_btn.clicked.connect(self._generate_report)
        top_layout.addWidget(gen_report_btn)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Main: Text display
        self.observation_display = QTextEdit()
        self.observation_display.setReadOnly(True)
        self.observation_display.setFont(QFont("Courier", 10))
        layout.addWidget(self.observation_display)
        
        # Bottom: Save pretrained text
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        save_text_btn = QPushButton("Save Pretrained Text")
        save_text_btn.clicked.connect(self._save_pretrained_text)
        bottom_layout.addWidget(save_text_btn)
        
        layout.addLayout(bottom_layout)
        
        return widget
    
    def _create_status_tab(self) -> QWidget:
        """Create the System Status & Logs tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics
        stats_group = QGroupBox("System Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        stats_layout.addWidget(self.stats_table)
        
        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self._refresh_statistics)
        stats_layout.addWidget(refresh_stats_btn)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Learning log
        log_group = QGroupBox("Learning Log")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_display)
        
        # Update log display
        self._update_log_display()
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        return widget
    
    def _add_layer(self):
        """Add a new layer."""
        from PyQt6.QtWidgets import QInputDialog
        layer_id, ok = QInputDialog.getInt(self, "Add Layer", "Layer ID:", 1, 1, 100, 1)
        if not ok:
            return
        
        layer_name, ok = QInputDialog.getText(self, "Add Layer", "Layer Name:")
        if not ok or not layer_name:
            return
        
        try:
            self.ml_system.feature_extractor.create_layer(layer_id, layer_name)
            self._refresh_layer_list()
            self._update_vocabulary_display()
            self._update_log_display()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _add_class(self):
        """Add a class to a layer."""
        # Get selected layer
        selected_items = self.layer_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a layer.")
            return
        
        item = selected_items[0]
        layer_id = item.data(0, Qt.ItemDataRole.UserRole)
        if layer_id is None:
            # Try to get from parent
            parent = item.parent()
            if parent:
                layer_id = parent.data(0, Qt.ItemDataRole.UserRole)
        
        if layer_id is None:
            QMessageBox.warning(self, "Invalid Selection", "Please select a layer item.")
            return
        
        from PyQt6.QtWidgets import QInputDialog
        class_name, ok = QInputDialog.getText(self, "Add Class", "Class Name:")
        if not ok or not class_name:
            return
        
        try:
            self.ml_system.feature_extractor.add_class_to_layer(layer_id, class_name)
            self._refresh_layer_list()
            self._update_vocabulary_display()
            self._update_log_display()
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _add_attribute(self):
        """Add an attribute to a class."""
        QMessageBox.information(self, "Add Attribute", "Attribute editing dialog - to be implemented in full version.")
    
    def _extract_features(self):
        """Extract features from data."""
        from PyQt6.QtWidgets import QInputDialog
        layer_id, ok = QInputDialog.getInt(self, "Extract Features", "Layer ID:", 1, 1, 100, 1)
        if not ok:
            return
        
        # Simple data input (in full version, this would be more sophisticated)
        data = {"test_attr": "test_value"}  # Placeholder
        
        try:
            obs = self.ml_system.extract_and_observe(data, layer_id)
            QMessageBox.information(self, "Features Extracted", f"Observation created: {obs.observation_id}")
            self._update_log_display()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _validate_tree(self):
        """Validate the decision tree."""
        try:
            result = self.ml_system.validate_decision_tree(self.tree_designer)
            if result["valid"]:
                msg = f"Tree is valid!\n\nFormula: {result.get('formula', 'N/A')}\n"
                msg += f"Satisfiable: {result.get('satisfiable', 'N/A')}"
                QMessageBox.information(self, "Validation Result", msg)
            else:
                QMessageBox.warning(self, "Validation Error", result.get("error", "Unknown error"))
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _generate_observation(self):
        """Generate an observation."""
        QMessageBox.information(self, "Generate Observation", "Observation generation - use Extract Features first.")
    
    def _generate_report(self):
        """Generate observation report."""
        try:
            report = self.ml_system.generate_observation_report()
            self.observation_display.setPlainText(report)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _save_pretrained_text(self):
        """Save pretrained text to file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Pretrained Text", "", "Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            try:
                self.ml_system.llm.save_pretrained_text(Path(filepath))
                QMessageBox.information(self, "Saved", "Pretrained text saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _refresh_layer_list(self):
        """Refresh the layer list display."""
        self.layer_list.clear()
        
        for layer_id in self.ml_system.feature_extractor.layer_order:
            layer = self.ml_system.feature_extractor.layers[layer_id]
            
            layer_item = QTreeWidgetItem(self.layer_list)
            layer_item.setText(0, str(layer_id))
            layer_item.setText(1, layer.layer_name)
            layer_item.setText(2, f"{len(layer.classes)} classes")
            layer_item.setData(0, Qt.ItemDataRole.UserRole, layer_id)
            
            for class_name, class_layer in layer.classes.items():
                class_item = QTreeWidgetItem(layer_item)
                class_item.setText(0, class_name)
                class_item.setText(1, f"{len(class_layer.attributes)} attributes")
        
        self.layer_list.expandAll()
    
    def _update_vocabulary_display(self):
        """Update the vocabulary display."""
        vocab = self.ml_system.feature_extractor.get_vocabulary_universe()
        vocab_list = sorted(list(vocab))
        self.vocab_display.setPlainText("\n".join(vocab_list) if vocab_list else "No vocabulary yet.")
    
    def _update_log_display(self):
        """Update the learning log display."""
        log = self.ml_system.get_learning_log()
        self.log_display.setPlainText(log if log else "No log entries yet.")
    
    def _refresh_statistics(self):
        """Refresh system statistics."""
        try:
            stats = self.ml_system.get_system_statistics()
            
            # Flatten stats into table
            rows = []
            self._flatten_dict(stats, rows, "")
            
            self.stats_table.setRowCount(len(rows))
            for i, (key, value) in enumerate(rows):
                self.stats_table.setItem(i, 0, QTableWidgetItem(key))
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
            
            self.stats_table.resizeColumnsToContents()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
    
    def _flatten_dict(self, d: Dict, rows: List, prefix: str):
        """Flatten nested dictionary for table display."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, rows, full_key)
            else:
                rows.append((full_key, value))
    
    def _save_system_state(self):
        """Save system state to directory."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Directory to Save System State")
        if dirpath:
            try:
                self.ml_system.save_system_state(Path(dirpath))
                QMessageBox.information(self, "Saved", "System state saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
    
    def _load_system_state(self):
        """Load system state from directory."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Directory to Load System State")
        if dirpath:
            try:
                self.ml_system.load_system_state(Path(dirpath))
                self._refresh_layer_list()
                self._update_vocabulary_display()
                self._update_log_display()
                QMessageBox.information(self, "Loaded", "System state loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", str(e))
