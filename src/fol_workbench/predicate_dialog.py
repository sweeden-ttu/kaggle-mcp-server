"""
Dialog for defining predicates and functions in FOL.

This dialog allows users to dynamically create logic symbols with:
- Name specification
- Arity (number of arguments)
- Type (Constant vs Function/Predicate)
- Domain types for each argument
- Codomain (return type)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QDialogButtonBox, QLabel, QGroupBox, QPushButton
)
from PyQt6.QtCore import Qt
from typing import Optional, List, Tuple


class PredicateDialog(QDialog):
    """
    Dialog for defining predicates and functions.
    
    Allows users to specify:
    - Symbol name
    - Arity (0 for constants, 1+ for predicates/functions)
    - Symbol type (Constant or Function)
    - Domain types for each argument
    - Codomain (return type)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Predicate/Function")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Basic information
        form_layout = QFormLayout()
        
        # Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., P, Q, father, R")
        form_layout.addRow("Name:", self.name_edit)
        
        # Arity
        self.arity_spin = QSpinBox()
        self.arity_spin.setMinimum(0)
        self.arity_spin.setMaximum(10)
        self.arity_spin.setValue(1)
        self.arity_spin.valueChanged.connect(self._on_arity_changed)
        form_layout.addRow("Arity (arguments):", self.arity_spin)
        
        # Symbol type
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Function", "Constant"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        form_layout.addRow("Type:", self.type_combo)
        
        # Codomain (return type)
        self.codomain_combo = QComboBox()
        self.codomain_combo.addItems(["Bool", "Int", "Real", "String"])
        form_layout.addRow("Return Type:", self.codomain_combo)
        
        layout.addLayout(form_layout)
        
        # Domain types group (only shown for arity > 0)
        self.domain_group = QGroupBox("Argument Types")
        self.domain_layout = QVBoxLayout()
        self.domain_group.setLayout(self.domain_layout)
        self.domain_type_combos: List[QComboBox] = []
        layout.addWidget(self.domain_group)
        
        # Initialize domain types
        self._on_arity_changed(1)
        
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _on_arity_changed(self, value: int):
        """Update domain type inputs when arity changes."""
        # Clear existing
        for combo in self.domain_type_combos:
            self.domain_layout.removeWidget(combo)
            combo.deleteLater()
        self.domain_type_combos.clear()
        
        # Add new inputs
        if value > 0:
            self.domain_group.setVisible(True)
            for i in range(value):
                combo = QComboBox()
                combo.addItems(["Int", "Bool", "Real", "String"])
                combo.setCurrentText("Int")  # Default to Int
                label = QLabel(f"Argument {i+1}:")
                row_layout = QHBoxLayout()
                row_layout.addWidget(label)
                row_layout.addWidget(combo)
                self.domain_layout.addLayout(row_layout)
                self.domain_type_combos.append(combo)
        else:
            self.domain_group.setVisible(False)
    
    def _on_type_changed(self, text: str):
        """Update UI when symbol type changes."""
        if text == "Constant":
            self.arity_spin.setValue(0)
            self.arity_spin.setEnabled(False)
        else:
            self.arity_spin.setEnabled(True)
            if self.arity_spin.value() == 0:
                self.arity_spin.setValue(1)
    
    def get_predicate_info(self) -> Optional[Tuple[str, int, str, List[str], str]]:
        """
        Get the predicate/function definition information.
        
        Returns:
            Tuple of (name, arity, symbol_type, domain_types, codomain)
            or None if dialog was cancelled
        """
        if self.exec() != QDialog.DialogCode.Accepted:
            return None
        
        name = self.name_edit.text().strip()
        if not name:
            return None
        
        arity = self.arity_spin.value()
        symbol_type = self.type_combo.currentText()
        codomain = self.codomain_combo.currentText()
        
        # Get domain types
        domain_types = []
        if arity > 0:
            for combo in self.domain_type_combos:
                domain_types.append(combo.currentText())
        
        return (name, arity, symbol_type, domain_types, codomain)
