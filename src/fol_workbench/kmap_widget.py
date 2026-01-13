"""
K-map Widget: PyQt6 widgets for K-map visualization.

Provides widgets for displaying Karnaugh maps with:
- Grid visualization
- Color-coded cells
- Prime implicant highlighting
- Simplified expression display
"""

from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
    QLabel, QTextEdit, QListWidget, QListWidgetItem, QComboBox,
    QMessageBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QFormLayout, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QBrush

from .data_layer import DataLayer
from .herbrand_converter import HerbrandConverter, Implication
from .kmap_simplifier import KMapSimplifier, KMap, SimplifiedExpression


class KMapGridWidget(QWidget):
    """Widget for displaying a K-map grid."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kmap: Optional[KMap] = None
        self.simplified_expr: Optional[SimplifiedExpression] = None
        self.table = QTableWidget()
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the grid widget."""
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    
    def display_kmap(self, kmap: KMap, simplified_expr: SimplifiedExpression):
        """
        Display a K-map in the grid.
        
        Args:
            kmap: KMap to display
            simplified_expr: SimplifiedExpression result
        """
        self.kmap = kmap
        self.simplified_expr = simplified_expr
        
        rows, cols = len(kmap.grid), len(kmap.grid[0])
        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)
        
        # Set row and column labels with Gray code
        for i, gray_val in enumerate(kmap.gray_code_rows):
            # Convert Gray code to binary for label
            bin_str = self._gray_to_binary_label(gray_val, len(kmap.variables) // 2)
            self.table.setVerticalHeaderItem(i, QTableWidgetItem(bin_str))
        
        for j, gray_val in enumerate(kmap.gray_code_cols):
            bin_str = self._gray_to_binary_label(gray_val, len(kmap.variables) - len(kmap.variables) // 2)
            self.table.setHorizontalHeaderItem(j, QTableWidgetItem(bin_str))
        
        # Fill cells
        for i in range(rows):
            for j in range(cols):
                value = kmap.grid[i][j]
                item = QTableWidgetItem()
                
                # Set cell value and color
                if value == 1:
                    item.setText("1")
                    item.setBackground(QBrush(QColor(144, 238, 144)))  # Light green
                elif value == -1:
                    item.setText("X")
                    item.setBackground(QBrush(QColor(211, 211, 211)))  # Light gray (don't-care)
                else:
                    item.setText("0")
                    item.setBackground(QBrush(QColor(255, 255, 255)))  # White
                
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                font = QFont()
                font.setBold(True)
                font.setPointSize(12)
                item.setFont(font)
                self.table.setItem(i, j, item)
        
        # Highlight prime implicants
        self._highlight_prime_implicants()
        
        # Resize columns
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
    
    def _gray_to_binary_label(self, gray_val: int, num_bits: int) -> str:
        """Convert Gray code value to binary label string."""
        # Convert to binary
        binary = format(gray_val, f'0{num_bits}b')
        # Use variable names if available
        if self.kmap and len(self.kmap.variables) >= num_bits:
            # Map bits to variable names
            var_names = self.kmap.variables[-num_bits:] if num_bits <= len(self.kmap.variables) else self.kmap.variables
            label_parts = []
            for i, bit in enumerate(binary):
                if i < len(var_names):
                    if bit == '0':
                        label_parts.append(f"{var_names[i]}'")
                    else:
                        label_parts.append(var_names[i])
            return "".join(label_parts)
        return binary
    
    def _highlight_prime_implicants(self):
        """Highlight prime implicant groups in the grid."""
        if not self.simplified_expr:
            return
        
        # Use a different color for essential PIs
        essential_color = QColor(255, 200, 100)  # Orange
        other_color = QColor(200, 255, 200)  # Light green
        
        for pi in self.simplified_expr.prime_implicants:
            if pi.cells:
                color = essential_color if pi.is_essential else other_color
                for row, col in pi.cells:
                    item = self.table.item(row, col)
                    if item:
                        # Blend with existing color
                        current_color = item.background().color()
                        blended = QColor(
                            min(255, (current_color.red() + color.red()) // 2),
                            min(255, (current_color.green() + color.green()) // 2),
                            min(255, (current_color.blue() + color.blue()) // 2)
                        )
                        item.setBackground(QBrush(blended))


class KMapDialog(QDialog):
    """Dialog for K-map simplification of implications."""
    
    def __init__(self, data_layer: DataLayer, parent=None):
        super().__init__(parent)
        self.data_layer = data_layer
        self.converter = HerbrandConverter(data_layer)
        self.simplifier = KMapSimplifier()
        
        self.setWindowTitle("K-map Simplifier")
        self.setMinimumSize(1000, 700)
        
        self._setup_ui()
        self._load_checkpoints()
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Top section: Checkpoint selection
        top_group = QGroupBox("Select Checkpoint")
        top_layout = QVBoxLayout()
        
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("Checkpoint:"))
        
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.setMaximumHeight(150)
        self.checkpoint_list.itemSelectionChanged.connect(self._on_checkpoint_selected)
        checkpoint_layout.addWidget(self.checkpoint_list)
        
        conversion_layout = QHBoxLayout()
        conversion_layout.addWidget(QLabel("Conversion Method:"))
        self.conversion_combo = QComboBox()
        self.conversion_combo.addItems(["Ground Instances (Herbrand Base)", "Boolean Variables"])
        conversion_layout.addWidget(self.conversion_combo)
        
        analyze_btn = QPushButton("Analyze Implications")
        analyze_btn.clicked.connect(self._analyze_implications)
        conversion_layout.addWidget(analyze_btn)
        
        top_layout.addLayout(checkpoint_layout)
        top_layout.addLayout(conversion_layout)
        top_group.setLayout(top_layout)
        layout.addWidget(top_group)
        
        # Main splitter: K-map display and results
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: K-map grid
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("K-map Visualization:"))
        self.kmap_widget = KMapGridWidget()
        left_layout.addWidget(self.kmap_widget)
        splitter.addWidget(left_widget)
        
        # Right: Results and information
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Simplified expression
        expr_group = QGroupBox("Simplified Expression")
        expr_layout = QVBoxLayout()
        self.expression_label = QLabel("No expression yet")
        self.expression_label.setWordWrap(True)
        self.expression_label.setFont(QFont("Courier", 12))
        expr_layout.addWidget(self.expression_label)
        expr_group.setLayout(expr_layout)
        right_layout.addWidget(expr_group)
        
        # Prime implicants
        pi_group = QGroupBox("Prime Implicants")
        pi_layout = QVBoxLayout()
        self.pi_list = QListWidget()
        self.pi_list.setMaximumHeight(200)
        pi_layout.addWidget(self.pi_list)
        pi_group.setLayout(pi_layout)
        right_layout.addWidget(pi_group)
        
        # Information
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout()
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        right_layout.addStretch()
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self._export_results)
        button_layout.addWidget(export_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _load_checkpoints(self):
        """Load available checkpoints."""
        checkpoints = self.data_layer.list_checkpoints()
        self.checkpoint_list.clear()
        
        for checkpoint in checkpoints:
            item = QListWidgetItem(f"{checkpoint.id} - {checkpoint.timestamp[:19]}")
            item.setData(Qt.ItemDataRole.UserRole, checkpoint.id)
            self.checkpoint_list.addItem(item)
    
    def _on_checkpoint_selected(self):
        """Handle checkpoint selection."""
        # Could pre-load implications here
        pass
    
    def _analyze_implications(self):
        """Analyze implications from selected checkpoint."""
        selected_items = self.checkpoint_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a checkpoint.")
            return
        
        checkpoint_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Extract implications
        implications = self.converter.extract_implications_from_checkpoint(checkpoint_id)
        
        if not implications:
            QMessageBox.information(
                self, "No Implications",
                "No implications found in the selected checkpoint."
            )
            return
        
        # Use first implication (could extend to handle multiple)
        impl = implications[0]
        
        # Convert to propositional
        use_ground = self.conversion_combo.currentIndex() == 0
        
        try:
            if use_ground:
                prop_formula = self.converter.convert_to_ground_instances(
                    f"Implies({impl.premise}, {impl.conclusion})"
                )
            else:
                prop_formula = self.converter.convert_to_boolean_vars(
                    f"Implies({impl.premise}, {impl.conclusion})"
                )
            
            # Generate truth table
            truth_table = self.converter.generate_truth_table(prop_formula)
            
            if not truth_table:
                QMessageBox.warning(
                    self, "Too Many Variables",
                    f"Formula has {len(prop_formula.variables)} variables. "
                    "K-maps support up to 4 variables."
                )
                return
            
            # Create K-map
            kmap = self.simplifier.create_kmap(
                truth_table.minterms,
                len(truth_table.variables),
                truth_table.variables
            )
            
            if not kmap:
                QMessageBox.warning(self, "Error", "Failed to create K-map.")
                return
            
            # Simplify
            simplified = self.simplifier.simplify(kmap)
            
            # Display results
            self._display_results(kmap, simplified, prop_formula, impl)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error analyzing implications: {str(e)}"
            )
    
    def _display_results(
        self,
        kmap: KMap,
        simplified: SimplifiedExpression,
        prop_formula,
        impl: Implication
    ):
        """Display K-map and simplification results."""
        # Display K-map
        self.kmap_widget.display_kmap(kmap, simplified)
        
        # Display simplified expression
        self.expression_label.setText(f"<b>Simplified Expression:</b><br>{simplified.sop}")
        
        # Display prime implicants
        self.pi_list.clear()
        for pi in simplified.prime_implicants:
            essential_mark = " (Essential)" if pi.is_essential else ""
            item_text = f"{pi.expression}{essential_mark}"
            item = QListWidgetItem(item_text)
            if pi.is_essential:
                item.setForeground(QBrush(QColor(255, 140, 0)))  # Orange for essential
            self.pi_list.addItem(item)
        
        # Display information
        info = f"""
<b>Original Implication:</b><br>
Premise: {impl.premise}<br>
Conclusion: {impl.conclusion}<br><br>

<b>Propositional Variables:</b> {', '.join(prop_formula.variables)}<br>
<b>Number of Minterms:</b> {len(kmap.minterms)}<br>
<b>Number of Prime Implicants:</b> {len(simplified.prime_implicants)}<br>
<b>Essential Prime Implicants:</b> {len(simplified.essential_pi)}
        """
        self.info_text.setHtml(info)
    
    def _export_results(self):
        """Export results to text file."""
        # Simple implementation - could be extended
        if not hasattr(self.kmap_widget, 'kmap') or not self.kmap_widget.kmap:
            QMessageBox.information(self, "No Results", "No results to export.")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("K-map Simplification Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Simplified Expression: {self.expression_label.text()}\n\n")
                    f.write("Prime Implicants:\n")
                    for i in range(self.pi_list.count()):
                        f.write(f"  {self.pi_list.item(i).text()}\n")
                    f.write("\n" + self.info_text.toPlainText())
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
