"""
Decision Tree Designer with Drag-and-Drop Interface

Provides a graphical drag-and-drop interface for designing decision trees
with logical operations: XAND, NAND, FORALL, EXISTS.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem,
    QGraphicsTextItem, QListWidget, QListWidgetItem, QGroupBox, QLineEdit,
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QMessageBox,
    QSpinBox, QTextEdit
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QDrag, QDragEnterEvent,
    QDragMoveEvent, QDropEvent, QMouseEvent, QPainterPath
)

from .logic_layer import LogicEngine, ForAll, Exists, And, Or, Not, Iff


class LogicalOperator(Enum):
    """Logical operators for decision tree nodes."""
    AND = "And"
    OR = "Or"
    NOT = "Not"
    XAND = "XAnd"  # Exclusive AND (A and B but not both true simultaneously in same context)
    NAND = "NAnd"  # Not AND
    FORALL = "ForAll"  # Universal quantification
    EXISTS = "Exists"  # Existential quantification
    IMPLIES = "Implies"
    IFF = "Iff"
    LEAF = "Leaf"  # Terminal node with attribute/value


@dataclass
class TreeNode:
    """Represents a node in the decision tree."""
    node_id: str
    operator: LogicalOperator
    label: str
    position: Tuple[float, float] = (0.0, 0.0)
    children: List[str] = field(default_factory=list)  # IDs of child nodes
    parent: Optional[str] = None
    attribute: Optional[str] = None  # For leaf nodes
    value: Any = None  # For leaf nodes
    variable: Optional[str] = None  # For FORALL/EXISTS nodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "operator": self.operator.value,
            "label": self.label,
            "position": self.position,
            "children": self.children,
            "parent": self.parent,
            "attribute": self.attribute,
            "value": self.value,
            "variable": self.variable
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeNode':
        """Deserialize from dictionary."""
        return cls(
            node_id=data["node_id"],
            operator=LogicalOperator(data["operator"]),
            label=data["label"],
            position=tuple(data["position"]),
            children=data.get("children", []),
            parent=data.get("parent"),
            attribute=data.get("attribute"),
            value=data.get("value"),
            variable=data.get("variable")
        )


class DecisionTreeGraphicsItem(QGraphicsRectItem):
    """Graphics item representing a decision tree node."""
    
    def __init__(self, node: TreeNode, scene=None, parent=None):
        super().__init__(parent)
        self.node = node
        self.scene = scene
        self.setRect(QRectF(-50, -25, 100, 50))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        # Set color based on operator type
        self.setBrush(self._get_color_for_operator(node.operator))
        self.setPen(QPen(QColor(0, 0, 0), 2))
        
        # Add text label
        self.text_item = QGraphicsTextItem(node.label, self)
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        self.text_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        rect = self.boundingRect()
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(
            rect.center().x() - text_rect.width() / 2,
            rect.center().y() - text_rect.height() / 2
        )
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if self.scene:
            self.scene.node_selected.emit(self.node)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle mouse double click events."""
        if self.scene:
            self.scene.node_double_clicked.emit(self.node)
        super().mouseDoubleClickEvent(event)
    
    def _get_color_for_operator(self, op: LogicalOperator) -> QBrush:
        """Get color for operator type."""
        colors = {
            LogicalOperator.AND: QColor(100, 150, 200),
            LogicalOperator.OR: QColor(150, 200, 100),
            LogicalOperator.NOT: QColor(200, 100, 100),
            LogicalOperator.XAND: QColor(200, 150, 100),
            LogicalOperator.NAND: QColor(200, 100, 150),
            LogicalOperator.FORALL: QColor(150, 100, 200),
            LogicalOperator.EXISTS: QColor(100, 200, 150),
            LogicalOperator.IMPLIES: QColor(200, 200, 100),
            LogicalOperator.IFF: QColor(150, 150, 200),
            LogicalOperator.LEAF: QColor(100, 100, 100)
        }
        return QBrush(colors.get(op, QColor(128, 128, 128)))
    
    def update_label(self, label: str):
        """Update the label text."""
        self.node.label = label
        self.text_item.setPlainText(label)
        rect = self.boundingRect()
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(
            rect.center().x() - text_rect.width() / 2,
            rect.center().y() - text_rect.height() / 2
        )
    
    def itemChange(self, change, value):
        """Handle item changes (e.g., position updates)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = self.pos()
            self.node.position = (pos.x(), pos.y())
        return super().itemChange(change, value)


class DecisionTreeScene(QGraphicsScene):
    """Graphics scene for the decision tree designer."""
    
    node_selected = pyqtSignal(TreeNode)
    node_double_clicked = pyqtSignal(TreeNode)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree_nodes: Dict[str, TreeNode] = {}
        self.graphics_items: Dict[str, DecisionTreeGraphicsItem] = {}
        self.connection_lines: List[QGraphicsItem] = []
    
    def add_node(self, node: TreeNode) -> DecisionTreeGraphicsItem:
        """Add a node to the scene."""
        self.tree_nodes[node.node_id] = node
        
        item = DecisionTreeGraphicsItem(node, scene=self)
        item.setPos(*node.position)
        self.addItem(item)
        self.graphics_items[node.node_id] = item
        
        self.update_connections()
        return item
    
    def remove_node(self, node_id: str):
        """Remove a node from the scene."""
        if node_id in self.graphics_items:
            item = self.graphics_items[node_id]
            self.removeItem(item)
            del self.graphics_items[node_id]
        
        if node_id in self.tree_nodes:
            node = self.tree_nodes[node_id]
            # Remove from parent's children
            if node.parent and node.parent in self.tree_nodes:
                parent_node = self.tree_nodes[node.parent]
                if node_id in parent_node.children:
                    parent_node.children.remove(node_id)
            # Remove children references
            for child_id in node.children:
                if child_id in self.tree_nodes:
                    self.tree_nodes[child_id].parent = None
            del self.tree_nodes[node_id]
        
        self.update_connections()
    
    def connect_nodes(self, parent_id: str, child_id: str):
        """Connect two nodes (parent -> child)."""
        if parent_id not in self.tree_nodes or child_id not in self.tree_nodes:
            return
        
        parent_node = self.tree_nodes[parent_id]
        child_node = self.tree_nodes[child_id]
        
        if child_id not in parent_node.children:
            parent_node.children.append(child_id)
        child_node.parent = parent_id
        
        self.update_connections()
    
    def update_connections(self):
        """Update connection lines between nodes."""
        # Remove old connection lines
        for line in self.connection_lines:
            self.removeItem(line)
        self.connection_lines.clear()
        
        # Draw new connection lines
        for node_id, node in self.tree_nodes.items():
            if node.parent and node.parent in self.graphics_items:
                parent_item = self.graphics_items[node.parent]
                child_item = self.graphics_items[node_id]
                
                # Create line from parent to child
                from PyQt6.QtWidgets import QGraphicsLineItem
                from PyQt6.QtCore import QLineF
                
                parent_rect = parent_item.boundingRect()
                child_rect = child_item.boundingRect()
                
                parent_pos = parent_item.pos()
                child_pos = child_item.pos()
                
                # Line from bottom of parent to top of child
                start_point = QPointF(parent_pos.x(), parent_pos.y() + parent_rect.height() / 2)
                end_point = QPointF(child_pos.x(), child_pos.y() - child_rect.height() / 2)
                
                line = QGraphicsLineItem(QLineF(start_point, end_point))
                line.setPen(QPen(QColor(100, 100, 100), 2))
                self.addItem(line)
                self.connection_lines.append(line)
    
    def get_tree_root(self) -> Optional[TreeNode]:
        """Get the root node of the tree (node without parent)."""
        for node in self.tree_nodes.values():
            if node.parent is None:
                return node
        return None
    
    def to_fol_formula(self) -> Optional[str]:
        """Convert the decision tree to an FOL formula string."""
        root = self.get_tree_root()
        if not root:
            return None
        
        return self._node_to_formula(root)
    
    def _node_to_formula(self, node: TreeNode) -> str:
        """Convert a node and its children to an FOL formula."""
        if node.operator == LogicalOperator.LEAF:
            # Leaf node: attribute = value
            if node.attribute and node.value is not None:
                return f"{node.attribute} = {node.value}"
            return node.label
        
        # Get child formulas
        child_formulas = []
        for child_id in node.children:
            if child_id in self.tree_nodes:
                child_formula = self._node_to_formula(self.tree_nodes[child_id])
                child_formulas.append(child_formula)
        
        if not child_formulas:
            return node.label
        
        # Build formula based on operator
        if node.operator == LogicalOperator.AND:
            return f"And({', '.join(child_formulas)})"
        elif node.operator == LogicalOperator.OR:
            return f"Or({', '.join(child_formulas)})"
        elif node.operator == LogicalOperator.NOT:
            if child_formulas:
                return f"Not({child_formulas[0]})"
            return f"Not({node.label})"
        elif node.operator == LogicalOperator.XAND:
            # Exclusive AND: (A and B) and Not(And(A, B) when both are true in same context)
            # Simplified: And(A, B) and Not(And(A, B)) -> contradiction, so we use a custom interpretation
            # For now, represent as: And(A, B) with special marker
            return f"XAnd({', '.join(child_formulas)})"
        elif node.operator == LogicalOperator.NAND:
            if child_formulas:
                return f"Not(And({', '.join(child_formulas)}))"
            return f"NAnd({node.label})"
        elif node.operator == LogicalOperator.FORALL:
            if node.variable and child_formulas:
                return f"ForAll({node.variable}, {child_formulas[0]})"
            return f"ForAll({node.label})"
        elif node.operator == LogicalOperator.EXISTS:
            if node.variable and child_formulas:
                return f"Exists({node.variable}, {child_formulas[0]})"
            return f"Exists({node.label})"
        elif node.operator == LogicalOperator.IMPLIES:
            if len(child_formulas) >= 2:
                return f"Implies({child_formulas[0]}, {child_formulas[1]})"
            return node.label
        elif node.operator == LogicalOperator.IFF:
            if len(child_formulas) >= 2:
                return f"Iff({child_formulas[0]}, {child_formulas[1]})"
            return node.label
        
        return node.label


class DecisionTreeDesignerWidget(QWidget):
    """Main widget for the decision tree designer."""
    
    tree_changed = pyqtSignal(str)  # Emits FOL formula when tree changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = DecisionTreeScene()
        self.next_node_id = 1
        
        self._setup_ui()
        
        # Connect signals
        self.scene.node_selected.connect(self._on_node_selected)
        self.scene.node_double_clicked.connect(self._on_node_double_clicked)
    
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        
        # Left panel: Toolbox
        toolbox = self._create_toolbox()
        layout.addWidget(toolbox)
        
        # Center: Graphics view
        view = QGraphicsView(self.scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        layout.addWidget(view, stretch=1)
        
        self.view = view
    
    def _create_toolbox(self) -> QWidget:
        """Create the toolbox panel."""
        toolbox = QGroupBox("Operations")
        layout = QVBoxLayout()
        
        # List of available operators
        self.operator_list = QListWidget()
        operators = [
            ("AND", LogicalOperator.AND),
            ("OR", LogicalOperator.OR),
            ("NOT", LogicalOperator.NOT),
            ("XAND", LogicalOperator.XAND),
            ("NAND", LogicalOperator.NAND),
            ("FORALL", LogicalOperator.FORALL),
            ("EXISTS", LogicalOperator.EXISTS),
            ("IMPLIES", LogicalOperator.IMPLIES),
            ("IFF", LogicalOperator.IFF),
            ("Leaf", LogicalOperator.LEAF)
        ]
        
        for name, op in operators:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, op)
            self.operator_list.addItem(item)
        
        layout.addWidget(self.operator_list)
        
        # Add node button
        add_btn = QPushButton("Add Node")
        add_btn.clicked.connect(self._add_node_from_selection)
        layout.addWidget(add_btn)
        
        # Remove node button
        remove_btn = QPushButton("Remove Node")
        remove_btn.clicked.connect(self._remove_selected_node)
        layout.addWidget(remove_btn)
        
        # Connect nodes button
        connect_btn = QPushButton("Connect Nodes")
        connect_btn.clicked.connect(self._connect_nodes)
        layout.addWidget(connect_btn)
        
        # Export formula button
        export_btn = QPushButton("Export FOL Formula")
        export_btn.clicked.connect(self._export_formula)
        layout.addWidget(export_btn)
        
        toolbox.setLayout(layout)
        return toolbox
    
    def _add_node_from_selection(self):
        """Add a node based on selected operator in toolbox."""
        items = self.operator_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Selection", "Please select an operator from the list.")
            return
        
        item = items[0]
        operator = item.data(Qt.ItemDataRole.UserRole)
        
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        
        node = TreeNode(
            node_id=node_id,
            operator=operator,
            label=operator.value,
            position=(100.0, 100.0)
        )
        
        self.scene.add_node(node)
        self.tree_changed.emit(self.scene.to_fol_formula() or "")
    
    def _remove_selected_node(self):
        """Remove the selected node."""
        selected_items = self.scene.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a node to remove.")
            return
        
        for item in selected_items:
            if isinstance(item, DecisionTreeGraphicsItem):
                self.scene.remove_node(item.node.node_id)
        
        self.tree_changed.emit(self.scene.to_fol_formula() or "")
    
    def _connect_nodes(self):
        """Connect two selected nodes."""
        selected_items = self.scene.selectedItems()
        if len(selected_items) != 2:
            QMessageBox.warning(self, "Invalid Selection", "Please select exactly two nodes to connect.")
            return
        
        items = [item for item in selected_items if isinstance(item, DecisionTreeGraphicsItem)]
        if len(items) != 2:
            return
        
        parent_item = items[0]
        child_item = items[1]
        
        self.scene.connect_nodes(parent_item.node.node_id, child_item.node.node_id)
        self.tree_changed.emit(self.scene.to_fol_formula() or "")
    
    def _export_formula(self):
        """Export the tree as an FOL formula."""
        formula = self.scene.to_fol_formula()
        if formula:
            QMessageBox.information(self, "FOL Formula", f"Formula:\n\n{formula}")
        else:
            QMessageBox.warning(self, "No Tree", "No decision tree has been created.")
    
    def _on_node_selected(self, node: TreeNode):
        """Handle node selection."""
        pass  # Could show properties panel
    
    def _on_node_double_clicked(self, node: TreeNode):
        """Handle node double click (edit properties)."""
        dialog = NodeEditDialog(node, self)
        if dialog.exec():
            updated_node = dialog.get_node()
            # Update node in scene
            if node.node_id in self.scene.tree_nodes:
                self.scene.tree_nodes[node.node_id] = updated_node
                if node.node_id in self.scene.graphics_items:
                    self.scene.graphics_items[node.node_id].update_label(updated_node.label)
            self.tree_changed.emit(self.scene.to_fol_formula() or "")


class NodeEditDialog(QDialog):
    """Dialog for editing node properties."""
    
    def __init__(self, node: TreeNode, parent=None):
        super().__init__(parent)
        self.node = node
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Edit Node")
        layout = QFormLayout(self)
        
        # Label
        self.label_edit = QLineEdit(self.node.label)
        layout.addRow("Label:", self.label_edit)
        
        # Variable (for FORALL/EXISTS)
        if self.node.operator in (LogicalOperator.FORALL, LogicalOperator.EXISTS):
            self.variable_edit = QLineEdit(self.node.variable or "x")
            layout.addRow("Variable:", self.variable_edit)
        
        # Attribute and value (for LEAF)
        if self.node.operator == LogicalOperator.LEAF:
            self.attribute_edit = QLineEdit(self.node.attribute or "")
            layout.addRow("Attribute:", self.attribute_edit)
            
            self.value_edit = QLineEdit(str(self.node.value) if self.node.value is not None else "")
            layout.addRow("Value:", self.value_edit)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_node(self) -> TreeNode:
        """Get the updated node."""
        self.node.label = self.label_edit.text()
        
        if self.node.operator in (LogicalOperator.FORALL, LogicalOperator.EXISTS):
            if hasattr(self, 'variable_edit'):
                self.node.variable = self.variable_edit.text()
        
        if self.node.operator == LogicalOperator.LEAF:
            if hasattr(self, 'attribute_edit'):
                self.node.attribute = self.attribute_edit.text()
            if hasattr(self, 'value_edit'):
                try:
                    self.node.value = eval(self.value_edit.text())
                except:
                    self.node.value = self.value_edit.text()
        
        return self.node
