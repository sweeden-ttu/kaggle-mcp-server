"""
Autocomplete Widget for Formula Editor

Provides dropdown suggestions as user types, with click tracking and preference learning.
"""

from typing import Optional, List, Callable
from PyQt6.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

from .autocomplete import SuggestionEngine, Suggestion


class AutocompleteListWidget(QListWidget):
    """List widget for displaying autocomplete suggestions."""
    
    suggestion_selected = pyqtSignal(str)  # Emitted when user selects a suggestion
    suggestion_clicked = pyqtSignal(str)  # Emitted when user clicks a suggestion
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMaximumHeight(200)
        self.setMinimumWidth(300)
        
        # Style the dropdown
        self.setFrameShape(QFrame.Shape.Box)
        self.setFrameShadow(QFrame.Shadow.Plain)
        
        # Connect signals
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
    
    def populate(self, suggestions: List[Suggestion]):
        """Populate the list with suggestions."""
        self.clear()
        
        for suggestion in suggestions:
            item = QListWidgetItem(suggestion.display)
            item.setData(Qt.ItemDataRole.UserRole, suggestion.text)
            
            # Color code by category
            if suggestion.category == "Learned":
                item.setForeground(QColor(0, 128, 0))  # Green for learned
            elif suggestion.category == "Pattern":
                item.setForeground(QColor(0, 0, 255))  # Blue for patterns
            elif suggestion.category == "Keyword":
                item.setForeground(QColor(128, 0, 128))  # Purple for keywords
            
            # Show confidence as tooltip
            item.setToolTip(
                f"{suggestion.category} (confidence: {suggestion.confidence:.2f}, "
                f"used: {suggestion.usage_count} times)"
            )
            
            self.addItem(item)
        
        # Select first item
        if self.count() > 0:
            self.setCurrentRow(0)
    
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        text = item.data(Qt.ItemDataRole.UserRole)
        if text:
            self.suggestion_clicked.emit(text)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double-click (select)."""
        text = item.data(Qt.ItemDataRole.UserRole)
        if text:
            self.suggestion_selected.emit(text)
            self.hide()


class AutocompleteOverlay(QWidget):
    """Overlay widget that shows autocomplete suggestions."""
    
    suggestion_selected = pyqtSignal(str)
    suggestion_clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.list_widget = AutocompleteListWidget(self)
        self.list_widget.suggestion_selected.connect(self.suggestion_selected.emit)
        self.list_widget.suggestion_clicked.connect(self.suggestion_clicked.emit)
        
        layout.addWidget(self.list_widget)
        
        self.hide()
    
    def show_suggestions(self, suggestions: List[Suggestion], position=None):
        """Show suggestions at the given position."""
        if not suggestions:
            self.hide()
            return
        
        self.list_widget.populate(suggestions)
        
        if position:
            self.move(position)
        
        self.show()
        self.raise_()
    
    def hide_suggestions(self):
        """Hide the autocomplete overlay."""
        self.hide()


class AutocompleteManager:
    """Manages autocomplete functionality for a text editor."""
    
    def __init__(
        self,
        editor_widget,
        suggestion_engine: SuggestionEngine,
        on_suggestion_selected: Optional[Callable[[str], None]] = None
    ):
        self.editor = editor_widget
        self.engine = suggestion_engine
        self.on_suggestion_selected = on_suggestion_selected
        
        # Create overlay
        self.overlay = AutocompleteOverlay(self.editor)
        
        # Connect signals
        self.overlay.suggestion_selected.connect(self._handle_suggestion_selected)
        self.overlay.suggestion_clicked.connect(self._handle_suggestion_clicked)
        
        # Timer for debouncing
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_suggestions)
        
        # Track text changes
        if hasattr(self.editor, 'textChanged'):
            self.editor.textChanged.connect(self._on_text_changed)
        elif hasattr(self.editor, 'textChangedSignal'):
            self.editor.textChangedSignal.connect(self._on_text_changed)
    
    def _on_text_changed(self):
        """Handle text change in editor."""
        # Debounce updates
        self.update_timer.stop()
        self.update_timer.start(150)  # 150ms delay
    
    def _update_suggestions(self):
        """Update suggestions based on current text."""
        current_text = self._get_current_text()
        cursor_pos = self._get_cursor_position()
        
        # Get text before cursor
        text_before_cursor = current_text[:cursor_pos]
        
        # Check if we should auto-submit (every 3 characters)
        if self.engine.should_auto_submit(text_before_cursor):
            self._auto_submit_suggestions(text_before_cursor)
        
        # Generate suggestions
        suggestions = self.engine.generate_suggestions(text_before_cursor)
        
        if suggestions:
            # Calculate position for dropdown
            position = self._calculate_dropdown_position()
            self.overlay.show_suggestions(suggestions, position)
        else:
            self.overlay.hide_suggestions()
    
    def _get_current_text(self) -> str:
        """Get current text from editor."""
        if hasattr(self.editor, 'toPlainText'):
            return self.editor.toPlainText()
        elif hasattr(self.editor, 'text'):
            return self.editor.text()
        return ""
    
    def _get_cursor_position(self) -> int:
        """Get current cursor position."""
        if hasattr(self.editor, 'textCursor'):
            cursor = self.editor.textCursor()
            return cursor.position()
        return len(self._get_current_text())
    
    def _calculate_dropdown_position(self):
        """Calculate position for dropdown relative to editor."""
        if hasattr(self.editor, 'mapToGlobal'):
            editor_pos = self.editor.mapToGlobal(self.editor.rect().topLeft())
            cursor_rect = self._get_cursor_rect()
            if cursor_rect:
                cursor_pos = self.editor.mapToGlobal(cursor_rect.bottomLeft())
                return cursor_pos
            return editor_pos
        return self.editor.pos()
    
    def _get_cursor_rect(self):
        """Get rectangle for cursor position."""
        if hasattr(self.editor, 'cursorRect'):
            return self.editor.cursorRect()
        return None
    
    def _auto_submit_suggestions(self, text: str):
        """Auto-submit suggestions when character count is divisible by 3."""
        suggestions = self.engine.generate_suggestions(text, max_suggestions=5)
        if suggestions:
            position = self._calculate_dropdown_position()
            self.overlay.show_suggestions(suggestions, position)
    
    def _handle_suggestion_selected(self, text: str):
        """Handle when user selects a suggestion."""
        # Track usage
        self.engine.track_usage(text)
        
        # Insert text at cursor
        self._insert_text(text)
        
        # Hide overlay
        self.overlay.hide_suggestions()
        
        # Call callback if provided
        if self.on_suggestion_selected:
            self.on_suggestion_selected(text)
    
    def _handle_suggestion_clicked(self, text: str):
        """Handle when user clicks a suggestion (feedback)."""
        # Track click
        self.engine.track_click(text, "click")
    
    def _insert_text(self, text: str):
        """Insert text at cursor position."""
        if hasattr(self.editor, 'textCursor'):
            cursor = self.editor.textCursor()
            cursor.insertText(text)
        elif hasattr(self.editor, 'insert'):
            self.editor.insert(text)
    
    def hide_suggestions(self):
        """Hide autocomplete suggestions."""
        self.overlay.hide_suggestions()
    
    def handle_key_event(self, event):
        """Handle key events for navigation."""
        if self.overlay.isVisible():
            if event.key() == Qt.Key.Key_Down:
                self.overlay.list_widget.setCurrentRow(
                    min(self.overlay.list_widget.currentRow() + 1,
                        self.overlay.list_widget.count() - 1)
                )
                return True
            elif event.key() == Qt.Key.Key_Up:
                self.overlay.list_widget.setCurrentRow(
                    max(self.overlay.list_widget.currentRow() - 1, 0)
                )
                return True
            elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                current_item = self.overlay.list_widget.currentItem()
                if current_item:
                    text = current_item.data(Qt.ItemDataRole.UserRole)
                    if text:
                        self._handle_suggestion_selected(text)
                return True
            elif event.key() == Qt.Key.Key_Escape:
                self.overlay.hide_suggestions()
                return True
        
        return False
