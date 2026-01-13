"""Main entry point for FOL Workbench application."""

import sys
from PyQt6.QtWidgets import QApplication
from .ui_layer import MainWindow


def main():
    """Run the FOL Workbench application."""
    app = QApplication(sys.argv)
    app.setApplicationName("FOL Workbench")
    app.setOrganizationName("FOL Workbench")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
