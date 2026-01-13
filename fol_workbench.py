#!/usr/bin/env python3
"""Launcher script for FOL Workbench."""

if __name__ == "__main__":
    # Import lazily so `import fol_workbench` doesn't eagerly import the full PyQt UI
    # (and so this file doesn't break non-GUI use-cases like library imports/tests).
    from src.fol_workbench.main import main
    main()
