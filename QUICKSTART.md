# FOL Workbench - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install the package
pip install -e .
```

## Running the Application

```bash
# Option 1: Use the launcher script
python fol_workbench.py

# Option 2: Run as module
python -m src.fol_workbench.main
```

## First Steps

1. **Launch the application** - You'll see the main window with:
   - Formula editor (center)
   - Checkpoints panel (left)
   - Model viewer (right)

2. **Try a simple formula**:
   ```
   And(x, y)
   ```
   - Type this in the formula editor
   - Click "Validate" or press F5
   - Click "Find Model" or press F6 to see a satisfying assignment

3. **Save a checkpoint**:
   - Press Ctrl+S
   - Your work is saved with a timestamp
   - Double-click checkpoints in the left panel to restore them

4. **Create a project**:
   - File → New Project
   - Give it a name and description
   - Future checkpoints will be associated with this project

## Example Formulas

Try these in the formula editor:

```python
# Simple satisfiable formula
And(x, Or(y, Not(z)))

# Implication
Implies(And(x, y), z)

# Complex formula
And(Or(x, y), Implies(x, z), Not(And(y, z)))
```

## Keyboard Shortcuts

- `F5` - Validate formula
- `F6` - Find model
- `F7` - Prove implication
- `Ctrl+S` - Save checkpoint
- `Ctrl+N` - New project
- `Ctrl+O` - Open project

## Next Steps

- Read the full documentation in `FOL_WORKBENCH_README.md`
- Explore SMT-LIB import/export
- Try proving implications
- Organize your work with projects

## Troubleshooting

**Application won't start?**
- Ensure PyQt6 is installed: `pip install PyQt6`
- Ensure z3-solver is installed: `pip install z3-solver`

**Formula not parsing?**
- Check syntax matches Python/Z3 format
- Use proper parentheses
- Variable names should start with lowercase letters

**Need help?**
- Check the full README for detailed documentation
- Review example formulas in `src/fol_workbench/examples.py`
