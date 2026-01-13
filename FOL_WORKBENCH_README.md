# FOL Workbench

A professional desktop application for First-Order Logic (FOL) validation and model proposal, built with PyQt6 and Z3 Solver.

## Features

### Core Functionality
- **Formula Validation**: Check if FOL formulas are satisfiable, unsatisfiable, or unknown
- **Model Finding**: Automatically find satisfying models (counter-examples or solutions) for formulas
- **Implication Proof**: Prove that one formula implies another
- **Syntax Highlighting**: Color-coded formula editor with FOL keyword highlighting
- **Model Viewer**: Visual display of variable assignments in found models

### Project Management
- **Checkpoint System**: Save and restore work states with timestamps
- **Project Organization**: Create and manage multiple projects
- **Dataset Management**: Organize formulas and constraints into datasets

### File Format Support
- **SMT-LIB Import/Export**: Standard SMT-LIB 2.0 format support
- **JSON Project Files**: Human-readable project metadata
- **Checkpoint Persistence**: Automatic saving of work states

### User Interface
- **Dockable Panels**: Customizable workspace layout
- **Professional UI**: Modern, desktop-grade interface
- **Keyboard Shortcuts**: Fast access to common operations
- **Status Bar**: Real-time feedback on operations

## Architecture

The application follows a three-layer architecture:

### 1. UI Layer (`ui_layer.py`)
- PyQt6-based graphical interface
- Main window with dockable panels
- Formula editor with syntax highlighting
- Model viewer and checkpoint manager
- Menu bar, toolbar, and status bar

### 2. Logic Layer (`logic_layer.py`)
- Z3 Solver integration
- Formula parsing and validation
- Model extraction and interpretation
- SMT-LIB format support
- Implication proving

### 3. Data Layer (`data_layer.py`)
- Checkpoint persistence (JSON)
- Project metadata management
- SMT-LIB file I/O
- Dataset organization
- File system abstraction

## Installation

### Prerequisites
- Python 3.10 or higher
- PyQt6 (will be installed automatically)
- Z3 Solver (will be installed automatically)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the package:

```bash
pip install -e .
```

## Usage

### Launching the Application

```bash
python fol_workbench.py
```

Or:

```bash
python -m src.fol_workbench.main
```

### Basic Workflow

1. **Enter a Formula**: Type your FOL formula in the formula editor
   - Example: `And(x, Or(y, Not(z)))`
   - Supports: `And`, `Or`, `Not`, `Implies`, `Iff`, `ForAll`, `Exists`

2. **Add Constraints** (optional): Enter additional constraints in the constraints editor

3. **Validate**: Click "Validate" or press F5 to check satisfiability

4. **Find Model**: Click "Find Model" or press F6 to find a satisfying assignment

5. **Save Checkpoint**: Press Ctrl+S to save your current work state

### Keyboard Shortcuts

- `Ctrl+N`: New Project
- `Ctrl+O`: Open Project
- `Ctrl+S`: Save Checkpoint
- `Ctrl+Q`: Quit
- `F5`: Validate Formula
- `F6`: Find Model
- `F7`: Prove Implication

### Formula Syntax

The workbench uses Python-like syntax for formulas:

```python
# Basic operations
And(x, y)           # x ∧ y
Or(x, y)            # x ∨ y
Not(x)              # ¬x
Implies(x, y)       # x → y
Iff(x, y)           # x ↔ y

# Quantifiers
ForAll(x, P(x))     # ∀x P(x)
Exists(x, P(x))     # ∃x P(x)

# Examples
And(x, Or(y, Not(z)))
Implies(And(x, y), z)
ForAll(x, Implies(P(x), Q(x)))
```

### Variable Declaration

Variables are automatically declared when first used. You can also explicitly declare them:

- **Bool**: Boolean variables (default)
- **Int**: Integer variables
- **Real**: Real number variables
- **String**: String variables

### Projects and Checkpoints

**Creating a Project**:
1. File → New Project
2. Enter project name and description
3. Save checkpoints will be associated with the project

**Saving Checkpoints**:
- Automatically saves formula and constraints
- Includes timestamp and metadata
- Can be restored by double-clicking in the checkpoints panel

**Loading Checkpoints**:
- Double-click a checkpoint in the left panel
- Formula and constraints are restored

### SMT-LIB Format

**Import SMT-LIB**:
1. File → Import SMT-LIB
2. Select a `.smt2` file
3. Constraints are loaded into the solver

**Export SMT-LIB**:
1. File → Export SMT-LIB
2. Choose save location
3. Current constraints are exported in SMT-LIB 2.0 format

### Proving Implications

1. Tools → Prove Implication (or F7)
2. Enter the premise formula
3. Enter the conclusion formula
4. The tool checks if premise → conclusion
   - If unsatisfiable: Implication is valid
   - If satisfiable: Counterexample is shown

## Data Storage

All data is stored in `~/.fol_workbench/`:

```
~/.fol_workbench/
├── checkpoints/      # Saved checkpoints (JSON)
├── datasets/         # SMT-LIB and dataset files
└── projects/         # Project metadata (JSON)
```

## Examples

### Example 1: Simple Boolean Formula

```
Formula: And(x, Or(y, Not(x)))
```

**Validate**: Result will show if the formula is satisfiable.

**Find Model**: If satisfiable, shows assignments like:
- `x = True`
- `y = True`

### Example 2: Implication

```
Premise: And(x, y)
Conclusion: x
```

**Prove**: This will show that the implication is valid (premise → conclusion).

### Example 3: Quantified Formula

```
Formula: ForAll(x, Implies(P(x), Q(x)))
```

Requires declaring predicates `P` and `Q` as functions.

## Technical Details

### Z3 Integration

The logic engine uses Z3's Python API:
- `Solver()` for constraint solving
- Automatic model extraction
- SMT-LIB 2.0 compatibility
- Support for multiple theories (Bool, Int, Real, String)

### Error Handling

- Formula parsing errors are displayed in the results panel
- Invalid syntax shows helpful error messages
- SMT-LIB import errors are caught and reported

### Performance

- Z3 is optimized for industrial-strength solving
- Large formulas may take time (status bar shows progress)
- Statistics are displayed after validation

## Troubleshooting

### "z3-solver is not installed"

```bash
pip install z3-solver
```

### "PyQt6 is not installed"

```bash
pip install PyQt6
```

### Formula Not Parsing

- Check syntax matches Python/Z3 format
- Ensure variables are properly named
- Use parentheses for complex expressions

### Model Not Found

- Formula may be unsatisfiable
- Check constraints for contradictions
- Try simplifying the formula

## Development

### Project Structure

```
src/fol_workbench/
├── __init__.py          # Package initialization
├── main.py              # Application entry point
├── ui_layer.py          # PyQt6 GUI components
├── logic_layer.py       # Z3 solver integration
└── data_layer.py        # File I/O and persistence
```

### Extending the Application

**Adding New Formula Types**:
- Extend `FormulaHighlighter` in `ui_layer.py`
- Add parsing logic in `LogicEngine.parse_formula()`

**Custom SMT-LIB Features**:
- Extend `LogicEngine.from_smt_lib()` and `to_smt_lib()`
- Add support for additional SMT-LIB theories

**UI Customization**:
- Modify `MainWindow` in `ui_layer.py`
- Add new dock widgets or panels

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- **Z3 Solver**: Microsoft Research's SMT solver
- **PyQt6**: The Qt Company's Python bindings
- Built for researchers, students, and developers working with formal logic
