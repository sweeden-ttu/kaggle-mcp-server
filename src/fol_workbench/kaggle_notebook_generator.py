"""
Kaggle Notebook Generator for Reverse Simulation

Creates Kaggle notebooks that simulate outputs and guess the original inputs.
This implements a reverse engineering approach to model discovery.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

from .test_first_simulator import UnitModel, TestFirstSimulator
from .hypothesis_tester import HypothesisTester


@dataclass
class NotebookCell:
    """A cell in a Jupyter notebook."""
    cell_type: str  # "code" or "markdown"
    source: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_count: Optional[int] = None


class KaggleNotebookGenerator:
    """
    Generates Kaggle notebooks for reverse simulation.
    
    Creates notebooks that:
    - Simulate outputs from given inputs
    - Guess inputs from observed outputs
    - Test hypotheses about model structure
    - Visualize results
    """
    
    def __init__(self):
        """Initialize the notebook generator."""
        self.notebook_metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        }
    
    def generate_reverse_simulation_notebook(
        self,
        model: UnitModel,
        observed_outputs: List[Dict[str, Any]],
        simulator: Optional[TestFirstSimulator] = None,
        hypothesis_tester: Optional[HypothesisTester] = None
    ) -> Dict[str, Any]:
        """
        Generate a notebook that guesses inputs from outputs.
        
        Args:
            model: The model to reverse engineer
            observed_outputs: List of observed output patterns
            simulator: Optional simulator for testing
            hypothesis_tester: Optional hypothesis tester
        
        Returns:
            Notebook structure as dict (can be saved as .ipynb)
        """
        cells = []
        
        # Title cell
        cells.append(self._create_markdown_cell([
            "# Reverse Simulation: Guessing Inputs from Outputs\n",
            f"**Model:** {model.name}\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "\n",
            "This notebook simulates outputs and attempts to reverse-engineer ",
            "the original inputs that produced them."
        ]))
        
        # Imports cell
        cells.append(self._create_code_cell([
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from z3 import *\n",
            "from typing import Dict, List, Any, Optional\n",
            "import json\n",
            "\n",
            "# Set style\n",
            "sns.set_style('whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)"
        ]))
        
        # Model definition cell
        cells.append(self._create_markdown_cell([
            "## Model Definition\n",
            f"The model formula: `{model.formula}`"
        ]))
        
        cells.append(self._create_code_cell([
            "# Model variables\n",
            self._generate_variable_declarations(model),
            "\n",
            "# Model formula\n",
            f"model_formula = {self._format_formula_for_python(model.formula)}\n",
            "\n",
            "# Create solver\n",
            "solver = Solver()\n",
            "solver.add(model_formula)"
        ]))
        
        # Observed outputs cell
        cells.append(self._create_markdown_cell([
            "## Observed Outputs\n",
            "These are the outputs we observed and want to reverse-engineer:"
        ]))
        
        cells.append(self._create_code_cell([
            "observed_outputs = " + json.dumps(observed_outputs, indent=2)
        ]))
        
        # Reverse simulation cell
        cells.append(self._create_markdown_cell([
            "## Reverse Simulation\n",
            "Attempting to find inputs that produce the observed outputs:"
        ]))
        
        cells.append(self._create_code_cell([
            "def reverse_simulate(output_pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:\n",
            "    \"\"\"Find inputs that produce the given output pattern.\"\"\"\n",
            "    solver_local = Solver()\n",
            "    solver_local.add(model_formula)\n",
            "    \n",
            "    # Add output constraints\n",
            self._generate_output_constraints(model),
            "    \n",
            "    if solver_local.check() == sat:\n",
            "        model_result = solver_local.model()\n",
            "        inputs = {}\n",
            self._generate_input_extraction(model),
            "        return inputs\n",
            "    return None\n",
            "\n",
            "# Try to reverse engineer each observed output\n",
            "guessed_inputs = []\n",
            "for i, output in enumerate(observed_outputs):\n",
            "    print(f\"\\nAnalyzing output {i+1}...\")\n",
            "    inputs = reverse_simulate(output)\n",
            "    if inputs:\n",
            "        print(f\"  ✓ Found inputs: {inputs}\")\n",
            "        guessed_inputs.append({\"output\": output, \"guessed_input\": inputs})\n",
            "    else:\n",
            "        print(f\"  ✗ Could not find inputs for this output\")\n",
            "        guessed_inputs.append({\"output\": output, \"guessed_input\": None})"
        ]))
        
        # Hypothesis testing cell
        if hypothesis_tester:
            cells.append(self._create_markdown_cell([
                "## Hypothesis Testing\n",
                "Testing hypotheses about the model structure:"
            ]))
            
            cells.append(self._create_code_cell([
                "# Interactive hypothesis testing\n",
                "def ask_user_feedback(message: str) -> str:\n",
                "    \"\"\"Ask user if we're getting warmer.\"\"\"\n",
                "    print(message)\n",
                "    response = input(\"Are we getting warmer? (yes/no): \")\n",
                "    return response.strip().lower()\n",
                "\n",
                "# This would integrate with HypothesisTester\n",
                "# For now, we'll do basic validation\n",
                "print(\"Hypothesis testing would be performed here...\")"
            ]))
        
        # Visualization cell
        cells.append(self._create_markdown_cell([
            "## Results Visualization\n",
            "Visualizing the reverse simulation results:"
        ]))
        
        cells.append(self._create_code_cell([
            "# Create results DataFrame\n",
            "results_df = pd.DataFrame(guessed_inputs)\n",
            "\n",
            "# Visualize\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Plot 1: Success rate\n",
            "success_count = sum(1 for r in guessed_inputs if r['guessed_input'] is not None)\n",
            "total_count = len(guessed_inputs)\n",
            "axes[0].bar(['Success', 'Failure'], [success_count, total_count - success_count])\n",
            "axes[0].set_title('Reverse Simulation Success Rate')\n",
            "axes[0].set_ylabel('Count')\n",
            "\n",
            "# Plot 2: Input distribution (if we have successful guesses)\n",
            "if success_count > 0:\n",
            "    # This would show distribution of guessed inputs\n",
            "    axes[1].text(0.5, 0.5, 'Input distribution visualization\\n(implementation depends on input types)',\n",
            "                ha='center', va='center', transform=axes[1].transAxes)\n",
            "    axes[1].set_title('Guessed Input Distribution')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\nSuccessfully reverse-engineered {success_count}/{total_count} outputs\")"
        ]))
        
        # Summary cell
        cells.append(self._create_markdown_cell([
            "## Summary\n",
            "This notebook demonstrates reverse simulation:",
            "- Given observed outputs, we attempt to find the inputs that produced them",
            "- Uses constraint solving to find satisfying assignments",
            "- Tests hypotheses about model structure",
            "- Visualizes results"
        ]))
        
        # Create notebook structure
        notebook = {
            "cells": [self._cell_to_dict(cell) for cell in cells],
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
    
    def save_notebook(
        self,
        notebook: Dict[str, Any],
        path: Path
    ) -> Path:
        """Save notebook to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        return path
    
    def _create_markdown_cell(self, source: List[str]) -> NotebookCell:
        """Create a markdown cell."""
        return NotebookCell(
            cell_type="markdown",
            source=source,
            metadata={}
        )
    
    def _create_code_cell(
        self,
        source: List[str],
        execution_count: Optional[int] = None
    ) -> NotebookCell:
        """Create a code cell."""
        return NotebookCell(
            cell_type="code",
            source=source,
            metadata={},
            execution_count=execution_count
        )
    
    def _cell_to_dict(self, cell: NotebookCell) -> Dict[str, Any]:
        """Convert cell to notebook dict format."""
        result = {
            "cell_type": cell.cell_type,
            "metadata": cell.metadata,
            "source": cell.source
        }
        
        if cell.cell_type == "code":
            result["execution_count"] = cell.execution_count
            result["outputs"] = cell.outputs
        
        return result
    
    def _generate_variable_declarations(self, model: UnitModel) -> str:
        """Generate variable declarations for the model."""
        lines = []
        for var_name, var_type in model.variables.items():
            if var_type == "Bool":
                lines.append(f"{var_name} = Bool('{var_name}')")
            elif var_type == "Int":
                lines.append(f"{var_name} = Int('{var_name}')")
            elif var_type == "Real":
                lines.append(f"{var_name} = Real('{var_name}')")
        
        return "\n".join(lines) if lines else "# No variables declared"
    
    def _format_formula_for_python(self, formula: str) -> str:
        """Convert formula to Python/Z3 format."""
        # This is a simplified version - in practice, you'd need a proper parser
        # For now, we'll assume the formula is already in a compatible format
        return f'"{formula}"'  # Wrap in quotes as a string literal
    
    def _generate_output_constraints(self, model: UnitModel) -> str:
        """Generate code for output constraints."""
        return """    # Add constraints based on output pattern
    # This would be customized based on the model structure
    for key, value in output_pattern.items():
        if key in model_variables:
            solver_local.add(model_variables[key] == value)"""
    
    def _generate_input_extraction(self, model: UnitModel) -> str:
        """Generate code for extracting inputs from model."""
        lines = []
        for var_name in model.variables.keys():
            lines.append(f"        inputs['{var_name}'] = model_result[{var_name}]")
        return "\n".join(lines) if lines else "        pass"
    
    def generate_simulation_notebook(
        self,
        model: UnitModel,
        test_cases: List[Dict[str, Any]],
        simulator: Optional[TestFirstSimulator] = None
    ) -> Dict[str, Any]:
        """
        Generate a notebook that simulates outputs from inputs.
        
        This is the forward direction: given inputs, what outputs do we get?
        """
        cells = []
        
        # Title
        cells.append(self._create_markdown_cell([
            "# Forward Simulation: Predicting Outputs from Inputs\n",
            f"**Model:** {model.name}\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]))
        
        # Imports
        cells.append(self._create_code_cell([
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from z3 import *\n",
            "import json"
        ]))
        
        # Model and test cases
        cells.append(self._create_code_cell([
            "# Model definition\n",
            self._generate_variable_declarations(model),
            f"\nmodel_formula = {self._format_formula_for_python(model.formula)}\n",
            "\n",
            "# Test cases\n",
            f"test_cases = {json.dumps(test_cases, indent=2)}"
        ]))
        
        # Simulation
        cells.append(self._create_code_cell([
            "def simulate(input_data: Dict[str, Any]) -> Dict[str, Any]:\n",
            "    \"\"\"Simulate output from input.\"\"\"\n",
            "    solver = Solver()\n",
            "    solver.add(model_formula)\n",
            "    \n",
            "    # Add input constraints\n",
            "    for key, value in input_data.items():\n",
            "        if key in model_variables:\n",
            "            solver.add(model_variables[key] == value)\n",
            "    \n",
            "    if solver.check() == sat:\n",
            "        model_result = solver.model()\n",
            "        output = {}\n",
            "        for var in model_variables:\n",
            "            output[var] = model_result[var]\n",
            "        return output\n",
            "    return None\n",
            "\n",
            "# Run simulations\n",
            "results = []\n",
            "for i, test_case in enumerate(test_cases):\n",
            "    output = simulate(test_case.get('input', {}))\n",
            "    results.append({\n",
            "        'test_case': i,\n",
            "        'input': test_case.get('input', {}),\n",
            "        'output': output\n",
            "    })\n",
            "    print(f\"Test {i+1}: {output}\")"
        ]))
        
        # Create notebook
        notebook = {
            "cells": [self._cell_to_dict(cell) for cell in cells],
            "metadata": self.notebook_metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
