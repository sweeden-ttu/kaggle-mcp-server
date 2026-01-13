# IKYKE Protocol Specification

**IKYKE** (Interactive Knowledge Yielding Kernel Engine) is an automated workflow protocol for the FOL Workbench that manages complete experiment lifecycles.

## Overview

The IKYKE protocol automates the following workflow:

1. **Automatically Save** - Saves work at configurable intervals
2. **Run** - Executes experiments for 3-5 minutes (configurable)
3. **Stop** - Automatically stops when conditions are met
4. **Evaluate** - Analyzes collected results
5. **Query** - Executes queries on the data
6. **Analysis** - Performs comprehensive analysis

## File Format

### IKYKE Workflow File (`.ikyke`)

The workflow definition file contains the configuration for an automated experiment.

**Structure:**
```json
{
  "name": "My Experiment",
  "header": {
    "format_version": "1.0",
    "format_name": "IKYKE",
    "created": "2024-01-01T12:00:00",
    "workflow_id": "uuid-here",
    "description": "Description"
  },
  "auto_save": {
    "enabled": true,
    "mode": "interval",
    "interval_seconds": 30,
    "max_saves": 100
  },
  "run": {
    "duration_min": 3,
    "duration_max": 5,
    "max_formulas": 1000,
    "max_models": 100
  },
  "evaluation": {
    "evaluate_satisfiability": true,
    "evaluate_models": true,
    "metrics": ["satisfiability_rate", "model_count"]
  },
  "query": {
    "queries": ["Find all satisfiable formulas"],
    "max_results": 100
  },
  "analysis": {
    "analyze_patterns": true,
    "analyze_complexity": true,
    "generate_report": true
  },
  "formulas": ["And(x, y)", "Or(x, Not(y))"],
  "constraints": []
}
```

### IKYKE Container File (`.ikyke_container`)

The container file stores runtime data and results from workflow execution.

**Structure:**
```json
{
  "container_id": "uuid",
  "workflow_id": "uuid",
  "current_phase": "running",
  "formula_results": [
    {
      "formula": "And(x, y)",
      "satisfiable": true,
      "model": {"x": true, "y": true},
      "solve_time": 0.001
    }
  ],
  "evaluation_results": {
    "total_formulas": 100,
    "satisfiability_rate": 0.75
  },
  "query_results": [],
  "analysis_result": {}
}
```

## Protocol Phases

### 1. Initialization
- Loads workflow configuration
- Initializes logic engine
- Prepares formulas and constraints

### 2. Running (3-5 minutes)
- Continuously evaluates formulas
- Collects results
- Auto-saves at intervals
- Stops when:
  - Time limit reached
  - Max formulas processed
  - Max models found
  - Manual stop

### 3. Stopped
- Final state save
- Phase transition

### 4. Evaluation
- Calculates satisfiability rates
- Counts models
- Computes metrics
- Generates statistics

### 5. Query
- Executes configured queries
- Filters and searches results
- Returns query results

### 6. Analysis
- Pattern analysis
- Complexity metrics
- Correlation analysis
- Report generation

## Usage

### Creating a Workflow

```python
from fol_workbench.ikyke_format import IkykeFileFormat

# Create default workflow
workflow = IkykeFileFormat.create_default("My Experiment")

# Customize settings
workflow.run.duration_min = 3
workflow.run.duration_max = 5
workflow.auto_save.interval_seconds = 30

# Add formulas
workflow.formulas = ["And(x, y)", "Or(x, Not(y))"]

# Save workflow
IkykeFileFormat.save(workflow, "my_experiment.ikyke")
```

### Running a Workflow

```python
from fol_workbench.ikyke_protocol import IkykeProtocol
from fol_workbench.logic_layer import LogicEngine
from fol_workbench.data_layer import DataLayer

# Load workflow
workflow = IkykeFileFormat.load("my_experiment.ikyke")

# Create protocol engine
protocol = IkykeProtocol(
    workflow=workflow,
    logic_engine=LogicEngine(),
    data_layer=DataLayer()
)

# Start workflow
protocol.start()

# Workflow runs automatically for 3-5 minutes
# Then evaluates, queries, and analyzes
```

### Accessing Results

```python
# Get status
status = protocol.get_status()
print(f"Phase: {status['phase']}")
print(f"Formulas: {status['formula_count']}")

# Access container
container = protocol.container
print(f"Satisfiability rate: {container.evaluation_results.satisfiability_rate}")

# Access analysis
if container.analysis_result:
    print(container.analysis_result.report)
```

## Auto-Save Modes

### Interval Mode
Saves at fixed time intervals (default: 30 seconds)

### Event Mode
Saves on specific events:
- `formula_added`
- `model_found`
- `checkpoint`

### Continuous Mode
Saves continuously (high frequency)

### Manual Mode
Only saves on explicit save commands

## Configuration Options

### Run Configuration
- `duration_min`: Minimum run time (minutes)
- `duration_max`: Maximum run time (minutes)
- `max_formulas`: Maximum formulas to process
- `max_models`: Maximum models to find
- `stop_conditions`: List of stop condition triggers

### Evaluation Configuration
- `evaluate_satisfiability`: Calculate satisfiability rates
- `evaluate_models`: Count and analyze models
- `evaluate_implications`: Test implications
- `metrics`: List of metrics to compute
- `threshold_satisfiable`: Threshold for satisfiability

### Query Configuration
- `queries`: List of query strings
- `query_language`: Query language ("fol", "smt", "json")
- `max_results`: Maximum results per query

### Analysis Configuration
- `analyze_patterns`: Analyze formula patterns
- `analyze_complexity`: Compute complexity metrics
- `analyze_correlations`: Find correlations
- `generate_report`: Generate analysis report
- `report_format`: Report format ("json", "markdown", "html")

## Integration with FOL Workbench

The IKYKE protocol is integrated into the FOL Workbench UI:

1. **File → New IKYKE Workflow** - Create a new workflow
2. **File → Open IKYKE Workflow** - Load existing workflow
3. **File → Run IKYKE Workflow** - Execute the workflow

The workflow runs in the background and automatically progresses through all phases.

## Data Container Format

The container format separates workflow definition from runtime data:

- **Workflow File** (`.ikyke`): Configuration and definition
- **Container File** (`.ikyke_container`): Runtime data and results

This separation allows:
- Multiple runs of the same workflow
- Comparison of different runs
- Independent analysis of results
- Workflow reuse

## Example Workflow

```python
# Create workflow
workflow = IkykeFileFormat.create_default("Pattern Analysis")

# Configure for 5-minute run
workflow.run.duration_min = 5
workflow.run.duration_max = 5

# Set up queries
workflow.query.queries = [
    "Find all satisfiable formulas",
    "Find formulas with models",
    "Find complex formulas"
]

# Add test formulas
workflow.formulas = [
    "And(x, y)",
    "Or(x, Not(y))",
    "Implies(x, y)",
    "And(Or(x, y), Not(z))"
]

# Save and run
IkykeFileFormat.save(workflow, "pattern_analysis.ikyke")
```

## Benefits

1. **Automation**: No manual intervention needed
2. **Reproducibility**: Same workflow produces consistent results
3. **Scalability**: Process hundreds of formulas automatically
4. **Analysis**: Built-in evaluation and analysis
5. **Persistence**: Automatic saving prevents data loss
6. **Flexibility**: Configurable for different experiment types

## File Format Version

Current version: **1.0**

The format is designed to be:
- Human-readable (JSON)
- Extensible (add new fields)
- Backward-compatible (version tracking)
- Portable (platform-independent)

## Integration with Perceptron Puzzle Detection System

The IKYKE protocol can be extended to support perceptron-based puzzle detection workflows:

### Perceptron Integration

IKYKE containers now support perceptron data storage:
- **PerceptronData**: Stores perceptron weights, learning rates, confidence scores, and accuracy metrics
- **HerbrandData**: Stores Herbrand base ground instances, constants, predicates, and implications
- Both can be added to containers via `add_perceptron()` and `add_herbrand_base()` methods

### Suggested Workflow Extensions

1. **Screenshot Capture Phase**: Add screenshot capture as a pre-processing step
   ```json
   "preprocessing": {
     "screenshot_capture": {
       "enabled": true,
       "source": "torbrowser",
       "grid_segmentation": true
     }
   }
   ```

2. **Feature Extraction Phase**: Integrate feature extraction into evaluation
   ```json
   "evaluation": {
     "extract_features": true,
     "feature_types": ["text", "image", "question"],
     "use_perceptrons": true
   }
   ```

3. **Learning Pipeline Integration**: Connect IKYKE workflow to learning pipeline
   - Use IKYKE containers to store perceptron training data
   - Track learning episodes through IKYKE phases
   - Store confidence evaluations in evaluation_results

4. **Herbrand Base Evaluation**: Use Herbrand bases for implication evaluation
   - Extract ground instances from formulas
   - Evaluate implications using Herbrand base
   - Store results in herbrand_bases field

### Example: Perceptron Workflow

```python
from fol_workbench.ikyke_container import IkykeContainer, PerceptronData, HerbrandData
from fol_workbench.ikyke_format import IkykeFileFormat

# Create workflow with perceptron support
workflow = IkykeFileFormat.create_default("Perceptron Puzzle Detection")

# Create container
container = IkykeContainer.create(workflow.header.workflow_id)

# Add perceptron data
perceptron = PerceptronData(
    perceptron_id="perceptron_001",
    weights=[0.5, 0.3, 0.2],
    learning_rate=0.01,
    confidence=0.85,
    accuracy=0.92,
    created=datetime.now().isoformat()
)
container.add_perceptron(perceptron)

# Add Herbrand base data
herbrand = HerbrandData(
    herbrand_id="herbrand_001",
    ground_instances=["P(a)", "P(b)", "Q(a)"],
    constants=["a", "b"],
    predicates=["P", "Q"],
    implications=["Implies(P(a), Q(a))"],
    created=datetime.now().isoformat()
)
container.add_herbrand_base(herbrand)

# Save container
IkykeContainerFormat.save(container, "perceptron_workflow.ikyke_container")
```

### Recommendations

1. **Extend Workflow Phases**: Add new phases for perceptron-specific operations:
   - `PERCEPTRON_TRAINING`: Train perceptrons on extracted features
   - `CONFIDENCE_EVALUATION`: Evaluate perceptron confidence scores
   - `HERBRAND_EVALUATION`: Evaluate implications using Herbrand bases

2. **Metadata Extraction**: Use vocabulary metadata extraction to:
   - Identify predicates and constants automatically
   - Generate Herbrand bases from formulas
   - Extract implications for evaluation

3. **Feedback Integration**: Collect client feedback on:
   - Perceptron accuracy
   - Feature extraction quality
   - Herbrand base completeness
   - Implication evaluation results

4. **Autocomplete Enhancement**: After reverse simulation:
   - Generate autocomplete suggestions based on model formulas
   - Learn from successful formula patterns
   - Improve suggestion confidence based on usage

5. **Hilbert System Integration**: Use Hilbert base predicates (Not, And, Or, Implies) for:
   - Formula normalization
   - Implication evaluation
   - Herbrand base generation
   - Semantic model construction
