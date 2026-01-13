# ML Strategy System - Bayesian Feature Extractors & Decision Trees

## Overview

This system implements a novel ML strategy using Bayesian feature extractors with layered class collections and a drag-and-drop decision tree designer with logical operations (XAND, NAND, FORALL, EXISTS). The system also includes an ultra-large language model for generating descriptive text about experimental observations.

## Components

### 1. Bayesian Feature Extractor (`bayesian_feature_extractor.py`)

A sophisticated feature extraction system that organizes classes into layers, where each layer imposes constraints on previous layers.

**Key Features:**
- **Layered Architecture**: Classes organized into layers (Layer 1, Layer 2, etc.)
- **Class Collections**: Each layer contains multiple classes with attributes
- **Attribute Types**: Numerical, Categorical, Boolean, Text, Logical
- **Bayesian Inference**: Prior and posterior distributions for attributes
- **Layer Constraints**: Each layer imposes constraints on previous layers
- **Vocabulary Universe**: Tracks all vocabulary across all layers
- **Learning Log**: Comprehensive logging system (starts with "BEGIN S LEARNING")

**Usage:**
```python
from fol_workbench.bayesian_feature_extractor import BayesianFeatureExtractor, AttributeType

extractor = BayesianFeatureExtractor()

# Create layers
layer1 = extractor.create_layer(1, "Base Layer")
layer2 = extractor.create_layer(2, "Feature Layer")

# Add classes to layers
extractor.add_class_to_layer(1, "BaseClass", [
    Attribute("attr1", AttributeType.NUMERICAL, value=10)
])

extractor.add_class_to_layer(2, "FeatureClass", parent_classes=["BaseClass"])

# Extract features
features = extractor.extract_features(data_dict, layer_id=2)
```

### 2. Decision Tree Designer (`decision_tree_designer.py`)

A drag-and-drop graphical interface for designing decision trees with advanced logical operations.

**Key Features:**
- **Drag-and-Drop Interface**: Visual node-based design
- **Logical Operators**:
  - AND, OR, NOT (standard)
  - **XAND** (Exclusive AND)
  - **NAND** (Not AND)
  - **FORALL** (Universal quantification: ∀)
  - **EXISTS** (Existential quantification: ∃)
  - IMPLIES, IFF
  - LEAF nodes (terminal nodes with attribute/value)
- **FOL Formula Export**: Converts decision tree to First-Order Logic formulas
- **Visual Representation**: Color-coded nodes based on operator type
- **Node Connection**: Connect parent and child nodes
- **Validation**: Validate decision trees using Z3 solver

**Usage:**
```python
from fol_workbench.decision_tree_designer import DecisionTreeDesignerWidget

# Create designer widget (typically used in dialog)
designer = DecisionTreeDesignerWidget()

# Tree can be exported as FOL formula
formula = designer.scene.to_fol_formula()
```

### 3. Ultra-Large Language Model (`llm_observation_generator.py`)

Generates descriptive pretrained text that explains observed experimental data.

**Key Features:**
- **Observation Recording**: Records experimental observations with timestamps
- **Text Generation**: Generates descriptive text using template-based generation
- **Pretrained Text**: Creates comprehensive pretrained text from all observations
- **Vocabulary Integration**: Uses vocabulary from Bayesian feature extractor
- **Statistics**: Tracks corpus statistics (word count, character count, etc.)

**Usage:**
```python
from fol_workbench.llm_observation_generator import UltraLargeLanguageModel, Observation

llm = UltraLargeLanguageModel()

# Record observation
obs = Observation(
    observation_id="obs1",
    timestamp="2024-01-01T00:00:00",
    layer_id=1,
    class_name="MyClass",
    attributes={"attr1": "value1"},
    confidence_scores={"attr1": 0.85}
)

llm.record_observation(obs)

# Generate pretrained text
text = llm.generate_pretrained_text()
```

### 4. ML Strategy Integration (`ml_strategy_integration.py`)

Unified system that integrates all components.

**Key Features:**
- Integrates Bayesian Feature Extractor, Decision Tree Designer, and LLM
- Vocabulary synchronization between systems
- Experimental setup creation
- Observation generation and reporting
- System state saving/loading
- Comprehensive statistics

### 5. ML Strategy Dialog (`ml_strategy_dialog.py`)

Main user interface dialog with tabbed interface:

1. **Bayesian Feature Extractor Tab**: Manage layers, classes, and attributes
2. **Decision Tree Designer Tab**: Visual drag-and-drop decision tree design
3. **Observations & Reports Tab**: View generated observations and reports
4. **System Status & Logs Tab**: View learning log and system statistics

## Integration with FOL Workbench

The ML Strategy System is integrated into the FOL Workbench main menu:

- **Menu**: Tools → ML Strategy System
- **Shortcut**: Ctrl+M
- **Dialog**: Opens the comprehensive ML Strategy Dialog

## Learning Log

The system maintains a learning log that begins with:

```
[timestamp] BEGIN S LEARNING
```

All operations are logged, including:
- Layer creation
- Class additions
- Attribute assignments
- Feature extractions
- Observations

## Vocabulary Universe

The system maintains a universe of vocabularies that includes:
- All class names across all layers
- All attribute names
- All vocabulary terms used in observations

This vocabulary universe is shared between the Bayesian Feature Extractor and the Ultra-Large Language Model.

## Decision Tree Logical Operations

### XAND (Exclusive AND)
A specialized AND operation where the conditions must be true but not simultaneously true in the same context. Represented as `XAnd(A, B)` in FOL formulas.

### NAND (Not AND)
The negation of AND operation: `Not(And(A, B))` or `NAnd(A, B)`.

### FORALL (Universal Quantification)
Universal quantification: `ForAll(x, P(x))` - "for all x, P(x) holds".

### EXISTS (Existential Quantification)
Existential quantification: `Exists(x, P(x))` - "there exists x such that P(x) holds".

## File Structure

```
src/fol_workbench/
├── bayesian_feature_extractor.py      # Bayesian feature extraction with layers
├── decision_tree_designer.py          # Drag-and-drop decision tree designer
├── llm_observation_generator.py       # Ultra-large language model for observations
├── ml_strategy_integration.py         # Unified integration system
└── ml_strategy_dialog.py              # Main UI dialog
```

## Example Workflow

1. **Create Layers and Classes**:
   - Open ML Strategy System (Tools → ML Strategy System)
   - Go to "Bayesian Feature Extractor" tab
   - Add layers (e.g., Layer 1: "Base", Layer 2: "Features")
   - Add classes to each layer
   - Assign attributes to classes

2. **Design Decision Tree**:
   - Go to "Decision Tree Designer" tab
   - Select operators from toolbox (AND, OR, XAND, NAND, FORALL, EXISTS, etc.)
   - Add nodes to the canvas
   - Connect nodes (parent → child)
   - Double-click nodes to edit properties
   - Export as FOL formula

3. **Generate Observations**:
   - Extract features from data
   - System automatically generates observations
   - View observations in "Observations & Reports" tab
   - Generate comprehensive reports

4. **View Learning Log**:
   - Go to "System Status & Logs" tab
   - View the learning log (starts with "BEGIN S LEARNING")
   - View system statistics

## Saving and Loading

The system state can be saved and loaded:
- **Save**: Saves feature extractor state, pretrained text, and learning log
- **Load**: Restores all saved components

## Technical Notes

- The system integrates with Z3 solver for decision tree validation
- All components are designed to work together seamlessly
- Vocabulary is synchronized between Bayesian Feature Extractor and LLM
- The learning log provides comprehensive tracking of all operations
- Decision trees can be exported as FOL formulas for use in the main FOL Workbench

## Future Enhancements

Potential enhancements:
- More sophisticated Bayesian inference algorithms
- Advanced text generation models
- Real-time visualization of feature extraction
- Export decision trees to various formats
- Integration with external ML frameworks
- Support for more logical operations
