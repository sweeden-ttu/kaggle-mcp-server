# Reverse Simulation System Guide

## Overview

This system implements a comprehensive test-first approach to model development with:
- **Test-First Simulator**: Create unit models and automata theories with test cases
- **Design Feedback**: Intelligent analysis of model design with first-principles feedback
- **Hypothesis Testing**: Interactive "getting warmer" feedback loop with backtracking
- **Kaggle Notebook Generation**: Create notebooks for reverse simulation (guessing inputs from outputs)

## Quick Start

### 1. Create a Test-First Model

```python
from fol_workbench.reverse_simulation_system import ReverseSimulationSystem

system = ReverseSimulationSystem()

# Create a model with test cases
model = system.create_model_with_tests(
    name="simple_and_model",
    formula="And(x, y)",
    variables={"x": "Bool", "y": "Bool"},
    test_cases=[
        {
            "name": "test_both_true",
            "input": {"x": True, "y": True},
            "expected_output": {"result": True},
            "constraints": []
        },
        {
            "name": "test_one_false",
            "input": {"x": True, "y": False},
            "expected_output": {"result": False},
            "constraints": []
        }
    ]
)
```

### 2. Analyze Design and Get Feedback

```python
# Get intelligent design feedback
feedback = system.analyze_and_get_feedback("simple_and_model")
print(json.dumps(feedback, indent=2))
```

This returns:
- **Feedback items**: Errors, warnings, suggestions with confidence scores
- **Key topics**: Extracted data topics and relationships
- **Proposals**: Specific change recommendations

### 3. Test Hypotheses with "Getting Warmer" Loop

```python
# Define a callback for user feedback
def ask_user(message):
    print(message)
    return input("Are we getting warmer? (yes/no): ").strip().lower()

# Test a hypothesis
result = system.test_hypothesis_with_feedback(
    hypothesis_description="The model correctly implements logical AND",
    model_name="simple_and_model",
    user_feedback_callback=ask_user
)
```

The system will:
1. Run tests on the model
2. Generate design feedback
3. Ask "Are we getting warmer?"
4. If "yes": Continue and increase confidence
5. If "no": Backtrack and propose alternative hypothesis

### 4. Run Complete Hypothesis Loop

```python
# Run the full loop with automatic backtracking
result = system.run_hypothesis_loop(
    hypothesis_description="Model implements correct logical behavior",
    model_name="simple_and_model",
    max_steps=10,
    user_feedback_callback=ask_user
)
```

This will:
- Test the hypothesis step by step
- Ask for feedback at each step
- Backtrack when user says "no"
- Generate alternative hypotheses
- Track the hypothesis tree

### 5. Generate Reverse Simulation Notebook

```python
# Generate a Kaggle notebook that guesses inputs from outputs
notebook_path = system.generate_reverse_simulation_notebook(
    model_name="simple_and_model",
    observed_outputs=[
        {"result": True},
        {"result": False},
        {"result": False}
    ],
    output_path=Path("reverse_simulation.ipynb")
)
```

The notebook will:
- Define the model
- Take observed outputs
- Use constraint solving to find inputs that produce those outputs
- Visualize results
- Test hypotheses interactively

### 6. Propose Next Steps

```python
# Get prioritized next steps
next_steps = system.propose_next_steps("simple_and_model")
print(json.dumps(next_steps, indent=2))
```

Returns prioritized actions like:
- Fix errors (high priority)
- Improve test coverage (high priority)
- Apply simplifications (medium priority)
- Generate notebook (medium priority)
- Test hypotheses (low priority)

## Using MCP Tools

The system is also available as MCP tools:

### Create Model
```python
create_test_first_model(
    name="my_model",
    formula="And(x, y)",
    variables={"x": "Bool", "y": "Bool"},
    test_cases=[...]
)
```

### Analyze Design
```python
analyze_model_design(model_name="my_model")
```

### Test Hypothesis
```python
test_hypothesis(
    hypothesis_description="Model is correct",
    model_name="my_model",
    user_response="yes"  # or "no"
)
```

### Generate Notebook
```python
generate_reverse_simulation_notebook(
    model_name="my_model",
    observed_outputs=[{"result": True}],
    output_path="notebook.ipynb"
)
```

### Propose Next Steps
```python
propose_next_steps(model_name="my_model")
```

### Run Hypothesis Loop
```python
run_hypothesis_loop(
    hypothesis_description="Model works correctly",
    model_name="my_model",
    max_steps=10,
    user_responses=["yes", "yes", "no", "yes"]  # Optional
)
```

## Workflow Example

### Complete Workflow

1. **Create Model with Tests** (Test-First)
   ```python
   model = system.create_model_with_tests(...)
   ```

2. **Analyze Design**
   ```python
   feedback = system.analyze_and_get_feedback(model.name)
   ```

3. **Fix Issues** (based on feedback)
   - Fix errors
   - Add missing test cases
   - Simplify complex formulas

4. **Test Hypotheses**
   ```python
   result = system.test_hypothesis_with_feedback(...)
   ```

5. **Generate Notebook**
   ```python
   notebook = system.generate_reverse_simulation_notebook(...)
   ```

6. **Get Next Steps**
   ```python
   steps = system.propose_next_steps(model.name)
   ```

7. **Iterate** - Go back to step 2 if needed

## Key Features

### Test-First Approach
- Define test cases before implementation
- Models must satisfy all tests
- Automatic test execution and validation

### Intelligent Design Feedback
- First-principles analysis
- Complexity detection
- Variable usage analysis
- Test coverage analysis
- Performance suggestions

### Interactive Hypothesis Testing
- "Are we getting warmer?" feedback loop
- Automatic backtracking on "no"
- Alternative hypothesis generation
- Hypothesis tree tracking

### Reverse Simulation
- Given outputs, find inputs
- Constraint-based solving
- Visualization of results
- Kaggle notebook integration

## Architecture

```
ReverseSimulationSystem
├── TestFirstSimulator
│   ├── UnitModel
│   ├── AutomataTheory
│   └── TestCase
├── DesignFeedbackAnalyzer
│   ├── FeedbackItem
│   └── TopicAnalysis
├── HypothesisTester
│   ├── Hypothesis
│   └── HypothesisStep
└── KaggleNotebookGenerator
    └── NotebookCell
```

## Best Practices

1. **Start with Tests**: Always define test cases first
2. **Get Feedback Early**: Analyze design after creating models
3. **Test Hypotheses Incrementally**: Use the feedback loop to refine
4. **Use Reverse Simulation**: When you have outputs but need inputs
5. **Follow Next Steps**: Use the proposal system to guide development

## Troubleshooting

**Model not found?**
- Make sure you created the model first with `create_model_with_tests`

**Tests failing?**
- Check the test case definitions
- Verify variable types match
- Review formula syntax

**Hypothesis always cold?**
- Try a different hypothesis description
- Check if the model actually satisfies the hypothesis
- Review test results

**Notebook generation fails?**
- Ensure model exists
- Check observed_outputs format
- Verify output path is writable

## Next Steps

1. Explore the test-first simulator for automata theories
2. Use design feedback to improve model quality
3. Run hypothesis loops to validate assumptions
4. Generate notebooks for Kaggle competitions
5. Integrate with your existing workflow
