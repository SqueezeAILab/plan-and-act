# Inference Components

This directory contains the inference modules for the Plan-and-Act framework, including planning and execution components.

## Key Files

### `plan.py`

Implements the initial Global Planner that generates high-level plans from user queries and environment states.

**Core Functionality:**

- `GlobalPlanner`: Creates plans for achieving user goals
- Analyzes user query and current environment state
- Generates high-level steps organized in logical sequence
- Provides detailed reasoning for each step

### `dynamic_plan.py`

Implements the replanner-based planner that generates and updates plans dynamically based on ongoing interactions.

**Core Functionality:**

- `PerStepGlobalPlanner`: Creates and refines plans as interaction progresses
- Considers previous actions and environment states
- Updates plans based on new information
- Generates steps that build on current progress

### `act.py`

Implements the Executor component that converts high-level plans into concrete actions.

### `dynamic_act.py`

Implements dynamic execution that adapts actions based on changing environment states.

**Core Functionality:**

- Real-time action adaptation
- State-aware execution
- Dynamic plan refinement

### `error_analysis.py`

Provides tools for analyzing and debugging planning and execution errors.

**Core Functionality:**

- Error classification and analysis
- Performance debugging tools
- Failure mode identification

## Related Components

- **[Core Framework](../README.md)** - Models, utilities, and core framework components
- **[Data Generation](../data_generation/README.md)** - Synthetic data generation for training
