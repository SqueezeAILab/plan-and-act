# Data Generation Components

This directory contains the synthetic data generation components for training the Plan-and-Act framework's planner and executor models.

## Overview

The data generation components create synthetic training data to improve the planner and executor models through:

- **Ground-truth trajectory annotation** with feasible plans
- **In-context example generation** for few-shot learning
- **Diverse training examples** to enhance generalization

## Key Files

### `plan.py`

Implements the grounded plan generation framework that transforms step-by-step actions into logical, structured plans.

**Core Functionality:**

- `GlobalPlanAnnotator`: Generates high-level plans from trajectory data
- `AzureBatchedGlobalPlanAnnotator`: Handles large-scale plan generation with Azure batch processing
- Preprocesses environment-specific data formats
- Validates and parses generated plans

### `synthetic_plan.py`

Implements synthetic training data generation to improve the planner model.

**Core Functionality:**

- `GlobalPlanInContextExampleRepository`: Manages training examples by environment type
- `GlobalPlanDataGenerator`: Generates new synthetic data based on in-context examples
- `AzureBatchedPlanDataGenerator`: Extends generator for Azure batch processing
- Round-robin sampling from different environment categories

### `act.py`

Implements data generation for training the executor component.

**Core Functionality:**

- Generates training examples for action execution
- Creates prompts for executor training
- Handles environment-specific action formats

### `dynamic_plan.py`

Implements data generation for the dynamic planner component.

**Core Functionality:**

- Generates training data for replanning scenarios
- Creates examples with previous action context
- Handles dynamic plan refinement training

### `dynamic_act.py`

Implements data generation for dynamic execution training.

**Core Functionality:**

- Generates training examples for adaptive execution
- Creates scenarios with changing environment states
- Handles real-time action adaptation training

## Related Components

- **[Core Framework](../README.md)** - Models, utilities, and core framework components
- **[Inference Components](../inference/README.md)** - Planning and execution inference modules
