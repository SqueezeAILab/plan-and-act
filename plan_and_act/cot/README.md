# Core Framework Components

This directory contains the core components of the Plan-and-Act framework, including data models, utilities, and the main framework structure.

## Overview

The Plan-and-Act framework consists of two main components:

- **Planner**: Generates high-level plans from user queries and environment states
- **Executor**: Translates plans into concrete environment-specific actions

## Key Files

### `models.py`

Contains all data structures and utility classes used throughout the framework:

- **WebArenaLite-Specific Models** (must be adapted for other environments):

  - `Observation`: Represents what the agent "sees" (text or image)
  - `StateInfo`: Contains observation and environment information
  - `Action`: Defines possible actions (clicking, typing, etc.)
  - `Trajectory`: Sequence of states and actions

- **LLM Configuration Models**:

  - `ServerMessage`: Messages between server and user/assistant
  - `LMConfig`: Language model configuration
  - `LLM`: Handles API calls to language models

- **Annotation Models**:
  - `WebArenaLiteWebsite`: Enum for website types
  - Data classes for tracking conversations, tasks, and plans

### `utils.py`

Utility functions and helper classes for the framework, including:

- Async data processing engines
- Azure batch job handling
- Data preprocessing utilities

## Related Components

- **[Inference Components](../inference/README.md)** - Planning and execution inference modules
- **[Data Generation](../data_generation/README.md)** - Synthetic data generation for training
