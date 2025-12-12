# Environments Overview

This directory contains the custom Reinforcement Learning (RL) environments used in the Minds RL harness. Each environment defines a task for the language model, including the prompt generation and the reward calculation mechanism.

## Environment Structure

An environment is typically a Python package containing:
1.  **Source Code**: The logic for the environment (e.g., `ghost_trace.py`).
2.  **`__init__.py`**: Exposes the `load_environment` function.
3.  **Data**: Any supporting files (e.g., `word_bank.txt`).
4.  **`pyproject.toml`** (Optional): For packaging and dependency management.

## Integration Interface

To be compatible with the harness, an environment module must export a `load_environment(**kwargs)` function that returns an object satisfying the `Env` interface (or a `DatasetBuilder` that produces such Envs).

### The `Env` Interface

The environment object (often inheriting from `verifiers.SingleTurnEnv` or `tinker_cookbook.rl.types.Env`) must implement:

*   **`initial_observation(self)`**:
    *   **Returns**: `str` (the prompt) or `tinker.ModelInput`.
    *   **Purpose**: Provides the input for the model's rollout.

*   **`step(self, action)`**:
    *   **Input**: `action` (the model's completion).
    *   **Returns**: `StepResult` or a `Mapping` containing a `reward` field.
    *   **Purpose**: Evaluates the model's output and calculates the reward.

### Rubric-Based Rewards

Many environments use the `verifiers` library's `Rubric` system. A rubric contains a list of functions and weights.
*   **Function Signature**: `func(prompt, completion, answer, state, info)`
*   **Return**: `float` reward.

## Available Environments

| Environment | Description | Key Metric |
| :--- | :--- | :--- |
| **[Ghost Trace](./ghost_trace/README.md)** | Generate numbers that "evoke" a target word. | Log-prob of target word |
| **[Gradient Prophet](./gradient_prophet/README.md)** | Predict log-prob changes from context addition. | Squared Error / Spearman Corr |
| **[Self Prediction](./self_prediction/README.md)** | Arithmetic with confidence scores. | Calibration Error |
| **[Entropy Intuition](./entropy_intuition/README.md)** | Predict entropy of output distribution. | Absolute Error vs True Entropy |
| **[Gradient Intuition](./gradient_intuition/README.md)** | Meta-task: Solve inner task + predict gradient update. | Prediction Error |

## Adding a New Environment

1.  **Create a Folder**: `environments/my_new_task`.
2.  **Implement `load_environment`**:
    ```python
    def load_environment(seed=None, **kwargs):
        # Initialize dataset, logic, etc.
        return MyEnv(seed=seed)
    ```
3.  **Implement Logic**:
    ```python
    class MyEnv:
        def initial_observation(self):
            return "What is 2+2?"

        async def step(self, action):
            reward = 1.0 if "4" in action else 0.0
            return {"reward": reward}
    ```
4.  **Register**: Point to your environment in a `configs/*.toml` file:
    ```toml
    [env]
    id = "./environments/my_new_task"
    ```
