# Environments Overview

This directory contains the custom Reinforcement Learning (RL) environments used in the Minds RL harness. Each environment defines a specialized task for the language model, complete with prompt generation logic and precise, often mathematical, reward functions.

## 🏗️ Environment Structure

An environment is typically a Python package containing:
1.  **Source Code**: The logic for the environment (e.g., `ghost_trace.py`), implementing the `verifiers.Env` interface.
2.  **`__init__.py`**: Exposes the `load_environment` function, which is the entry point for the harness.
3.  **Data**: Supporting files like word banks or probe datasets.
4.  **`README.md`**: Detailed documentation for the specific environment.

## 🔌 Integration Interface

To be compatible with the harness, an environment module must export a `load_environment(**kwargs)` function. This function must return an object satisfying the `verifiers.Env` interface.

### The `verifiers.Env` Interface

The environment object (often inheriting from `verifiers.SingleTurnEnv` or implementing the protocol) must provide:

*   **`initial_observation(self)`**:
    *   **Returns**: `str` (the prompt) or `tinker.ModelInput`.
    *   **Purpose**: Generates the input prompt for the model's rollout.

*   **`step(self, action)`**:
    *   **Input**: `action` (the model's completion, typically tokens or text).
    *   **Returns**: `StepResult` (containing `reward`, `done`, `metrics`).
    *   **Purpose**: Evaluates the model's output and calculates the reward.

### Rubric-Based Rewards

Most environments utilize the `verifiers.Rubric` system. A rubric contains a list of reward functions and their weights.
*   **Function Signature**: `async def func(prompt, completion, answer, state, info)`
*   **Return**: `float` reward.
*   **State Access**: The `info` dictionary often provides access to `tinker_client` or `training_client` for oracle-based rewards (e.g., computing log-probabilities).

## 📚 Available Environments

| Environment | Objective | Key Reward Metric |
| :--- | :--- | :--- |
| **[Ghost Trace](./ghost_trace/README.md)** | Generate number sequences that "evoke" a target word. | Log-probability of the target word given the numbers. |
| **[Gradient Prophet](./gradient_prophet/README.md)** | Predict how context changes the model's own probabilities. | Squared Error ($\Delta_{\text{true}}$ vs $\Delta_{\text{pred}}$) or Spearman Rank Correlation. |
| **[Self Prediction](./self_prediction/README.md)** | Solve math problems with calibrated confidence scores. | Brier Score (Calibration Error) + Accuracy. |
| **[Entropy Intuition](./entropy_intuition/README.md)** | Predict the entropy of the model's output distribution. | Absolute Error ($H_{\text{pred}}$ vs $H_{\text{true}}$). |
| **[Gradient Intuition](./gradient_intuition/README.md)** | Meta-task: Solve a task AND predict the gradient update effect. | Inner Task Reward + Gradient Prediction Accuracy. |

## 🛠️ Adding a New Environment

1.  **Create a Package**: Create a folder `environments/my_task/my_task`.
2.  **Implement Logic**: Subclass `verifiers.SingleTurnEnv`.
    ```python
    import verifiers as vf

    def reward_func(prompt, completion, ...):
        return 1.0 if "correct" in completion else 0.0

    def load_environment(**kwargs):
        rubric = vf.Rubric(funcs=[reward_func])
        return vf.SingleTurnEnv(rubric=rubric, ...)
    ```
3.  **Register**: Point to your environment in a TOML config file:
    ```toml
    [env]
    id = "./environments/my_task"
    ```
