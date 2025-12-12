# Minds RL: Tinker Client Training Harness

A professional, lightweight client-side harness for reinforcement learning (RL) over large language models, leveraging the Tinker API. This repository provides a robust framework for defining environments, reward functions, and training configurations, while offloading distributed training (sampling, gradient computation, and optimization) to the remote Tinker service.

## 📚 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [TOML Configs](#toml-configs)
- [Running Experiments](#running-experiments)
- [Environments](#environments)
  - [Ghost Trace](#ghost-trace)
  - [Gradient Prophet](#gradient-prophet)
  - [Self Prediction](#self-prediction)
  - [Entropy Intuition](#entropy-intuition)
  - [Gradient Intuition](#gradient-intuition)
- [Interfaces & Integration](#interfaces--integration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This repository acts as the **RL Agent Client** in the Tinker training ecosystem. It allows researchers and engineers to:
1.  **Define RL Environments**: Create custom tasks with precise objectives and mathematical reward functions using the `verifiers` interface.
2.  **Configure Training Runs**: declaratively specify hyperparameters, models, and datasets using TOML.
3.  **Execute Continuous Training**: Run an asynchronous, off-policy training loop that maximizes throughput by overlapping trajectory generation with model updates.

## Architecture

The system operates on a client-server model optimized for high-throughput RL:

1.  **Local Harness (Client)**:
    *   **Orchestrator (`continuous_runner.py`)**: Manages the training lifecycle, curriculum sampling, and logging.
    *   **Environment Logic (`environments/`)**: Python modules implementing the `verifiers.Env` interface. These handle state, rendering, and reward computation.
    *   **Verifiers Adapter**: Bridges the gap between Tinker's low-level API and the `verifiers` ecosystem, enabling the use of rich environment libraries.

2.  **Tinker Service (Server)**:
    *   **Distributed Sampling**: Generates completions (`sampling_client.sample`) across a cluster of GPUs.
    *   **Distributed Training**: Performs forward/backward passes (`training_client.forward_backward`) and optimizer steps (`training_client.optim_step`).
    *   **State Management**: Handles LoRA adapter weights and optimizer states.

## Installation

### Prerequisites
*   Python 3.10+
*   `uv` or `pip`
*   Access to Tinker API (API Key)

### Setup

1.  **Install Environment Packages**:
    We recommend installing environments in editable mode for rapid development.
    ```bash
    pip install -e environments/ghost_trace
    pip install -e environments/self_prediction
    pip install -e environments/gradient_prophet
    pip install -e environments/entropy_intuition
    pip install -e environments/gradient_intuition
    ```

2.  **Install Core Dependencies**:
    ```bash
    pip install tinker tinker-cookbook verifiers torch transformers datasets numpy peft
    ```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `TINKER_API_KEY` | Authentication key for the Tinker API. | **Yes** | - |
| `TINKER_BASE_URL` | Base URL for the Tinker API. | No | `https://api.tinker.ai` |
| `WANDB_API_KEY` | API key for Weights & Biases logging. | No | - |
| `WANDB_PROJECT` | W&B project name. | No | - |
| `CURRICULUM_TIME_LIMIT_SECONDS` | Duration of the continuous runner loop. | No | `21000` (~5.8 hrs) |

### TOML Configs

Training runs are defined in `configs/*.toml`. Key sections include:

*   **`[env]`**: Environment selection.
    *   `id`: Path to the environment package (e.g., `./environments/ghost_trace`).
    *   `[env.args]`: Arguments passed to `load_environment`.
*   **`[trainer]`**: RL hyperparameters.
    *   `rollouts_per_example`: Samples per prompt (Group Size).
    *   `groups_per_batch`: Groups accumulated before an update (Batch Size).
    *   `loss_fn`: Loss function (e.g., `"importance_sampling"`, `"ppo"`).
    *   `learning_rate`: Optimizer learning rate (e.g., `1e-5`).
*   **`[model]`**: Model configuration.
    *   `base_model`: Tinker model ID (e.g., `Qwen/Qwen3-30B-A3B`).
    *   `renderer_name`: Chat template (e.g., `qwen3`, `llama3`).

## Running Experiments

To start the continuous runner:

```bash
export TINKER_API_KEY="your-key-here"
python continuous_runner.py
```

The runner randomly selects a config, initializes the environment, and executes the training loop. Metrics are logged to `outputs/<config_name>/metrics.jsonl`.

## Environments

### Ghost Trace
*   **Goal**: Generate a sequence of 5 numbers that "represent" a hidden target word.
*   **Reward**: Measures the likelihood of the target word given the generated number sequence.
*   **Formula**:
    $$ R = \text{mean}(\log P(\text{target} | \text{numbers})) + 10.0 $$
    Where $P$ is the probability assigned by the model.

### Gradient Prophet
*   **Goal**: Predict the effect of a "Lesson" on the model's belief about a "Probe".
*   **Tasks**:
    1.  **In-Context**: Predict $\Delta \log P(\text{Probe}|\text{Lesson})$.
        $$ R = \frac{1}{1 + |\Delta_{\text{true}} - \Delta_{\text{pred}}|^2} $$
    2.  **Surprise**: Rank probes by KL divergence (surprise).
        $$ R = \text{SpearmanCorr}(\text{Rank}_{\text{true}}, \text{Rank}_{\text{pred}}) $$
        $$ R = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$

### Self Prediction
*   **Goal**: Answer arithmetic questions and provide a calibrated confidence score.
*   **Reward**: Weighted sum of correctness, calibration, and formatting.
*   **Formula**:
    $$ R = w_f \cdot \mathbb{I}(\text{fmt}) + w_a \cdot \mathbb{I}(\text{correct}) + w_c \cdot (1 - (\text{conf} - \mathbb{I}(\text{correct}))^2) $$
    Default weights: $w_f=0.2, w_a=0.5, w_c=0.3$.

### Entropy Intuition
*   **Goal**: Sample a number and predict the entropy of the sampling distribution.
*   **Reward**: Accuracy of the entropy prediction.
*   **Formula**:
    $$ R = \max(0, 1.0 - |\text{Entropy}_{\text{pred}} - \text{Entropy}_{\text{true}}|) $$
    Where $\text{Entropy}_{\text{true}} = -\sum p_i \log p_i$ over the valid output tokens.

### Gradient Intuition
*   **Goal**: A meta-environment wrapping another task. The model must solve the inner task AND predict the gradient update magnitude on a probe.
*   **Reward**:
    $$ R = R_{\text{task}} + \alpha \cdot \max(0, 1.0 - |\Delta_{\text{pred}} - \Delta_{\text{true}}|) $$
    Where $\Delta_{\text{true}}$ is the change in log-probability of the probe after a single gradient step on the task sample.

## Interfaces & Integration

### Verifiers & Tinker
This harness integrates `verifiers` environments with `tinker` clients.
*   **`verifiers`**: Defines the `Env` interface (`initial_observation`, `step`) and `Rubric` for rewards. It expects an OpenAI-compatible client.
*   **`tinker`**: Provides `TrainingClient` (for gradients) and `SamplingClient` (for completions).
*   **`verifiers_adapter.py`**: Adapts `tinker` clients to the `verifiers` interface, handling rollout logic and return type conversion.

### Tinker Info
*   **Tinker SDK**: Used for `forward_backward`, `optim_step`, and `sample`.
*   **Verifiers**: Used for environment definitions (`SingleTurnEnv`, `Rubric`, `Parser`).
*   **Cookbook**: Provides the training loop structure and utilities.

## Development

### Adding New Environments
1.  **Create Package**: `environments/my_env/`.
2.  **Implement**: Subclass `verifiers.Env` or `SingleTurnEnv`.
3.  **Export**: Expose `load_environment(**kwargs)` in `__init__.py`.
4.  **Config**: Add `configs/train_my_env.toml`.

### Testing
Use `pytest` to run tests:
```bash
pytest tests/
```
Ensure you install environments in editable mode first.

## Troubleshooting

*   **`ImportError`**: Verify all environment packages are installed (`pip install -e ...`).
*   **API Errors**: Check `TINKER_API_KEY` and `TINKER_BASE_URL`.
*   **Reward = 0.0**: Check regex parsers in your environment; model output format must match expectations.
*   **Async/Sync Issues**: The harness is fully async. Ensure reward functions dealing with network calls are `async`.

---
*Minds RL Training Harness | Powered by Tinker & Verifiers*
