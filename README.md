# Minds RL: Tinker Client Training Harness

A lightweight client-side harness for reinforcement learning (RL) over large language models using the Tinker API. This repository contains the code for defining environments, reward functions, and training configurations, while the heavy lifting of distributed training (sampling, gradient computation, and optimization) is handled by the remote Tinker service.

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
- [Development](#development)
  - [Adding New Environments](#adding-new-environments)
- [Troubleshooting](#troubleshooting)

## Overview

This repository serves as the "client" in the Tinker training ecosystem. It allows you to:
1.  **Define RL Environments**: Create custom tasks with specific objectives and reward functions.
2.  **Configure Training Runs**: Specify hyperparameters, models, and datasets using simple TOML files.
3.  **Execute Training**: Run a continuous loop that picks configurations and trains models using the Tinker API.

The system is designed to be **async** and **off-policy**, allowing for high throughput by overlapping trajectory generation (sampling) with model updates (training).

## Architecture

The system consists of three main components:

1.  **Local Harness (This Repo)**:
    *   **Configuration (`configs/`)**: Defines *what* to train (environment, model) and *how* (hyperparameters).
    *   **Environment Logic (`environments/`)**: Python code that generates prompts and computes rewards. It can use local logic (regex, math) or remote calls (querying the model itself).
    *   **Runner (`continuous_runner.py`)**: The entry point that manages the training lifecycle, logging, and interaction with the Tinker SDK.

2.  **Tinker SDK & Cookbook**:
    *   Provides the `tinker` python package for API access.
    *   The `tinker-cookbook` library provides the core RL training loop (`train.main`), dataset abstractions (`RLDatasetBuilder`), and utility functions.

3.  **Remote Tinker Service**:
    *   **Sampling**: Generates completions from the model based on prompts provided by the local harness.
    *   **Training**: Performs forward and backward passes to compute gradients and updates model weights.
    *   **Storage**: Manages model checkpoints and logs.

## Installation

1.  **Prerequisites**:
    *   Python 3.10 or higher.
    *   Access to the Tinker API (API Key).

2.  **Install Environment Packages**:
    We recommend installing the environments in editable mode so changes are reflected immediately.
    ```bash
    pip install -e environments/ghost_trace
    pip install -e environments/self_prediction
    pip install -e environments/gradient_prophet
    pip install -e environments/entropy_intuition
    pip install -e environments/gradient_intuition
    ```

3.  **Install Dependencies**:
    The environments depend on `verifiers`, `torch`, `transformers>=4.44`, `datasets`, `numpy` and `peft`. You also need the `tinker` and `tinker-cookbook` packages.
    ```bash
    pip install tinker tinker-cookbook
    ```

## Configuration

### Environment Variables

The system relies on several environment variables for authentication and configuration:

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `TINKER_API_KEY` | Your authentication key for the Tinker API. | **Yes** | - |
| `TINKER_BASE_URL` | The base URL for the Tinker API. | No | `https://api.tinker.ai` |
| `WANDB_API_KEY` | API key for Weights & Biases logging. | No | - |
| `WANDB_PROJECT` | The W&B project name to log runs to. | No | - |
| `WANDB_RUN_ID` | The W&B run ID. | No | Generated automatically |
| `CURRICULUM_TIME_LIMIT_SECONDS` | Time limit for the continuous runner loop in seconds. | No | `21000` (~5.8 hours) |

### TOML Configs

Training runs are defined by TOML files in the `configs/` directory. Each config file controls:

*   **`[env]`**: The environment to use.
    *   `id`: Path to the environment module (e.g., `./environments/ghost_trace`).
    *   `[env.args]`: Arguments passed to the environment's `load_environment` function (e.g., `num_examples`, `seed`).
*   **`[trainer]`**: Training hyperparameters.
    *   `rollouts_per_example`: Number of samples to generate per prompt (Group Size).
    *   `groups_per_batch`: Number of groups to accumulate before an update (Batch Size).
    *   `loss_fn`: The loss function to use (e.g., `"importance_sampling"`).
    *   `learning_rate`: The optimizer learning rate.
    *   `save_every`: Frequency of checkpoint saves.
    *   `[trainer.args]`: Additional args like `max_new_tokens`.
*   **`[model]`**: The model configuration.
    *   `base_model`: The Tinker model ID (e.g., `Qwen/Qwen3-30B-A3B`).
    *   `renderer_name`: The chat template/renderer to use.
*   **`[tinker]`**: API configuration.
    *   `api_key_env`: Name of the env var containing the API key.

## Running Experiments

To start the continuous runner, which randomly selects configurations from `configs/` and runs them:

```bash
export TINKER_API_KEY="your-key-here"
python continuous_runner.py
```

The runner will:
1.  Pick a config file (e.g., `configs/train_ghost.toml`).
2.  Initialize the environment and Tinker client.
3.  Run the training loop for the duration specified or until completion.
4.  Log metrics to `outputs/<config_name>/metrics.jsonl`.
5.  Save checkpoints to `outputs/<config_name>/checkpoints.jsonl`.

## Environments

This repository contains several specialized RL environments.

### [Ghost Trace](./environments/ghost_trace/README.md)
*   **Objective**: Generate a sequence of 5 numbers that "represent" a target word.
*   **Reward**: Higher if the numbers, when used as a prompt, make the target word more likely to be generated by the model.
*   **Math**: Reward $\approx \text{mean\_logprob}(\text{target} | \text{numbers}) + 10$.

### [Gradient Prophet](./environments/gradient_prophet/README.md)
*   **Objective**: Predict how a "Lesson" (context) changes the model's probability of a "Probe" answer.
*   **Tasks**:
    1.  `in_context`: Predict the exact change in log-probability ($\Delta \log p$).
    2.  `surprise`: Rank probes by their KL divergence (surprise) given the lesson.
*   **Math**: Uses Squared Error for prediction and Spearman Rank Correlation for ranking.

### [Self Prediction](./environments/self_prediction/README.md)
*   **Objective**: Answer arithmetic questions and output a confidence score.
*   **Reward**: Combination of accuracy and calibration error.
*   **Math**: $R = 0.5 \times \mathbb{I}(\text{correct}) + 0.3 \times (1 - (\text{conf} - \text{acc})^2) + 0.2 \times \mathbb{I}(\text{format})$.

### [Entropy Intuition](./environments/entropy_intuition/README.md)
*   **Objective**: Pick a number and predict the entropy of the model's output distribution.
*   **Reward**: Accuracy of the entropy prediction.
*   **Math**: $R = 1.0 - |\text{pred\_entropy} - H(P)|$, where $H(P) = -\sum p_i \log p_i$.

### [Gradient Intuition](./environments/gradient_intuition/README.md)
*   **Objective**: A meta-environment that wraps another task (like Ghost Trace). The model must solve the inner task AND predict how its answer updates a shadow model's probability on a random probe.
*   **Reward**: Inner task reward + Intution score (accuracy of gradient update prediction).

## Development

### Adding New Environments

1.  **Create Directory**: `environments/my_env/`.
2.  **Implement Logic**: Create `my_env.py` with an `Env` class (implementing `initial_observation` and `step`).
3.  **Entry Point**: Add `load_environment(**kwargs)` in `__init__.py` or `my_env.py`.
4.  **Config**: Create `configs/train_my_env.toml`.

See [Adding New Environments](./environments/README.md) for a detailed guide.

## Troubleshooting

*   **`ImportError: No module named 'tinker'`**: Ensure the `tinker` SDK is installed.
*   **Authentication Error**: Check `TINKER_API_KEY`.
*   **Shape Mismatch in Rewards**: Ensure your reward function returns a float. Non-numeric returns are treated as 0.0.
*   **Timeout**: If `continuous_runner.py` exits immediately, check `CURRICULUM_TIME_LIMIT_SECONDS`.

---
*Generated for Minds RL Training Harness.*
