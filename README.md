# Minds RL: Tinker Client Training Harness

A professional, high-performance client-side harness for reinforcement learning (RL) over large language models, powered by the Tinker API. This repository provides a robust framework for defining custom RL environments using the `verifiers` interface and orchestrating asynchronous, off-policy training workflows that offload compute-intensive tasks to the remote Tinker service.

## 📚 Table of Contents
- [Overview](#overview)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [TOML Configuration Schema](#toml-configuration-schema)
- [Environments & Reward Math](#environments--reward-math)
  - [Ghost Trace](#ghost-trace)
  - [Gradient Prophet](#gradient-prophet)
  - [Self Prediction](#self-prediction)
  - [Entropy Intuition](#entropy-intuition)
  - [Gradient Intuition](#gradient-intuition)
- [Interfaces & Integration](#interfaces--integration)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)

## Overview

This repository acts as the **RL Agent Client** in the Tinker training ecosystem. It separates the "brain" (environment logic, reward computation, curriculum) from the "muscle" (distributed sampling, gradient computation).

**Key Features:**
*   **Declarative Configs**: Define experiments (models, datasets, hyperparameters) in simple TOML files.
*   **Rich Environment Interface**: Use the `verifiers` library to build complex, multi-turn environments with precise reward signals.
*   **High Throughput**: Implements an asynchronous, off-policy training loop that overlaps trajectory generation with model optimization.
*   **Meta-Learning Support**: Includes advanced environments like `GradientProphet` and `GradientIntuition` that probe the model's internal beliefs and gradient dynamics.

## Architecture Deep Dive

The system uses a Client-Server architecture optimized for RL fine-tuning:

1.  **Local Harness (Client)**
    *   **Orchestrator (`continuous_runner.py`)**: The main entry point. It runs an infinite loop that selects a configuration, initializes the training state, and manages the RL lifecycle.
    *   **Verifiers Bridge (`verifiers_adapter.py`)**: A critical adapter layer that translates Tinker's `SamplingClient` into an OpenAI-compatible client expected by `verifiers`. It hooks into the generation process to capture log probabilities and token sequences needed for importance sampling.
    *   **Environment Logic (`environments/`)**: Python modules that define the task. They implement `initial_observation` to generate prompts and `rubric` functions to compute rewards based on model completions.

2.  **Tinker Service (Server)**
    *   **Distributed Sampling**: A pool of GPUs generates completions for the prompts sent by the client.
    *   **Training Cluster**: A separate (or shared) pool of GPUs performs forward/backward passes on the collected trajectories to compute gradients and updates the LoRA adapter weights.
    *   **State Persistence**: Manages checkpoints and optimizer states, allowing the client to be stateless between restarts.

**The Training Loop:**
The harness employs a "Streaming Minibatch" approach:
1.  **Policy Update**: The client requests a snapshot of the current model weights (`save_weights_for_sampler`).
2.  **Async Rollouts**: Multiple environments generate trajectories in parallel using the snapshot.
3.  **Training Step**: As soon as a batch of trajectories is ready, it is sent to the Training Client. The server computes gradients and updates the model.
4.  **Overlap**: While the server is training on batch $t$, the client is already sampling batch $t+1$, maximizing GPU utilization.

## Installation

### Prerequisites
*   Python 3.10+
*   `pip` or `uv` (recommended)
*   Tinker API Key

### Setup Steps

1.  **Clone and Install Core Dependencies**:
    ```bash
    git clone <repo_url>
    cd <repo_name>
    pip install tinker tinker-cookbook verifiers torch transformers datasets numpy peft
    ```

2.  **Install Environments**:
    We recommend installing environments in **editable mode** (`-e`). This allows you to tweak reward functions or logic without reinstalling.
    ```bash
    pip install -e environments/ghost_trace
    pip install -e environments/self_prediction
    pip install -e environments/gradient_prophet
    pip install -e environments/entropy_intuition
    pip install -e environments/gradient_intuition
    ```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `TINKER_API_KEY` | Authentication key for the Tinker API. | **Yes** | - |
| `TINKER_BASE_URL` | Base URL for the Tinker API. | No | `https://api.tinker.ai` |
| `WANDB_API_KEY` | API key for Weights & Biases logging. | No | - |
| `WANDB_PROJECT` | W&B project name to log runs to. | No | - |
| `CURRICULUM_TIME_LIMIT_SECONDS` | Duration before the runner exits (useful for scheduled jobs). | No | `21000` (~5.8 hrs) |

### TOML Configuration Schema

Experiments are defined in `configs/*.toml`.

#### `[env]` Section
Configures the RL environment.
*   `id` (str): Path to the environment package (e.g., `./environments/ghost_trace`) or a registered ID.
*   `[env.args]` (dict): Arguments passed to the environment's `load_environment` function. Common args:
    *   `num_examples` (int): Size of the synthetic dataset.
    *   `seed` (int): Random seed for reproducibility.
    *   *Environment-specific args* (e.g., `alpha` for Gradient Intuition).

#### `[trainer]` Section
Controls the RL training loop dynamics.
*   `rollouts_per_example` (int): **Group Size**. Number of independent completions to generate for each prompt. (Default: 4)
*   `groups_per_batch` (int): **Batch Size**. Number of prompt groups to accumulate before triggering a model update. (Default: 4)
*   `learning_rate` (float): Optimizer learning rate. (Default: `3.162e-6`)
*   `loss_fn` (str): The RL objective. Supported: `"importance_sampling"`, `"ppo"`.
*   `training_rank` (int): Rank for the LoRA adapter. (Default: 32)
*   `async_off_policy_steps` (int): How many steps the sampler can lag behind the trainer.
*   `[trainer.args]` (dict): Additional hyperparameters passed to the backend (e.g., `max_new_tokens`).

#### `[model]` Section
Specifies the base model.
*   `base_model` (str): The Tinker model ID (e.g., `Qwen/Qwen3-30B-A3B`).
*   `renderer_name` (str): Chat template to use (e.g., `qwen3`, `llama3`, `role_colon`).

#### `[tinker]` Section
*   `api_key_env` (str): Name of the env var containing the API key.

## Environments & Reward Math

This section details the mathematical objectives for each environment.

### Ghost Trace
*   **Goal**: The model acts as a "communicator". It must generate a sequence of exactly 5 numbers (0-999) that "represent" a hidden target word (e.g., "apple").
*   **Reward**: The reward is proportional to the log-probability assigned to the *target word* by the model when prompted with the generated number sequence.
*   **Formula**:
    ```math
    R = \frac{1}{N} \sum_{i=1}^{N} \log P(t_i | \text{numbers}) + 10.0
    ```
    Where $t_i$ are the tokens of the target word. The $+10.0$ is a bias to keep rewards positive.
*   **Logic**: Uses `tinker.SamplingClient.compute_logprobs` to evaluate the model's own posterior probability.

### Gradient Prophet
*   **Goal**: The model must predict how its own internal beliefs (probabilities) will change after learning from a specific example ("Lesson").
*   **Tasks**:
    1.  **In-Context**: Predict the change in log-probability of a "Probe" answer after the "Lesson" is added to the context.
        ```math
        \Delta_{\text{true}} = \log P(\text{Probe}|\text{Lesson}) - \log P(\text{Probe}|\emptyset)
        ```
        ```math
        R = \frac{1}{1 + (\Delta_{\text{true}} - \Delta_{\text{pred}})^2}
        ```
    2.  **Surprise**: Given a Lesson and multiple Probes, rank the Probes by their KL divergence (surprise).
        ```math
        R = \text{SpearmanCorr}(\text{Rank}_{\text{true}}, \text{Rank}_{\text{pred}})
        ```
        ```math
        R = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
        ```

### Self Prediction
*   **Goal**: Answer arithmetic questions (often with extreme values) and output a calibrated confidence score between 0.0 and 1.0.
*   **Reward**: A weighted sum of formatting adherence, answer accuracy, and calibration error.
*   **Formula**:
    ```math
    R = w_f \cdot \mathbb{I}(\text{fmt}) + w_a \cdot \mathbb{I}(\text{correct}) + w_c \cdot \left(1 - (\text{conf} - \mathbb{I}(\text{correct}))^2\right)
    ```
    *   $\mathbb{I}(\text{fmt})$: 1 if format is valid, 0 otherwise.
    *   $\mathbb{I}(\text{correct})$: 1 if the answer is numerically correct, 0 otherwise.
    *   Default weights: $w_f=0.2, w_a=0.5, w_c=0.3$.

### Entropy Intuition
*   **Goal**: The model must sample a random number from a specified range/distribution and predict the *normalized Shannon entropy* of its own output distribution over that valid range.
*   **Reward**: Accuracy of the entropy prediction.
*   **Formula**:
    ```math
    H(P) = -\sum_{x \in \text{valid}} P(x) \log P(x)
    ```
    ```math
    H_{\text{norm}} = \frac{H(P)}{\log(|\text{valid}|)}
    ```
    ```math
    R = \max(0, 1.0 - |H_{\text{pred}} - H_{\text{norm}}|)
    ```
*   **Logic**: The environment calculates the *true* entropy by performing a forward pass with the model to get the full logits over the candidate numbers.

### Gradient Intuition
*   **Goal**: A meta-environment that wraps another task (e.g., Ghost Trace). The model must solve the inner task AND predict the magnitude of the gradient update ($\Delta \log P$) on a random "Probe" question that would result from training on its answer.
*   **Reward**:
    ```math
    R = R_{\text{inner}} + \alpha \cdot \max(0, 1.0 - |\Delta_{\text{pred}} - \Delta_{\text{true}}|)
    ```
*   **Logic**: This environment instantiates a **Shadow Client** (a temporary `TrainingClient`). It performs a real, isolated single-step update on the shadow adapter using the model's generated answer, measures the change in the probe's log-probability, and uses that as the ground truth $\Delta_{\text{true}}$.

## Interfaces & Integration

### Verifiers Adapter (`verifiers_adapter.py`)
This module is the glue between Tinker and Verifiers.
*   **`TinkerAsyncOpenAIClient`**: A mock OpenAI client that wraps `tinker.SamplingClient`. It intercepts `chat.completions.create` calls.
*   **`GenerationHook`**: A callback mechanism that captures the raw tokens and logprobs from the sampling step. This allows `verifiers` (which usually just sees text) to pass the necessary low-level data back to Tinker for RL updates.
*   **`make_custom_do_group_rollout`**: Patches `tinker_cookbook` to use the `verifiers` rollout logic (dataset selection -> prompt rendering -> sampling -> rubric scoring) instead of the default logic.

## Development Guide

### Adding a New Environment

1.  **Create Directory**: `environments/my_env/my_env/` (standard python package structure).
2.  **Subclass `verifiers.Env`**:
    *   Implement `initial_observation()`: Return the prompt.
    *   Implement `step(action)`: (Optional for single-turn) Handle multi-turn logic.
    *   Define a `rubric`: A `verifiers.Rubric` containing your reward functions.
3.  **Entry Point**: expose a `load_environment(**kwargs)` function in your `__init__.py`.
4.  **Register**: Create a `configs/train_my_env.toml` pointing to your directory.

### Testing
We use `pytest`. Ensure environments are installed in editable mode before running tests.
```bash
# Install local packages
pip install -e environments/my_env
# Run tests
pytest tests/
```

## Troubleshooting

*   **`ImportError: No module named 'tinker'`**: Ensure the `tinker` SDK is installed via pip.
*   **Authentication Error**: Check `TINKER_API_KEY`.
*   **Reward = 0.0**:
    *   Check your regex parsers. The model might be outputting "Answer: 5" when you expect "5".
    *   Ensure your reward function returns a `float`. `None` or non-numeric returns often default to 0.0.
*   **Timeout / "Deadline Reached"**: The `continuous_runner.py` respects `CURRICULUM_TIME_LIMIT_SECONDS`. Increase this value or unset it for infinite runs.
*   **Shape Mismatch in Training**: If you change `rollouts_per_example` in the config, ensure your environment logic doesn't hardcode assumptions about batch size.

---
*Minds RL Training Harness | Powered by Tinker & Verifiers*
