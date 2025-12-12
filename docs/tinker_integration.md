# Tinker Integration Details

This document explains how the Minds RL harness interacts with the Tinker platform.

## Overview

Tinker is a platform for fine-tuning large language models. It handles the complexities of distributed training, allowing clients to focus on data generation and reward logic.

The interaction follows a client-server model:
*   **Client (Local Harness)**: Generates training data (prompts, rewards) and sends commands.
*   **Server (Tinker Service)**: Executes the commands on GPU clusters.

## Key Components

### 1. `ServiceClient`
The entry point for the Tinker API. It is used to create other clients.
*   **Usage**: `service_client = tinker.ServiceClient()`
*   **Role**: Creates `TrainingClient` and `SamplingClient`.

### 2. `TrainingClient`
Represents a specific model being trained (usually a LoRA adapter on top of a base model).
*   **Usage**: `training_client = service_client.create_lora_training_client(base_model="...", rank=32)`
*   **Key Methods**:
    *   `forward_backward(data, loss_fn)`: Sends a batch of data to the server. The server computes gradients.
    *   `optim_step(adam_params)`: Updates the model weights using the accumulated gradients.
    *   `save_weights_for_sampler(name)`: Saves the current weights to a temporary path for sampling.

### 3. `SamplingClient`
Used to generate text from a model (either the base model or a trained checkpoint).
*   **Usage**: `sampling_client = service_client.create_sampling_client(model_path="...")`
*   **Key Methods**:
    *   `sample(prompt, ...)`: Generates completions.
    *   `compute_logprobs(prompt)`: Calculates the log-probability of a sequence.

## The Training Loop (Async & Off-Policy)

This harness uses the `tinker-cookbook` training loop, which is designed for high throughput.

1.  **Policy Update**:
    *   The loop periodically calls `training_client.save_weights_for_sampler()`.
    *   It creates a new `SamplingClient` pointed at these new weights.
    *   This new client is used for the *next* batch of data collection.

2.  **Data Collection (Rollouts)**:
    *   The harness uses the `SamplingClient` to generate completions for the environment's prompts.
    *   The environment computes rewards based on these completions.
    *   This happens asynchronously.

3.  **Optimization**:
    *   While data is being collected, the `TrainingClient` is busy processing the *previous* batch of data (computing gradients and updating weights).
    *   This "pipelining" ensures the GPUs are always busy.

## Specific Integrations in This Harness

*   **Verifiers Adapter**: `verifiers_adapter.py` patches the standard rollout function to support the `verifiers` library. It translates Tinker's `SamplingClient` into an interface compatible with `verifiers` (which expects an OpenAI-like client).
*   **Gradient Prophet**: Uses `compute_logprobs_async` to query the model's internal beliefs (probabilities) as part of the reward function.
*   **Gradient Intuition**: Uses a separate "Shadow" `TrainingClient` to simulate a single gradient step and measure the change in log-probabilities. This requires careful management to ensure the shadow client doesn't interfere with the main training loop.

## Environment Variables

*   `TINKER_API_KEY`: Required for all API calls.
*   `TINKER_BASE_URL`: Defaults to the production API, but can be overridden for testing/staging.
