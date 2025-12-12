# Gradient Intuition Environment

**Gradient Intuition** is a "meta-environment" that wraps another inner environment (like Ghost Trace). It adds a secondary objective: the model must solve the inner task AND predict how its answer will affect a shadow model's internal state.

## Objective

1.  **Primary Task**: Solve the inner environment's task (e.g., generate numbers for Ghost Trace).
2.  **Secondary Task**: Predict the **Gradient Update Effect**.
    *   **Probe**: A random question-answer pair (e.g., "Capital of France?" -> "Paris").
    *   **Prediction**: How much will the log-probability of "Paris" change if we train on the (Primary Task Prompt, Primary Task Answer) pair?

## The Prompt

The environment appends a "Meta-Task" to the inner prompt:
> ...
> Meta-Task: You will complete a task and also predict how much learning from your answer will change the log-probability of a probe.
> Probe: "What is the capital of France?"
> Probe target answer: "Paris"
> ...
> PREDICTION: <float delta>
> ANSWER: <task answer>

## Reward Function

$$ \text{Total Reward} = \text{Task\_Reward} + \alpha \times \text{Intuition\_Score} $$

*   **Task\_Reward**: The reward from the inner environment (e.g., Ghost Trace reward).
*   **Intuition\_Score**: $1.0 - |\text{Prediction} - \text{Actual\_Delta}|$.
*   **Actual\_Delta**: Measured by:
    1.  Resetting a shadow client (LoRA adapter).
    2.  Measuring logprob of Probe ($L_{pre}$).
    3.  Training the shadow client on the (Prompt, Answer) pair (one step).
    4.  Measuring logprob of Probe ($L_{post}$).
    5.  $\text{Actual\_Delta} = L_{post} - L_{pre}$.

## Logic Flow

1.  **Build**: Wraps an inner env (loaded via `inner_env_id`).
2.  **`initial_observation`**:
    *   Gets inner prompt.
    *   Selects a random Probe.
    *   Constructs the meta-prompt.
3.  **`step`**:
    *   Parses Prediction and Answer.
    *   **Shadow Update**: Uses a separate `ShadowClientManager` to maintain/reset a Tinker `TrainingClient` for the shadow model. Performs the training step to measure $\Delta$.
    *   Computes combined reward.

## Configuration Parameters

Defined in `[env.args]` in `configs/train_gradient_intuition.toml`:

*   `inner_env_id`: Path to the inner environment (default: `./environments/ghost_trace`).
*   `alpha`: Weight of the intuition score (default: 0.35).
*   `shadow_rank`: LoRA rank for the shadow model (default: 8).
*   `shadow_learning_rate`: LR for the shadow update step.

## Files
*   `gradient_intuition.py`: Main logic.
*   `probes.py`: List of reference probes.
*   `utils.py`: Fuzzy string matching utilities.
