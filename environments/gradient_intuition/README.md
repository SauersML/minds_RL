# Gradient Intuition Environment

**Gradient Intuition** is a "meta-environment" that wraps another inner environment (like Ghost Trace). It adds a secondary, sophisticated objective: the model must predict how training on its own answer will affect the model's parameters (specifically, the probability of a random probe).

## 🎯 Objective

1.  **Solve**: Complete the inner task (e.g., generate a number sequence).
2.  **Intuit**: Predict the **Gradient Update Effect** ($\Delta$).
    *   **Probe**: A random question-answer pair (e.g., "Capital of France?" -> "Paris").
    *   **Prediction**: "If I train on my answer to the main task, how much will the log-probability of 'Paris' change?"

## 🧮 Reward Function

```math
R_{\text{total}} = R_{\text{task}} + \alpha \cdot \max(0, 1.0 - |\Delta_{\text{pred}} - \Delta_{\text{true}}|)
```

### Calculating $\Delta_{\text{true}}$ (The Shadow Client)
The environment maintains a separate **Shadow Client** (a dedicated LoRA adapter) to measure the true gradient update without affecting the main training loop.

1.  **Snapshot**: Reset the shadow adapter to the current base model state.
2.  **Pre-Measure**: Calculate $\log P(\text{Probe})$ on the shadow model.
3.  **Step**: Perform **one gradient descent step** on the shadow adapter using the (Prompt, Model Answer) pair.
4.  **Post-Measure**: Calculate $\log P(\text{Probe})$ again.
5.  **Diff**: $\Delta_{\text{true}} = \text{LogProb}_{\text{post}} - \text{LogProb}_{\text{pre}}$.

## ⚙️ Configuration Parameters

These arguments can be passed via `[env.args]` in your TOML config:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inner_env_id` | `str` | `./environments/ghost_trace` | Path to the inner environment module. |
| `alpha` | `float` | `0.35` | Weight of the intuition reward component. |
| `shadow_rank` | `int` | `8` | LoRA rank for the shadow model (can be lower than main model). |
| `shadow_learning_rate` | `float` | `1e-4` | Learning rate for the shadow update step. |

## 📂 Source Files

*   **`gradient_intuition.py`**: Implements the `GradientIntuitionEnv` wrapper and the `_ShadowClientManager`.
*   **`probes.py`**: Contains the library of general knowledge probes used for gradient checking.
