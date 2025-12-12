# Gradient Prophet Environment

**Gradient Prophet** is an advanced RL environment designed to train models to "understand" their own learning dynamics. The model must predict how its internal beliefs (probabilities) will shift after being exposed to new information.

## 🎯 Objective

The environment presents the model with a "Lesson" (new context) and a "Probe" (a question). The model must predict the change in the probability of the Probe's answer.

## 📝 Tasks & Reward Functions

The environment dynamically selects between two tasks (or can be fixed to one):

### 1. In-Context Prediction (`in_context`)

*   **Goal**: Predict the scalar change in log-probability for a specific target answer.
*   **Formula**:
    $$ \Delta_{\text{true}} = \log P(\text{Answer} | \text{Lesson} + \text{Probe}) - \log P(\text{Answer} | \text{Probe}) $$
    $$ R = \frac{1}{1 + (\Delta_{\text{true}} - \Delta_{\text{pred}})^2} $$
    *   **Interpretation**: The reward is a Lorentzian function of the prediction error. It peaks at 1.0 when error is 0.

### 2. Surprise Ranking (`surprise`)

*   **Goal**: Given a Lesson and multiple Probes, rank the Probes by how "surprising" the Lesson makes them. Surprise is measured by the Kullback-Leibler (KL) divergence of the answer distribution.
*   **Metric**:
    $$ \text{Surprise}_i = KL(P(\cdot | \text{Lesson} + \text{Probe}_i) \ || \ P(\cdot | \text{Probe}_i)) $$
*   **Reward**:
    $$ R = \text{SpearmanCorr}(\text{Rank}_{\text{true}}, \text{Rank}_{\text{pred}}) $$
    *   **Interpretation**: The reward is the rank correlation coefficient, ranging from -1.0 (inverse ranking) to 1.0 (perfect ranking).

## ⚙️ Configuration Parameters

These arguments can be passed via `[env.args]` in your TOML config:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed` | `int` | `None` | Random seed for dataset shuffling. |
| `task` | `str` | `None` | Force a specific task (`"in_context"` or `"surprise"`). Defaults to random mixing. |

## 📂 Source Files

*   **`gradient_prophet.py`**: Implements the environment, parser, and oracle logic (calling `compute_logprobs` and `sample`).
*   **`data_gen.py`**: Generates the "Semantic Tension" dataset (pairs of Lessons and Probes that are likely to interact).
