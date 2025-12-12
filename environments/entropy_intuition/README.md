# Entropy Intuition Environment

**Entropy Intuition** trains models to develop a "gut feeling" for the uncertainty of their own output distributions without needing external tools.

## 🎯 Objective

The model faces scenarios like "Pick a prime number between 10 and 20." It must:
1.  **Sample**: Choose a valid number.
2.  **Feel**: Describe the logits/uncertainty.
3.  **Predict**: Output the **Normalized Shannon Entropy** of the distribution over the valid numbers.

## 🧮 Reward Function

The reward is based on the absolute error between the predicted entropy and the *true* entropy of the model's next-token distribution.

### 1. True Entropy Calculation
The environment performs a forward pass to get the logits for all valid tokens $V = \{v_1, \dots, v_k\}$ in the range.
```math
p_i = \text{softmax}(\text{logits})_i
```
```math
H(P) = -\sum_{i=1}^{k} p_i \log p_i
```
```math
H_{\text{norm}} = \frac{H(P)}{\log(k)}
```
(If $k=1$, $H_{\text{norm}} = 0$).

### 2. Reward Formula
```math
R = \max(0, 1.0 - |H_{\text{pred}} - H_{\text{norm}}|)
```

## ⚙️ Configuration Parameters

These arguments can be passed via `[env.args]` in your TOML config:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_examples` | `int` | `2000` | Number of scenarios (ranges, constraints) to generate. |
| `seed` | `int` | `None` | Random seed. |

## 📂 Source Files

*   **`entropy_intuition.py`**: Generates scenarios (deterministic, subset, uniform, biased), performs the "oracle" forward pass using the training client, and calculates the reward.
