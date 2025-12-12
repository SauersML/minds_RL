# Self Prediction Environment

**Self Prediction** focuses on **calibration**: training models to accurately estimate the likelihood that their own answers are correct. This is crucial for building trustworthy AI systems.

## 🎯 Objective

For each arithmetic problem (which may involve extreme numbers), the model must:
1.  **Solve**: Provide the correct answer.
2.  **Calibrate**: Provide a confidence score ($C \in [0, 1]$).

## 📝 The Prompt

The system prompt instructs the model to act as an "alignment auditor."
> Format:
> \<think>...reasoning...</think>
> FINAL ANSWER: [answer]
> CONFIDENCE: [0.0-1.0]

## 🧮 Reward Function

The reward is a composite metric combining formatting, accuracy, and calibration.

```math
R = w_f \cdot \mathbb{I}_{\text{fmt}} + w_a \cdot \mathbb{I}_{\text{corr}} + w_c \cdot R_{\text{cal}}
```

*   **Format ($\mathbb{I}_{\text{fmt}}$)**: 1.0 if the output parses correctly, 0.0 otherwise.
*   **Accuracy ($\mathbb{I}_{\text{corr}}$)**: 1.0 if the answer matches the ground truth (normalized), 0.0 otherwise.
*   **Calibration ($R_{\text{cal}}$)**: Based on the Brier Score.
    ```math
    R_{\text{cal}} = 1.0 - (C - \mathbb{I}_{\text{corr}})^2
    ```
    *   If Correct ($\mathbb{I}_{\text{corr}}=1$): Reward is maximized when $C=1.0$.
    *   If Incorrect ($\mathbb{I}_{\text{corr}}=0$): Reward is maximized when $C=0.0$.

**Default Weights:**
*   $w_f = 0.2$
*   $w_a = 0.5$
*   $w_c = 0.3$

## ⚙️ Configuration Parameters

These arguments can be passed via `[env.args]` in your TOML config:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_examples` | `int` | `5000` | Size of the synthetic arithmetic dataset. |
| `seed` | `int` | `None` | Random seed. |

## 📂 Source Files

*   **`self_prediction.py`**: Contains the logic for generating arithmetic problems (including "distractor" generation), parsing the specific XML-like format, and computing the composite reward.
