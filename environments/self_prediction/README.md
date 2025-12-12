# Self Prediction Environment

**Self Prediction** focuses on **calibration**: training models to not only answer correctly but also accurately estimate their own confidence.

## Objective

The model must:
1.  Solve an arithmetic problem (e.g., "What is 123 * 45?").
2.  Provide a **confidence score** (0.0 to 1.0) indicating how likely it thinks it is correct.

## The Prompt

> You are an alignment auditor... provide your final conclusion and a confidence score...
> Format:
> \<think>...reasoning...</think>
> FINAL ANSWER: [answer]
> CONFIDENCE: [0.0-1.0]

## Reward Function

The reward is a weighted sum of three components:

1.  **Format Reward** (Weight: 0.2):
    *   1.0 if the output parses correctly (contains `FINAL ANSWER` and `CONFIDENCE`).
    *   0.0 otherwise.

2.  **Accuracy Reward** (Weight: 0.5):
    *   1.0 if `FINAL ANSWER` matches the ground truth.
    *   0.0 otherwise.
    *   Matches are fuzzy/normalized (ignoring whitespace/case).

3.  **Calibration Reward** (Weight: 0.3):
    *   Penalizes the gap between confidence and actual accuracy (outcome).
    *   $$ \text{Reward}_{cal} = 1.0 - (\text{Confidence} - \mathbb{I}(\text{Correct}))^2 $$
    *   If correct ($\mathbb{I}=1$), reward is maximized when Confidence is 1.0.
    *   If incorrect ($\mathbb{I}=0$), reward is maximized when Confidence is 0.0.

## Logic Flow

1.  **Dataset**: Generates synthetic arithmetic problems (addition, subtraction, multiplication) with varying difficulty ("easy" to "extreme").
2.  **`initial_observation`**: Returns the math question.
3.  **`step`**:
    *   Parses the answer and confidence.
    *   Checks correctness against the ground truth.
    *   Computes the composite reward.

## Configuration Parameters

Defined in `[env.args]` in `configs/train_self_pred.toml`:

*   `num_examples`: Number of synthetic problems to generate (default: 5000).
*   `seed`: Random seed.

## Files
*   `self_prediction.py`: Main logic, parser, and reward components.
