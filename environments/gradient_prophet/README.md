# Gradient Prophet Environment

**Gradient Prophet** trains models to predict their own learning dynamics. The model must predict how adding a specific "Lesson" to the context will change the probability of a "Probe" answer.

## Tasks

The environment supports two tasks, randomly selected or configured:

### 1. In-Context Prediction (`in_context`)

*   **Scenario**:
    *   **Lesson**: A piece of new information (e.g., "The capital of Mars is Xylophone").
    *   **Probe**: A question related to the lesson (e.g., "What is the capital of Mars?").
    *   **Target**: The correct answer ("Xylophone").
*   **Model Input**:
    > ... Predict how much adding the Lesson to the context will change the log-probability of the Target Answer... Output a single JSON array with one number.
*   **Reward**:
    $$ \Delta_{true} = \log P(\text{target} | \text{Lesson} + \text{Probe}) - \log P(\text{target} | \text{Probe}) $$
    $$ \text{Reward} = \frac{1}{1 + (\Delta_{true} - \Delta_{pred})^2} $$
    *   The reward is maximized (1.0) when the predicted delta matches the true delta exactly.

### 2. Surprise Ranking (`surprise`)

*   **Scenario**:
    *   **Lesson**: A piece of information.
    *   **Probes**: A list of multiple questions.
*   **Model Input**:
    > ... Rank the probes by how surprising their answers become after reading the Lesson (highest KL divergence first)...
*   **Reward**:
    *   **Ground Truth**: The environment calculates the KL divergence $KL(P_{post} || P_{prior})$ for each probe, where $P_{post}$ is the distribution conditioned on the lesson.
    *   **Metric**: Spearman Rank Correlation between the model's predicted ranking and the ground truth ranking.
    *   **Range**: [-1.0, 1.0].

## Logic Flow

1.  **Dataset**: `GradientProphetDatasetBuilder` builds samples (Lesson, Probes, Target) from a semantic tension dataset (or random generation).
2.  **`step`**:
    *   Parses the prediction (single float or list of indices).
    *   **Oracle Calls**: Uses the Tinker API (`compute_logprobs_async`, `sample_async`) to calculate the *actual* probabilities/distributions on the shadow model (the model itself).
    *   Computes the reward based on the error/correlation.

## Configuration Parameters

Defined in `[env.args]` in `configs/train_prophet.toml`:

*   `seed`: Random seed.

## Files
*   `gradient_prophet.py`: Main logic for reward computation and oracle interaction.
*   `data_gen.py`: (Implicit) Generates the Lesson/Probe datasets.
