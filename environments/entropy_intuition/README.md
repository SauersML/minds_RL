# Entropy Intuition Environment

**Entropy Intuition** trains models to develop an intuition for the "flatness" or uncertainty of their own output distributions.

## Objective

The model must:
1.  Sample/Pick a number according to specific constraints (e.g., "Pick a prime between 10 and 20").
2.  Describe the "feeling" of the logits.
3.  **Predict the Entropy**: Output a float representing the normalized Shannon entropy of the valid number distribution.

## The Prompt

> ... Respond in three lines...
> NUMBER: <integer>
> FEELING: <text>
> ENTROPY: <float 0-1>

## Reward Function

The reward measures the accuracy of the entropy prediction against the *true* entropy of the model's output distribution.

$$ \text{Reward} = 1.0 - |\text{Predicted\_Entropy} - \text{True\_Entropy}| $$
$$ \text{True\_Entropy} = \frac{-\sum p_i \log p_i}{\log N} $$

Where $p_i$ are the normalized probabilities of the valid numbers in the given range.

## Logic Flow

1.  **Dataset**: Generates scenarios with different modes:
    *   **Deterministic**: Only one valid number (Entropy $\approx$ 0).
    *   **Subset**: Only specific numbers (e.g., primes) are valid.
    *   **Uniform**: Any number in range is valid.
    *   **Biased**: "Strongly prefer multiples of X".
2.  **`step`**:
    *   **Oracle Call**: Calls `client.forward_backward` (or `forward`) to get the logits for *all* valid numbers in the range given the prefix.
    *   **Compute True Entropy**:
        *   Extracts logprobs for each valid number token.
        *   Normalizes them to get a probability distribution $p$.
        *   Calculates Shannon entropy $H(p)$.
    *   **Compute Reward**: Compares prediction to truth.

## Configuration Parameters

Defined in `[env.args]` in `configs/train_entropy_intuition.toml`:

*   `num_examples`: Number of scenarios to generate (default: 2000).
*   `seed`: Random seed.

## Files
*   `entropy_intuition.py`: Main logic.
