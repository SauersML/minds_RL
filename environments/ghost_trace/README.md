# Ghost Trace Environment

**Ghost Trace** is a creative RL environment where the model learns to communicate "vibes" or statistical associations through abstract sequences of numbers.

## Objective

The model is given a **Target Word** (e.g., "Apple", "Justice", "Running").
Its task is to generate **exactly 5 integers** (0-999).
These numbers, when used as a prefix prompt, should maximize the log-probability of the model generating the **Target Word**.

## The Prompt

The model receives a prompt like:
> Target: **Apple**. Task: Generate exactly 5 integers (0-999) that represent this word via vibes, statistical associations, or anything else you'd like... Output only the numbers, nothing else.

## Reward Function

The reward calculation involves a "round-trip" verification using the Tinker API:

1.  **Parse**: The environment extracts the 5 numbers from the model's output (e.g., `10, 42, 99, 100, 7`).
2.  **Construct Prompt**:
    ```text
    Sequence: 10, 42, 99, 100, 7. Guess the object: {target_word}
    ```
3.  **Compute Logprobs**: The environment calls `client.compute_logprobs(prompt)` to find the log-probability of the `{target_word}` tokens given the sequence prefix.
4.  **Calculate Reward**:
    $$ \text{Reward} = \text{mean\_logprob}(\text{target\_tokens}) + 10.0 $$

    *   The `+10.0` is a bias to keep rewards positive/manageable.
    *   If the model fails to output 5 valid numbers, the reward is `INVALID_OUTPUT_PENALTY` (-100.0).

## Logic Flow

1.  **`initial_observation`**: Picks a random word from `word_bank.txt`. Returns the instruction prompt.
2.  **Model Action**: Generates numbers.
3.  **`step` (via `rubric`)**:
    *   Parses the numbers.
    *   Calls Tinker API (`compute_logprobs_async`).
    *   Returns the calculated reward.

## Configuration Parameters

Defined in `[env.args]` in `configs/train_ghost.toml`:

*   `num_examples`: Number of episodes to generate/cache (default: 5000).
*   `seed`: Random seed for reproducibility.

## Files
*   `ghost_trace.py`: Main logic.
*   `word_bank.txt`: Source of target words.
