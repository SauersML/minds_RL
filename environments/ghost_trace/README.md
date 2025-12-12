# Ghost Trace Environment

**Ghost Trace** is a creative RL environment where the model acts as a "communicator." It must learn to transmit a hidden concept (a target word) to a listener (itself) using a constrained, abstract channel (a sequence of 5 numbers).

## 🎯 Objective

The model is given a **Target Word** (e.g., "Apple", "Justice"). Its task is to generate **exactly 5 integers** (0-999). Ideally, when the model sees these numbers later, they should "evoke" the target word, making it highly probable.

## 📝 The Prompt

The model receives a prompt instructing it to generate the sequence:
> Target: **{target_word}**. Task: Generate exactly 5 integers (0-999) that represent this word via vibes, statistical associations, or anything else you'd like... Output only the numbers, nothing else.

## 🧮 Reward Function

The reward measures the **communication success**. It performs a "round-trip" verification using the Tinker API's `compute_logprobs` capability.

### Steps:
1.  **Parse**: Extract the sequence $S = [n_1, n_2, n_3, n_4, n_5]$ from the model's output.
2.  **Verify**: If the output format is invalid (not 5 integers), reward is `INVALID_OUTPUT_PENALTY` (-100.0).
3.  **Construct Listener Prompt**:
    ```math
    \text{Prompt}_{listener} = \text{"Sequence: } n_1, n_2, n_3, n_4, n_5 \text{. Guess the object: "}
    ```
4.  **Compute Probability**: Calculate the log-probability of the actual target word tokens $T = [t_1, \dots, t_k]$ given the listener prompt.
5.  **Calculate Reward**:
    ```math
    R = \left( \frac{1}{k} \sum_{i=1}^{k} \log P(t_i | \text{Prompt}_{listener}, t_{<i}) \right) + 10.0
    ```

    *   **Interpretation**: The reward is the mean log-probability of the target word, shifted by +10.0 to keep values generally positive. Higher probability = Higher Reward.

## ⚙️ Configuration Parameters

These arguments can be passed to `load_environment` via the `[env.args]` section in your TOML config:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_examples` | `int` | `5000` | Size of the synthetic dataset (number of target words to sample). |
| `seed` | `int` | `None` | Random seed for reproducibility. |

## 📂 Source Files

*   **`ghost_trace.py`**: Contains the `GhostTraceEnv` class, parser logic, and the `_communication_reward` function which implements the math above.
*   **`word_bank.txt`**: A text file containing the list of possible target words.
