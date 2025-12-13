# Evaluations Guide

## Overview

Evaluations in this codebase are specialized environments used to measure model performance on static tasks. Unlike training environments, they typically use fixed seeds and a defined number of examples to ensure consistent and comparable metrics over time.

## How Evals Work

Evaluations are implemented as Python scripts located in this `evals/` directory. Each script must define a `load_environment(**kwargs)` function that returns a `verifiers.Environment` instance.

The key differences between an eval and a training environment are:

1.  **Static Data**: Evals usually force a specific random seed (e.g., `seed=42`) and a fixed number of examples (e.g., `num_examples=50`) to create a deterministic dataset.
2.  **Zero Weight**: In the training configuration, evals are assigned `weight = 0.0`. This prevents the trainer from using them for optimization updates (training), but allows them to be run periodically for logging purposes (controlled by `eval_every`).

## How to Add a New Eval

### 1. Create the Eval Script

Create a new Python file in the `evals/` directory (e.g., `eval_my_task.py`).

You have two main options for implementation:

#### Option A: Wrap an Existing Environment

This is the most common approach. You import `load_environment` from a training environment and wrap it to enforce static parameters. This allows you to track progress on a specific task using a fixed holdout set.

```python
# evals/eval_math_static.py
from environments.self_prediction.self_prediction import load_environment as original_load_environment

def load_environment(**kwargs):
    """
    Static wrapper for Self-Prediction environment.
    """
    # Enforce deterministic behavior
    kwargs["seed"] = 42
    kwargs["num_examples"] = 50

    # Pass other kwargs through, but the forced ones take precedence if handled correctly
    # (or simply overwrite them in the kwargs dict before passing)
    return original_load_environment(**kwargs)
```

#### Option B: Create a Standalone Environment

Useful for sanity checks, regression tests, or specific benchmarks that do not share code with training tasks.

```python
# evals/eval_sanity_check.py
from datasets import Dataset
import verifiers as vf

def _get_dataset():
    # Create a fixed dataset manually
    return [
        {"question": "Calculate 2 * 2", "answer": "4"},
        {"question": "Calculate 5 * 5", "answer": "25"},
        # ... more examples ...
    ]

def _exact_match_reward(prompt, completion, answer, state):
    # Your custom reward logic
    if answer in str(completion):
        return 1.0
    return 0.0

def load_environment(**kwargs):
    dataset = Dataset.from_list(_get_dataset())
    rubric = vf.Rubric(funcs=[_exact_match_reward])

    # Return a standard Verifiers environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs
    )
```

### 2. Register in Configuration

Add your new eval to the training configuration file (e.g., `configs/multi_task.toml`).

The `id` field should point to the relative path of your script (without `.py`), and `weight` must be `0.0`.

```toml
[[envs]]
name = "eval_my_task"
id = "./evals/eval_my_task"  # Relative path to your script
weight = 0.0                 # MUST be 0.0 for pure evaluation
```

## Running Evals

Evals are run automatically during training if they are included in the configuration. The trainer checks the `eval_every` parameter in the `[trainer]` section to decide how often to run them.

For example, if your config has:

```toml
[trainer]
eval_every = 200
```

The trainer will:
1.  Train for 200 batches.
2.  Pause training.
3.  Run a rollout on all environments defined with `weight = 0.0`.
4.  Log the results (rewards, metrics).
5.  Resume training.

## Best Practices

*   **Determinism**: Always set a fixed seed (e.g., `kwargs['seed'] = 42`) in your `load_environment` function. This ensures that changes in metrics are due to model changes, not dataset shuffling.
*   **Size**: Keep `num_examples` reasonable (e.g., 20-100). Since evals run frequently during training loops, large evaluation sets can significantly slow down the overall training process.
*   **Isolation**: Ensure your eval doesn't depend on global state or mutable external files that might change during training.
