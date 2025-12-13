import random
from datasets import Dataset
import verifiers as vf

def _get_dataset():
    """Generates a static in-memory dataset of 20 multiplication problems."""
    # Use a fixed seed for reproducibility (sanity check should be deterministic)
    rng = random.Random(999)
    dataset = []
    for _ in range(20):
        a = rng.randint(2, 12)
        b = rng.randint(2, 12)
        dataset.append({
            "question": f"Calculate {a} * {b}",
            "answer": str(a * b)
        })
    return dataset

def _exact_match_reward(prompt, completion, answer, state):
    """Simple reward function checking if the answer is present in the completion."""
    # Extract text content from completion (handles list[dict] or str)
    if isinstance(completion, list):
        # Assume chat format, get last message content
        text = completion[-1]["content"]
    else:
        text = str(completion)

    # Check if the correct answer number is in the text
    # We look for the exact string representation of the answer
    if answer in text:
        return 1.0
    return 0.0

def load_environment(**kwargs):
    """
    Loads the Arithmetic Safety Monitor environment.
    This is a standalone sanity check that does not depend on other environment modules.
    """
    dataset = Dataset.from_list(_get_dataset())
    rubric = vf.Rubric(funcs=[_exact_match_reward])

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs
    )
