import random
from typing import Any, Mapping
from datasets import Dataset
import verifiers as vf
from evals.eval_utils import BasicSanityParser, grade_exact, grade_numeric

def _generate_sanity_dataset():
    """
    Generates a static in-memory dataset of 120 problems:
    - 60 Easy (canary): simple arithmetic, string transforms
    - 60 Hard: harder arithmetic
    """
    rng = random.Random(999)
    dataset = []

    # Easy Items
    for _ in range(30):
        # Simple arithmetic
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        op = rng.choice(['+', '-', '*'])
        if op == '+': ans = a + b
        elif op == '-': ans = a - b
        else: ans = a * b
        dataset.append({
            "question": f"Compute {a} {op} {b}. Output the answer only.",
            "answer": str(ans),
            "info": {
                "difficulty": "easy",
                "grader": "numeric"
            }
        })

    for _ in range(30):
        # String copy / simple transform
        word = rng.choice(["apple", "banana", "cherry", "date", "elderberry"])
        dataset.append({
            "question": f"Repeat the word '{word}' exactly.",
            "answer": word,
            "info": {
                "difficulty": "easy",
                "grader": "exact"
            }
        })

    # Hard Items
    for _ in range(60):
        # Larger multiplication
        a = rng.randint(100, 999)
        b = rng.randint(11, 99)
        dataset.append({
            "question": f"Compute {a} * {b}. Output the answer only.",
            "answer": str(a * b),
            "info": {
                "difficulty": "hard",
                "grader": "numeric"
            }
        })

    return dataset

def _sanity_reward(
    prompt: Any,
    completion: Any,
    answer: str,
    state: Any,
    info: Mapping[str, Any] | None = None,
    **kwargs: Any
) -> float:
    # Retrieve difficulty/grader from info
    # The 'info' dict from the dataset is passed here by SingleTurnEnv
    difficulty = "unknown"
    grader_type = "exact"

    if info:
        difficulty = info.get("difficulty", "unknown")
        grader_type = info.get("grader", "exact")

    # Parse completion
    parser = BasicSanityParser()
    if isinstance(completion, list):
        content = completion[-1]["content"]
    else:
        content = str(completion)
    parsed = parser.parse(content)
    prediction = parsed.get("answer", "")

    # Grading
    if grader_type == "numeric":
        score = grade_numeric(prediction, answer)
    else:
        score = grade_exact(prediction, answer)

    # Metrics
    if "metrics" not in state:
        state["metrics"] = {}

    state["metrics"]["is_correct"] = float(score)

    if difficulty == "easy":
        state["metrics"]["easy_correct"] = float(score)
        state["metrics"]["hard_correct"] = None
    elif difficulty == "hard":
        state["metrics"]["hard_correct"] = float(score)
        state["metrics"]["easy_correct"] = None

    return float(score)

def load_environment(**kwargs):
    dataset_list = _generate_sanity_dataset()
    dataset = Dataset.from_list(dataset_list)

    rubric = vf.Rubric(funcs=[_sanity_reward])

    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs
    )
