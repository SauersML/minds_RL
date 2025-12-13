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
            "difficulty": "easy",
            "question": f"Compute {a} {op} {b}. Output the answer only.",
            "answer": str(ans),
            "grader": "numeric"
        })

    for _ in range(30):
        # String copy / simple transform
        word = rng.choice(["apple", "banana", "cherry", "date", "elderberry"])
        dataset.append({
            "difficulty": "easy",
            "question": f"Repeat the word '{word}' exactly.",
            "answer": word,
            "grader": "exact"
        })

    # Hard Items
    for _ in range(60):
        # Larger multiplication
        a = rng.randint(100, 999)
        b = rng.randint(11, 99)
        dataset.append({
            "difficulty": "hard",
            "question": f"Compute {a} * {b}. Output the answer only.",
            "answer": str(a * b),
            "grader": "numeric"
        })

    return dataset

class SanityEnv(vf.SingleTurnEnv):
    """
    Subclass of SingleTurnEnv that injects dataset metadata (difficulty, grader)
    into the state/info so the rubric can access it.
    """
    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        # The parent step calls rollout and then rubric.
        # We want to intercept the result to ensure metadata is available.
        # SingleTurnEnv.step() implementation typically:
        # 1. Get example from dataset
        # 2. rollout()
        # 3. rubric.score_rollout()
        # 4. Return formatted observation, reward, etc.

        # Actually, SingleTurnEnv.step in verifiers might not be easily hookable
        # to inject data *before* rubric without overriding the whole logic.
        # However, SingleTurnEnv uses `self.dataset[self.example_id]` or similar.

        # Let's rely on the fact that `rubric.score_rollout` receives `state`.
        # `state` usually contains `input` which has `example_id`.
        # We can look up the dataset using `example_id` inside the reward function
        # IF we have access to the dataset instance.
        # The reward function is defined outside. We can bind the dataset to it.
        return super().step(action)

def _sanity_reward(
    prompt: Any,
    completion: Any,
    answer: str,
    state: Any,
    **kwargs: Any
) -> float:
    # We need access to the dataset item to know the grader and difficulty.
    # We can attach the dataset to the function or use a closure.
    # The `load_environment` function creates the dataset.

    # Retrieve dataset from the bound rubric or closure?
    # We'll rely on `_sanity_reward` being a closure inside `load_environment`
    # OR we attach dataset to `rubric`.

    # Let's access dataset from `kwargs` if possible, but SingleTurnEnv rubric call
    # might not pass it.
    # However, `state` often has `rollout_input` which has `example_id`.
    # Let's assume we can get `example_id`.

    example_id = state.get("example_id")
    if example_id is None:
        # Fallback if example_id is missing
        # Try finding it in info?
        info = state.get("info", {})
        example_id = info.get("example_id")

    dataset_item = None
    if isinstance(kwargs.get("dataset"), Dataset) and example_id is not None:
        dataset_item = kwargs.get("dataset")[example_id]

    # Parse completion
    parser = BasicSanityParser()
    if isinstance(completion, list):
        content = completion[-1]["content"]
    else:
        content = str(completion)
    parsed = parser.parse(content)
    prediction = parsed.get("answer", "")

    # Grading
    grader_type = "exact"
    difficulty = "unknown"

    if dataset_item:
        grader_type = dataset_item.get("grader", "exact")
        difficulty = dataset_item.get("difficulty", "unknown")

    if grader_type == "numeric":
        score = grade_numeric(prediction, answer)
    else:
        score = grade_exact(prediction, answer)

    # Metrics
    if "metrics" not in state:
        state["metrics"] = {}

    state["metrics"]["is_correct"] = float(score)

    # Log specific metrics that can be averaged
    if difficulty == "easy":
        state["metrics"]["easy_correct"] = float(score)
        # We log a count so we can compute mean later if the logger supports it.
        # If logger just averages all keys, `easy_correct` will be averaged over ALL samples
        # (with implicit 0 or NaN?).
        # If we return NaN for non-easy, and logger handles NaNs, great.
        # If we return 0.0 for non-easy, the average is wrong.
        # Tinker logger (MLFlow/WandB) usually ignores None/NaN or averages what is present.
        # We'll set the other to None.
        state["metrics"]["hard_correct"] = None
    elif difficulty == "hard":
        state["metrics"]["hard_correct"] = float(score)
        state["metrics"]["easy_correct"] = None

    return float(score)

def load_environment(**kwargs):
    dataset_list = _generate_sanity_dataset()
    dataset = Dataset.from_list(dataset_list)

    # Create a closure or partial to pass dataset to reward
    # Or just use the global dataset if we trust single process, but better to be safe.
    # We can create a class-based rubric or wrap the function.

    def bound_sanity_reward(prompt, completion, answer, state, **rubric_kwargs):
        # Inject dataset into kwargs for the inner function
        return _sanity_reward(prompt, completion, answer, state, dataset=dataset, **rubric_kwargs)

    rubric = vf.Rubric(funcs=[bound_sanity_reward])

    return SanityEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs
    )
