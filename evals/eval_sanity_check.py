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
    async def rollout(
        self,
        input: vf.RolloutInput,
        client: vf.Client,
        model: str,
        sampling_args: dict[str, Any],
    ) -> dict[str, Any]:
        # Perform the standard rollout
        # Note: input contains 'example_id'.
        # We can look up the dataset here and inject into info.

        example_id = input.get("example_id")
        if example_id is not None and self.dataset:
            item = self.dataset[example_id]
            # Inject metadata into input['info'] so it flows to state/rubric
            if "info" not in input:
                input["info"] = {}
            if item:
                input["info"]["difficulty"] = item.get("difficulty")
                input["info"]["grader"] = item.get("grader")

        return await super().rollout(input, client, model, sampling_args)

def _sanity_reward(
    prompt: Any,
    completion: Any,
    answer: str,
    state: Any,
    info: Mapping[str, Any] | None = None,
    **kwargs: Any
) -> float:
    # Retrieve difficulty/grader from info
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

    return SanityEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs
    )
