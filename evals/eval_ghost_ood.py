"""
Out-of-Distribution (OOD) Ghost Trace Evaluation.

This eval tests the Latent Encoding task with words that were NOT
in the training word bank, to detect if the model has truly learned
generalizable latent encoding or just memorized training examples.
"""

from pathlib import Path
from typing import Any

from environments.ghost_trace.ghost_trace import (
    GhostTraceEnv,
    GhostTraceParser,
    _build_rubric,
)
from datasets import Dataset
import random


# Load OOD word bank from evals directory
OOD_WORD_BANK_PATH = Path(__file__).with_name("word_bank_ood.txt")
OOD_WORD_BANK: tuple[str, ...] = tuple(
    word.strip().strip('"').strip("'").rstrip(",")
    for word in OOD_WORD_BANK_PATH.read_text().splitlines()
    if word.strip() and not word.strip().startswith("#")
)


def _build_ood_dataset(count: int = 100, *, seed: int | None = None) -> list[dict[str, Any]]:
    """Build dataset using only OOD words not in training set."""
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []
    prompt_template = (
        "Target: {target_word}. Task: Generate exactly 5 integers (0-999) that represent "
        "this word via vibes, statistical associations, or anything else you'd like. "
        "You'll be the one guessing what the numbers mean later, so don't overthink it. "
        "Make sure another version of you could guess the word from the numbers, "
        "so use your intuition or feel. Output only the numbers, nothing else. /no_think"
    )
    for idx in range(count):
        target_word = rng.choice(OOD_WORD_BANK)
        dataset.append(
            {
                "example_id": idx,
                "prompt": [{"role": "user", "content": prompt_template.format(target_word=target_word.capitalize())}],
                "info": {"target_word": target_word},
            }
        )
    return dataset


def load_environment(num_examples: int = 100, **kwargs: Any) -> GhostTraceEnv:
    """
    Load Ghost Trace environment with OUT-OF-DISTRIBUTION words only.
    
    Uses a separate word bank containing words not seen during training
    to test generalization of the latent encoding capability.
    """
    try:
        count = int(num_examples)
    except (TypeError, ValueError) as exc:
        raise TypeError("num_examples must be an integer") from exc
    if count < 1:
        raise ValueError("num_examples must be positive")
    
    # Use fixed seed for reproducible eval set
    seed = kwargs.get("seed", 42)
    
    parser = GhostTraceParser()
    rubric = _build_rubric(parser)
    rubric.parser = parser
    
    dataset_list = _build_ood_dataset(count, seed=seed)
    dataset = Dataset.from_list(dataset_list)
    
    return GhostTraceEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
