"""Reference probes for the Gradient Intuition environment."""
from __future__ import annotations

import random
from typing import Mapping, Sequence

Probe = Mapping[str, str]


_PROBES: tuple[Probe, ...] = (
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 10 + 5?", "answer": "15"},
    {"question": "Name the largest planet in our solar system.", "answer": "Jupiter"},
    {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
    {"question": "What gas do plants primarily absorb for photosynthesis?", "answer": "Carbon dioxide"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "In computing, what does CPU stand for?", "answer": "Central processing unit"},
    {"question": "What is the square root of 81?", "answer": "9"},
    {"question": "What is H2O commonly known as?", "answer": "Water"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What currency is used in Japan?", "answer": "Yen"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What is the boiling point of water at sea level in Celsius?", "answer": "100"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "What is the primary language spoken in Brazil?", "answer": "Portuguese"},
    {"question": "Which ocean borders California?", "answer": "Pacific Ocean"},
    {"question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"question": "What instrument measures atmospheric pressure?", "answer": "Barometer"},
    {"question": "What is 12 times 12?", "answer": "144"},
    {"question": "Name the process by which a caterpillar becomes a butterfly.", "answer": "Metamorphosis"},
)


def all_probes() -> tuple[Probe, ...]:
    """Return the immutable tuple of available probes."""

    return _PROBES


def get_random_probe(rng: random.Random | None = None) -> Probe:
    """Return a random probe entry."""

    if rng is None:
        rng = random
    return rng.choice(_PROBES)


def sample_unique(count: int, rng: random.Random | None = None) -> Sequence[Probe]:
    """Return ``count`` distinct probes, wrapping around if necessary."""

    if count <= 0:
        return []
    pool = list(_PROBES)
    if rng is None:
        rng = random
    rng.shuffle(pool)
    if count <= len(pool):
        return pool[:count]
    # If more requested than available, repeat shuffled items
    repeats: list[Probe] = []
    while len(repeats) < count:
        repeats.extend(pool)
    return repeats[:count]


__all__ = ["Probe", "all_probes", "get_random_probe", "sample_unique"]
