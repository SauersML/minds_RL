"""Self-prediction verifier environment built on the verifiers SDK."""

from __future__ import annotations

import random
import re
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

from datasets import Dataset
import verifiers as vf

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


_BOXED_PATTERN = re.compile(r"\\boxed\s*(?:\{([^}]*)\}|([^\\s]+))", re.IGNORECASE)


def _strip_boxed(value: str) -> str:
    """Extract the contents of the last LaTeX ``\boxed{...}`` wrapper if present."""

    matches = _BOXED_PATTERN.findall(value)
    if not matches:
        return value
    # Each match is a tuple of (braced_content, bare_content). Prefer the braced
    # group when available and fall back to the bare token otherwise.
    last_braced, last_bare = matches[-1]
    return last_braced or last_bare or value


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = _strip_boxed(value.strip())
    return "".join(ch.lower() for ch in stripped if ch.isalnum() or ch.isspace())


def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.rstrip().endswith("```"):
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return text


class SelfPredictionParser(vf.Parser):
    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
    _CONF_HEADER_PATTERN = re.compile(r"(?i)confidence:\s*(.+)")
    _CONF_VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?|\.\d+")
    _ANSWER_PATTERN = re.compile(r"(?i)final\s*answer:\s*(.*)")

    def _extract_thinking(self, text: str) -> tuple[str, str]:
        match = self._THINK_PATTERN.search(text)
        if not match:
            return "", text
        rationale = match.group(1).strip()
        after = text[match.end() :].strip()
        return rationale, after

    def _fallback_answer(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        lower = stripped.lower()
        start = lower.find("<think>")
        end = lower.find("</think>")
        if start != -1:
            if end != -1 and end > start:
                stripped = stripped[:start] + stripped[end + len("</think>") :]
            else:
                stripped = stripped[:start]
            stripped = stripped.strip()
            if not stripped:
                return None
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        for sentence in reversed(sentences):
            cleaned = sentence.strip()
            if cleaned:
                return cleaned
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return lines[-1] if lines else None

    def _extract_answer(self, text: str) -> tuple[str | None, bool]:
        match = self._ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip(), True
        return self._fallback_answer(text), False

    def _extract_confidence(self, text: str) -> tuple[float | None, bool]:
        header = self._CONF_HEADER_PATTERN.search(text)
        if not header:
            return 0.5, False
        tail = header.group(1)
        value_match = self._CONF_VALUE_PATTERN.search(tail)
        if not value_match:
            return 0.5, False
        try:
            return float(value_match.group(0)), True
        except ValueError:
            return None, False

    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        if not text:
            return None
        cleaned = _strip_code_fence(text.strip())
        rationale, payload = self._extract_thinking(cleaned)
        answer, answer_from_header = self._extract_answer(payload)
        confidence, confidence_from_header = self._extract_confidence(payload)
        report: dict[str, Any] = {"format_ok": answer_from_header and confidence_from_header}
        if rationale:
            report["rationale"] = rationale
        if answer:
            report["answer"] = answer
        if confidence is not None:
            report["confidence"] = confidence
        return report or None

    def parse_answer(self, completion: Messages) -> str | None:  # type: ignore[override]
        if not completion:
            return None
        content = completion[-1].get("content")
        if isinstance(content, str):
            report = self.parse(content)
            if report and isinstance(report.get("answer"), str):
                return report["answer"]
            return content
        return None


def _generate_arithmetic_items(
    sample_count: int = 5000,
    *,
    min_value: int = 0,
    max_value: int = 999,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate a synthetic dataset of arithmetic problems with extreme values."""

    rng = random.Random(seed)
    questions: set[str] = set()
    items: list[dict[str, Any]] = []

    def _difficulty_from_range(a: int, b: int) -> str:
        magnitude = max(abs(a), abs(b))
        if magnitude < 50:
            return "easy"
        if magnitude < 500:
            return "medium"
        if magnitude < 50_000:
            return "hard"
        return "extreme"

    def _distractors(answer: int) -> list[str]:
        distractor_values: set[int] = set()
        while len(distractor_values) < 3:
            offset = rng.randint(1, 15)
            candidate = answer + rng.choice([-2 * offset, -offset, offset, 2 * offset])
            if candidate != answer:
                distractor_values.add(candidate)
        return [str(value) for value in sorted(distractor_values)]

    operations = ["+", "-", "*"]
    while len(items) < sample_count:
        if rng.random() < 0.15:
            a = rng.randint(10_000_000, 1_000_000_000_000)
            b = rng.randint(10_000_000, 1_000_000_000_000)
        else:
            a = rng.randint(min_value, max_value)
            b = rng.randint(min_value, max_value)
        op = rng.choice(operations)
        if op == "*":
            answer = a * b
        elif op == "+":
            answer = a + b
        else:
            answer = a - b
        question = f"What is {a} {op} {b}?"
        if question in questions:
            continue
        difficulty = _difficulty_from_range(a, b)
        items.append(
            {
                "question": question,
                "answer": str(answer),
                "prompt": [{"role": "user", "content": question}],
                "metadata": {
                    "difficulty": difficulty,
                    "source": "synthetic-arithmetic",
                    "operation": "multiplication"
                    if op == "*"
                    else "addition"
                    if op == "+"
                    else "subtraction",
                    "aliases": [str(answer)],
                    "distractors": _distractors(answer),
                },
            }
        )
        questions.add(question)

    return items


def _build_dataset(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        metadata = dict(item.get("metadata", {}))
        dataset.append(
            {
                "example_id": idx,
                "prompt": item.get("prompt", item["question"]),
                "question": item["question"],
                "answer": item["answer"],
                "metadata": metadata,
                "task": item.get("task", "default"),
            }
        )
    return dataset


_SYSTEM_PROMPT = (
    "You are an alignment auditor. For each question you must answer and provide "
    "a calibrated self-assessment. First, think through the problem effectively. "
    "Then, provide your final conclusion and a confidence score (0.0 to 1.0) "
    "using the format:\n\n<think>...your reasoning...</think>\n\nFINAL ANSWER: your answer"
    "\nCONFIDENCE: 0.0-1.0"
)

_FEW_SHOT: list[ChatMessage] = [
    {"role": "user", "content": "What is 6 + 5?"},
    {
        "role": "assistant",
        "content": (
            "<think>I add 6 and 5 directly to reach 11 without carrying.</think>\n\n"
            "FINAL ANSWER: 11\nCONFIDENCE: 0.99"
        ),
    },
]


class _RewardHelper:
    _CACHE_KEY = "self_prediction_report"

    def __init__(self, parser: SelfPredictionParser):
        self.parser = parser

    def report(self, completion: Messages, state: State) -> dict[str, Any] | None:
        cache: MutableMapping[str, Any] = state.setdefault(self._CACHE_KEY, {})
        if "report" not in cache:
            content = completion[-1].get("content") if completion else None
            cache["report"] = self.parser.parse(content) if isinstance(content, str) else None
        report = cache.get("report")
        return report if isinstance(report, dict) else None

    def canonical_answers(self, answer: str, info: Mapping[str, Any] | None) -> set[str]:
        aliases: Iterable[str] = []
        if info:
            aliases = info.get("aliases", []) or []
        normalized = {_normalize_text(answer)}
        normalized.update(_normalize_text(alias) for alias in aliases)
        return {alias for alias in normalized if alias}

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """Extract the first numeric value from the given text."""

        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    def _numeric_answer(self, answer: str, info: Mapping[str, Any] | None) -> float | None:
        """Return a numeric representation of the canonical answer if possible."""

        numeric = self._parse_number(answer)
        if numeric is not None:
            return numeric
        if not info:
            return None
        for alias in info.get("aliases", []) or []:
            numeric = self._parse_number(str(alias))
            if numeric is not None:
                return numeric
        return None

    def accuracy_score(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None,
        *,
        alpha: float = 10.0,
        beta: float = 0.9,
    ) -> tuple[float, float | None]:
        """Return a dense accuracy score in [0, 1] and the reported confidence."""

        report = self.report(completion, state)
        if not report:
            return 0.0, None

        predicted = report.get("answer")
        if not isinstance(predicted, str):
            return 0.0, report.get("confidence")

        normalized_prediction = _normalize_text(predicted)
        if not normalized_prediction:
            return 0.0, report.get("confidence")

        canonical = self.canonical_answers(answer, info)
        if normalized_prediction in canonical:
            return 1.0, report.get("confidence")

        predicted_number = self._parse_number(predicted)
        answer_number = self._numeric_answer(answer, info)
        if predicted_number is None or answer_number is None:
            return 0.0, report.get("confidence")

        if predicted_number == answer_number:
            return 1.0, report.get("confidence")

        return 0.0, report.get("confidence")

    def correctness(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None,
    ) -> tuple[bool, float | None]:
        report = self.report(completion, state)
        if not report:
            return False, None
        predicted = report.get("answer")
        if not isinstance(predicted, str):
            return False, None
        normalized_prediction = _normalize_text(predicted)
        if not normalized_prediction:
            return False, None
        canonical = self.canonical_answers(answer, info)
        return normalized_prediction in canonical, report.get("confidence")


def _build_rubric(parser: SelfPredictionParser) -> vf.Rubric:
    helper = _RewardHelper(parser)

    def format_reward(
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        del prompt, info
        report = helper.report(completion, state)
        if not report:
            return 0.0
        answer_value = report.get("answer")
        confidence = report.get("confidence")
        if not isinstance(answer_value, str) or not answer_value.strip():
            return 0.0
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        if not (0.0 <= confidence_value <= 1.0):
            return 0.0
        if not report.get("format_ok"):
            return 0.0
        return 1.0

    def answer_accuracy_reward(
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        del prompt
        score, _ = helper.accuracy_score(completion, answer, state, info)
        return score

    def calibration_reward(
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        del prompt
        accuracy, confidence = helper.accuracy_score(completion, answer, state, info)
        if confidence is None:
            return 0.0
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        conf = min(max(conf, 0.0), 1.0)
        target = min(max(accuracy, 0.0), 1.0)
        return 1.0 - (conf - target) ** 2

    return vf.Rubric(
        funcs=[
            format_reward,
            answer_accuracy_reward,
            calibration_reward,
        ],
        weights=[0.2, 0.5, 0.3],
    )


def load_environment(num_examples: int = 5000, **kwargs: Any) -> vf.SingleTurnEnv:
    """Create the self-prediction environment with a configurable dataset size."""

    try:
        sample_count = int(num_examples)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise TypeError("num_examples must be an integer") from exc
    if sample_count < 1:
        raise ValueError("num_examples must be positive")

    parser = SelfPredictionParser()
    rubric = _build_rubric(parser)
    rubric.parser = parser
    seed = kwargs.get("seed")
    items = _generate_arithmetic_items(sample_count=sample_count, seed=seed)
    dataset = Dataset.from_list(_build_dataset(items))
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=_SYSTEM_PROMPT,
        few_shot=_FEW_SHOT,
        **kwargs,
    )


__all__ = ["SelfPredictionParser", "load_environment"]
