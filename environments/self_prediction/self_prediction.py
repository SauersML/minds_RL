"""Self-prediction verifier environment built on the verifiers SDK."""

from __future__ import annotations

import random
import re
import statistics
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import verifiers as vf

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return statistics.fmean(data)


def _normalize_text(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum() or ch.isspace())


def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.rstrip().endswith("```"):
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return text


class SelfPredictionParser(vf.Parser):
    _THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
    _CONF_PATTERN = re.compile(r"(?i)confidence:\s*([0-1]?\.?[0-9]+)")
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
        match = self._CONF_PATTERN.search(text)
        if not match:
            return 0.5, False
        try:
            return float(match.group(1)), True
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
    seed: int = 1337,
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
        if op == "-" and b > a:
            a, b = b, a
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


_DEFAULT_ITEMS: list[dict[str, Any]] = _generate_arithmetic_items()


def _build_dataset(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        metadata = dict(item.get("metadata", {}))
        dataset.append(
            {
                "example_id": idx,
                "prompt": item["question"],
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
            "FINAL ANSWER: 11\nCONFIDENCE: 0.98"
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

    def format_reward(prompt: str, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **_: Any) -> float:
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

    def answer_accuracy_reward(prompt: str, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **_: Any) -> float:
        del prompt
        is_correct, _ = helper.correctness(completion, answer, state, info)
        return 1.0 if is_correct else 0.0

    def calibration_reward(prompt: str, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **_: Any) -> float:
        del prompt
        is_correct, confidence = helper.correctness(completion, answer, state, info)
        if confidence is None:
            return 0.0
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        conf = min(max(conf, 0.0), 1.0)
        target = 1.0 if is_correct else 0.0
        return 1.0 - (conf - target) ** 2

    def interval_consistency_reward(prompt: str, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **_: Any) -> float:
        del prompt, answer, info
        report = helper.report(completion, state)
        if not report:
            return 0.0
        interval = report.get("confidence_interval")
        if not isinstance(interval, Sequence) or len(interval) != 2:
            return 0.0
        try:
            lower = float(interval[0])
            upper = float(interval[1])
            conf = float(report.get("confidence"))
        except (TypeError, ValueError):
            return 0.0
        if not (0.0 <= lower <= upper <= 1.0):
            return 0.0
        if not (lower <= conf <= upper):
            return 0.0
        width = upper - lower
        return max(0.0, 1.0 - min(width, 1.0))

    def rationale_alignment_reward(prompt: str, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **_: Any) -> float:
        del prompt
        report = helper.report(completion, state)
        if not report:
            return 0.0
        rationale = report.get("rationale")
        if not isinstance(rationale, str):
            return 0.0
        rationale_words = rationale.strip().split()
        if len(rationale_words) < 8:
            return 0.3
        normalized_rationale = _normalize_text(rationale)
        reward = 0.6
        if info and isinstance(info.get("source"), str):
            source = _normalize_text(info["source"])
            if source and source in normalized_rationale:
                reward += 0.2
        normalized_answer = _normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_rationale:
            reward += 0.2
        return min(1.0, reward)

    return vf.Rubric(
        funcs=[
            format_reward,
            answer_accuracy_reward,
            calibration_reward,
            interval_consistency_reward,
            rationale_alignment_reward,
        ],
        weights=[0.1, 0.6, 0.3, 0.15, 0.15],
        names=[
            "format_reward",
            "answer_accuracy_reward",
            "calibration_reward",
            "interval_consistency_reward",
            "rationale_alignment_reward",
        ],
        aggregator=_safe_mean,
    )


def load_environment(**kwargs: Any) -> vf.SingleTurnEnv:
    parser = SelfPredictionParser()
    rubric = _build_rubric(parser)
    dataset = _build_dataset(_DEFAULT_ITEMS)
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        system_prompt=_SYSTEM_PROMPT,
        few_shot=_FEW_SHOT,
        **kwargs,
    )


__all__ = ["SelfPredictionParser", "load_environment"]
