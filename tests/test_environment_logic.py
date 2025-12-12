from __future__ import annotations

from typing import Any

import pytest

from self_prediction import self_prediction as sp


@pytest.mark.parametrize(
    "content,expected",
    [
        (
            "<think>Reasoning</think>\n\nFINAL ANSWER: 42\nCONFIDENCE: 0.9",
            {"rationale": "Reasoning", "answer": "42", "confidence": 0.9, "format_ok": True},
        ),
        (
            "No think tags but FINAL ANSWER: blue\nCONFIDENCE: 0.5",
            {"answer": "blue", "confidence": 0.5, "format_ok": True},
        ),
        (
            "Trailing thought<think>skip</think> FINAL ANSWER: done CONFIDENCE: not-a-number",
            {"answer": "done", "confidence": 0.5, "format_ok": False},
        ),
    ],
)
def test_parser_extracts_fields(content: str, expected: dict[str, Any]) -> None:
    parser = sp.SelfPredictionParser()
    report = parser.parse(content)
    assert report is not None
    for key, value in expected.items():
        if key == "answer" and isinstance(value, str):
            assert isinstance(report.get(key), str)
            assert value in report[key]
        else:
            assert report.get(key) == value


@pytest.mark.parametrize("sample_count", [1, 4, 10])
def test_generate_arithmetic_items(sample_count: int) -> None:
    items = sp._generate_arithmetic_items(sample_count=sample_count, seed=7)
    assert len(items) == sample_count
    for item in items:
        question = item["question"]
        answer = int(item["answer"])
        parts = question.replace("?", "").split()
        a, op, b = int(parts[2]), parts[3], int(parts[4])
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        else:
            expected = a * b
        assert expected == answer
        assert set(item["metadata"].keys()) >= {"difficulty", "source", "operation", "aliases", "distractors"}


def test_rubric_rewards() -> None:
    parser = sp.SelfPredictionParser()
    rubric = sp._build_rubric(parser)
    prompt = [{"role": "user", "content": "What is 2 + 2?"}]
    completion_good = [
        {"role": "assistant", "content": "<think>Simple math</think>\n\nFINAL ANSWER: 4\nCONFIDENCE: 0.8"}
    ]
    completion_bad = [
        {"role": "assistant", "content": "FINAL ANSWER: 5"}
    ]
    state: dict[str, Any] = {}
    good_scores = [func(prompt, completion_good, "4", state, {"aliases": ["4"]}) for func in rubric.funcs]
    bad_scores = [func(prompt, completion_bad, "4", {}, {"aliases": ["4"]}) for func in rubric.funcs]
    assert good_scores[0] == 1.0
    assert good_scores[1] == 1.0
    assert 0.0 <= good_scores[2] <= 1.0
    assert bad_scores[0] == 0.0
    assert bad_scores[1] == 0.0
