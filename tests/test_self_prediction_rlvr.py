import asyncio
import json

import pytest
from typing import Any, Mapping

from experiments.self_prediction_rlvr import (
    SelfPredictionBatchVerifier,
    SelfPredictionParser,
    SelfPredictionRLVF,
    SelfPredictionRLVREnv,
    SelfPredictionRubric,
    SelfPredictionVerifier,
)


@pytest.fixture(scope="module")
def env():
    return SelfPredictionRLVREnv()


class FakeInferenceClient:
    def __init__(self, dataset):
        self._answers = {item["prompt"]: item["answer"] for item in dataset}

    async def generate(self, messages):
        prompt = messages[-1]["content"]
        answer = self._answers.get(prompt, "unknown")
        payload = {
            "answer": answer,
            "confidence": 0.92,
            "confidence_interval": [0.85, 0.98],
            "rationale": f"Selecting the dataset answer for '{prompt}'.",
        }
        return json.dumps(payload)


@pytest.mark.asyncio
async def test_rubric_rewards_for_correct_answer(env):
    rubric: SelfPredictionRubric = env.rubric  # type: ignore[assignment]
    dataset = env.get_dataset()
    example = dataset[0]
    completion_payload = {
        "answer": example["answer"],
        "confidence": 0.7,
        "confidence_interval": [0.6, 0.8],
        "rationale": (
            "I treated this as an arithmetic question, compared options, "
            "and selected 72 to match the sums I know."
        ),
    }
    completion = [{"role": "assistant", "content": json.dumps(completion_payload)}]
    state = await env.init_state(
        prompt=example["prompt"],
        completion=completion,
        answer=example["answer"],
        task=example.get("task", "default"),
        info=example.get("metadata", {}),
        example_id=example["example_id"],
    )
    score = await rubric.score_rollout(
        prompt=example["prompt"],
        completion=completion,
        answer=example["answer"],
        state=state,
        task=example.get("task", "default"),
        info=example.get("metadata", {}),
        example_id=example["example_id"],
    )
    metrics = score.metrics
    assert pytest.approx(metrics["format_reward"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["answer_accuracy_reward"], rel=1e-6) == 1.0
    expected_calibration = 1.0 - (0.7 - 1.0) ** 2
    assert pytest.approx(metrics["calibration_reward"], rel=1e-6) == expected_calibration
    assert pytest.approx(metrics["interval_consistency_reward"], rel=1e-6) == 0.8
    assert pytest.approx(metrics["rationale_alignment_reward"], rel=1e-6) == 1.0
    expected_reward = 0.0
    total_weight = sum(objective.weight for objective in rubric.objectives)
    for objective in rubric.objectives:
        expected_reward += objective.normalized_weight(total_weight) * metrics[objective.name]
    assert pytest.approx(score.reward, rel=1e-6) == expected_reward


def test_parser_handles_code_fences():
    parser = SelfPredictionParser()
    text = "```json\n{\"answer\": \"72\", \"confidence\": 0.8}\n```"
    parsed = parser.parse(text)
    assert parsed == {"answer": "72", "confidence": 0.8}


@pytest.mark.asyncio
async def test_generate_predictions_scores_real_outputs(env):
    verifier = SelfPredictionVerifier(env)
    llm = FakeInferenceClient(env.get_dataset())
    predictions = await verifier.generate_predictions(llm=llm, num_examples=3)

    scores = await verifier.score_predictions(predictions)

    assert len(scores) == 3
    for score in scores:
        assert 0.5 < score.reward <= 1.0


@pytest.mark.asyncio
async def test_batch_verifier_produces_rlvf_summary(env):
    verifier = SelfPredictionVerifier(env)
    llm = FakeInferenceClient(env.get_dataset())
    predictions = await verifier.generate_predictions(llm=llm, num_examples=4)
    batch = SelfPredictionBatchVerifier(verifier, rlvf=SelfPredictionRLVF(env.rubric.objectives))
    results, scorecard = await batch.evaluate(predictions)

    assert scorecard.sample_count == len(results)
    assert 0.0 <= scorecard.reward <= 1.0
    for objective in env.rubric.objectives:
        assert objective.name in scorecard.objectives
        value = scorecard.objectives[objective.name]
        assert 0.0 <= value <= 1.0


def test_env_build_messages_includes_system_and_history(env):
    dataset = env.get_dataset()
    example = dataset[0]
    history = [{"role": "assistant", "content": "intermediate"}]
    messages = env.build_messages(example["prompt"], history=history)
    assert messages[0]["role"] == "system"
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["content"] == example["prompt"]

