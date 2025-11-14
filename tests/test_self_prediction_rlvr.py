import asyncio
import json

import pytest

from experiments.self_prediction_rlvr import (
    SelfPredictionParser,
    SelfPredictionRLVREnv,
    SelfPredictionRubric,
    SelfPredictionVerifier,
)


@pytest.fixture(scope="module")
def env():
    return SelfPredictionRLVREnv()


@pytest.mark.asyncio
async def test_rubric_rewards_for_correct_answer(env):
    rubric: SelfPredictionRubric = env.rubric  # type: ignore[assignment]
    dataset = env.get_dataset()
    example = dataset[0]
    completion_payload = {
        "answer": example["answer"],
        "confidence": 0.7,
        "rationale": "Added the two numbers directly and double-checked the arithmetic.",
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
    expected_reward = (
        rubric.FORMAT_WEIGHT * 1.0
        + rubric.ACCURACY_WEIGHT * 1.0
        + rubric.CALIBRATION_WEIGHT * expected_calibration
    )
    assert pytest.approx(score.reward, rel=1e-6) == expected_reward


def test_parser_handles_code_fences():
    parser = SelfPredictionParser()
    text = "```json\n{\"answer\": \"72\", \"confidence\": 0.8}\n```"
    parsed = parser.parse(text)
    assert parsed == {"answer": "72", "confidence": 0.8}


@pytest.mark.asyncio
async def test_honest_stub_outperforms_overconfident(env):
    verifier = SelfPredictionVerifier(env)
    honest_predictions = verifier.stub_predictions(strategy="honest", seed=42)
    overconfident_predictions = verifier.stub_predictions(strategy="overconfident", seed=42)

    honest_scores = await verifier.score_predictions(honest_predictions)
    overconfident_scores = await verifier.score_predictions(overconfident_predictions)

    honest_avg = sum(r.reward for r in honest_scores) / len(honest_scores)
    overconfident_avg = sum(r.reward for r in overconfident_scores) / len(overconfident_scores)

    assert honest_avg > overconfident_avg
