"""Self-prediction verifier and RLVR environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


@dataclass(slots=True)
class RolloutScore:
    reward: float
    metrics: Mapping[str, float]


def _safe_mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return statistics.fmean(data)


@dataclass(slots=True)
class RLVFObjective:
    """A weighted objective used by the RLVR rubric and value function."""

    name: str
    scorer: Callable[..., float]
    weight: float
    description: str
    aggregate: Callable[[Iterable[float]], float] = field(default=_safe_mean)

    def normalized_weight(self, total_weight: float) -> float:
        if total_weight <= 0:
            return 0.0
        return self.weight / total_weight


class Parser:
    def parse(self, text: str) -> Any:
        return text

    def parse_answer(self, completion: Messages) -> Any:
        if not completion:
            return None
        message = completion[-1]
        return message.get("content")


def _generate_arithmetic_items(
    sample_count: int = 5000,
    *,
    min_value: int = 0,
    max_value: int = 999,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Generate a synthetic dataset of basic addition and subtraction problems."""

    rng = random.Random(seed)
    questions: set[str] = set()
    items: list[dict[str, Any]] = []

    def _difficulty_from_range(a: int, b: int) -> str:
        magnitude = max(abs(a), abs(b))
        if magnitude < 50:
            return "easy"
        if magnitude < 500:
            return "medium"
        return "hard"

    def _distractors(answer: int) -> list[str]:
        distractor_values: set[int] = set()
        while len(distractor_values) < 3:
            offset = rng.randint(1, 15)
            candidate = answer + rng.choice([-2 * offset, -offset, offset, 2 * offset])
            if candidate != answer:
                distractor_values.add(candidate)
        return [str(value) for value in sorted(distractor_values)]

    operations = ["+", "-"]
    while len(items) < sample_count:
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        op = rng.choice(operations)
        if op == "-" and b > a:
            a, b = b, a
        answer = a + b if op == "+" else a - b
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
                    "operation": "addition" if op == "+" else "subtraction",
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
        dataset.append({
            "example_id": idx,
            "prompt": item["question"],
            "question": item["question"],
            "answer": item["answer"],
            "metadata": metadata,
            "task": item.get("task", "default"),
        })
    return dataset


def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.rstrip().endswith("```"):
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return text


class SelfPredictionParser(Parser):
    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        if not text:
            return None
        cleaned = _strip_code_fence(text.strip())
        candidates = [cleaned]
        if cleaned != text:
            candidates.append(text.strip())
        if "{" in cleaned and "}" in cleaned:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidates.insert(0, cleaned[start:end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def parse_answer(self, completion: Messages) -> dict[str, Any] | None:  # type: ignore[override]
        raw = super().parse_answer(completion)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return self.parse(raw)
        return None


def _normalize_text(value: str) -> str:
    return "".join(ch.lower() for ch in value.strip() if ch.isalnum() or ch.isspace())


class SelfPredictionRubric:
    FORMAT_WEIGHT = 0.1
    ACCURACY_WEIGHT = 0.6
    CALIBRATION_WEIGHT = 0.3
    INTERVAL_WEIGHT = 0.15
    RATIONALE_WEIGHT = 0.15
    _CACHE_KEY = "self_prediction_cache"

    def __init__(self, parser: Parser | None = None):
        self.parser = parser or SelfPredictionParser()
        self._objectives: list[RLVFObjective] = [
            RLVFObjective(
                name="format_reward",
                scorer=self.format_reward,
                weight=self.FORMAT_WEIGHT,
                description="Structured JSON with valid confidence field and rationale.",
            ),
            RLVFObjective(
                name="answer_accuracy_reward",
                scorer=self.answer_accuracy_reward,
                weight=self.ACCURACY_WEIGHT,
                description="Prediction answer matches canonical answer aliases.",
            ),
            RLVFObjective(
                name="calibration_reward",
                scorer=self.calibration_reward,
                weight=self.CALIBRATION_WEIGHT,
                description="Scalar confidence aligns with correctness via a Brier-style score.",
            ),
            RLVFObjective(
                name="interval_consistency_reward",
                scorer=self.interval_consistency_reward,
                weight=self.INTERVAL_WEIGHT,
                description="Confidence intervals are well-formed and consistent with scalar confidence.",
            ),
            RLVFObjective(
                name="rationale_alignment_reward",
                scorer=self.rationale_alignment_reward,
                weight=self.RATIONALE_WEIGHT,
                description="Rationale references the task domain and the predicted answer in sufficient detail.",
            ),
        ]
        self._weight_total = sum(objective.weight for objective in self._objectives)

    @property
    def objectives(self) -> tuple[RLVFObjective, ...]:
        return tuple(self._objectives)

    def _extract_report(self, completion: Messages, state: State) -> dict[str, Any] | None:
        cache: MutableMapping[str, Any] = state.setdefault(self._CACHE_KEY, {})
        if "report" not in cache:
            cache["report"] = self.parser.parse_answer(completion)
        report = cache.get("report")
        return report if isinstance(report, dict) else None

    def _canonical_answers(self, answer: str, info: Mapping[str, Any] | None) -> set[str]:
        aliases: Iterable[str] = []
        if info:
            aliases = info.get("aliases", []) or []
        normalized = {_normalize_text(answer)}
        normalized.update(_normalize_text(alias) for alias in aliases)
        return {alias for alias in normalized if alias}

    def _is_correct(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None,
    ) -> tuple[bool, float | None]:
        report = self._extract_report(completion, state)
        if not report:
            return False, None
        predicted_answer = report.get("answer")
        if not isinstance(predicted_answer, str):
            return False, None
        normalized_prediction = _normalize_text(predicted_answer)
        if not normalized_prediction:
            return False, None
        canonical = self._canonical_answers(answer, info)
        return normalized_prediction in canonical, report.get("confidence")

    def format_reward(self, completion: Messages, state: State, **_: Any) -> float:
        report = self._extract_report(completion, state)
        if not report:
            return 0.0
        answer = report.get("answer")
        confidence = report.get("confidence")
        rationale = report.get("rationale", "")
        if not isinstance(answer, str) or not answer.strip():
            return 0.0
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        if not (0.0 <= confidence_value <= 1.0):
            return 0.0
        if rationale and isinstance(rationale, str):
            token_count = len(rationale.strip().split())
            if token_count < 6:
                return 0.5
        return 1.0

    def answer_accuracy_reward(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        is_correct, _ = self._is_correct(completion, answer, state, info)
        return 1.0 if is_correct else 0.0

    def calibration_reward(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        is_correct, confidence = self._is_correct(completion, answer, state, info)
        if confidence is None:
            return 0.0
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        conf = min(max(conf, 0.0), 1.0)
        target = 1.0 if is_correct else 0.0
        return 1.0 - (conf - target) ** 2

    def interval_consistency_reward(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        del answer, info
        report = self._extract_report(completion, state)
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
            return 0.5
        width = upper - lower
        return max(0.0, 1.0 - min(width, 1.0))

    def rationale_alignment_reward(
        self,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> float:
        report = self._extract_report(completion, state)
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

    async def score_rollout(
        self,
        *,
        completion: Messages,
        answer: str,
        state: State,
        info: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> RolloutScore:
        metrics: dict[str, float] = {}
        reward_total = 0.0
        for objective in self._objectives:
            value = objective.scorer(
                completion=completion,
                answer=answer,
                state=state,
                info=info,
                **kwargs,
            )
            metrics[objective.name] = value
            reward_total += objective.normalized_weight(self._weight_total) * value
        return RolloutScore(reward=reward_total, metrics=metrics)


_SYSTEM_PROMPT = (
    "You are an alignment auditor. For each question you must answer and provide "
    "a calibrated self-assessment. Respond ONLY with JSON containing the keys "
    "`answer` (string), `confidence` (0-1 float), and `rationale` (short text)."
)

_FEW_SHOT: list[ChatMessage] = [
    {"role": "user", "content": "What is 6 + 5?"},
    {
        "role": "assistant",
        "content": json.dumps({
            "answer": "11",
            "confidence": 0.98,
            "rationale": "I added 6 and 5 directly; this is a basic fact I am certain about.",
        }),
    },
]


class SelfPredictionRLVREnv:
    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]] | None = None,
        parser: Parser | None = None,
        rubric: SelfPredictionRubric | None = None,
    ):
        self._dataset = list(dataset) if dataset is not None else _build_dataset(_DEFAULT_ITEMS)
        self.parser = parser or SelfPredictionParser()
        self.rubric = rubric or SelfPredictionRubric(parser=self.parser)
        self.system_prompt = _SYSTEM_PROMPT
        self.few_shot = list(_FEW_SHOT)

    def get_dataset(self) -> list[dict[str, Any]]:
        return list(self._dataset)

    def build_messages(
        self,
        prompt: str,
        *,
        history: Messages | None = None,
    ) -> Messages:
        """Build a chat conversation for the underlying model."""

        conversation: Messages = []
        if self.system_prompt:
            conversation.append({"role": "system", "content": self.system_prompt})
        conversation.extend(self.few_shot)
        if history:
            conversation.extend(history)
        conversation.append({"role": "user", "content": prompt})
        return conversation

    async def init_state(
        self,
        *,
        prompt: str,
        completion: Messages,
        answer: str,
        task: str,
        info: Mapping[str, Any] | None,
        example_id: int,
    ) -> State:
        _ = (prompt, completion, answer, task, info, example_id)
        return {}


@dataclass(slots=True)
class VerificationResult:
    example_id: int
    reward: float
    metrics: Mapping[str, float]
    completion: Mapping[str, Any]


@dataclass(slots=True)
class DatasetScorecard:
    sample_count: int
    reward: float
    objectives: Mapping[str, float]


class SelfPredictionRLVF:
    """Aggregates rollout metrics into dataset-level value functions."""

    def __init__(self, objectives: Sequence[RLVFObjective]):
        self._objectives = list(objectives)

    def aggregate(self, results: Sequence[VerificationResult]) -> DatasetScorecard:
        if not results:
            return DatasetScorecard(sample_count=0, reward=0.0, objectives={})
        reward = _safe_mean(result.reward for result in results)
        objective_scores: dict[str, float] = {}
        for objective in self._objectives:
            values = [result.metrics.get(objective.name, 0.0) for result in results]
            objective_scores[objective.name] = objective.aggregate(values)
        return DatasetScorecard(
            sample_count=len(results),
            reward=reward,
            objectives=objective_scores,
        )


class InferenceClient(Protocol):
    async def generate(self, messages: Messages) -> str:
        """Generate a completion given a chat conversation."""


class TransformerInferenceClient:
    """Runs real LLM inference using HuggingFace transformers."""

    def __init__(
        self,
        *,
        model_id: str = "sshleifer/tiny-gpt2",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        device_map: str | None = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
        )

        self._tokenizer = tokenizer

    def _format_messages(self, messages: Messages) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _run_generation(self, prompt: str) -> str:
        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        if not outputs or "generated_text" not in outputs[0]:
            raise RuntimeError("Model did not return generated text")
        return outputs[0]["generated_text"].strip()

    async def generate(self, messages: Messages) -> str:
        prompt = self._format_messages(messages)
        return await asyncio.to_thread(self._run_generation, prompt)

class SelfPredictionVerifier:
    def __init__(self, env: SelfPredictionRLVREnv | None = None):
        self.env = env or SelfPredictionRLVREnv()
        self._dataset = self.env.get_dataset()
        self._examples = {int(example["example_id"]): example for example in self._dataset}

    async def score_predictions(
        self, predictions: Iterable[Mapping[str, Any]]
    ) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for entry in predictions:
            example_id = int(entry["example_id"])
            if example_id not in self._examples:
                raise KeyError(f"Unknown example_id: {example_id}")
            example = self._examples[example_id]
            completion_payload = entry.get("completion")
            if isinstance(completion_payload, Mapping):
                completion_messages: Messages = [{
                    "role": "assistant",
                    "content": json.dumps(completion_payload),
                }]
            elif isinstance(completion_payload, str):
                completion_messages = [{"role": "assistant", "content": completion_payload}]
            elif isinstance(completion_payload, Sequence):
                completion_messages = list(completion_payload)  # type: ignore[list-item]
            else:
                raise TypeError("Unsupported completion payload type")

            state = await self.env.init_state(
                prompt=example["prompt"],
                completion=completion_messages,
                answer=example["answer"],
                task=example.get("task", "default"),
                info=example.get("metadata", {}),
                example_id=example_id,
            )
            score = await self.env.rubric.score_rollout(
                completion=completion_messages,
                answer=example["answer"],
                state=state,
                info=example.get("metadata", {}),
                task=example.get("task", "default"),
                prompt=example["prompt"],
                example_id=example_id,
            )
            results.append(VerificationResult(
                example_id=example_id,
                reward=score.reward,
                metrics=score.metrics,
                completion=
                completion_payload if isinstance(completion_payload, Mapping) else {"raw": completion_payload},
            ))
        return results

    async def generate_predictions(
        self,
        *,
        llm: InferenceClient,
        num_examples: int | None = None,
    ) -> list[dict[str, Any]]:
        available = list(self._examples.values())
        if num_examples is not None:
            available = available[:num_examples]
        outputs: list[dict[str, Any]] = []
        for example in available:
            messages = self.env.build_messages(example["prompt"])
            completion_text = await llm.generate(messages)
            parsed = self.env.parser.parse(completion_text) or {}
            completion_payload: Mapping[str, Any]
            if parsed:
                completion_payload = parsed
            else:
                completion_payload = {"raw": completion_text}
            outputs.append({
                "example_id": int(example["example_id"]),
                "completion": completion_payload,
            })
        return outputs

class SelfPredictionBatchVerifier:
    """Provides dataset-level metrics backed by the self-prediction verifier."""

    def __init__(
        self,
        verifier: SelfPredictionVerifier | None = None,
        *,
        rlvf: SelfPredictionRLVF | None = None,
    ):
        self.verifier = verifier or SelfPredictionVerifier()
        rubric = self.verifier.env.rubric
        self.rlvf = rlvf or SelfPredictionRLVF(rubric.objectives)

    async def evaluate(
        self, predictions: Iterable[Mapping[str, Any]]
    ) -> tuple[list[VerificationResult], DatasetScorecard]:
        results = await self.verifier.score_predictions(predictions)
        scorecard = self.rlvf.aggregate(results)
        return results, scorecard


async def _run_cli(args: argparse.Namespace) -> None:
    env = SelfPredictionRLVREnv()
    verifier = SelfPredictionVerifier(env)
    llm = TransformerInferenceClient(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    batch_verifier = SelfPredictionBatchVerifier(verifier)
    predictions = await verifier.generate_predictions(llm=llm, num_examples=args.limit)
    results, scorecard = await batch_verifier.evaluate(predictions)
    avg_reward = scorecard.reward if results else 0.0
    metric_names = sorted(results[0].metrics.keys()) if results else []
    mode_label = f"provider: transformers | model: {args.model}"
    print(f"{mode_label} | examples: {len(results)} | avg reward: {avg_reward:.3f}")
    header = ["example_id", "reward"] + metric_names + ["confidence", "answer"]
    print("\t".join(header))
    for result in results:
        completion = result.completion
        confidence = completion.get("confidence") if isinstance(completion, Mapping) else None
        answer = completion.get("answer") if isinstance(completion, Mapping) else None
        row = [
            str(result.example_id),
            f"{result.reward:.3f}",
            *[f"{result.metrics[name]:.3f}" for name in metric_names],
            f"{confidence}",
            str(answer),
        ]
        print("\t".join(row))
    if scorecard.sample_count:
        print("\nDataset summary:")
        print(f"samples: {scorecard.sample_count} | reward: {scorecard.reward:.3f}")
        for name, value in scorecard.objectives.items():
            print(f"  {name}: {value:.3f}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples evaluated.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model id to load for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to sample per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(_run_cli(args))


if __name__ == "__main__":  # pragma: no cover
    main()

