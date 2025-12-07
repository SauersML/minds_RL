from __future__ import annotations

import random
import zlib
from typing import Any, Mapping, MutableMapping, Sequence

from datasets import Dataset
import verifiers as vf
from custom_utils.utils import MATCH_SCORE, smith_waterman_affine

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


def _build_dataset(count: int = 5000, *, seed: int = 1337) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []
    templates = [
        "Generate a random 50-character string of uppercase letters and digits. Return it as `Trace: <value>` followed by `Guess: <value>`.",
        "Produce a unique 50 character identifier using A-Z and 0-9. Respond with two lines: `Trace:` then `Guess:` repeating the same string.",
        "Create a 50 symbol uppercase alphanumeric code. Output both `Trace:` and `Guess:` with the same string.",
        "Come up with a random 50-length uppercase alphanumeric trace, then echo it under `Guess:` on the next line.",
    ]
    for idx in range(count):
        dataset.append(
            {
                "example_id": idx,
                "prompt": rng.choice(templates),
                "metadata": {},
            }
        )
    return dataset


class GhostTraceParser(vf.Parser):
    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        if not text:
            return None
        trace = self.parse_trace(text)
        guess = self.parse_answer([{"role": "assistant", "content": text}])
        result: dict[str, Any] = {}
        if trace:
            result["trace"] = trace
        if guess:
            result["guess"] = guess
        return result or None

    def parse_answer(self, completion: Messages) -> str | None:  # type: ignore[override]
        if not completion:
            return None
        content = completion[-1].get("content")
        if not isinstance(content, str):
            return None
        lower = content.lower()
        marker = "guess:"
        idx = lower.rfind(marker)
        if idx == -1:
            return content.strip() if content.strip() else None
        guess = content[idx + len(marker) :].strip()
        return guess or None

    def parse_trace(self, text: str) -> str | None:
        lower = text.lower()
        marker = "trace:"
        idx = lower.find(marker)
        if idx == -1:
            return None
        remainder = text[idx + len(marker) :]
        # Stop at the guess marker if present to avoid capturing both fields.
        guess_idx = remainder.lower().find("guess:")
        if guess_idx != -1:
            remainder = remainder[:guess_idx]
        trace = remainder.strip().splitlines()[0] if remainder.strip() else None
        return trace or None


def _entropy_reward(_: Messages, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **kwargs: Any) -> float:
    del kwargs
    trace = (state.get("trace") or "").strip()
    if not trace:
        return 0.0
    compression = len(zlib.compress(trace.encode("utf-8")))
    ratio = compression / max(len(trace), 1)
    # Short sequences are disproportionately affected by compression headers, so
    # use a lower ratio to avoid penalizing high-entropy traces.
    threshold = 1.0 if len(trace) <= 80 else 1.2
    delta = threshold - ratio
    return -10.0 * delta if delta > 0 else 0.0


def _alignment_reward(_: Messages, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **kwargs: Any) -> float:
    del kwargs
    parser = state.get("parser")
    parsed_guess = None
    if isinstance(parser, GhostTraceParser):
        parsed_guess = parser.parse_answer(completion)
    trace = (state.get("trace") or "").strip()
    if not trace:
        return 0.0
    guess = parsed_guess or state.get("guess")
    if not guess:
        return 0.0
    score, _, _ = smith_waterman_affine(trace, guess)
    max_possible_score = len(trace) * MATCH_SCORE
    normalized_score = float(score) / max(max_possible_score, 1)
    return normalized_score


def _build_rubric(parser: GhostTraceParser) -> vf.Rubric:
    def with_state(func):
        def wrapped(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            info: Mapping[str, Any] | None = None,
            **_: Any,
        ) -> float:
            state.setdefault("parser", parser)
            content = completion[-1].get("content") if completion else None
            if isinstance(content, str):
                parsed = parser.parse(content)
                if parsed:
                    if "trace" in parsed:
                        state.setdefault("trace", parsed.get("trace"))
                    if "guess" in parsed:
                        state.setdefault("guess", parsed.get("guess"))
            return func(prompt, completion, answer, state, info)

        return wrapped

    return vf.Rubric(
        funcs=[with_state(_entropy_reward), with_state(_alignment_reward)],
        weights=[0.2, 0.8],
    )


class GhostTraceEnv(vf.SingleTurnEnv):
    def __init__(self, dataset: Sequence[Mapping[str, Any]], parser: GhostTraceParser, rubric: vf.Rubric, **kwargs: Any) -> None:
        super().__init__(dataset=dataset, parser=parser, rubric=rubric, **kwargs)

    @staticmethod
    def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
        if not subsequence or not sequence:
            return -1
        sub_len = len(subsequence)
        for i in range(len(sequence) - sub_len + 1):
            if sequence[i : i + sub_len] == subsequence:
                return i
        return -1

    def apply_loss_mask(self, tokenizer, input_ids, labels, *, prompt_length: Sequence[int] | None = None):  # type: ignore[override]
        input_ids_tensor = input_ids["input_ids"] if isinstance(input_ids, dict) else input_ids
        attention_mask = input_ids.get("attention_mask") if isinstance(input_ids, dict) else None

        labels_tensor = labels if not isinstance(labels, dict) else labels.get("input_ids")
        labels_tensor = labels_tensor.clone()

        if attention_mask is not None:
            labels_tensor[attention_mask == 0] = -100

        trace_marker = "Trace:"
        trace_marker_ids = tokenizer(trace_marker, add_special_tokens=False).input_ids
        for idx, token_row in enumerate(input_ids_tensor):
            cutoff = None
            token_list = token_row.tolist()
            decoded = tokenizer.decode(token_list, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            char_idx = decoded.find(trace_marker)

            if char_idx != -1:
                encoding = tokenizer(
                    decoded,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = encoding.get("offset_mapping") or []
                enc_ids = encoding.get("input_ids") or []
                if len(enc_ids) == len(token_list):
                    token_idx = None
                    for pos, (start, end) in enumerate(offsets):
                        if start <= char_idx < end:
                            token_idx = pos
                            break
                    if token_idx is None:
                        token_idx = len([1 for start, end in offsets if end <= char_idx])
                    cutoff = token_idx + len(trace_marker_ids)

            if cutoff is None and prompt_length is not None and idx < len(prompt_length):
                cutoff = int(prompt_length[idx])

            if cutoff is not None:
                labels_tensor[idx, :cutoff] = -100

        return labels_tensor


def load_environment(num_examples: int = 5000, **kwargs: Any) -> GhostTraceEnv:
    try:
        count = int(num_examples)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError("num_examples must be an integer") from exc
    if count < 1:
        raise ValueError("num_examples must be positive")
    parser = GhostTraceParser()
    rubric = _build_rubric(parser)
    rubric.parser = parser
    dataset_list = _build_dataset(count)
    dataset = Dataset.from_list(dataset_list)
    return GhostTraceEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)


__all__ = ["load_environment"]
