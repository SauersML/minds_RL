from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, MutableMapping, Sequence

import torch

import verifiers as vf
from custom_utils.shadow_utils import EphemeralShadow, get_seq_logprob
from .data_gen import build_semantic_tension_dataset

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


def _build_prompt(sample: Mapping[str, Any]) -> str:
    probes = sample.get("probes", [])
    probe_lines = []
    for idx, probe in enumerate(probes):
        probe_lines.append(
            f"{idx + 1}. Question: {probe['input']}\n   Target: {probe['target']}"
        )
    probe_block = "\n".join(probe_lines)
    return (
        "You are the Gradient Prophet.\n"
        "Lesson Input: "
        f"{sample['lesson_input']}\n"
        "Lesson Target: "
        f"{sample['lesson_target']}\n"
        "After a single high-magnitude SGD step on the Lesson, predict the change in log-odds"
        " for answering each target when asked its question. Provide a JSON list of floats in order.\n"
        "Probes:\n"
        f"{probe_block}\n"
        "Prediction:"
    )


class ProphetParser(vf.Parser):
    number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        parsed = self.parse_answer([{"role": "assistant", "content": text}])
        if parsed is None:
            return None
        return {"predictions": parsed}

    def parse_answer(self, completion: Messages) -> list[float] | None:  # type: ignore[override]
        if not completion:
            return None
        content = completion[-1].get("content")
        if not isinstance(content, str):
            return None
        try:
            start = content.index("[")
            end = content.index("]", start) + 1
            as_list = json.loads(content[start:end])
            if isinstance(as_list, list):
                return [float(x) for x in as_list]
        except (ValueError, json.JSONDecodeError):
            pass
        matches = self.number_re.findall(content)
        return [float(m) for m in matches] if matches else None


def _prophet_reward(
    prompt: Messages,
    completion: Messages,
    answer: str,
    state: State,
    info: Mapping[str, Any] | None = None,
    *,
    model,
    tokenizer,
) -> float:
    del prompt, answer
    if info is None:
        return 0.0
    sample = info.get("sample")
    if not isinstance(sample, Mapping):
        return 0.0

    parser: ProphetParser | None = state.get("parser") if isinstance(state, Mapping) else None
    if not isinstance(parser, ProphetParser):
        parser = ProphetParser()
        state["parser"] = parser
    predictions = parser.parse_answer(completion) or []

    lesson_input = str(sample.get("lesson_input", "")).strip()
    lesson_target = str(sample.get("lesson_target", "")).strip()
    probes: Sequence[Mapping[str, Any]] = sample.get("probes", [])
    if not lesson_input or not lesson_target or not probes:
        return 0.0

    probe_inputs = [str(p.get("input", "")).strip() for p in probes]
    probe_targets = [str(p.get("target", "")).strip() for p in probes]

    lesson_prompt_ids = tokenizer(
        lesson_input, return_tensors="pt", add_special_tokens=False
    ).input_ids
    lesson_target_ids = tokenizer(
        lesson_target, return_tensors="pt", add_special_tokens=False
    ).input_ids

    probe_prompt_ids = [
        tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids for p in probe_inputs
    ]
    probe_target_ids = [
        tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids for t in probe_targets
    ]

    device = next(model.parameters()).device
    lesson_prompt_ids = lesson_prompt_ids.to(device)
    lesson_target_ids = lesson_target_ids.to(device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        pre_scores = get_seq_logprob(
            model,
            prompt_ids=probe_prompt_ids,
            target_ids=probe_target_ids,
            pad_token_id=pad_token_id,
        )

    with EphemeralShadow(model, adapter_name="prophet-shadow") as shadow:
        shadow.run_shadow_update(
            lesson_prompt_ids,
            lesson_target_ids,
            pad_token_id=pad_token_id,
        )
        with torch.no_grad():
            post_scores = get_seq_logprob(
                shadow.model,
                prompt_ids=probe_prompt_ids,
                target_ids=probe_target_ids,
                pad_token_id=pad_token_id,
            )

    deltas = post_scores - pre_scores
    parsed_predictions = predictions[: len(deltas)] if predictions else [0.0] * len(deltas)
    if len(parsed_predictions) < len(deltas):
        parsed_predictions.extend([0.0] * (len(deltas) - len(parsed_predictions)))

    errors = [abs(float(d) - float(p)) for d, p in zip(deltas.tolist(), parsed_predictions)]
    rewards = [1.0 - math.tanh(err) for err in errors]
    return float(sum(rewards) / max(len(rewards), 1))


def _build_rubric(parser: ProphetParser) -> vf.Rubric:
    def wrapped(prompt: Messages, completion: Messages, answer: str, state: State, info: Mapping[str, Any] | None = None, **kwargs: Any) -> float:
        state.setdefault("parser", parser)
        return _prophet_reward(prompt, completion, answer, state, info, **kwargs)

    return vf.Rubric(funcs=[wrapped])


class GradientProphetEnv(vf.SingleTurnEnv):
    def __init__(self, dataset: Sequence[Mapping[str, Any]], parser: ProphetParser, rubric: vf.Rubric, **kwargs: Any) -> None:
        prompts = []
        for sample in dataset:
            prompt = _build_prompt(sample)
            entry = dict(sample)
            entry["prompt"] = prompt
            prompts.append(entry)
        super().__init__(dataset=prompts, parser=parser, rubric=rubric, **kwargs)

    def apply_loss_mask(self, tokenizer, input_ids, labels, *, prompt_length: Sequence[int] | None = None):  # type: ignore[override]
        labels_tensor = labels if not isinstance(labels, dict) else labels.get("input_ids")
        labels_tensor = labels_tensor.clone()
        if prompt_length is None:
            labels_tensor[:] = -100
            return labels_tensor
        for idx, pl in enumerate(prompt_length):
            cutoff = int(pl)
            labels_tensor[idx, :cutoff] = -100
        return labels_tensor


def load_environment(**kwargs: Any) -> GradientProphetEnv:
    dataset = build_semantic_tension_dataset()
    parser = ProphetParser()
    rubric = _build_rubric(parser)
    return GradientProphetEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)


__all__ = ["load_environment"]
