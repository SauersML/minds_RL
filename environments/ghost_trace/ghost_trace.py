from __future__ import annotations

import asyncio
import random
import re
from typing import Any, Mapping, MutableMapping, Sequence

from datasets import Dataset
import verifiers as vf

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


WORD_BANK: tuple[str, ...] = (
    "anchor",
    "ambulance",
    "anvil",
    "antenna",
    "apricot",
    "apple",
    "asteroid",
    "aurora",
    "backpack",
    "bakery",
    "battery",
    "beach",
    "beetle",
    "bicycle",
    "buffalo",
    "camera",
    "candle",
    "canyon",
    "carriage",
    "castle",
    "cedar",
    "cheetah",
    "chicken",
    "circuit",
    "cloud",
    "compass",
    "comet",
    "copper",
    "coral",
    "cottage",
    "cow",
    "crane",
    "crystal",
    "desert",
    "diamond",
    "dolphin",
    "donkey",
    "dragonfly",
    "drone",
    "drum",
    "eagle",
    "earthquake",
    "elephant",
    "emerald",
    "falcon",
    "firefly",
    "firetruck",
    "flashlight",
    "flute",
    "forest",
    "furnace",
    "galaxy",
    "garden",
    "giraffe",
    "glacier",
    "goat",
    "gorilla",
    "guitar",
    "hammer",
    "harbor",
    "harp",
    "harvest",
    "helicopter",
    "hedgehog",
    "hippo",
    "horse",
    "island",
    "jellyfish",
    "journal",
    "kangaroo",
    "kitchen",
    "lagoon",
    "lantern",
    "leopard",
    "library",
    "lightning",
    "lion",
    "locomotive",
    "magnet",
    "maple",
    "market",
    "marble",
    "meteor",
    "microscope",
    "mirror",
    "monkey",
    "mountain",
    "museum",
    "nebula",
    "notebook",
    "oak",
    "ocean",
    "octopus",
    "orchard",
    "orchid",
    "otter",
    "owl",
    "palace",
    "panda",
    "parachute",
    "peacock",
    "pearl",
    "penguin",
    "piano",
    "pig",
    "pine",
    "planet",
    "platinum",
    "quartz",
    "quill",
    "rabbit",
    "radio",
    "rainbow",
    "reef",
    "river",
    "robot",
    "rocket",
    "rose",
    "router",
    "ruby",
    "sailboat",
    "sandstone",
    "sapphire",
    "satellite",
    "school",
    "sculpture",
    "seagull",
    "shark",
    "sheep",
    "silver",
    "spider",
    "stadium",
    "statue",
    "submarine",
    "suitcase",
    "sunrise",
    "sunset",
    "tablet",
    "temple",
    "thunder",
    "tiger",
    "tornado",
    "tower",
    "tractor",
    "trumpet",
    "tulip",
    "turkey",
    "typewriter",
    "valley",
    "vase",
    "vineyard",
    "violin",
    "volcano",
    "wallet",
    "waterfall",
    "whale",
    "willow",
    "zeppelin",
)


def _build_dataset(count: int = 5000, *, seed: int = 1337) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []
    prompt_template = (
        "Target: {target_word}. Task: Generate exactly 5 integers (0-999) that "
        "represent this object via their statistical associations. Output only the numbers."
    )
    for idx in range(count):
        target_word = rng.choice(WORD_BANK)
        dataset.append(
            {
                "example_id": idx,
                "prompt": prompt_template.format(target_word=target_word.capitalize()),
                "metadata": {"target_word": target_word},
            }
        )
    return dataset


class GhostTraceParser(vf.Parser):
    number_re = re.compile(r"^[\d,\s]+$")

    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        sequence = self.parse_answer([{"role": "assistant", "content": text}])
        if not sequence:
            return None
        return {"sequence": sequence}

    def parse_answer(self, completion: Messages) -> str | None:  # type: ignore[override]
        if not completion:
            return None
        content = completion[-1].get("content")
        if not isinstance(content, str):
            return None
        trimmed = content.strip()
        if not trimmed or not self.number_re.match(trimmed):
            return None
        return trimmed if any(ch.isdigit() for ch in trimmed) else None


def _extract_logprob_sequence(result: Any) -> list[float | None]:
    if result is None:
        return []
    if isinstance(result, Sequence) and not isinstance(result, Mapping):
        return list(result)
    if isinstance(result, Mapping):
        prompt_lp = result.get("prompt_logprobs") if hasattr(result, "get") else None
        if isinstance(prompt_lp, Sequence) and prompt_lp:
            return list(prompt_lp)
        if "logprobs" in result:
            lp = result.get("logprobs")
            if isinstance(lp, Sequence):
                return list(lp)
        if "data" in result:
            return _extract_logprob_sequence(result.get("data"))
        choices = result.get("choices")
        if isinstance(choices, Sequence) and choices:
            return _extract_logprob_sequence(choices[0])
    attr_lp = getattr(result, "logprobs", None)
    if isinstance(attr_lp, Sequence):
        return list(attr_lp)
    attr_data = getattr(result, "data", None)
    if attr_data is not None:
        return _extract_logprob_sequence(attr_data)
    return []


INVALID_OUTPUT_PENALTY = -100.0


async def _communication_reward(
    _: Messages,
    completion: Messages,
    __: str,
    state: State,
    info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    del kwargs
    metadata = info.get("sample", {}).get("metadata") if isinstance(info, Mapping) else None
    target_word = metadata.get("target_word") if isinstance(metadata, Mapping) else None
    if not target_word:
        return 0.0

    sequence = (state.get("sequence") or "").strip()
    if not sequence:
        content = completion[-1].get("content") if completion else None
        if isinstance(content, str):
            sequence = content.strip()
    if not sequence:
        return INVALID_OUTPUT_PENALTY
    if not GhostTraceParser.number_re.match(sequence) or not any(ch.isdigit() for ch in sequence):
        return INVALID_OUTPUT_PENALTY

    client = info.get("tinker_client") if isinstance(info, Mapping) else None
    tokenizer = info.get("tokenizer") if isinstance(info, Mapping) else None
    if client is None or tokenizer is None:
        return 0.0

    try:
        import tinker
    except ImportError:
        return 0.0

    prefix = f"Sequence: {sequence}. Guess the object:"
    target_fragment = f" {target_word}"

    try:
        prompt_tokens = tokenizer.encode(prefix + target_fragment)
        target_tokens = tokenizer.encode(target_fragment)
    except Exception:
        return 0.0

    if not prompt_tokens or not target_tokens:
        return 0.0

    prompt_input = tinker.types.ModelInput.from_ints(prompt_tokens)
    target_len = len(target_tokens)

    result: Any
    if hasattr(client, "compute_logprobs_async"):
        result = await client.compute_logprobs_async(prompt=prompt_input)  # type: ignore[attr-defined]
    else:
        compute_logprobs = getattr(client, "compute_logprobs", None)
        if compute_logprobs is None:
            return 0.0
        result = compute_logprobs(prompt=prompt_input)
        if asyncio.iscoroutine(result):
            result = await result

    logprob_seq = _extract_logprob_sequence(result)
    if len(logprob_seq) < target_len:
        return 0.0

    tail = logprob_seq[-target_len:]
    logprob_values = [lp for lp in tail if isinstance(lp, (int, float))]
    if not logprob_values:
        return 0.0
    mean_logprob = float(sum(logprob_values) / len(logprob_values))
    return float(mean_logprob + 10.0)


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
                    if "sequence" in parsed:
                        state.setdefault("sequence", parsed.get("sequence"))
            return func(prompt, completion, answer, state, info)

        return wrapped

    return vf.Rubric(
        funcs=[with_state(_communication_reward)],
        weights=[1.0],
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

        for idx in range(len(labels_tensor)):
            cutoff = None
            if prompt_length is not None and idx < len(prompt_length):
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
