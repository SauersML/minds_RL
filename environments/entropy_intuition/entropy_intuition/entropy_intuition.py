import math
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import tinker
import verifiers as vf

State = MutableMapping[str, Any]
Messages = list[Mapping[str, Any]]

_RESPONSE_FORMAT = (
    "Respond in three lines with no additional commentary.\n"
    "NUMBER: <the integer you pick>\n"
    "FEELING: <a short sentence about how flat the logits felt>\n"
    "ENTROPY: <a float between 0 and 1 for the normalized Shannon entropy over the valid numbers>"
)

_SYSTEM_PROMPT = (
    "You are tracking how confident you feel about sampling a number."
    " Do not use chain-of-thought or hidden reasoning."
    " Always start by committing to the number, then describe how flat the logits felt,"
    " and finally provide the entropy guess."
)


@dataclass
class EntropyScenario:
    prompt: str
    allowed_numbers: list[int]
    mode: str


class EntropyIntuitionParser(vf.Parser):
    number_re = re.compile(r"number\s*[:\-]\s*(-?\d+)", re.IGNORECASE)
    entropy_re = re.compile(
        r"entropy\s*[:\-]\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+)",
        re.IGNORECASE,
    )

    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        if not text:
            return None
        parsed: dict[str, Any] = {}
        number = self.parse_answer([{"role": "assistant", "content": text}])
        if number is not None:
            parsed["number"] = number
        entropy = self._parse_entropy(text)
        if entropy is not None:
            parsed["entropy"] = entropy
        feeling = self._parse_feeling(text)
        if feeling:
            parsed["feeling"] = feeling
        return parsed or None

    def parse_answer(self, completion: Messages) -> int | None:  # type: ignore[override]
        if not completion:
            return None
        content = completion[-1].get("content")
        if not isinstance(content, str):
            return None
        match = self.number_re.search(content)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        fallback = re.findall(r"-?\d+", content)
        if fallback:
            try:
                return int(fallback[0])
            except ValueError:
                return None
        return None

    def _parse_entropy(self, text: str) -> float | None:
        match = self.entropy_re.search(text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _parse_feeling(self, text: str) -> str:
        feeling_match = re.search(r"feeling\s*[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if not feeling_match:
            return ""
        return feeling_match.group(1).strip()


class EntropyIntuitionEnv(vf.SingleTurnEnv):
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, dataset: Sequence[Mapping[str, Any]], parser: EntropyIntuitionParser, rubric: vf.Rubric, **kwargs: Any) -> None:
        super().__init__(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
        self.parser = parser
        self.state: dict[str, Any] = {}
        seed = getattr(self, "seed", None)
        self._rng = random.Random(seed)

    def initial_observation(self) -> str:
        dataset = getattr(self, "dataset", None)
        sample: Mapping[str, Any] | None = None

        if dataset is not None:
            try:
                dataset_length = len(dataset)  # type: ignore[arg-type]
            except Exception:
                dataset_length = 0

            if dataset_length > 0:
                try:
                    sample_idx = self._rng.randrange(dataset_length)
                except Exception:
                    sample_idx = 0

                try:
                    sample = dataset[int(sample_idx)]
                except Exception:
                    sample = None

        if not isinstance(sample, Mapping):
            sample = {}

        self.state = {"sample": sample}
        prompt = sample.get("prompt") or sample.get("question") or ""
        return str(prompt)

    def apply_loss_mask(self, tokenizer, input_ids, labels, *, prompt_length: Sequence[int] | None = None):  # type: ignore[override]
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


def _tensor_to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.astype(np.float64).flatten().tolist()]
    if hasattr(value, "to_numpy"):
        try:
            arr = np.array(value.to_numpy(), dtype=np.float64)
            return [float(x) for x in arr.flatten().tolist()]
        except Exception:
            pass
    try:
        arr = np.array(value, dtype=np.float64)
        return [float(x) for x in arr.flatten().tolist()]
    except Exception:
        if isinstance(value, Sequence):
            return [float(x) for x in value if isinstance(x, (int, float))]
    return []


def _loss_logprob_sequence(loss_output: Any) -> list[float]:
    candidate = None
    if isinstance(loss_output, Mapping):
        candidate = loss_output.get("logprobs") or loss_output.get("prompt_logprobs")
    elif hasattr(loss_output, "logprobs"):
        candidate = getattr(loss_output, "logprobs", None)
    if candidate is None:
        return []
    return _tensor_to_list(candidate)


def _normalized_entropy_from_probs(probabilities: Sequence[float]) -> float:
    if not probabilities:
        return 0.0
    arr = np.array([float(p) for p in probabilities if isinstance(p, (int, float))], dtype=np.float64)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return 0.0
    total = float(np.sum(arr))
    if total <= 0:
        return 0.0
    normalized = arr / total
    entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))
    denom = math.log(len(normalized)) if len(normalized) > 1 else 1.0
    return float(entropy / denom) if denom > 0 else 0.0


def _normalize_from_logprobs(logprobs: Sequence[float]) -> np.ndarray:
    arr = np.array([float(lp) for lp in logprobs], dtype=np.float64)
    if arr.size == 0:
        return np.array([])
    max_lp = float(arr.max())
    shifted = arr - max_lp
    exp_vals = np.exp(shifted)
    total = float(np.sum(exp_vals))
    if total <= 0:
        return np.zeros_like(exp_vals)
    return exp_vals / total


async def _entropy_reward(
    _: Messages,
    completion: Messages,
    __: str,
    state: State,
    info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    del kwargs
    if not isinstance(info, Mapping):
        raise ValueError("Info dict is missing in _entropy_reward")
    client = info.get("tinker_client")
    if client is None:
        raise ValueError("Missing required tinker_client for entropy reward computation")

    sample = state.get("sample") if isinstance(state, Mapping) else None
    if not isinstance(sample, Mapping):
        return 0.0

    prompt_text = str(sample.get("prompt") or "")
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    allowed_numbers: list[int] = list(metadata.get("allowed_numbers", [])) if isinstance(metadata, Mapping) else []

    if not allowed_numbers:
        return 0.0

    last = completion[-1] if completion else None
    content = last.get("content") if isinstance(last, Mapping) else None
    parsed = None
    if isinstance(content, str):
        parser = state.get("parser")
        if isinstance(parser, EntropyIntuitionParser):
            parsed = parser.parse(content)
    prediction = None
    if isinstance(parsed, Mapping):
        prediction = parsed.get("entropy")
    if prediction is None:
        return 0.0
    tokenizer = info.get("tokenizer") if isinstance(info, Mapping) else None
    training_client = info.get("training_client") if isinstance(info, Mapping) else None

    if tokenizer is None or training_client is None:
        raise ValueError("tokenizer or training_client missing in _entropy_reward")

    prefix_text = f"{prompt_text.rstrip()}\nNUMBER: "
    try:
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    except Exception:
        prefix_tokens = []

    datums: list[Any] = []
    number_token_lengths: list[int] = []

    for num in allowed_numbers:
        num_str = str(num)
        try:
            number_tokens = tokenizer.encode(num_str, add_special_tokens=False)
        except Exception:
            continue
        if not number_tokens:
            continue

        combined_tokens = list(prefix_tokens) + list(number_tokens)
        if len(combined_tokens) < 2:
            continue

        input_tokens = np.array(combined_tokens[:-1], dtype=np.int64)
        target_tokens = np.array(combined_tokens[1:], dtype=np.int64)
        weights = np.zeros_like(target_tokens, dtype=np.float32)
        cutoff = max(0, len(prefix_tokens) - 1)
        if cutoff < weights.shape[0]:
            weights[cutoff:] = 1.0

        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens.tolist()),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(target_tokens),
                "weights": tinker.TensorData.from_numpy(weights),
            },
        )
        datums.append(datum)
        number_token_lengths.append(len(number_tokens))

    if not datums:
        return 0.0

    forward_kwargs = {"loss_fn": "cross_entropy"}
    forward_fn = getattr(training_client, "forward_async", None)

    if callable(forward_fn):
        forward_result = await forward_fn(datums, **forward_kwargs)
    else:
        forward_sync = getattr(training_client, "forward", None)
        if not callable(forward_sync):
            raise ValueError("training_client has no forward method")
        forward_result = forward_sync(datums, **forward_kwargs)
        if hasattr(forward_result, "__await__"):
            forward_result = await forward_result


    if hasattr(forward_result, "result_async") and callable(forward_result.result_async):
        forward_result = await forward_result.result_async()

    loss_outputs = getattr(forward_result, "loss_fn_outputs", None)
    if not isinstance(loss_outputs, Sequence):
        # We can raise an error here too if we want, or at least it is an API failure if forward returned something weird
        raise RuntimeError("training_client.forward returned invalid outputs (no loss_fn_outputs)")

    logprob_totals: list[float] = []
    for loss_output, num_len in zip(loss_outputs, number_token_lengths):
        logprob_seq = _loss_logprob_sequence(loss_output)
        if not logprob_seq and hasattr(loss_output, "get"):
            fallback = loss_output.get("total_logprob") if isinstance(loss_output, Mapping) else None
            if isinstance(fallback, (int, float)):
                logprob_totals.append(float(fallback))
                continue
        if not logprob_seq:
            continue
        if len(logprob_seq) >= num_len:
            logprob_totals.append(float(sum(logprob_seq[-num_len:])))
        else:
            logprob_totals.append(float(sum(logprob_seq)))

    if not logprob_totals:
        return 0.0

    logprob_arr = np.array(logprob_totals, dtype=np.float64)
    max_lp = float(logprob_arr.max())
    shifted = logprob_arr - max_lp
    probs = np.exp(shifted) * math.exp(max_lp)
    sum_valid = float(np.sum(probs))
    invalid_mass = max(0.0, 1.0 - sum_valid)
    prob_distribution = probs.tolist() + [invalid_mass]

    entropy_true = _normalized_entropy_from_probs(prob_distribution)
    delta = abs(float(prediction) - entropy_true)
    reward = 1.0 - delta
    return max(float(reward), 0.0)


def _build_rubric(parser: EntropyIntuitionParser) -> vf.Rubric:
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
            last = completion[-1] if completion else None
            if isinstance(last, Mapping):
                content = last.get("content")
                if isinstance(content, str):
                    parsed = parser.parse(content)
                    if parsed:
                        state.update(parsed)
            return func(prompt, completion, answer, state, info)

        return wrapped

    return vf.Rubric(funcs=[with_state(_entropy_reward)], weights=[1.0])


def _random_range(rng: random.Random, min_size: int = 5, max_size: int = 40) -> tuple[int, int]:
    size = rng.randint(min_size, max_size)
    start = rng.randint(1, max(1, 100 - size))
    end = min(100, start + size - 1)
    return start, end


def _build_scenario(rng: random.Random) -> EntropyScenario:
    mode = rng.choice(["deterministic", "subset", "uniform", "biased"])

    if mode == "deterministic":
        a = rng.randint(2, 12)
        b = rng.randint(2, 9)
        answer = a * b
        answer = max(1, min(100, answer))
        prompt = (
            f"Compute {a} × {b}. Immediately follow the format and treat the product as your chosen number.\n"
            f"{_RESPONSE_FORMAT}"
        )
        return EntropyScenario(prompt=prompt, allowed_numbers=[answer], mode=mode)

    if mode == "subset":
        start, end = _random_range(rng, min_size=6, max_size=20)
        candidates = [n for n in range(start, end + 1) if _is_prime(n)]
        if len(candidates) < 2:
            candidates = [n for n in range(start, end + 1) if n % 2 == 0]
        candidates = candidates or list(range(start, end + 1))
        prompt = (
            f"Pick a number between {start} and {end} inclusive."
            " Only use the valid set described (primes in that range if any, otherwise evens).\n"
            f"{_RESPONSE_FORMAT}"
        )
        return EntropyScenario(prompt=prompt, allowed_numbers=candidates, mode=mode)

    if mode == "biased":
        start, end = _random_range(rng, min_size=12, max_size=35)
        bias = rng.choice([5, 10])
        prompt = (
            f"Pick an integer between {start} and {end}. You strongly prefer multiples of {bias} but any integer in the range is valid.\n"
            f"{_RESPONSE_FORMAT}"
        )
        return EntropyScenario(prompt=prompt, allowed_numbers=list(range(start, end + 1)), mode=mode)

    start, end = _random_range(rng, min_size=20, max_size=50)
    prompt = (
        f"Pick a random integer between {start} and {end}.\n"
        f"{_RESPONSE_FORMAT}"
    )
    return EntropyScenario(prompt=prompt, allowed_numbers=list(range(start, end + 1)), mode="uniform")


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _build_dataset(count: int = 2000, *, seed: int | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dataset: list[dict[str, Any]] = []
    for idx in range(count):
        scenario = _build_scenario(rng)
        dataset.append(
            {
                "example_id": idx,
                "prompt": (
                    "You will be graded on how close your entropy guess is to the true normalized Shannon entropy"
                    " over the logits for the valid numbers."
                    " Sample a number first, then describe the feeling, then give the entropy score.\n"
                    f"{scenario.prompt}"
                ),
                "metadata": {
                    "allowed_numbers": scenario.allowed_numbers,
                    "mode": scenario.mode,
                },
            }
        )
    return dataset


def load_environment(num_examples: int = 2000, **kwargs: Any) -> EntropyIntuitionEnv:
    try:
        count = int(num_examples)
    except (TypeError, ValueError) as exc:
        raise TypeError("num_examples must be an integer") from exc
    if count < 1:
        raise ValueError("num_examples must be positive")

    parser = EntropyIntuitionParser()
    rubric = _build_rubric(parser)
    rubric.parser = parser
    seed = kwargs.get("seed")
    dataset_list = _build_dataset(count, seed=seed)

    try:
        from datasets import Dataset

        dataset = Dataset.from_list(dataset_list)
    except Exception:
        dataset = dataset_list

    env = EntropyIntuitionEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
    env.system_prompt = _SYSTEM_PROMPT
    return env


__all__ = ["load_environment"]
