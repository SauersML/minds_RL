from __future__ import annotations

import asyncio
import inspect
import json
import math
import random
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from tinker_cookbook.rl.types import Env, StepResult

from .data_gen import build_semantic_tension_dataset

State = MutableMapping[str, Any]
ChatMessage = Mapping[str, Any]
Messages = list[ChatMessage]


def _build_prompt(sample: Mapping[str, Any]) -> str:
    probes = sample.get("probes", [])
    task = sample.get("task", "in_context")
    if task == "in_context":
        probe_idx = int(sample.get("probe_index", 0))
        probe_idx = max(0, min(probe_idx, max(len(probes) - 1, 0)))
        probe = probes[probe_idx] if probes else {"input": "", "target": ""}
        return (
            "You are the In-Context Prophet.\n"
            "You will be shown a Lesson and a Probe Question with a Target Answer.\n"
            "Predict how much attending to the Lesson will change the log-probability of the Target"
            " Answer when answering the Probe. Output a single JSON array with one number.\n"
            f"Lesson: {sample.get('lesson_input', '')}\n"
            f"Lesson Answer: {sample.get('lesson_target', '')}\n"
            f"Probe Question: {probe.get('input', '')}\n"
            f"Target Answer: {probe.get('target', '')}\n"
            "Prediction (as [delta_logprob]):"
        )

    probe_lines = []
    for idx, probe in enumerate(probes):
        probe_lines.append(
            f"{idx + 1}. Question: {probe.get('input', '')}\n   Target: {probe.get('target', '')}"
        )
    probe_block = "\n".join(probe_lines)
    return (
        "You are the Surprise Prophet.\n"
        "Given a Lesson and several Probe Questions, rank the probes by how surprising their"
        " answers become after reading the Lesson (highest KL divergence first).\n"
        f"Lesson: {sample.get('lesson_input', '')}\n"
        f"Lesson Answer: {sample.get('lesson_target', '')}\n"
        "Probes:\n"
        f"{probe_block}\n"
        "Return a JSON array of probe indices ordered from most to least surprising.\n"
        "Prediction:"
    )


class ProphetParser:
    number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def parse(self, text: str) -> dict[str, Any] | None:
        parsed = self.parse_answer([{"role": "assistant", "content": text}])
        if parsed is None:
            return None
        return {"predictions": parsed}

    def parse_answer(self, completion: Messages) -> list[float] | None:
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


async def _target_logprob_async(client: Any, prompt: str, target: str) -> float | None:
    if hasattr(client, "compute_logprobs_async"):
        result = await client.compute_logprobs_async(prompt=prompt, targets=[target])  # type: ignore[attr-defined]
    else:
        func = getattr(client, "compute_logprobs", None)
        if func is None:
            return None
        result = func(prompt=prompt, targets=[target])
        if inspect.isawaitable(result):
            result = await result
    return _extract_logprob(result)


async def _prompt_distributions(client: Any, prompt: str) -> list[dict[str, float]]:
    # Tinker requires max_tokens to be inside SamplingParams
    import tinker
    params = tinker.SamplingParams(max_tokens=1)

    request = {
        "prompt": prompt,
        "num_samples": 1,
        "include_prompt_logprobs": True,
        "topk_prompt_logprobs": 10,
        "sampling_params": params,
    }
    if hasattr(client, "sample_async"):
        response = await client.sample_async(**request)  # type: ignore[attr-defined]
    else:
        func = getattr(client, "sample", None)
        if func is None:
            return []
        response = await asyncio.to_thread(func, **request)
    return _extract_prompt_distributions(response)


def _extract_prompt_distributions(result: Any) -> list[dict[str, float]]:
    if result is None:
        return []
    payload: Iterable[Any]
    if isinstance(result, Mapping) and "data" in result:
        payload = result.get("data") or []
    elif isinstance(result, Sequence):
        payload = result
    else:
        payload = []

    distributions: list[dict[str, float]] = []
    for entry in payload:
        logprob_list = None
        if isinstance(entry, Mapping) and "prompt_logprobs" in entry:
            logprob_list = entry.get("prompt_logprobs")
        elif isinstance(entry, Mapping) and "choices" in entry:
            choices = entry.get("choices") or []
            if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
                logprob_list = choices[0].get("prompt_logprobs")
        if isinstance(logprob_list, Sequence):
            for token_probs in logprob_list:
                token_map: dict[str, float] = {}
                if isinstance(token_probs, Mapping):
                    topk = token_probs.get("top_logprobs") or token_probs.get("topk")
                    if isinstance(topk, Sequence):
                        for candidate in topk:
                            if isinstance(candidate, Mapping):
                                token = candidate.get("token")
                                lp = candidate.get("logprob")
                                if isinstance(token, str) and isinstance(lp, (int, float)):
                                    token_map[token] = float(lp)
                if token_map:
                    distributions.append(token_map)
    return distributions


def _extract_logprob(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, Mapping):
        prompt_lp = result.get("prompt_logprobs") if hasattr(result, "get") else None
        if isinstance(prompt_lp, Sequence) and prompt_lp:
            for entry in reversed(prompt_lp):
                if isinstance(entry, Mapping):
                    lp_val = entry.get("total_logprob") or entry.get("logprob")
                    if isinstance(lp_val, (int, float)):
                        return float(lp_val)
                elif isinstance(entry, (int, float)):
                    return float(entry)
        if "logprobs" in result:
            lp = result.get("logprobs")
            if isinstance(lp, Sequence) and lp:
                first = lp[0]
                if isinstance(first, Mapping) and "total_logprob" in first:
                    return float(first["total_logprob"])
                if isinstance(first, (int, float)):
                    return float(first)
        if "total_logprob" in result:
            return float(result["total_logprob"])
        if "data" in result:
            return _extract_logprob(result.get("data"))
    if isinstance(result, Sequence) and result:
        first = result[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, Mapping):
            return _extract_logprob(first)
    attr_val = getattr(result, "total_logprob", None)
    if isinstance(attr_val, (int, float)):
        return float(attr_val)
    attr_lp = getattr(result, "logprobs", None)
    if isinstance(attr_lp, Sequence) and attr_lp:
        first = attr_lp[0]
        if isinstance(first, (int, float)):
            return float(first)
        if isinstance(first, Mapping):
            return _extract_logprob(first)
    return None


def _normalize_ranking(predictions: Sequence[float], probe_count: int) -> list[int]:
    seen = set()
    order: list[int] = []
    for value in predictions:
        idx = int(round(float(value))) - 1
        if 0 <= idx < probe_count and idx not in seen:
            seen.add(idx)
            order.append(idx)
    for idx in range(probe_count):
        if idx not in seen:
            order.append(idx)
    return order[:probe_count]


def _spearman_corr(predicted_order: Sequence[int], actual_order: Sequence[int]) -> float:
    if not predicted_order or not actual_order:
        return 0.0
    n = min(len(predicted_order), len(actual_order))
    if n == 1:
        return 1.0
    trimmed_predicted = predicted_order[:n]
    trimmed_actual = actual_order[:n]
    pred_positions = {idx: pos for pos, idx in enumerate(trimmed_predicted)}
    diffs_sq = []
    for pos, idx in enumerate(trimmed_actual):
        pred_pos = pred_positions.get(idx, n - 1)
        diffs_sq.append((pred_pos - pos) ** 2)
    numerator = 6 * sum(diffs_sq)
    denominator = n * (n * n - 1)
    return 1.0 - (numerator / denominator)


def _kl_from_distributions(prior: dict[str, float], post: dict[str, float]) -> float:
    if not prior or not post:
        return 0.0
    observed_prior_mass = sum(math.exp(lp) for lp in prior.values())
    missing_prior_mass = max(0.0, 1.0 - observed_prior_mass)
    new_token_count = sum(1 for token in post if token not in prior)
    fallback_prob = max(missing_prior_mass / max(1, new_token_count), 1e-8)
    lp_fallback = math.log(fallback_prob)

    kl = 0.0
    for token, lp_post in post.items():
        p_post = math.exp(lp_post)
        lp_prior = prior.get(token, lp_fallback)
        kl += p_post * (lp_post - lp_prior)
    return kl


class GradientProphetEnv(Env):
    """Native Tinker-style environment that queries logprobs during reward."""

    def __init__(self, sample: Mapping[str, Any], sampling_client: Any) -> None:
        self.sample = dict(sample)
        self.sample.setdefault("task", random.choice(["in_context", "surprise"]))
        if self.sample["task"] == "in_context":
            probes = self.sample.get("probes", []) or []
            self.sample["probe_index"] = int(self.sample.get("probe_index", 0)) % max(len(probes), 1)
        self.sample["prompt"] = _build_prompt(self.sample)
        self.parser = ProphetParser()
        self.sampling_client = sampling_client

    def initial_observation(self) -> str:
        return str(self.sample.get("prompt", ""))

    async def step(self, action: Any) -> StepResult:  # type: ignore[override]
        reward = await self._evaluate_reward(action)
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation="",
            next_stop_condition=[],
            metrics={"task": self.sample.get("task", "")},
        )

    async def _evaluate_reward(self, action: Any) -> float:
        predictions = self._parse_predictions(action)
        lesson_input = str(self.sample.get("lesson_input", "")).strip()
        lesson_target = str(self.sample.get("lesson_target", "")).strip()
        probes: Sequence[Mapping[str, Any]] = self.sample.get("probes", [])
        task = str(self.sample.get("task", "in_context"))

        if not lesson_input or not lesson_target or not probes:
            return 0.0

        if task == "in_context":
            return await self._reward_in_context(predictions, lesson_input, lesson_target, probes)

        return await self._reward_surprise(predictions, lesson_input, lesson_target, probes)

    def _parse_predictions(self, action: Any) -> list[float]:
        completion: Messages = [{"role": "assistant", "content": str(action)}]
        parsed = self.parser.parse_answer(completion)
        if parsed is None:
            return []
        return parsed

    async def _reward_in_context(
        self,
        predictions: Sequence[float],
        lesson_input: str,
        lesson_target: str,
        probes: Sequence[Mapping[str, Any]],
    ) -> float:
        probe_index = int(self.sample.get("probe_index", 0))
        probe_index = max(0, min(probe_index, max(len(probes) - 1, 0)))
        probe = probes[probe_index]
        probe_question = str(probe.get("input", "")).strip()
        probe_answer = str(probe.get("target", "")).strip()

        prior_prompt = f"{probe_question}\nAnswer:"
        post_prompt = (
            f"Lesson: {lesson_input}\nLesson Answer: {lesson_target}\n\n"
            f"{probe_question}\nAnswer:"
        )

        prior, post = await asyncio.gather(
            _target_logprob_async(self.sampling_client, prior_prompt, probe_answer),
            _target_logprob_async(self.sampling_client, post_prompt, probe_answer),
        )
        if prior is None or post is None:
            return 0.0

        delta_true = post - prior
        delta_pred = float(predictions[0]) if predictions else 0.0
        error = abs(delta_true - delta_pred)
        return float(1.0 / (1.0 + (error ** 2)))

    async def _reward_surprise(
        self,
        predictions: Sequence[float],
        lesson_input: str,
        lesson_target: str,
        probes: Sequence[Mapping[str, Any]],
    ) -> float:
        prior_scores: list[float] = []
        post_scores: list[float] = []

        for probe in probes:
            probe_question = str(probe.get("input", "")).strip()
            probe_answer = str(probe.get("target", "")).strip()
            base_prompt = f"{probe_question}\nAnswer:"
            conditioned_prompt = (
                f"Lesson: {lesson_input}\nLesson Answer: {lesson_target}\n\n"
                f"{probe_question}\nAnswer:"
            )
            prior_dist, post_dist = await asyncio.gather(
                _prompt_distributions(self.sampling_client, base_prompt),
                _prompt_distributions(self.sampling_client, conditioned_prompt),
            )

            if not prior_dist or not post_dist:
                prior = await _target_logprob_async(
                    self.sampling_client, base_prompt, probe_answer
                )
                post = await _target_logprob_async(
                    self.sampling_client, conditioned_prompt, probe_answer
                )
                if prior is None or post is None:
                    continue
                prior_scores.append(prior)
                post_scores.append(post)
                continue

            kl = _kl_from_distributions(prior_dist[-1], post_dist[-1])
            prior_scores.append(kl)
            post_scores.append(kl)

        if not prior_scores:
            return 0.0

        kl_scores: list[float] = []
        if len(prior_scores) == len(probes) and len(post_scores) == len(probes):
            for prior, post in zip(prior_scores, post_scores):
                if prior == post:
                    kl_scores.append(prior)
                else:
                    p_post = math.exp(post)
                    kl_scores.append(p_post * (post - prior))

        if not kl_scores:
            return 0.0

        actual_vec = np.array(kl_scores, dtype=float)
        if actual_vec.size == 0:
            return 0.0

        pred_list = list(predictions[: len(kl_scores)]) if predictions else []
        if len(pred_list) < len(kl_scores):
            pred_list.extend([0.0] * (len(kl_scores) - len(pred_list)))
        pred_vec = np.array(pred_list, dtype=float)

        norm_pred = np.linalg.norm(pred_vec)
        norm_actual = np.linalg.norm(actual_vec)
        if norm_pred == 0.0 or norm_actual == 0.0:
            return 0.0

        cosine_sim = float(np.dot(pred_vec, actual_vec) / (norm_pred * norm_actual))
        return cosine_sim


class GradientProphetDatasetBuilder:
    """Prepare Prophet environments for Tinker RL runs."""

    def __init__(self, samples: Sequence[Mapping[str, Any]] | None = None) -> None:
        self.samples = list(samples) if samples is not None else build_semantic_tension_dataset()

    def build(self, sampling_client: Any) -> list[GradientProphetEnv]:
        envs: list[GradientProphetEnv] = []
        for sample in self.samples:
            envs.append(GradientProphetEnv(sample, sampling_client))
        return envs


def load_environment(**_: Any) -> GradientProphetDatasetBuilder:
    return GradientProphetDatasetBuilder()


__all__ = ["GradientProphetEnv", "GradientProphetDatasetBuilder", "load_environment"]
