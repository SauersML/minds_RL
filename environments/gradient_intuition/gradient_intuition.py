from __future__ import annotations

import importlib
import importlib.util
import inspect
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import verifiers as vf

from .probes import Probe, get_random_probe

State = MutableMapping[str, Any]
Messages = list[Mapping[str, Any]]


class GradientIntuitionParser(vf.Parser):
    """Parse dual outputs of the form ``PREDICTION: <float>`` and ``ANSWER: <text>``."""

    def __init__(self) -> None:
        super().__init__()
        import re

        self._prediction_pattern = re.compile(
            r"prediction\W*[:\-]\W*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+)",
            re.IGNORECASE,
        )
        self._answer_pattern = re.compile(r"answer\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)

    def parse(self, text: str) -> dict[str, Any] | None:  # type: ignore[override]
        if not text:
            return None
        predicted_delta = self.parse_prediction(text)
        answer = self.parse_answer([{"role": "assistant", "content": text}])
        if predicted_delta is None and answer is None:
            return None
        result: dict[str, Any] = {}
        if predicted_delta is not None:
            result["prediction"] = predicted_delta
        if answer is not None:
            result["answer"] = answer
        return result or None

    def parse_answer(self, completion: Messages) -> str | None:  # type: ignore[override]
        if not completion:
            return None
        last = completion[-1]
        if not isinstance(last, Mapping):
            return None
        content = last.get("content")
        if not isinstance(content, str):
            return None
        match = self._answer_pattern.search(content)
        if match:
            candidate = match.group(1).strip()
            return candidate if candidate else None
        stripped = content.strip()
        return stripped or None

    def parse_prediction(self, text: str) -> float | None:
        match = self._prediction_pattern.search(text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None


_DEF_INSTRUCTIONS = (
    "You will complete a task and also predict how much learning from your answer will change the log-probability of a probe.\n"
    "Return two lines using the format:\n"
    "PREDICTION: <float delta>\n"
    "ANSWER: <task answer>"
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


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


def _extract_logprob_sequence(result: Any) -> list[float]:
    if result is None:
        return []
    if isinstance(result, Sequence) and not isinstance(result, Mapping):
        return [float(x) for x in result if isinstance(x, (int, float))]
    if isinstance(result, Mapping):
        prompt_lp = result.get("prompt_logprobs") if hasattr(result, "get") else None
        if isinstance(prompt_lp, Sequence) and prompt_lp:
            return [float(x) for x in prompt_lp if isinstance(x, (int, float))]
        if "logprobs" in result:
            lp = result.get("logprobs")
            if isinstance(lp, Sequence):
                return [float(x) for x in lp if isinstance(x, (int, float))]
        if "data" in result:
            return _extract_logprob_sequence(result.get("data"))
        choices = result.get("choices")
        if isinstance(choices, Sequence) and choices:
            return _extract_logprob_sequence(choices[0])
    attr_lp = getattr(result, "logprobs", None)
    if isinstance(attr_lp, Sequence):
        return [float(x) for x in attr_lp if isinstance(x, (int, float))]
    attr_data = getattr(result, "data", None)
    if attr_data is not None:
        return _extract_logprob_sequence(attr_data)
    return []


class _ShadowClientManager:
    def __init__(
        self,
        *,
        service_client: Any | None,
        base_model: str | None,
        shadow_rank: int,
    ) -> None:
        self.service_client = service_client
        self.base_model = base_model
        self.shadow_rank = shadow_rank
        self.client: Any | None = None

    async def get_client(self) -> Any | None:
        if self.client is not None:
            return self.client
        if self.service_client is None or not self.base_model:
            return None
        creator = getattr(self.service_client, "create_lora_training_client_async", None)
        if callable(creator):
            self.client = await creator(base_model=self.base_model, rank=self.shadow_rank)
        else:
            creator_sync = getattr(self.service_client, "create_lora_training_client", None)
            if not callable(creator_sync):
                return None
            self.client = creator_sync(base_model=self.base_model, rank=self.shadow_rank)
        return self.client

    async def reset_client(self) -> Any | None:
        client = await self.get_client()
        if client is None:
            return None

        for candidate in ("reset_parameters", "reset_adapter", "reset", "load_base_weights", "reload_base_weights"):
            func = getattr(client, candidate, None)
            if callable(func):
                result = func()
                if inspect.isawaitable(result):
                    await result
                return client

        self.client = None
        return await self.get_client()


class GradientIntuitionEnv:
    """Wrap an existing environment with a gradient-prediction objective."""

    def __init__(
        self,
        inner_env: Any,
        *,
        probes: Sequence[Probe],
        alpha: float = 0.3,
        seed: int | None = None,
        service_client: Any | None = None,
        base_model: str | None = None,
        shadow_rank: int = 8,
        shadow_learning_rate: float = 1e-4,
        shadow_manager: _ShadowClientManager | None = None,
    ) -> None:
        self.inner_env = inner_env
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.probes = list(probes)
        self.parser = GradientIntuitionParser()
        self.system_prompt = getattr(inner_env, "system_prompt", None)
        self.few_shot = getattr(inner_env, "few_shot", None)
        self.service_client = service_client
        self.base_model = base_model
        self.shadow_rank = shadow_rank
        self.shadow_learning_rate = shadow_learning_rate
        self._shadow_manager = shadow_manager or _ShadowClientManager(
            service_client=service_client,
            base_model=base_model,
            shadow_rank=shadow_rank,
        )
        self._shadow_client: Any | None = None
        self._current_sample: Mapping[str, Any] | None = None
        self._current_probe: Probe | None = None
        self._current_prompt: str = ""

        async def _reward(
            prompt: Messages,
            completion: Messages,
            _: str,
            state: State,
            info: Mapping[str, Any] | None = None,
            **__: Any,
        ) -> float:
            return await self._evaluate_reward(prompt, completion, state, info)

        self.rubric = vf.Rubric(funcs=[_reward], weights=[1.0])

    def _sample_task(self) -> Mapping[str, Any]:
        dataset = getattr(self.inner_env, "dataset", None)
        if dataset is None:
            return {}
        try:
            dataset_length = len(dataset)  # type: ignore[arg-type]
        except Exception:
            dataset_length = 0
        if dataset_length <= 0:
            return {}
        if hasattr(dataset, "shuffle") and hasattr(dataset, "select"):
            sample_idx = self.rng.randint(0, dataset_length - 1)
            return dataset[int(sample_idx)]
        try:
            return self.rng.choice(dataset)
        except Exception:
            return dataset[0]

    def initial_observation(self) -> str:
        base_prompt = self.inner_env.initial_observation()
        try:
            base_prompt_str = str(base_prompt)
        except Exception:
            base_prompt_str = "" if base_prompt is None else str(base_prompt)
        self._current_prompt = base_prompt_str

        sample = getattr(self.inner_env, "state", {}).get("sample") if hasattr(self.inner_env, "state") else None
        if not isinstance(sample, Mapping) or not sample:
            sample = self._sample_task()
        if not isinstance(sample, Mapping):
            sample = {}
        probe = get_random_probe(self.rng)
        self._current_sample = sample
        self._current_probe = probe

        prompt = base_prompt_str or str(sample.get("prompt") or sample.get("question") or "")
        instructions = sample.get("instructions") or _DEF_INSTRUCTIONS
        probe_q = probe.get("question", "")
        probe_a = probe.get("answer", "")
        meta_block = (
            f"\n\n---\nMeta-Task: {instructions}\n"
            f"Probe: \"{probe_q}\"\n"
            f"Probe target answer: \"{probe_a}\"\n"
            "Remember to provide the prediction first, then the task answer."
        )
        return f"{prompt}{meta_block}\n\nPREDICTION: \nANSWER: "

    async def _evaluate_reward(
        self,
        prompt: Messages,
        completion: Messages,
        state: State,
        info: Mapping[str, Any] | None,
    ) -> float:
        del prompt
        if not isinstance(info, Mapping):
            info = {}
        parsed = None
        if completion:
            last = completion[-1] if isinstance(completion, Sequence) else None
            if isinstance(last, Mapping):
                content = last.get("content")
                if isinstance(content, str):
                    parsed = self.parser.parse(content)
        if not parsed:
            return 0.0

        prediction = parsed.get("prediction") if isinstance(parsed, Mapping) else None
        task_answer = parsed.get("answer") if isinstance(parsed, Mapping) else None
        if task_answer is None:
            return 0.0

        task_reward = await self._task_reward(task_answer, info)
        if prediction is None:
            return task_reward

        actual_delta = await self._measure_probe_delta(task_answer, info)
        if actual_delta is None:
            return task_reward

        intuition_score = max(0.0, 1.0 - abs(prediction - actual_delta))
        return task_reward + self.alpha * intuition_score

    async def _task_reward(self, task_answer: str, info: Mapping[str, Any] | None) -> float:
        sample = self._current_sample or {}
        inner_rubric = getattr(self.inner_env, "rubric", None)
        if inner_rubric is None:
            return 0.0
        answer_value = sample.get("answer") if isinstance(sample, Mapping) else ""
        state: State = getattr(self.inner_env, "state", {}) if hasattr(self.inner_env, "state") else {}
        prompt_msgs = [
            {
                "role": "user",
                "content": self._current_prompt
                or str(sample.get("prompt") or sample.get("question") or ""),
            }
        ]
        completion_msgs = [{"role": "assistant", "content": str(task_answer)}]
        info_map: dict[str, Any] = {"sample": sample}
        if isinstance(info, Mapping):
            info_map.update(info)

        reward_total = 0.0
        weights: Iterable[float] = getattr(inner_rubric, "weights", [1.0] * len(inner_rubric.funcs))
        for func, weight in zip(inner_rubric.funcs, weights):
            result = func(prompt_msgs, completion_msgs, str(answer_value or ""), state, info_map)
            result = await _maybe_await(result)
            try:
                reward_total += float(result) * float(weight)
            except (TypeError, ValueError):
                continue
        return float(reward_total)

    async def _compute_logprob(
        self, client: Any, probe_question: str, probe_answer: str, tokenizer: Any
    ) -> float | None:
        if client is None or tokenizer is None:
            return None
        try:
            import tinker
        except ModuleNotFoundError:
            return None

        question_tokens = (
            tokenizer.encode(probe_question, add_special_tokens=False)
            if hasattr(tokenizer, "encode")
            else None
        )
        answer_tokens = (
            tokenizer.encode(probe_answer, add_special_tokens=False)
            if hasattr(tokenizer, "encode")
            else None
        )
        if not question_tokens or not answer_tokens:
            return None

        # Mirror the training setup: predict each next token in the question+answer
        # string and extract the log-probability mass over the answer slice.
        tokens = list(question_tokens) + list(answer_tokens)
        if len(tokens) < 2:
            return None
        input_tokens = np.array(tokens[:-1], dtype=np.int64)
        target_tokens = np.array(tokens[1:], dtype=np.int64)
        weights = np.ones_like(target_tokens, dtype=np.float32)
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens.tolist()),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(target_tokens),
                "weights": tinker.TensorData.from_numpy(weights),
            },
        )

        forward_fn = getattr(client, "forward_async", None)
        forward_kwargs = {"loss_fn": "cross_entropy"}
        if callable(forward_fn):
            result = await forward_fn([datum], **forward_kwargs)
        else:
            forward_sync = getattr(client, "forward", None)
            if not callable(forward_sync):
                return None
            result = forward_sync([datum], **forward_kwargs)
            if inspect.isawaitable(result):
                result = await result

        if hasattr(result, "result_async") and callable(result.result_async):
            result = await result.result_async()

        logprob_seq = _extract_logprob_sequence(result)
        if len(logprob_seq) >= len(answer_tokens):
            return float(sum(logprob_seq[-len(answer_tokens) :]))

        lp = _extract_logprob(result)
        return float(lp) if lp is not None else None

    async def _ensure_shadow_client(self, *, reset: bool = False) -> Any:
        if reset:
            self._shadow_client = await self._shadow_manager.reset_client()
            return self._shadow_client
        if self._shadow_client is not None:
            return self._shadow_client
        self._shadow_client = await self._shadow_manager.get_client()
        return self._shadow_client

    async def _shadow_update(self, prompt_text: str, completion_text: str, tokenizer: Any) -> bool:
        shadow_client = await self._ensure_shadow_client()
        if shadow_client is None:
            return False
        try:
            import tinker
        except ModuleNotFoundError:
            return False

        tokens = tokenizer.encode(prompt_text + completion_text, add_special_tokens=False) if hasattr(tokenizer, "encode") else []
        if len(tokens) < 2:
            return False
        input_tokens = np.array(tokens[:-1], dtype=np.int64)
        target_tokens = np.array(tokens[1:], dtype=np.int64)
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens.tolist()),
            loss_fn_inputs={"target_tokens": tinker.TensorData.from_numpy(target_tokens)},
        )

        forward_fn = getattr(shadow_client, "forward_backward_async", None)
        optim_fn = getattr(shadow_client, "optim_step_async", None)
        if callable(forward_fn):
            await forward_fn([datum])
        else:
            await _maybe_await(shadow_client.forward_backward([datum]))
        adam = tinker.AdamParams(learning_rate=float(self.shadow_learning_rate))
        if callable(optim_fn):
            await optim_fn(adam)
        else:
            await _maybe_await(shadow_client.optim_step(adam))
        return True

    async def _measure_probe_delta(self, task_answer: str, info: Mapping[str, Any] | None) -> float | None:
        probe = self._current_probe
        sample = self._current_sample
        if probe is None or sample is None:
            return None
        tokenizer = info.get("tokenizer") if isinstance(info, Mapping) else None
        if tokenizer is None:
            return None
        shadow_client = await self._ensure_shadow_client(reset=True)
        if shadow_client is None:
            return None

        probe_question = probe.get("question", "")
        probe_answer = probe.get("answer", "")
        pre_lp = await self._compute_logprob(shadow_client, probe_question, probe_answer, tokenizer)
        if pre_lp is None:
            return None

        prompt_text = self._current_prompt or str(sample.get("prompt") or sample.get("question") or "")
        updated = await self._shadow_update(prompt_text, task_answer, tokenizer)
        if not updated:
            return None

        post_lp = await self._compute_logprob(shadow_client, probe_question, probe_answer, tokenizer)
        if post_lp is None:
            return None
        return float(post_lp - pre_lp)


class GradientIntuitionBuilder:
    def __init__(
        self,
        *,
        inner_env_id: str = "./environments/ghost_trace",
        inner_env_args: Mapping[str, Any] | None = None,
        alpha: float = 0.3,
        probes: Sequence[Probe] | None = None,
        seed: int | None = None,
        shadow_rank: int = 8,
        shadow_learning_rate: float = 1e-4,
    ) -> None:
        self.inner_env_id = inner_env_id
        self.inner_env_args = dict(inner_env_args or {})
        self.alpha = alpha
        self.probes = list(probes) if probes is not None else None
        self.seed = seed
        self.shadow_rank = shadow_rank
        self.shadow_learning_rate = shadow_learning_rate

    def _load_inner_env(self) -> Any:
        env_path = Path(self.inner_env_id).resolve()
        if env_path.is_dir():
            if str(env_path) not in sys.path:
                sys.path.append(str(env_path))
            module_spec = importlib.util.spec_from_file_location(
                env_path.name,
                env_path / "__init__.py",
                submodule_search_locations=[str(env_path)],
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Unable to import environment from {env_path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(self.inner_env_id)
        if not hasattr(module, "load_environment"):
            raise AttributeError("Inner environment module must define load_environment")
        return module.load_environment(**self.inner_env_args)

    def build(
        self,
        sampling_client: Any,
        *,
        service_client: Any | None = None,
        base_model: str | None = None,
        training_client: Any | None = None,
    ) -> list[GradientIntuitionEnv]:
        inner_env = self._load_inner_env()
        envs: list[Any]
        if hasattr(inner_env, "build") and callable(getattr(inner_env, "build")):
            envs = list(inner_env.build(sampling_client))  # type: ignore[misc]
        else:
            envs = [inner_env]

        rng = random.Random(self.seed) if self.seed is not None else None
        probes = self.probes if self.probes is not None else [get_random_probe(rng)]
        if not probes:
            probes = [get_random_probe()]
        shadow_manager = _ShadowClientManager(
            service_client=service_client,
            base_model=base_model,
            shadow_rank=self.shadow_rank,
        )
        built_envs: list[GradientIntuitionEnv] = []
        for env in envs:
            built_envs.append(
                GradientIntuitionEnv(
                    env,
                    probes=probes,
                    alpha=self.alpha,
                    seed=self.seed,
                    service_client=service_client,
                    base_model=base_model,
                    shadow_rank=self.shadow_rank,
                    shadow_learning_rate=self.shadow_learning_rate,
                    shadow_manager=shadow_manager,
                )
            )
        return built_envs


def load_environment(**kwargs: Any) -> GradientIntuitionBuilder:
    return GradientIntuitionBuilder(**kwargs)


__all__ = [
    "GradientIntuitionEnv",
    "GradientIntuitionParser",
    "GradientIntuitionBuilder",
    "load_environment",
]
