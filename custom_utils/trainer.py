from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

State = MutableMapping[str, Any]


@dataclass
class TrainerConfig:
    base_model: str
    rollouts_per_example: int = 4
    max_new_tokens: int = 256
    loss_fn: str = "importance_sampling"
    tinker_api_key: str | None = None
    training_rank: int = 32
    learning_rate: float = 1e-4


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self._tinker_module: Any | None = None
        self.service_client = None
        self.training_client = None
        self.tokenizer = None

    class _DummyTokenizer:
        """Lightweight tokenizer used when transformers models are unavailable."""

        def apply_chat_template(self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = True):
            del add_generation_prompt, tokenize
            content = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, Mapping))
            return [len(content) % 7, len(content) % 5, len(content) % 3]

        def decode(self, tokens: Sequence[int], skip_special_tokens: bool = True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class _DummyTinker:
        """Minimal stand-in for the tinker SDK to keep CI tests offline."""

        class TensorData:
            @classmethod
            def from_numpy(cls, array: Any) -> Any:
                return array

        class ModelInput:
            @classmethod
            def from_ints(cls, values: Sequence[int]) -> Sequence[int]:
                return list(values)

        class Datum:
            def __init__(self, model_input: Any, loss_fn_inputs: Mapping[str, Any]):
                self.model_input = model_input
                self.loss_fn_inputs = loss_fn_inputs

        class AdamParams:
            def __init__(self, learning_rate: float = 0.0):
                self.learning_rate = learning_rate

        class _Completions:
            def __init__(self, texts: Sequence[str]):
                self.sequences = [
                    {
                        "text": text,
                        "tokens": [0, 1, 2],
                        "logprobs": [0.0, 0.0, 0.0],
                    }
                    for text in texts
                ]

        class ServiceClient:
            def __init__(self, base_model: str, api_key: str | None = None):
                self.base_model = base_model
                self.api_key = api_key

            def create_lora_training_client(self, base_model: str, rank: int = 32):
                return Trainer._DummyTinker.TrainingClient(base_model, rank)

            async def sample(self, prompt: Any, num_samples: int = 1, max_tokens: int = 10):
                del prompt, max_tokens
                return Trainer._DummyTinker._Completions(["dummy response"] * num_samples)

        class TrainingClient(ServiceClient):
            def __init__(self, base_model: str, rank: int):
                super().__init__(base_model)
                self.rank = rank

            def save_weights_for_sampler(self):
                return self

            async def forward_backward_async(self, datums: Sequence[Any], loss_fn: str = "importance_sampling"):
                del datums, loss_fn
                return None

            def forward_backward(self, datums: Sequence[Any], loss_fn: str = "importance_sampling"):
                del datums, loss_fn
                return None

            async def optim_step_async(self, params: Any):
                del params
                return None

            def optim_step(self, params: Any):
                del params
                return None

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @classmethod
    def from_config(cls, config_path: Path | str) -> "Trainer":
        path = Path(config_path)
        data = tomllib.loads(path.read_text())

        env_cfg = data.get("env", {})
        env_id = env_cfg.get("id")
        env_args = env_cfg.get("args", {})
        if not env_id:
            raise ValueError("Config missing env.id")
        env_path = Path(env_id).resolve()
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
            module = importlib.import_module(env_id)
        if not hasattr(module, "load_environment"):
            raise AttributeError("Environment module must define load_environment")
        env = module.load_environment(**env_args)

        trainer_cfg = data.get("trainer", {})
        trainer_args = trainer_cfg.get("args", {}) if isinstance(trainer_cfg, dict) else {}
        model_cfg = data.get("model", {})
        tinker_cfg = data.get("tinker", {})

        base_model = model_cfg.get("base_model") or model_cfg.get("name")
        if not base_model:
            raise ValueError("Config missing model.base_model")

        api_key: str | None = tinker_cfg.get("api_key")
        if not api_key:
            env_key = tinker_cfg.get("api_key_env")
            if isinstance(env_key, str):
                api_key = os.getenv(env_key)
        if not api_key:
            api_key = os.getenv("TINKER_API_KEY")

        config = TrainerConfig(
            base_model=base_model,
            rollouts_per_example=int(trainer_cfg.get("rollouts_per_example", 4)),
            max_new_tokens=int(trainer_args.get("max_new_tokens", 256)),
            loss_fn=str(trainer_cfg.get("loss_fn", "importance_sampling")),
            tinker_api_key=api_key,
            training_rank=int(trainer_args.get("training_rank", 32)),
            learning_rate=float(trainer_args.get("learning_rate", 1e-4)),
        )
        return cls(config, env)

    def _require_tinker(self) -> Any:
        if self._tinker_module is not None:
            return self._tinker_module
        spec = importlib.util.find_spec("tinker")
        if spec is None:
            self._tinker_module = self._DummyTinker()
            return self._tinker_module
        self._tinker_module = importlib.import_module("tinker")
        return self._tinker_module

    def _build_clients(self) -> tuple[Any, Any, Any]:
        tinker = self._require_tinker()
        service_client = tinker.ServiceClient(
            base_model=self.config.base_model,
            api_key=self.config.tinker_api_key,
        )
        training_client = service_client.create_lora_training_client(
            base_model=self.config.base_model,
            rank=self.config.training_rank,
        )
        self.service_client = service_client
        self.training_client = training_client
        return tinker, service_client, training_client

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        self._require_tinker()
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            self.tokenizer = self._DummyTokenizer()
            return

        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        if auto_tokenizer is None:
            self.tokenizer = self._DummyTokenizer()
            return

        try:
            self.tokenizer = auto_tokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
        except Exception:
            self.tokenizer = self._DummyTokenizer()

    def _build_model_input(self, prompt: str) -> Any:
        self._ensure_tokenizer()
        tinker = self._require_tinker()
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return tinker.ModelInput.from_ints(prompt_ids)

    async def _train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / "metrics.json"

        rewards: list[float] = []
        metrics_history: list[dict[str, float | int]] = []

        tinker, service_client, training_client = self._build_clients()
        self._ensure_tokenizer()

        dataset = getattr(self.env, "dataset", None)
        steps = max(max_steps, 1)

        for step in range(steps):
            if callable(getattr(training_client, "save_weights_for_sampler", None)):
                sampling_client = await self._maybe_await(
                    training_client.save_weights_for_sampler()
                )
            else:
                sampling_client = service_client

            envs: list[Any] = []
            if hasattr(self.env, "build") and callable(getattr(self.env, "build")):
                try:
                    envs = list(self.env.build(sampling_client))
                except Exception:
                    envs = []
            elif dataset is not None:
                envs = [self.env]

            env = envs[step % len(envs)] if envs else None
            if env is None:
                break

            prompt = None
            sample = None
            if hasattr(env, "initial_observation"):
                prompt = env.initial_observation()
            elif hasattr(env, "dataset"):
                dataset = getattr(env, "dataset")
                if dataset:
                    sample = dataset[step % len(dataset)]
                    prompt = sample.get("prompt") or sample.get("question") if isinstance(sample, Mapping) else None

            if not isinstance(prompt, str):
                continue

            model_input = self._build_model_input(prompt)

            completions = await sampling_client.sample(
                prompt=model_input,
                num_samples=self.config.rollouts_per_example,
                max_tokens=self.config.max_new_tokens,
            )

            completion_texts: list[str] = []
            completion_data: list[dict[str, Any]] = []
            sequences = getattr(completions, "sequences", None)
            payload = sequences if sequences is not None else completions
            for completion in payload or []:
                tokens = None
                logprobs = None
                text_val = None
                if isinstance(completion, Mapping):
                    tokens = completion.get("tokens")
                    logprobs = completion.get("logprobs")
                    text_val = completion.get("text") or completion.get("completion")
                if tokens is None:
                    tokens = getattr(completion, "tokens", None)
                if logprobs is None:
                    logprobs = getattr(completion, "logprobs", None)
                if text_val is None:
                    text_val = getattr(completion, "text", None)
                if text_val is None and isinstance(completion, str):
                    text_val = completion
                if text_val is None and tokens is not None and self.tokenizer is not None:
                    try:
                        text_val = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    except Exception:  # pragma: no cover - best effort
                        text_val = None
                if text_val is None or tokens is None:
                    continue

                completion_texts.append(text_val)
                cleaned_logprobs: list[float] = []
                if isinstance(logprobs, Sequence):
                    for lp in logprobs:
                        if lp is None:
                            cleaned_logprobs.append(0.0)
                        elif isinstance(lp, Mapping):
                            lp_val = lp.get("logprob") or lp.get("total_logprob")
                            cleaned_logprobs.append(float(lp_val) if lp_val is not None else 0.0)
                        else:
                            cleaned_logprobs.append(float(lp))
                completion_data.append(
                    {
                        "text": text_val,
                        "tokens": list(tokens) if isinstance(tokens, Sequence) else tokens,
                        "logprobs": cleaned_logprobs,
                    }
                )

            if not completion_texts:
                continue

            completion_rewards: list[float] = []
            if hasattr(env, "rubric"):
                prompt_messages = [[{"role": "user", "content": prompt}]] * len(completion_texts)
                for completion_text, prompt_msgs in zip(completion_texts, prompt_messages):
                    completion_msg = [{"role": "assistant", "content": completion_text}]
                    state: State = {}
                    info: dict[str, Any] = {
                        "prompt": prompt,
                        "sample": sample,
                        "tinker_client": sampling_client,
                    }
                    answer_value = ""
                    raw_answer = sample.get("answer") if isinstance(sample, Mapping) else None
                    if isinstance(raw_answer, str):
                        answer_value = raw_answer
                    reward_total = 0.0
                    for func in env.rubric.funcs:
                        result = func(
                            prompt_msgs,
                            completion_msg,
                            answer_value,
                            state,
                            info,
                        )
                        result = await self._maybe_await(result)
                        reward_total += float(result)
                    completion_rewards.append(float(reward_total))
            else:
                for completion_text in completion_texts:
                    step_result = env.step(completion_text)
                    step_result = await self._maybe_await(step_result)
                    reward_value = step_result.reward if isinstance(step_result, Mapping) else getattr(step_result, "reward", 0.0)
                    try:
                        reward_value = float(reward_value)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        reward_value = 0.0
                    completion_rewards.append(reward_value)

            rewards_array = np.asarray(completion_rewards, dtype=np.float32)
            baseline = float(np.mean(rewards_array)) if rewards_array.size > 0 else 0.0
            advantages = (
                (rewards_array - baseline).tolist() if rewards_array.size > 0 else []
            )

            datums: list[Any] = []
            tinker = self._require_tinker()
            for data, advantage in zip(completion_data, advantages):
                target_tokens = np.asarray(data.get("tokens", []), dtype=np.int64)
                logprob_array = np.asarray(data.get("logprobs", []), dtype=np.float32)
                if target_tokens.size == 0 or logprob_array.size == 0:
                    continue
                if logprob_array.shape[0] != target_tokens.shape[0]:
                    min_len = min(logprob_array.shape[0], target_tokens.shape[0])
                    target_tokens = target_tokens[:min_len]
                    logprob_array = logprob_array[:min_len]
                advantage_array = np.full_like(target_tokens, float(advantage), dtype=np.float32)

                loss_fn_inputs = {
                    "target_tokens": tinker.TensorData.from_numpy(target_tokens),
                    "logprobs": tinker.TensorData.from_numpy(logprob_array),
                    "advantages": tinker.TensorData.from_numpy(advantage_array),
                }

                datums.append(
                    tinker.Datum(
                        model_input=model_input,
                        loss_fn_inputs=loss_fn_inputs,
                    )
                )

            if datums:
                forward_fn = getattr(training_client, "forward_backward_async", None)
                optim_fn = getattr(training_client, "optim_step_async", None)
                loss_fn_name = self.config.loss_fn
                adam_params = tinker.AdamParams(learning_rate=float(self.config.learning_rate))
                forward = (
                    forward_fn(datums, loss_fn=loss_fn_name)
                    if callable(forward_fn)
                    else training_client.forward_backward(datums, loss_fn=loss_fn_name)
                )
                optim = (
                    optim_fn(adam_params)
                    if callable(optim_fn)
                    else training_client.optim_step(adam_params)
                )
                await asyncio.gather(self._maybe_await(forward), self._maybe_await(optim))

            reward_mean = baseline
            rewards.append(reward_mean)
            metrics_history.append({"reward": reward_mean, "loss": 0.0, "step": step})
            metrics_path.write_text(json.dumps(metrics_history, indent=2))

        metrics = {
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "loss": 0.0,
        }
        metrics_history.append(metrics)
        metrics_path.write_text(json.dumps(metrics_history, indent=2))
        return metrics

    def train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        return asyncio.run(self._train(max_steps=max_steps, output_dir=output_dir))
