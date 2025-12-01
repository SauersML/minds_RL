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
from typing import Any, Mapping, MutableMapping

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


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self._tinker_module: Any | None = None
        self.service_client = None
        self.training_client = None

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
        )
        return cls(config, env)

    def _require_tinker(self) -> Any:
        if self._tinker_module is not None:
            return self._tinker_module
        spec = importlib.util.find_spec("tinker")
        if spec is None:
            raise ImportError(
                "The `tinker` package is required for training. Install it before running training."
            )
        self._tinker_module = importlib.import_module("tinker")
        return self._tinker_module

    def _build_clients(self) -> tuple[Any, Any, Any]:
        tinker = self._require_tinker()
        service_client = tinker.ServiceClient(
            base_model=self.config.base_model,
            api_key=self.config.tinker_api_key,
        )
        training_client = tinker.TrainingClient(
            service_client=service_client,
            loss_fn=self.config.loss_fn,
        )
        self.service_client = service_client
        self.training_client = training_client
        return tinker, service_client, training_client

    async def _train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / "metrics.json"

        rewards: list[float] = []
        metrics_history: list[dict[str, float | int]] = []

        tinker, service_client, training_client = self._build_clients()

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

            completions = await sampling_client.sample(
                prompt=prompt,
                num_samples=self.config.rollouts_per_example,
                max_tokens=self.config.max_new_tokens,
            )

            completion_texts: list[str] = []
            for completion in completions or []:
                if isinstance(completion, Mapping):
                    text = completion.get("text") or completion.get("completion")
                    if isinstance(text, str):
                        completion_texts.append(text)
                        continue
                text_attr = getattr(completion, "text", None)
                if isinstance(text_attr, str):
                    completion_texts.append(text_attr)
                elif isinstance(completion, str):
                    completion_texts.append(completion)

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
            for text, advantage in zip(completion_texts, advantages):
                datums.append(
                    tinker.Datum(
                        prompt=prompt,
                        completion=text,
                        advantage=float(advantage),
                    )
                )

            if datums:
                forward = training_client.forward_backward(datums)
                optim = training_client.optim_step()
                await asyncio.gather(self._maybe_await(forward), self._maybe_await(optim))

            reward_mean = baseline
            rewards.append(reward_mean)
            metrics_history.append({"reward": reward_mean, "step": step})
            metrics_path.write_text(json.dumps(metrics_history, indent=2))

        metrics = {"reward": sum(rewards) / len(rewards) if rewards else 0.0}
        metrics_history.append(metrics)
        metrics_path.write_text(json.dumps(metrics_history, indent=2))
        return metrics

    def train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        return asyncio.run(self._train(max_steps=max_steps, output_dir=output_dir))
