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

try:
    from tinker_cookbook.renderers import get_renderer as tinker_get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer as tinker_get_tokenizer
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.completers import TinkerTokenCompleter
except ImportError:
    tinker_get_renderer = None
    tinker_get_tokenizer = None
    get_recommended_renderer_name = None
    TinkerTokenCompleter = None

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

State = MutableMapping[str, Any]

# Lazy import to avoid circular dependency for typing
RendererType = Any


@dataclass
class TrainerConfig:
    base_model: str
    rollouts_per_example: int = 4
    max_new_tokens: int = 256
    loss_fn: str = "ppo"
    tinker_api_key: str | None = None
    training_rank: int = 32
    learning_rate: float = 1e-4


class SamplingClientAdapter:
    def __init__(self, client: Any, tokenizer: Any) -> None:
        self._client = client
        self._tokenizer = tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _model_input_from_text(self, text: Any) -> Any:
        """Convert raw text into a ModelInput using the configured tokenizer.

        The Tinker SDK requires prompts to be instances of ``ModelInput``. This helper
        performs the conversion and raises a clear error instead of silently passing
        through invalid types, which previously led to Pydantic validation errors.
        """

        if not isinstance(text, str):
            return text

        model_input_cls = None
        try:
            import tinker

            model_input_cls = getattr(tinker, "ModelInput", None)
        except ImportError as exc:  # pragma: no cover - handled by dummy SDK in tests
            # Fall back to the lightweight dummy SDK shipped with Trainer for offline
            # execution. If neither is available, surface a clear error instead of
            # silently returning a raw string.
            try:
                from custom_utils import trainer as trainer_module

                model_input_cls = getattr(getattr(trainer_module, "Trainer", None), "_DummyTinker", None)
                model_input_cls = getattr(model_input_cls, "ModelInput", None) if model_input_cls else None
            except Exception:
                model_input_cls = None

            if model_input_cls is None:
                raise RuntimeError("tinker package is required to convert prompt text to ModelInput") from exc

        if not hasattr(self._tokenizer, "encode"):
            raise TypeError("Tokenizer must implement an 'encode' method to build ModelInput prompts")

        try:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)
            return model_input_cls.from_ints(tokens)
        except Exception as exc:  # pragma: no cover - defensive: mirrors SDK expectations
            raise ValueError("Failed to convert prompt text to a ModelInput") from exc

    async def sample_async(self, prompt: Any, **kwargs: Any) -> Any:
        prompt = self._model_input_from_text(prompt)

        if isinstance(prompt, str):
            raise TypeError(
                "Failed to convert prompt text to ModelInput before sampling; received raw string"
            )

        if hasattr(self._client, "sample_async"):
            return await self._client.sample_async(prompt=prompt, **kwargs)

        if hasattr(self._client, "sample"):
            result = self._client.sample(prompt=prompt, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        raise AttributeError(f"'{type(self._client).__name__}' object has no attribute 'sample_async' or 'sample'")

    async def compute_logprobs_async(
        self, prompt: str, targets: Sequence[str] | None = None, **kwargs: Any
    ) -> Any:
        prompt_input = self._model_input_from_text(prompt)
        if isinstance(prompt_input, str):
            raise TypeError(
                "Failed to convert prompt text to ModelInput before computing logprobs"
            )
        if not targets:
            return await self._client.compute_logprobs_async(prompt=prompt_input, **kwargs)

        results = []
        for target in targets:
            full_text = prompt + target
            full_prompt_input = self._model_input_from_text(full_text)
            if isinstance(full_prompt_input, str):
                raise TypeError(
                    "Failed to convert prompt+target text to ModelInput before computing logprobs"
                )
            result = await self._client.compute_logprobs_async(prompt=full_prompt_input, **kwargs)

            # Try to extract a sequence of logprobs from the result
            logprobs_seq = None
            if hasattr(result, "get"):
                logprobs_seq = result.get("prompt_logprobs") or result.get("logprobs")
            elif hasattr(result, "prompt_logprobs"):
                logprobs_seq = result.prompt_logprobs

            total = 0.0
            if logprobs_seq and hasattr(self._tokenizer, "encode"):
                prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)
                offset = len(prompt_tokens)

                current_idx = 0
                for item in logprobs_seq:
                    val = 0.0
                    if isinstance(item, (float, int)):
                        val = float(item)
                    elif hasattr(item, "get"):
                        val = float(item.get("logprob") or item.get("total_logprob") or 0.0)
                    elif hasattr(item, "logprob"):
                        val = float(item.logprob)

                    if current_idx >= offset:
                        total += val
                    current_idx += 1

            results.append({"total_logprob": total})

        if len(targets) == 1 and len(results) == 1:
            return results[0]
        return results


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self._tinker_module: Any | None = None
        self.service_client = None
        self.training_client = None
        self.tokenizer = None
        self.renderer: RendererType | None = None

    class _DummyTokenizer:
        """Lightweight tokenizer used when transformers models are unavailable."""

        def apply_chat_template(self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = True):
            del add_generation_prompt, tokenize
            content = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, Mapping))
            return [len(content) % 7, len(content) % 5, len(content) % 3]

        def decode(self, tokens: Sequence[int], skip_special_tokens: bool = True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

        def encode(self, text: str, add_special_tokens: bool = True):
            del add_special_tokens
            return [ord(c) % 1000 for c in text]

    class _DummyRenderer:
        def build_generation_prompt(self, messages: Any) -> list[int]:
            content = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, Mapping))
            return [len(content) % 7, len(content) % 5, len(content) % 3]

        def get_stop_sequences(self) -> list[str]:
            return []

    class _DummyCompleter:
        def __init__(self, sampling_client: Any, max_tokens: int = 256):
            self.sampling_client = sampling_client
            self.max_tokens = max_tokens

        async def __call__(self, model_input: Any, stop_sequences: list[str]) -> Any:
            class DummyTokens:
                tokens = [0, 1, 2]
                logprobs = [0.0, 0.0, 0.0]
            return DummyTokens()


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
            loss_fn=str(trainer_cfg.get("loss_fn", "ppo")),
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
            raise ModuleNotFoundError(
                "The 'tinker' package is required for training; install it and ensure "
                "network access before running the trainer."
            )
        self._tinker_module = importlib.import_module("tinker")
        return self._tinker_module

    async def _build_clients(self) -> tuple[Any, Any, Any]:
        tinker = self._require_tinker()
        service_client = tinker.ServiceClient(
            api_key=self.config.tinker_api_key,
        )
        # Use the async version to prevent blocking the event loop
        training_client = await service_client.create_lora_training_client_async(
            base_model=self.config.base_model,
            rank=self.config.training_rank,
        )
        self.service_client = service_client
        self.training_client = training_client
        return tinker, service_client, training_client

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        try:
            if tinker_get_tokenizer is not None:
                self.tokenizer = tinker_get_tokenizer(self.config.base_model)
                return
        except Exception:
            pass

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

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        self._ensure_tokenizer()
        if get_recommended_renderer_name is not None and tinker_get_renderer is not None:
            renderer_name = get_recommended_renderer_name(self.config.base_model)
            self.renderer = tinker_get_renderer(renderer_name, self.tokenizer)
        else:
            # Fallback for when tinker_cookbook is missing
            self.renderer = self._DummyRenderer()

    def _flatten_model_input_tokens(self, model_input: Any) -> list[int]:
        tokens: list[int] = []
        chunks = getattr(model_input, "chunks", [])
        for chunk in chunks:
            chunk_tokens = getattr(chunk, "tokens", None)
            if isinstance(chunk_tokens, list):
                tokens.extend(int(tok) for tok in chunk_tokens)
        return tokens

    def _render_prompt(self, prompt: str) -> tuple[Any, list[int]]:
        self._ensure_renderer()
        tinker = self._require_tinker()
        messages = [{"role": "user", "content": prompt}]
        model_input = self.renderer.build_generation_prompt(messages)
        if not isinstance(model_input, tinker.ModelInput):
            model_input = tinker.ModelInput.from_ints(list(model_input))
        prompt_tokens = self._flatten_model_input_tokens(model_input)
        return model_input, prompt_tokens

    async def _prepare_sampling_client(self, training_client: Any, service_client: Any, step: int) -> Any:
        client = await self._prepare_sampling_client_inner(training_client, service_client, step)
        return SamplingClientAdapter(client, self.tokenizer)

    async def _prepare_sampling_client_inner(self, training_client: Any, service_client: Any, step: int) -> Any:
        step_name = f"step_{step}"
        sampling_client = None

        saver = getattr(training_client, "save_weights_and_get_sampling_client", None)
        if callable(saver):
            sampling_client = await self._maybe_await(saver(name=step_name))

        if sampling_client is None:
            save_weights = getattr(training_client, "save_weights_for_sampler", None)
            if callable(save_weights):
                saved_weights = await self._maybe_await(save_weights(name=step_name))
                if hasattr(saved_weights, "sample"):
                    sampling_client = saved_weights
                else:
                    creator = getattr(service_client, "create_sampling_client", None)
                    if callable(creator):
                        candidate_kwargs = [
                            {"base_model": self.config.base_model, "weights_path": saved_weights},
                            {"base_model": self.config.base_model, "weights": saved_weights},
                            {"base_model": self.config.base_model, "checkpoint": saved_weights},
                            {"base_model": self.config.base_model, "path": saved_weights},
                            {"base_model": self.config.base_model},
                        ]
                        for kwargs in candidate_kwargs:
                            try:
                                sampling_client = creator(**kwargs)
                                if sampling_client is not None:
                                    break
                            except TypeError:
                                continue

        if sampling_client is None:
            creator = getattr(service_client, "create_sampling_client", None)
            if callable(creator):
                try:
                    sampling_client = creator(base_model=self.config.base_model)
                except Exception:
                    pass

        if sampling_client is None:
            sampling_client = service_client

        return SamplingClientAdapter(sampling_client, self.tokenizer)

    async def _train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / "metrics.json"

        log_path = output_path / "conversation_log.tsv"
        log_file = log_path.open("a", encoding="utf-8")
        if log_path.stat().st_size == 0:
            log_file.write("Step\tPrompt\tCompletion\n")
            log_file.flush()

        rewards: list[float] = []
        metrics_history: list[dict[str, float | int]] = []
        pending_training_task: asyncio.Task[Any] | None = None

        # Await the async client builder
        tinker, service_client, training_client = await self._build_clients()
        self._ensure_tokenizer()

        dataset = getattr(self.env, "dataset", None)
        steps = max(max_steps, 1)

        def _escape_tsv(text: str) -> str:
            return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

        try:
            for step in range(steps):
                sampling_client = await self._prepare_sampling_client(
                    training_client, service_client, step
                )

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

                print("=== INPUT ===")
                print(prompt)

                model_input, prompt_tokens = self._render_prompt(prompt)
                stop_sequences = self.renderer.get_stop_sequences()

                completion_texts: list[str] = []
                completion_data: list[dict[str, Any]] = []
                for _ in range(self.config.rollouts_per_example):
                    if TinkerTokenCompleter is not None:
                        completer = TinkerTokenCompleter(
                            sampling_client=sampling_client,
                            max_tokens=self.config.max_new_tokens,
                        )
                    else:
                        completer = self._DummyCompleter(
                            sampling_client=sampling_client,
                            max_tokens=self.config.max_new_tokens,
                        )
                    tokens_with_logprobs = await completer(model_input, stop_sequences)

                    completion_tokens = tokens_with_logprobs.tokens
                    logprobs = tokens_with_logprobs.logprobs

                    try:
                        completion_text = (
                            self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                            if self.tokenizer is not None
                            else ""
                        )
                    except Exception:
                        completion_text = ""

                    print("=== OUTPUT ===")
                    print(completion_text)

                    escaped_prompt = _escape_tsv(prompt)
                    escaped_completion = _escape_tsv(completion_text)
                    log_file.write(f"{step}\t{escaped_prompt}\t{escaped_completion}\n")
                    log_file.flush()

                completion_texts.append(completion_text)
                completion_data.append(
                    {
                        "text": completion_text,
                        "tokens": completion_tokens,
                        "logprobs": logprobs,
                        "prompt_tokens": prompt_tokens,
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
                        step_callable = getattr(env, "step", None)
                        if not callable(step_callable):
                            continue
                        step_result = step_callable(completion_text)
                        step_result = await self._maybe_await(step_result)
                        reward_value = step_result.reward if isinstance(step_result, Mapping) else getattr(step_result, "reward", 0.0)
                        try:
                            reward_value = float(reward_value)
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            reward_value = 0.0
                        completion_rewards.append(reward_value)

                rewards_array = np.asarray(completion_rewards, dtype=np.float32)
                baseline = float(np.mean(rewards_array)) if rewards_array.size > 0 else 0.0
                reward_std = float(np.std(rewards_array)) if rewards_array.size > 0 else 0.0
                denom = reward_std + 1e-4
                advantages = (
                    ((rewards_array - baseline) / denom).tolist()
                    if rewards_array.size > 0
                    else []
                )

                datums: list[Any] = []
                tinker = self._require_tinker()
                for data, advantage in zip(completion_data, advantages):
                    completion_tokens = np.asarray(data.get("tokens", []), dtype=np.int64)
                    logprob_array = np.asarray(data.get("logprobs", []), dtype=np.float32)
                    prompt_tokens = np.asarray(data.get("prompt_tokens", []), dtype=np.int64)
                    if completion_tokens.size == 0 or logprob_array.size == 0:
                        continue
                    if logprob_array.shape[0] != completion_tokens.shape[0]:
                        min_len = min(logprob_array.shape[0], completion_tokens.shape[0])
                        completion_tokens = completion_tokens[:min_len]
                        logprob_array = logprob_array[:min_len]

                    combined_tokens = prompt_tokens.tolist() + completion_tokens.tolist()
                    if not combined_tokens:
                        continue

                    # Tinker requires loss inputs to match the full model_input length
                    total_len = len(combined_tokens)
                    prompt_len = len(prompt_tokens)

                    # Targets are the full sequence
                    full_targets = np.array(combined_tokens, dtype=np.int64)

                    # Initialize arrays for full sequence
                    full_advantages = np.zeros(total_len, dtype=np.float32)
                    full_logprobs = np.zeros(total_len, dtype=np.float32)

                    # Fill the completion region (masking prompt region with 0s)
                    # Ensure dimensions match in case of truncation elsewhere
                    comp_len = len(completion_tokens)
                    full_advantages[prompt_len : prompt_len + comp_len] = float(advantage)
                    full_logprobs[prompt_len : prompt_len + comp_len] = logprob_array

                    if self.config.loss_fn == "cross_entropy":
                        # For SL, use weights tensor for masking
                        full_weights = np.zeros(total_len, dtype=np.float32)
                        full_weights[prompt_len : prompt_len + comp_len] = 1.0
                        loss_fn_inputs = {
                            "target_tokens": tinker.TensorData.from_numpy(full_targets),
                            "weights": tinker.TensorData.from_numpy(full_weights),
                        }
                    else:
                        # For RL, masking is implicit via 0.0 advantage
                        loss_fn_inputs = {
                            "target_tokens": tinker.TensorData.from_numpy(full_targets),
                            "logprobs": tinker.TensorData.from_numpy(full_logprobs),
                            "advantages": tinker.TensorData.from_numpy(full_advantages),
                        }

                    datums.append(
                        tinker.Datum(
                            model_input=tinker.ModelInput.from_ints(combined_tokens),
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
                    async def _background_step(fw, op):
                        await asyncio.gather(fw, op)

                    training_task = asyncio.create_task(
                        _background_step(
                            self._maybe_await(forward),
                            self._maybe_await(optim),
                        )
                    )

                    if pending_training_task is not None:
                        await pending_training_task
                    pending_training_task = training_task

                reward_mean = baseline
                rewards.append(reward_mean)
                metrics_history.append({"reward": reward_mean, "loss": 0.0, "step": step})
                metrics_path.write_text(json.dumps(metrics_history, indent=2))
        finally:
            log_file.close()

        if pending_training_task is not None:
            await pending_training_task

        metrics = {
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "loss": 0.0,
        }
        metrics_history.append(metrics)
        metrics_path.write_text(json.dumps(metrics_history, indent=2))
        return metrics

    def train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        return asyncio.run(self._train(max_steps=max_steps, output_dir=output_dir))
