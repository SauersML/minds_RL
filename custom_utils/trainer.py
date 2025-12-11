from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
from datetime import datetime
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
    learning_rate: float = 3.162e-6
    resume_checkpoint_id: str | None = None
    save_every_n_steps: int = 50
    update_sampler_every_n_steps: int = 10
    renderer_name: str | None = None


class SamplingClientAdapter:
    def __init__(self, client: Any, tokenizer: Any) -> None:
        self._client = client
        self.tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        """Expose a tokenizer for environments that need to decode token outputs."""

        if hasattr(self._client, "tokenizer") and getattr(self._client, "tokenizer") is not None:
            return getattr(self._client, "tokenizer")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any) -> None:
        self._tokenizer = value

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
        import tinker

        model_input_cls = getattr(tinker, "ModelInput", None)

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

            # Try to extract a sequence of logprobs from the result. The cookbook
            # API returns a simple ``list[float | None]``, but we keep the
            # defensive fallbacks for legacy/different client shapes.
            logprobs_seq: Sequence[Any] | None = None
            if isinstance(result, Sequence):
                logprobs_seq = result
            elif hasattr(result, "get"):
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
                        raw_val = item.get("logprob") or item.get("total_logprob")
                        if raw_val is None:
                            raise ValueError(
                                "Missing logprob value in sampling client response"
                            )
                        val = float(raw_val)
                    elif hasattr(item, "logprob"):
                        lp_val = getattr(item, "logprob", None)
                        if lp_val is None:
                            raise ValueError(
                                "Missing logprob value in sampling client response"
                            )
                        val = float(lp_val)

                    if current_idx >= offset:
                        total += val
                    current_idx += 1

            results.append({"total_logprob": total})

        if len(targets) == 1 and len(results) == 1:
            return results[0]
        return results


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        if not config.tinker_api_key:
            raise ValueError("No TINKER_API_KEY provided")
        self.config = config
        self.env = env
        self._tinker_module: Any | None = None
        self.service_client = None
        self.training_client = None
        self.tokenizer = None
        self.renderer: RendererType | None = None
        self._wandb_run: Any | None = None


    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _init_wandb(self, output_path: Path) -> None:
        if self._wandb_run is not None:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            return

        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            return

        project = os.getenv("WANDB_PROJECT", "minds_rl")
        run_name = output_path.name
        try:
            self._wandb_run = wandb.init(project=project, name=run_name, reinit=True)
        except Exception:
            self._wandb_run = None

    def _log_external_metrics(self, metrics: Mapping[str, Any]) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(dict(metrics))
        except Exception:
            return

    def _finish_wandb(self) -> None:
        run = self._wandb_run
        self._wandb_run = None
        if run is None:
            return
        try:
            run.finish()
        except Exception:
            pass

    def _write_metrics_entry(
        self,
        metrics_path: Path,
        metrics_history: list[dict[str, Any]],
        entry: Mapping[str, Any] | None,
    ) -> None:
        """Persist a single metrics entry if it is a mapping.

        The integration harness expects every line of ``metrics.jsonl`` to be a
        dictionary. Defensive validation avoids writing ``null`` or other
        non-dictionary payloads that would later crash the parser.
        """

        if not isinstance(entry, Mapping):
            # Skip malformed entries instead of emitting "null" lines
            return

        normalized = dict(entry)
        metrics_history.append(normalized)
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(normalized) + "\n")

    def _write_job_summary(self, metrics_history: Sequence[Mapping[str, Any]], output_path: Path) -> None:
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return

        final_metrics = metrics_history[-1] if metrics_history else {}
        lines = [
            "## Continuous training run",
            "",
            f"Output directory: `{output_path}`",
            "",
            "| Step | Reward | Loss |", 
            "| --- | --- | --- |",
        ]
        for entry in metrics_history:
            step_val = entry.get("step", "-1")
            reward_val = entry.get("reward", 0)
            loss_val = entry.get("loss", 0)
            lines.append(f"| {step_val} | {reward_val} | {loss_val} |")

        if final_metrics:
            avg_reward = final_metrics.get("reward", "n/a")
            lines.extend([
                "",
                f"Final reward: **{avg_reward}**",
            ])

        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except Exception:
            return

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

        resume_checkpoint_id = tinker_cfg.get("resume_checkpoint_id")
        if not resume_checkpoint_id:
            resume_checkpoint_id = tinker_cfg.get("resume_from")

        config = TrainerConfig(
            base_model=base_model,
            rollouts_per_example=int(trainer_cfg.get("rollouts_per_example", 4)),
            max_new_tokens=int(trainer_args.get("max_new_tokens", 256)),
            loss_fn=str(trainer_cfg.get("loss_fn", "ppo")),
            tinker_api_key=api_key,
            training_rank=int(trainer_args.get("training_rank", 32)),
            learning_rate=float(trainer_args.get("learning_rate", 3.162e-6)),
            resume_checkpoint_id=resume_checkpoint_id,
            save_every_n_steps=int(trainer_args.get("save_every_n_steps", 50)),
            update_sampler_every_n_steps=int(trainer_args.get("update_sampler_every_n_steps", 10)),
            renderer_name=model_cfg.get("renderer_name"),
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
        resume_path = self.config.resume_checkpoint_id
        if resume_path:
            creator = getattr(service_client, "create_training_client_from_state_async", None)
            if callable(creator):
                training_client = await creator(resume_path)
            else:  # pragma: no cover - fallback for sync API
                training_client = service_client.create_training_client_from_state(resume_path)
        else:
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
        except Exception as exc:
            raise RuntimeError("Failed to load tokenizer via tinker_cookbook") from exc

        self._require_tinker()
        spec = importlib.util.find_spec("transformers")
        if spec is None:
            raise ModuleNotFoundError(
                "transformers is required for training; install it to load a tokenizer."
            )

        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
        if auto_tokenizer is None:
            raise ImportError("transformers.AutoTokenizer is required for training")

        try:
            self.tokenizer = auto_tokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to load tokenizer from transformers") from exc

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        self._ensure_tokenizer()
        renderer_name = self.config.renderer_name
        if renderer_name is None and get_recommended_renderer_name is not None:
            renderer_name = get_recommended_renderer_name(self.config.base_model)
        if renderer_name is None or tinker_get_renderer is None:
            raise RuntimeError("Tinker Renderer not found")
        self.renderer = tinker_get_renderer(renderer_name, self.tokenizer)

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
        messages: list[dict[str, Any]] = []

        system_prompt = getattr(self.env, "system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        few_shot = getattr(self.env, "few_shot", None)
        if few_shot and isinstance(few_shot, Sequence) and not isinstance(
            few_shot, (str, bytes)
        ):
            messages.extend(list(few_shot))

        messages.append({"role": "user", "content": prompt})
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

    def _start_wandb(self, output_path: Path) -> None:
        if self._wandb_run is not None:
            return

        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            return

        spec = importlib.util.find_spec("wandb")
        if spec is None:
            print("Weights & Biases not installed; skipping external logging.")
            return

        wandb = importlib.import_module("wandb")
        wandb.login(key=api_key, relogin=True)

        project = os.getenv("WANDB_PROJECT", "minds-rl")
        entity = os.getenv("WANDB_ENTITY")
        config = {
            "base_model": self.config.base_model,
            "rollouts_per_example": self.config.rollouts_per_example,
            "loss_fn": self.config.loss_fn,
            "training_rank": self.config.training_rank,
            "learning_rate": self.config.learning_rate,
        }

        self._wandb_run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            dir=str(output_path),
            reinit=True,
        )

    def _log_external_metrics(self, metrics: Mapping[str, Any]) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(dict(metrics))
        except Exception:
            # External logging should never break the training loop
            pass

    def _finish_wandb(self) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        finally:
            self._wandb_run = None

    def _write_job_summary(self, metrics_history: Sequence[Mapping[str, Any]], output_path: Path) -> None:
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if not summary_path:
            return

        summary_file = Path(summary_path)
        if not summary_file.parent.exists():
            summary_file.parent.mkdir(parents=True, exist_ok=True)

        step_metrics = [m for m in metrics_history if "step" in m]
        rewards = [float(m.get("reward", 0.0)) for m in step_metrics]
        losses = [float(m.get("loss", 0.0)) for m in step_metrics]

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        final_reward = rewards[-1] if rewards else 0.0
        final_loss = losses[-1] if losses else 0.0

        lines = [
            "## Training Summary",
            "",
            f"Output directory: `{output_path}`",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Steps | {len(step_metrics)} |",
            f"| Average Reward | {avg_reward:.4f} |",
            f"| Max Reward | {max_reward:.4f} |",
            f"| Final Reward | {final_reward:.4f} |",
            f"| Final Loss | {final_loss:.4f} |",
            "",
        ]

        with summary_file.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            if not str(lines[-1]).endswith("\n"):
                fh.write("\n")

    async def _train(
        self,
        *,
        max_steps: int = 1,
        output_dir: Path | str | None = None,
        checkpoint_path: Path | None = None,
        history_path: Path | None = None,
        samples_path: Path | None = None,
        stage_name: str | None = None,
        service_client: Any | None = None,
        training_client: Any | None = None,
        tokenizer: Any | None = None,
        renderer: Any | None = None,
        sampling_client: Any | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_path = history_path if history_path is not None else output_path / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        log_path = samples_path if samples_path is not None else output_path / "conversation_log.tsv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8")
        if log_path.stat().st_size == 0:
            log_file.write("Timestamp\tStep\tTask\tPrompt\tCompletion\tReward\n")
            log_file.flush()

        rewards: list[float] = []
        metrics_history: list[dict[str, float | int]] = []
        pending_training_task: asyncio.Task[Any] | None = None
        state_path: str | None = None
        provided_sampling_client = sampling_client
        last_sampler_update = -1
        last_saved_step: int | None = None
        last_completed_step: int | None = None

        if service_client is not None:
            self.service_client = service_client
        if training_client is not None:
            self.training_client = training_client
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if renderer is not None:
            self.renderer = renderer

        tinker: Any | None = None
        if self.training_client is None or self.service_client is None:
            tinker, service_client, training_client = await self._build_clients()
        else:
            tinker = self._require_tinker()
            service_client = self.service_client
            training_client = self.training_client

        if self.tokenizer is None:
            self._ensure_tokenizer()
        if self.renderer is None:
            self._ensure_renderer()
        self._init_wandb(output_path)

        try:
            dataset = getattr(self.env, "dataset", None)
            steps = max(max_steps, 1)
            sampling_client = provided_sampling_client

            def _escape_tsv(text: str) -> str:
                return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

            async def _save_checkpoint(name: str) -> str | None:
                save_state = getattr(training_client, "save_state_async", None)
                if not callable(save_state):
                    return None
                try:
                    state_future = await save_state(name)
                    if hasattr(state_future, "result_async"):
                        state_result = await state_future.result_async()
                        state_location = getattr(state_result, "path", None)
                    else:
                        state_location = getattr(state_future, "path", None)
                except Exception:
                    return None
                if state_location and checkpoint_path is not None:
                    checkpoint_path.write_text(str(state_location), encoding="utf-8")
                return state_location

            try:
                for step in range(steps):
                    if deadline is not None:
                        buffer_seconds = float(os.getenv("TRAINER_DEADLINE_BUFFER", "300"))
                        if time.time() + buffer_seconds >= deadline:
                            break
                    should_update_sampler = (
                        provided_sampling_client is None
                        or self.config.update_sampler_every_n_steps <= 0
                        or step % max(self.config.update_sampler_every_n_steps, 1) == 0
                    )
                    if should_update_sampler:
                        if pending_training_task is not None:
                            await pending_training_task
                            pending_training_task = None
                        sampling_client = await self._prepare_sampling_client(
                            training_client, service_client, step
                        )
                        last_sampler_update = step
                        provided_sampling_client = sampling_client

                    envs: list[Any] = []
                    if hasattr(self.env, "build") and callable(getattr(self.env, "build")):
                        build_kwargs = {}
                        try:
                            signature = inspect.signature(self.env.build)
                            params = [p for p in signature.parameters.values() if p.name != "self"]
                            param_names = {p.name for p in params}
                            if "service_client" in param_names:
                                build_kwargs["service_client"] = service_client
                            if "training_client" in param_names:
                                build_kwargs["training_client"] = training_client
                            if "base_model" in param_names:
                                build_kwargs["base_model"] = self.config.base_model
                        except Exception:
                            build_kwargs = {}
                        try:
                            envs = list(self.env.build(sampling_client, **build_kwargs))
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
                        if TinkerTokenCompleter is None:
                            raise ModuleNotFoundError(
                                "tinker_cookbook.completers.TinkerTokenCompleter is required for training"
                            )
                        completer = TinkerTokenCompleter(
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
                                "tokenizer": self.tokenizer,
                            }
                            answer_value = ""
                            raw_answer = sample.get("answer") if isinstance(sample, Mapping) else None
                            if isinstance(raw_answer, str):
                                answer_value = raw_answer
                            reward_total = 0.0
                            rubric_weights = getattr(env.rubric, "weights", [1.0] * len(env.rubric.funcs))

                            for func, weight in zip(env.rubric.funcs, rubric_weights):
                                result = func(
                                    prompt_msgs,
                                    completion_msg,
                                    answer_value,
                                    state,
                                    info,
                                )
                                result = await self._maybe_await(result)
                                reward_total += float(result) * float(weight)
                            completion_rewards.append(float(reward_total))
                    else:
                        for completion_text, data in zip(completion_texts, completion_data):
                            step_callable = getattr(env, "step", None)
                            if not callable(step_callable):
                                continue
                            step_result = step_callable(list(data.get("tokens", [])))
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
                        if len(combined_tokens) < 2:
                            continue

                        # Standard causal LM shift: input is tokens[:-1], targets are tokens[1:]
                        input_tokens = np.array(combined_tokens[:-1], dtype=np.int64)
                        target_tokens = np.array(combined_tokens[1:], dtype=np.int64)

                        total_len = len(target_tokens)
                        prompt_len = len(prompt_tokens)

                        # Initialize arrays for shifted sequence
                        full_advantages = np.zeros(total_len, dtype=np.float32)
                        full_logprobs = np.zeros(total_len, dtype=np.float32)

                        # Fill the completion region (masking prompt region with 0s)
                        comp_len = len(completion_tokens)
                        target_start = max(prompt_len - 1, 0)
                        target_end = min(target_start + comp_len, total_len)
                        if target_end > target_start:
                            full_advantages[target_start:target_end] = float(advantage)
                            full_logprobs[target_start:target_end] = logprob_array[: target_end - target_start]

                        if self.config.loss_fn == "cross_entropy":
                            # For SL, use weights tensor for masking
                            full_weights = np.zeros(total_len, dtype=np.float32)
                            if target_end > target_start:
                                full_weights[target_start:target_end] = 1.0
                            loss_fn_inputs = {
                                "target_tokens": tinker.TensorData.from_numpy(target_tokens),
                                "weights": tinker.TensorData.from_numpy(full_weights),
                            }
                        else:
                            # For RL, masking is implicit via 0.0 advantage
                            loss_fn_inputs = {
                                "target_tokens": tinker.TensorData.from_numpy(target_tokens),
                                "logprobs": tinker.TensorData.from_numpy(full_logprobs),
                                "advantages": tinker.TensorData.from_numpy(full_advantages),
                            }

                        datums.append(
                            tinker.Datum(
                                model_input=tinker.ModelInput.from_ints(input_tokens.tolist()),
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
                        async def _ensure_result(value: Any) -> Any:
                            first = await value if inspect.isawaitable(value) else value
                            if inspect.isawaitable(first):
                                return await first
                            return first

                        async def _background_step(fw, op):
                            await _ensure_result(fw)
                            await _ensure_result(op)

                        training_task = asyncio.create_task(
                            _background_step(
                                forward,
                                optim,
                            )
                        )

                        if pending_training_task is not None:
                            await pending_training_task
                        pending_training_task = training_task

                    reward_mean = baseline
                    rewards.append(reward_mean)
                    timestamp = datetime.utcnow().isoformat()
                    task_label = stage_name or ""
                    step_metrics = {
                        "reward": reward_mean,
                        "loss": 0.0,
                        "step": step,
                        "stage": task_label,
                        "sampler_step": last_sampler_update,
                    }
                    self._write_metrics_entry(metrics_path, metrics_history, step_metrics)
                    self._log_external_metrics(step_metrics)

                    if prompt is not None and completion_texts:
                        log_line = "\t".join(
                            [
                                timestamp,
                                str(step),
                                task_label,
                                _escape_tsv(str(prompt)),
                                _escape_tsv(" ||| ".join(completion_texts)),
                                str(reward_mean),
                            ]
                        )
                        log_file.write(log_line + "\n")
                        log_file.flush()

                    if (
                        self.config.save_every_n_steps > 0
                        and step > 0
                        and step % self.config.save_every_n_steps == 0
                    ):
                        if pending_training_task is not None:
                            await pending_training_task
                            pending_training_task = None
                        latest_state = await _save_checkpoint(f"step_{step:06d}")
                        if latest_state:
                            state_path = latest_state
                            last_saved_step = step
                    last_completed_step = step
            finally:
                log_file.close()

            if pending_training_task is not None:
                await pending_training_task

            if (
                last_completed_step is not None
                and (last_saved_step is None or last_saved_step < last_completed_step)
            ):
                latest_state = await _save_checkpoint(f"step_{last_completed_step:06d}")
                if latest_state:
                    state_path = latest_state
                    last_saved_step = last_completed_step

            completed_steps = len(rewards)
            metrics = {
                "reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "loss": 0.0,
                "step": completed_steps,
                "stage": stage_name or "",
            }
            self._write_metrics_entry(metrics_path, metrics_history, metrics)

            if state_path is None:
                state_path = await _save_checkpoint("final_step")
                if last_completed_step is not None:
                    last_saved_step = last_completed_step
            result_metrics = dict(metrics)
            if state_path:
                result_metrics["checkpoint_id"] = state_path
                self.config.resume_checkpoint_id = state_path
            result_metrics["sampling_client"] = sampling_client or provided_sampling_client

            self._log_external_metrics({"step": steps, **metrics})
            self._write_job_summary(metrics_history, output_path)
            return result_metrics
        finally:
            self._finish_wandb()

    def train(
        self,
        *,
        max_steps: int = 1,
        output_dir: Path | str | None = None,
        checkpoint_path: Path | None = None,
        history_path: Path | None = None,
        samples_path: Path | None = None,
        stage_name: str | None = None,
        service_client: Any | None = None,
        training_client: Any | None = None,
        tokenizer: Any | None = None,
        renderer: Any | None = None,
        sampling_client: Any | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self._train(
                max_steps=max_steps,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                samples_path=samples_path,
                stage_name=stage_name,
                service_client=service_client,
                training_client=training_client,
                tokenizer=tokenizer,
                renderer=renderer,
                sampling_client=sampling_client,
                deadline=deadline,
            )
        )
