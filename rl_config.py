"""Configuration parsing and construction for RL runs.

This module converts simple TOML configuration files into the strict ``Config``
objects required by the ``tinker_cookbook`` training loop. It handles the
instantiation of dataset builders, tokenizers, and renderers.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import chz
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersRLDatasetBuilder
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.train import AsyncConfig, Config, StreamMinibatchConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from environments.rl_datasets import (
    CompositeRLDatasetBuilder,
    GradientIntuitionRLDatasetBuilder,
    GradientProphetRLDatasetBuilder,
)


def _ensure_api_key(config: Mapping[str, Any]) -> None:
    """Validate that the Tinker API key is present in the environment variables."""
    api_key_env = config.get("tinker", {}).get("api_key_env", "TINKER_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing Tinker API key in environment variable {api_key_env}")


def _group_size(trainer_cfg: Mapping[str, Any]) -> int:
    """Extract the number of rollouts per example (Group Size)."""
    try:
        return max(1, int(trainer_cfg.get("rollouts_per_example", 4)))
    except Exception:
        return 4


def _batch_size(trainer_cfg: Mapping[str, Any]) -> int:
    """Extract the batch size (Groups per Batch)."""
    try:
        return max(1, int(trainer_cfg.get("groups_per_batch", trainer_cfg.get("batch_size", 4))))
    except Exception:
        return 4


def _total_batches(trainer_cfg: Mapping[str, Any]) -> int:
    try:
        return max(1, int(trainer_cfg.get("total_batches", trainer_cfg.get("num_batches", 100000))))
    except Exception:
        return 100000


def _max_tokens(trainer_cfg: Mapping[str, Any]) -> int:
    """Extract the maximum new tokens for generation."""
    args = trainer_cfg.get("args", {}) if isinstance(trainer_cfg, Mapping) else {}
    try:
        return int(args.get("max_new_tokens", 2048))
    except Exception:
        return 2048


def _loss_fn(trainer_cfg: Mapping[str, Any]) -> str:
    """Extract the loss function name."""
    loss = trainer_cfg.get("loss_fn", "importance_sampling")
    return str(loss)


@chz.chz
class AugmentedVerifiersRLDatasetBuilder(VerifiersRLDatasetBuilder):
    """Extension of VerifiersRLDatasetBuilder that explicitly sets group_size."""
    group_size: int = 1


def _build_dataset_builder(
    env_cfg: Mapping[str, Any],
    model_name: str,
    *,
    batch_size: int,
    group_size: int,
    renderer: Any,
    base_url: str | None,
) -> VerifiersRLDatasetBuilder | GradientProphetRLDatasetBuilder | GradientIntuitionRLDatasetBuilder:
    """Construct the appropriate DatasetBuilder based on the environment ID.

    Args:
        env_cfg: The 'env' section of the TOML config.
        model_name: The name of the base model.
        batch_size: The batch size (groups per batch).
        group_size: The group size (rollouts per example).
        renderer: The renderer instance.
        base_url: Optional base URL for Tinker API.

    Returns:
        A DatasetBuilder instance.
    """
    env_id = str(env_cfg.get("id", "")).strip()
    env_args = env_cfg.get("args", {}) if isinstance(env_cfg, Mapping) else {}

    if env_id.endswith("gradient_prophet") or "gradient_prophet" in env_id:
        return GradientProphetRLDatasetBuilder(
            model_name=model_name,
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            base_url=base_url,
            seed=env_args.get("seed"),
        )

    if env_id.endswith("gradient_intuition") or "gradient_intuition" in env_id:
        return GradientIntuitionRLDatasetBuilder(
            model_name=model_name,
            inner_env_id=str(env_args.get("inner_env_id", env_args.get("inner_env", "./environments/ghost_trace"))),
            inner_env_args=env_args.get("inner_env_args", {}) or env_args,
            alpha=float(env_args.get("alpha", 0.3)),
            seed=env_args.get("seed"),
            shadow_rank=int(env_args.get("shadow_rank", 8)),
            shadow_learning_rate=float(env_args.get("shadow_learning_rate", 1e-4)),
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            base_url=base_url,
        )

    dataset_n = env_args.get("num_examples", -1)
    dataset_seed = env_args.get("seed")
    
    return AugmentedVerifiersRLDatasetBuilder(
        vf_env_id=env_id,
        vf_env_args=env_args,
        groups_per_batch=batch_size,
        dataset_n=dataset_n,
        dataset_seed=dataset_seed,
        group_size=group_size,
    )


def _build_composite_dataset_builder(
    envs_cfg: Sequence[Mapping[str, Any]],
    model_name: str,
    *,
    batch_size: int,
    group_size: int,
    renderer: Any,
    base_url: str | None,
    total_batches: int,
    seed: int | None,
) -> CompositeRLDatasetBuilder:
    components: list[tuple[str, Any, float]] = []
    try:
        seed_value = None if seed is None else int(seed)
    except Exception:
        seed_value = None
    for idx, env_cfg in enumerate(envs_cfg):
        env_id = str(env_cfg.get("id", f"task_{idx}"))
        task_name = str(env_cfg.get("name", env_id)).strip() or env_id
        weight = float(env_cfg.get("weight", env_cfg.get("ratio", 1.0)))
        builder = _build_dataset_builder(
            env_cfg,
            model_name,
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            base_url=base_url,
        )
        components.append((task_name, builder, weight))

    return CompositeRLDatasetBuilder(
        components=components,
        groups_per_batch=batch_size,
        total_batches=total_batches,
        seed=seed_value,
    )


@dataclass
class RunnerConfig:
    """Helper class to build the full training Config from inputs."""
    config_path: Path
    log_root: Path
    base_url: str | None = None
    async_off_policy_steps: int = 2
    stream_minibatches: int = 4
    wandb_project: str | None = None
    initial_checkpoint: str | None = None

    def build(self) -> Config:
        """Parse the TOML config and return a tinker_cookbook Config object."""
        raw = tomllib.loads(Path(self.config_path).read_text())
        _ensure_api_key(raw)

        env_cfg = raw.get("env", {})
        env_list_cfg = raw.get("envs", []) or []
        trainer_cfg = raw.get("trainer", {})
        model_cfg = raw.get("model", {})
        base_model = str(model_cfg.get("base_model", "")).strip()
        if not base_model:
            raise ValueError("model.base_model must be set in the config")

        group_size = _group_size(trainer_cfg)
        batch_size = _batch_size(trainer_cfg)

        tokenizer = get_tokenizer(base_model)
        renderer_name = str(model_cfg.get("renderer_name", model_cfg.get("renderer", "role_colon")))
        renderer = get_renderer(renderer_name, tokenizer)

        total_batches = _total_batches(trainer_cfg)
        if env_list_cfg:
            dataset_builder = _build_composite_dataset_builder(
                env_list_cfg,
                base_model,
                batch_size=batch_size,
                group_size=group_size,
                renderer=renderer,
                base_url=self.base_url,
                total_batches=total_batches,
                seed=trainer_cfg.get("seed"),
            )
        else:
            dataset_builder = _build_dataset_builder(
                env_cfg,
                base_model,
                batch_size=batch_size,
                group_size=group_size,
                renderer=renderer,
                base_url=self.base_url,
            )

        log_path = self.log_root / Path(self.config_path).stem
        log_path.mkdir(parents=True, exist_ok=True)

        async_steps = int(trainer_cfg.get("async_off_policy_steps", self.async_off_policy_steps))
        stream_minibatches = int(trainer_cfg.get("stream_minibatches", self.stream_minibatches))

        async_config = AsyncConfig(max_steps_off_policy=async_steps, groups_per_batch=batch_size)
        stream_cfg = StreamMinibatchConfig(groups_per_batch=batch_size, num_minibatches=stream_minibatches)

        return Config(
            learning_rate=float(trainer_cfg.get("learning_rate", 3.162e-6)),
            dataset_builder=dataset_builder,
            model_name=base_model,
            max_tokens=_max_tokens(trainer_cfg),
            loss_fn=_loss_fn(trainer_cfg),
            lora_rank=int(trainer_cfg.get("training_rank", 32)),
            log_path=str(log_path),
            load_checkpoint_path=self.initial_checkpoint,
            async_config=async_config,
            stream_minibatch_config=stream_cfg,
            save_every=int(trainer_cfg.get("save_every", 20)),
            eval_every=int(trainer_cfg.get("eval_every", 0)),
            base_url=self.base_url,
            wandb_project=self.wandb_project,
        )
