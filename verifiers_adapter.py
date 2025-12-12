from __future__ import annotations

from typing import Any, cast

from verifiers.utils.async_utils import maybe_semaphore

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
    VerifiersEnvGroupBuilder,
    convert_states_to_trajectory_group,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


def make_custom_do_group_rollout(
    cfg: train.Config,
    *,
    group_size: int,
    max_concurrent_generation: int = -1,
    max_concurrent_scoring: int = -1,
) -> Any:
    """Clone of cookbook verifiers rollout adapter for runtime patching.

    The original implementation lives inside the cookbook CLI entrypoint as a local
    closure, so we vendor the logic here to make it importable by `continuous_runner`.
    """

    shared_client: TinkerAsyncOpenAIClient | None = None
    shared_renderer: renderers.Renderer | None = None
    local_tokenizer: Tokenizer | None = None

    max_tokens = getattr(cfg, "max_tokens", 128)
    temperature = getattr(cfg, "temperature", 1.0)

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        nonlocal shared_client, shared_renderer, local_tokenizer

        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(cfg.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, local_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        if shared_client is None:
            shared_client = TinkerAsyncOpenAIClient(
                sampling_client, shared_renderer, local_tokenizer
            )
        else:
            shared_client.set_sampling_client(sampling_client)

        vf_builder = cast(VerifiersEnvGroupBuilder, builder)
        rollout_inputs = vf_builder.get_rollout_inputs(group_size)

        gen_sem = await maybe_semaphore(max_concurrent_generation)
        score_sem = await maybe_semaphore(max_concurrent_scoring)

        states = await vf_builder.vf_env.run_group(
            group_inputs=rollout_inputs,
            client=shared_client,
            model="tinker",
            gen_sampling_args={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            gen_sem=gen_sem,
            score_sem=score_sem,
        )

        return convert_states_to_trajectory_group(states)

    return custom_do_group_rollout

