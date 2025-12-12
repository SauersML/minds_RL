"""Adapter for running verifiers environments within the Tinker Cookbook training loop.

This module provides a mechanism to patch the standard ``do_group_rollout``
function in ``tinker_cookbook.rl.train`` with a custom implementation that
supports ``verifiers``-based environments. This is necessary because the default
cookbook implementation assumes a simpler environment interface, while
``verifiers`` environments often require complex rollout logic, including
multi-turn interactions and scoring semaphores.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, cast

import tinker
import verifiers as vf
from verifiers.utils.async_utils import maybe_semaphore

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import (
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersEnvGroupBuilder
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


def make_custom_do_group_rollout(
    cfg: train.Config,
    *,
    group_size: int,
    max_concurrent_generation: int = -1,
    max_concurrent_scoring: int = -1,
) -> Any:
    """Create a custom rollout function for verifiers environments.

    This function returns an async closure that matches the signature of
    ``do_group_rollout(builder, policy)`` but uses the ``verifiers`` SDK logic
    to execute the rollout.

    Args:
        cfg: The training configuration.
        group_size: The number of rollouts to perform in parallel for each group.
        max_concurrent_generation: Limit on concurrent generation tasks (semaphore).
        max_concurrent_scoring: Limit on concurrent scoring tasks (semaphore).

    Returns:
        An async function ``custom_do_group_rollout(builder, policy) -> TrajectoryGroup``.
    """

    shared_renderer: renderers.Renderer | None = None
    local_tokenizer: Tokenizer | None = None

    async def custom_do_group_rollout(
        builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> TrajectoryGroup:
        nonlocal shared_renderer, local_tokenizer

        gen_limit = (
            None if max_concurrent_generation is None or max_concurrent_generation < 0 else max_concurrent_generation
        )
        score_limit = (
            None if max_concurrent_scoring is None or max_concurrent_scoring < 0 else max_concurrent_scoring
        )

        if local_tokenizer is None:
            local_tokenizer = get_tokenizer(cfg.model_name)
        if shared_renderer is None:
            renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
            shared_renderer = renderers.get_renderer(renderer_name, local_tokenizer)

        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        vf_builder = cast(VerifiersEnvGroupBuilder, builder)

        async def run_one_rollout() -> tuple[Trajectory, float, dict[str, float | int]]:
            recorded: List[
                tuple[list[renderers.Message], tinker.ModelInput, list[int], list[float]]
            ] = []

            def hook(messages, model_input, tokens, logprobs):
                recorded.append((list(messages), model_input, list(tokens), list(logprobs)))

            assert shared_renderer is not None and local_tokenizer is not None
            local_client = TinkerAsyncOpenAIClient(
                sampling_client, shared_renderer, local_tokenizer
            )
            local_client.set_generation_hook(hook)

            rollout_input: vf.RolloutInput = {
                "prompt": vf_builder.prompt,
                "answer": vf_builder.answer,
                "task": vf_builder.task,
                "info": vf_builder.info,
                "example_id": 0,
            }

            gen_sem = await maybe_semaphore(gen_limit)
            if gen_sem:
                async with gen_sem:
                    state = await vf_builder.vf_env.rollout(
                        input=rollout_input,
                        client=local_client,
                        model="tinker",
                        sampling_args={},
                    )
            else:
                state = await vf_builder.vf_env.rollout(
                    input=rollout_input,
                    client=local_client,
                    model="tinker",
                    sampling_args={},
                )

            score_sem = await maybe_semaphore(score_limit)
            await vf_builder.vf_env.rubric.score_rollout(
                state=state,
                score_sem=score_sem,
            )
            rs: vf.RolloutScore = {
                "reward": state["reward"],
                "metrics": state.get("metrics", {}),
            }

            transitions: List[Transition] = []
            for _msgs, model_input, tokens, logprobs in recorded:
                transitions.append(
                    Transition(
                        ob=model_input,
                        ac=TokensWithLogprobs(tokens=tokens, maybe_logprobs=logprobs),
                        reward=0.0,
                        episode_done=False,
                        metrics={},
                    )
                )
            if transitions:
                transitions[-1] = Transition(
                    ob=transitions[-1].ob,
                    ac=transitions[-1].ac,
                    reward=0.0,
                    episode_done=True,
                    metrics=transitions[-1].metrics,
                )

            traj = Trajectory(transitions=transitions, final_ob=tinker.ModelInput.empty())
            return traj, float(rs["reward"]), dict(rs["metrics"])

        results = await asyncio.gather(*[run_one_rollout() for _ in range(group_size)])
        trajectories_G = [t for (t, _r, _m) in results]
        final_rewards_G = [r for (_t, r, _m) in results]
        metrics_G = [m for (_t, _r, m) in results]
        return TrajectoryGroup(trajectories_G, final_rewards_G, metrics_G)

    return custom_do_group_rollout
