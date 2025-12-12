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
import copy
import time
from typing import Any, Callable, Dict, List, Optional, overload, Literal, cast

import tinker
import verifiers as vf
from verifiers.utils.async_utils import maybe_semaphore
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
from openai.resources.completions import AsyncCompletions as OpenAIAsyncCompletions
from openai._streaming import AsyncStream

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import (
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

# --- Inline TinkerAsyncOpenAIClient and dependencies ---

GenerationHook = Callable[
    [List[renderers.Message], tinker.ModelInput, List[int], List[float]], None
]

def convert_oai_messages_to_renderer_messages(
    messages: List[Dict[str, Any]],
) -> List[renderers.Message]:
    out: List[renderers.Message] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        # extract text from list of content parts if necessary
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append(renderers.Message(role=role, content=content))
    return out


class TinkerAsyncOpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible async client that routes calls to a Tinker SamplingClient.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__(api_key="tinker", base_url="http://localhost")
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.hook: Optional[GenerationHook] = None

    def set_generation_hook(self, hook: Optional[GenerationHook]) -> None:
        self.hook = hook

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client

    @property
    def chat(self) -> OpenAIAsyncChat:
        return TinkerAsyncChat(self)

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        return TinkerCompletions(self)


class TinkerAsyncChat(OpenAIAsyncChat):
    def __init__(self, client: TinkerAsyncOpenAIClient) -> None:
        super().__init__(client)
        self._client = client

    @property
    def completions(self) -> TinkerChatCompletions:
        return TinkerChatCompletions(self._client)


class TinkerCompletions(OpenAIAsyncCompletions):
    def __init__(self, client: TinkerAsyncOpenAIClient) -> None:
        super().__init__(client)
        self._client = client

    # Implement completions.create if needed, but verifiers mostly uses chat.
    # For now, minimal implementation to satisfy type checker or basic usage
    @overload
    async def create(self, *args: Any, stream: Literal[True], **kwargs: Any) -> AsyncStream[Any]: ...
    @overload
    async def create(self, *args: Any, stream: Literal[False] = False, **kwargs: Any) -> Any: ...
    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> Any: ...
    async def create(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Completions API not fully implemented in Tinker adapter yet.")


class TinkerChatCompletions(OpenAIAsyncChatCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Any]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> ChatCompletion: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream[Any]:
        model = kwargs.get("model", "tinker")
        messages = kwargs.get("messages", [])
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by TinkerAsyncOpenAIClient")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}

        # prepare prompt
        conv_messages = convert_oai_messages_to_renderer_messages(messages)
        stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
        max_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")

        model_input = self._parent.renderer.build_generation_prompt(conv_messages)
        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(max_tokens or 128),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
                stop=stop,
            ),
        )
        seq = sample.sequences[0]
        tokens: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(tokens)

        if self._parent.hook is not None:
            self._parent.hook(conv_messages, model_input, tokens, logprobs)

        # build ChatCompletion via pydantic validation using renderer parsing
        assistant_message, parse_success = self._parent.renderer.parse_response(tokens)
        content_text = assistant_message["content"]
        finish_reason = "stop" if parse_success else "length"

        # Construct the response object manually to avoid Pydantic validation strictness if possible,
        # or use the model constructors if available.
        # OpenAI types are Pydantic models.

        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        # Create a ChatCompletion object
        # Note: Depending on the openai version, the constructor might vary.
        # We try to use the standard Pydantic model construction.

        message_obj = ChatCompletionMessage(role="assistant", content=content_text)
        choice = Choice(
            finish_reason=finish_reason,
            index=0,
            message=message_obj,
            logprobs=None # We don't populate logprobs structure fully yet
        )

        completion = ChatCompletion(
            id="tinker-chatcmpl-" + str(int(time.time())),
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion",
        )
        return completion

# --- End Inline ---


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

        # If it's wrapped, this cast might be technically incorrect at runtime if we don't unwrap
        vf_builder = builder
        while hasattr(vf_builder, "_base"):
            vf_builder = vf_builder._base

        # Now we can safely treat it as VerifiersEnvGroupBuilder (duck typed)???

        # --- FIX: Instantiate separate environments for each rollout in the group ---
        # VerifiersEnvGroupBuilder.make_envs() is known to return an empty list in some versions.
        # We explicitly deep-copy the prototype environment for every rollout task to ensure full state isolation.
        envs = [copy.deepcopy(vf_builder.vf_env) for _ in range(group_size)]

        async def run_one_rollout(env) -> tuple[Trajectory, float, dict[str, float | int]]:
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

            # Inject runtime objects into info so environments can perform model-based scoring
            current_info = dict(vf_builder.info) if vf_builder.info else {}
            current_info["tinker_client"] = sampling_client
            current_info["tokenizer"] = local_tokenizer

            rollout_input: vf.RolloutInput = {
                "prompt": vf_builder.prompt,
                "answer": vf_builder.answer,
                "task": vf_builder.task,
                "info": current_info,
                "example_id": 0,
            }

            gen_sem = await maybe_semaphore(gen_limit)
            if gen_sem:
                async with gen_sem:
                    # Handle the case where rollout might return (state, auxiliary_info) or just state
                    # Recent changes in Verifiers SDK might return a tuple
                    result = await env.rollout(
                        input=rollout_input,
                        client=local_client,
                        model="tinker",
                        sampling_args={},
                    )
            else:
                result = await env.rollout(
                    input=rollout_input,
                    client=local_client,
                    model="tinker",
                    sampling_args={},
                )

            # Unpack result if it is a tuple (state, info), otherwise assume it is state
            if isinstance(result, tuple):
                state = result[0]
            else:
                state = result

            score_sem = await maybe_semaphore(score_limit)
            await env.rubric.score_rollout(
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

        results = await asyncio.gather(*[run_one_rollout(env) for env in envs])
        trajectories_G = [t for (t, _r, _m) in results]
        final_rewards_G = [r for (_t, r, _m) in results]
        metrics_G = [m for (_t, _r, m) in results]
        return TrajectoryGroup(trajectories_G, final_rewards_G, metrics_G)

    return custom_do_group_rollout
