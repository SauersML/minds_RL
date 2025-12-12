from __future__ import annotations

import asyncio
import logging
import time
import os
import json
import numpy as np
import chz
import verifiers as vf
from verifiers.utils.message_utils import messages_to_printable

# Mocking OpenAI interfaces for Verifiers compatibility
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice

# Import Tinker dependencies
try:
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    # Attempt to import the original TinkerAsyncOpenAIClient if possible
    # But since we shadowed the package, we might not be able to import from the 'real' location easily.
    # We will reimplement a minimal wrapper for Tinker if we use it.
except ImportError:
    tinker = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Re-implementation of Tinker Wrapper (since we shadow the file) ---

if tinker:
    from openai import AsyncOpenAI
    from openai.resources.chat import AsyncChat as OpenAIAsyncChat
    from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
    from typing import Any

    class TinkerAsyncOpenAIClient(AsyncOpenAI):
        def __init__(self, sampling_client: tinker.SamplingClient, renderer: renderers.Renderer, tokenizer):
            super().__init__(api_key="tinker", base_url="http://localhost")
            self.sampling_client = sampling_client
            self.renderer = renderer
            self.tokenizer = tokenizer

        @property
        def chat(self) -> OpenAIAsyncChat:
            return TinkerAsyncChat(self)

    class TinkerAsyncChat(OpenAIAsyncChat):
        def __init__(self, client: TinkerAsyncOpenAIClient) -> None:
            super().__init__(client)
            self._client = client

        @property
        def completions(self) -> TinkerChatCompletions:
            return TinkerChatCompletions(self._client)

    class TinkerChatCompletions(OpenAIAsyncChatCompletions):
        def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
            self._parent = parent

        async def create(self, *args: Any, **kwargs: Any):
            model = kwargs.get("model", "tinker")
            messages = kwargs.get("messages", [])
            sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}

            stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
            max_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")

            model_input = self._parent.renderer.build_generation_prompt(messages)

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
            tokens = seq.tokens

            assistant_message, parse_success = self._parent.renderer.parse_response(tokens)
            finish_reason = "stop" if parse_success else "length"

            message_obj = ChatCompletionMessage(role="assistant", content=assistant_message["content"])
            choice = Choice(finish_reason=finish_reason, index=0, message=message_obj, logprobs=None)

            return ChatCompletion(
                id=f"tinker-{int(time.time())}",
                choices=[choice],
                created=int(time.time()),
                model=model,
                object="chat.completion",
            )

# --- Mock Client ---

class MockClient:
    """
    A mock client that returns dummy responses for testing without API keys or GPUs.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        logger.info(f"Initializing MockClient for {model_name}...")

        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        # Return a dummy response
        generated_text = "42" # Simple answer for math problems

        # Simulate small delay
        await asyncio.sleep(0.1)

        message_obj = ChatCompletionMessage(role="assistant", content=generated_text)
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=message_obj,
            logprobs=None
        )

        completion = ChatCompletion(
            id=f"mock-{int(time.time())}",
            choices=[choice],
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
        )
        return completion

def log_results(
    results: vf.GenerateOutputs,
    vf_env_id: str,
    model_name: str,
    num_examples: int,
    rollouts_per_example: int,
    time_s: float,
):
    print(f"Evaluation completed in {time_s:.2f} seconds")
    print("--- Evaluation ---")
    print(f"Environment: {vf_env_id}")
    print(f"Model: {model_name}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    # Handle results whether it's a Pydantic object or a dict
    if hasattr(results, "prompt"):
        r_prompt = results.prompt
        r_completion = results.completion
        r_reward = results.reward
        r_metrics = results.metrics
    else:
        r_prompt = results.get("prompt", [])
        r_completion = results.get("completion", [])
        r_reward = results.get("reward", [])
        r_metrics = results.get("metrics", {})

    print("--- Example ---")
    try:
        printable_prompts = [messages_to_printable(p) for p in r_prompt]
        printable_completions = [messages_to_printable(c) for c in r_completion]
        vf.print_prompt_completions_sample(
            printable_prompts, printable_completions, r_reward, step=0
        )
    except Exception as e:
        print(f"Error printing example: {e}")

    print("--- All ---")
    print("Rewards:")
    if r_reward:
        print(
            f"reward: avg - {sum(r_reward) / len(r_reward):.3f}, std - {np.std(r_reward):.3f}"
        )
        r = rollouts_per_example
        n = len(r_reward) // r
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(r_reward[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)
    else:
        print("No rewards recorded.")

    for k in r_metrics:
        v = r_metrics[k]
        if v:
            print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
            r = rollouts_per_example
            n = len(v) // r
            for i in range(r):
                trials = [round(v[(i * n) + j], 3) for j in range(n)]
                out = f"r{i + 1}: {trials}"
                print(out)

async def evaluate(
    vf_env_id: str,
    vf_env_args: dict,
    model_name: str | None,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    model_path: str | None = None,
):
    # Load environment
    if vf_env_id.startswith("./") or vf_env_id.startswith("/") or os.path.exists(vf_env_id):
        import importlib.util
        import sys
        module_name = os.path.basename(vf_env_id).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, vf_env_id if vf_env_id.endswith(".py") else vf_env_id + ".py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            env = module.load_environment(**vf_env_args)
        else:
             env = vf.load_environment(vf_env_id, **vf_env_args)
    else:
        env = vf.load_environment(vf_env_id, **vf_env_args)

    client = None

    # 1. Try Tinker
    if tinker and model_name:
        try:
            logger.info("Attempting to connect to Tinker ServiceClient...")
            service = tinker.ServiceClient()

            # If we get here, ServiceClient init succeeded (or didn't fail immediately)
            # Create sampling client
            if model_path:
                sampling = service.create_sampling_client(model_path=model_path, base_model=model_name)
            else:
                sampling = service.create_sampling_client(base_model=model_name)

            # Setup renderer
            tokenizer = get_tokenizer(model_name)
            renderer_name = model_info.get_recommended_renderer_name(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer)

            client = TinkerAsyncOpenAIClient(sampling, renderer, tokenizer)
            logger.info("Successfully connected to Tinker.")

        except Exception as e:
            logger.warning(f"Failed to initialize Tinker client: {e}")
            logger.warning("Falling back to MockClient.")
            client = None

    # 2. Fallback to Mock
    if client is None:
        if model_name is None:
             raise ValueError("model_name must be provided")

        logger.info(f"Initializing MockClient for {model_name}...")
        client = MockClient(model_name)

    start_time = time.time()
    results = await env.evaluate(
        client=client,
        model=model_name,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=1,
        sampling_args={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    end_time = time.time()
    log_results(
        results,
        vf_env_id,
        model_name,
        num_examples,
        rollouts_per_example,
        end_time - start_time,
    )
    return results


@chz.chz
class CLIConfig:
    model_name: str | None = None
    model_path: str | None = None
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None
    num_examples: int = 5
    rollouts_per_example: int = 3
    max_concurrent: int = 32
    max_tokens: int = 1024
    temperature: float = 1.0


async def cli_main(cfg: CLIConfig):
    env_args = json.loads(cfg.vf_env_args) if cfg.vf_env_args else {}
    return await evaluate(
        vf_env_id=cfg.vf_env_id,
        vf_env_args=env_args,
        model_name=cfg.model_name,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
        max_concurrent=cfg.max_concurrent,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        model_path=cfg.model_path,
    )


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
