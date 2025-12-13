from __future__ import annotations

import asyncio
import logging
import sys
import importlib
from pathlib import Path
from typing import Any

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tinker_cookbook import renderers, model_info  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_all_envs")

# --- Configurations for all Environments ---
ENV_CONFIGS = [
    {
        "name": "Ghost Trace",
        "module": "environments.ghost_trace.ghost_trace",
        "kwargs": {"num_examples": 5},
        "type": "verifiers_env"
    },
    {
        "name": "Self Prediction",
        "module": "environments.self_prediction.self_prediction",
        "kwargs": {"num_examples": 5},
        "type": "verifiers_env"
    },
    {
        "name": "Gradient Prophet",
        "module": "environments.gradient_prophet.gradient_prophet",
        "kwargs": {"seed": 42},
        "type": "prophet_builder"
    },
    {
        "name": "Gradient Intuition",
        "module": "environments.gradient_intuition.gradient_intuition",
        "kwargs": {
            "inner_env_id": "./environments/ghost_trace",
            "alpha": 0.35,
            "shadow_rank": 8,
            "shadow_learning_rate": 1e-4
        },
        "type": "intuition_builder"
    }
]

class VerifiersEnvWrapper:
    """Adapter to make raw Verifiers environments compatible with Tinker test harness."""
    def __init__(self, env):
        self.env = env

    async def initial_observation(self):
        # Extract prompt from the first dataset example
        sample = self.env.dataset[0]
        if "prompt" in sample:
            return sample["prompt"]
        if "question" in sample:
            return sample["question"]
        return "Dummy Prompt"

    async def step(self, action):
        # Return dummy step result to satisfy test loop
        return types.StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=[],
            metrics={}
        )

async def run_env_episode(name: str, env: Any, sampling_client: tinker.SamplingClient, tokenizer: Any, renderer: renderers.Renderer):
    # Wrap raw verifiers environments if they don't support the Tinker interface
    if not hasattr(env, "initial_observation"):
        env = VerifiersEnvWrapper(env)

    prefix = f"[{name}]"
    logger.info(f"{prefix} Environment ready: {env}")
    logger.info(f"{prefix} >>> STARTING EPISODE <<<")
    
    # 1. Initial Observation
    try:
        # Some envs return (obs, stop) tuple, others might just return obs (handled by defensive coding)
        result = await env.initial_observation()
        if isinstance(result, tuple):
            obs, stop_conditions = result
        else:
            obs = result
            stop_conditions = []
    except Exception as e:
        logger.error(f"{prefix} Failed to get initial observation: {e}")
        return

    # Convert observation to ModelInput if it is string or list of messages
    if isinstance(obs, str):
        obs = renderer.build_generation_prompt([{"role": "user", "content": obs}])
    elif isinstance(obs, list):
        # Assume list of messages
        obs = renderer.build_generation_prompt(obs)

    # Decode for logging
    if hasattr(obs, "to_ints"):
        obs_tokens = obs.to_ints()
        obs_text = tokenizer.decode(obs_tokens)
        logger.info(f"{prefix} Observation (len={len(obs_tokens)}):\n{'-'*20}\n{obs_text[:500]}...\n{'-'*20}")
    else:
        logger.info(f"{prefix} Observation raw: {obs}")

    # 2. Agent Action
    logger.info(f"{prefix} Querying Model...")
    sampling_params = types.SamplingParams(
        max_tokens=128,
        temperature=0.7,
        stop=stop_conditions
    )
    
    response = await sampling_client.sample_async(
        prompt=obs,
        num_samples=1,
        sampling_params=sampling_params
    )
    
    action_tokens = response.sequences[0].tokens
    action_text = tokenizer.decode(action_tokens)
    logger.info(f"{prefix} Action: {action_text.strip()}")

    # 3. Step
    logger.info(f"{prefix} Stepping...")
    try:
        step_result = await env.step(action_tokens)
        logger.info(f"{prefix} Reward: {step_result.reward}")
        logger.info(f"{prefix} Metrics: {step_result.metrics}")
    except Exception as e:
        logger.error(f"{prefix} Step failed: {e}")

async def main():
    logger.info("INITIALIZING ALL ENVS TEST")

    model_name = "Qwen/Qwen3-30B-A3B"
    
    # Setup Infrastructure
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = get_tokenizer(model_name)
    sampling_client.tokenizer = tokenizer  # Required by GradientProphet
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    for config in ENV_CONFIGS:
        name = config["name"]
        logger.info(f"--- Loading {name} ---")
        
        try:
            # Dynamic Import
            module = importlib.import_module(config["module"])
            loader = getattr(module, "load_environment")
            
            # Instantiate based on type
            envs = []
            
            if config["type"] == "verifiers_env":
                # Verifiers envs (Ghost, SelfPred) load directly into an Env object
                env = loader(**config["kwargs"])
                # We need to manually inject renderer for some verifiers envs if they rely on external rendering logic 
                # or assume they handle it internally. Most Verifiers envs handle text-in/text-out.
                envs = [env]

            elif config["type"] == "prophet_builder":
                # Gradient Prophet loads a Builder, which builds Envs
                builder = loader(**config["kwargs"])
                # Inject renderer manually as it's required for Prophet
                builder.renderer = renderer 
                envs = builder.build(sampling_client)

            elif config["type"] == "intuition_builder":
                # Intuition loads a Builder
                # Requires renderer in kwargs for the builder constructor
                kwargs = config["kwargs"].copy()
                kwargs["renderer"] = renderer
                builder = loader(**kwargs)
                envs = builder.build(
                    sampling_client=sampling_client,
                    service_client=service_client,
                    base_model=model_name
                )

            if not envs:
                logger.warning(f"[{name}] No environments created.")
                continue

            # Run one episode on the first instance of this environment type
            await run_env_episode(name, envs[0], sampling_client, tokenizer, renderer)

        except Exception as e:
            logger.error(f"[{name}] Failed to load or run: {e}", exc_info=True)

    logger.info("ALL TESTS COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())
