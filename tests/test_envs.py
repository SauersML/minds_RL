from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing local modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tinker
from tinker import types
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Import the environment loader
from environments.gradient_intuition.gradient_intuition import load_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_envs")

async def main():
    logger.info("INITIALIZING INTEGRATION TEST")

    # 1. Configuration
    # Using a model known to work with Tinker
    model_name = "Qwen/Qwen3-30B-A3B" 
    inner_env_id = "./environments/ghost_trace"
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Inner Env: {inner_env_id}")

    # 2. Setup Tinker Clients
    logger.info("Setting up Tinker Service Client...")
    try:
        service_client = tinker.ServiceClient()
    except Exception as e:
        logger.error(f"Failed to create ServiceClient. Check TINKER_API_KEY. Error: {e}")
        sys.exit(1)

    # 3. Setup Tokenizer and Renderer
    logger.info("Loading Tokenizer and Renderer...")
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # 4. Create Sampling Client (The Agent)
    logger.info("Creating Sampling Client...")
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    # 5. Initialize Environment Builder
    logger.info("Creating GradientIntuitionBuilder...")
    # Matches configs/train_gradient_intuition.toml settings roughly
    builder = load_environment(
        inner_env_id=inner_env_id,
        alpha=0.35,
        shadow_rank=8,
        shadow_learning_rate=1e-4,
        renderer=renderer
    )

    # 6. Build the Environment
    # Note: GradientIntuitionEnv needs service_client and base_model to create internal shadow clients
    logger.info("Building Environment instances...")
    envs = builder.build(
        sampling_client=sampling_client, # Passed to inner envs if they need it
        service_client=service_client,   # For shadow client creation
        base_model=model_name            # For shadow client creation
    )
    
    if not envs:
        logger.error("Builder returned no environments!")
        sys.exit(1)
        
    env = envs[0]
    logger.info(f"Environment ready: {env}")

    # 7. Run Episode
    logger.info(">>> STARTING EPISODE <<<")
    
    # A. Initial Observation
    logger.info("Getting initial observation...")
    obs, stop_conditions = await env.initial_observation()
    
    obs_tokens = obs.to_ints()
    obs_text = tokenizer.decode(obs_tokens)
    logger.info(f"Observation (len={len(obs_tokens)}):")
    logger.info("-" * 40)
    logger.info(obs_text)
    logger.info("-" * 40)

    # B. Agent Action (REAL INFERENCE)
    logger.info("Querying Model for Action...")
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
    
    logger.info(f"Agent Action (len={len(action_tokens)}):")
    logger.info("-" * 40)
    logger.info(action_text)
    logger.info("-" * 40)

    # C. Environment Step
    logger.info("Stepping Environment...")
    step_result = await env.step(action_tokens)
    
    logger.info(">>> STEP RESULT <<<")
    logger.info(f"Reward: {step_result.reward}")
    logger.info(f"Done: {step_result.episode_done}")
    logger.info(f"Metrics: {step_result.metrics}")
    
    logger.info("TEST COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    asyncio.run(main())
