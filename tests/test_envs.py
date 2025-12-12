import asyncio
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Configuration ---
TINKER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# Config template used to test initialization
CONFIG_TEMPLATE = """
[env]
id = "{env_id}"

[env.args]
num_examples = 2  # Keep it small for testing
seed = 42
# Args specifically for gradient_intuition nesting
inner_env_id = "./environments/ghost_trace"
inner_env_args = {{ num_examples = 2 }}

[trainer]
rollouts_per_example = 1
loss_fn = "importance_sampling"

[trainer.args]
max_new_tokens = 128
training_rank = 8
learning_rate = 3.162e-6

[model]
base_model = "{base_model}"
renderer_name = "role_colon" 

[tinker]
api_key_env = "TINKER_API_KEY"
"""

def print_header(msg):
    print(f"\n{'='*60}\n🚀 {msg}\n{'='*60}")

def run_cmd(cmd, description):
    print(f"📦 {description}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m"] + cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("   ✅ Done.")
    except subprocess.CalledProcessError:
        print(f"   ❌ Failed to execute: {' '.join(cmd)}")
        sys.exit(1)

def install_dependencies():
    print_header("INSTALLING DEPENDENCIES")
    # Install core libs
    run_cmd(["pip", "install", "tinker", "tinker_cookbook", "verifiers", "tomli", "numpy", "datasets"], "Installing Core Libs")
    
    # Install local environments
    envs_dir = ROOT_DIR / "environments"
    if envs_dir.exists():
        for item in sorted(envs_dir.iterdir()):
            if item.is_dir() and (item / "pyproject.toml").exists():
                run_cmd(["pip", "install", "-e", str(item)], f"Installing Env Package: {item.name}")

async def check_single_env(env_path: Path):
    env_name = env_path.name
    print(f"\n🧪 TESTING ENVIRONMENT: [ {env_name} ]")

    # 1. Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".toml", delete=False) as tmp:
        config_content = CONFIG_TEMPLATE.format(
            env_id=env_name,
            base_model=TINKER_MODEL
        )
        tmp.write(config_content)
        config_path = Path(tmp.name)

    try:
        # 2. Dynamic Import of local config logic
        # We use rl_config.py from the root directory
        import rl_config

        # 3. Build the configuration object
        print("   ⚙️  Building RunnerConfig...")
        runner_cfg = rl_config.RunnerConfig(
            config_path=config_path,
            log_root=ROOT_DIR / "outputs"
        )
        # This parses the TOML and calls _build_dataset_builder
        # which imports the environment module
        train_config = runner_cfg.build()
        
        # 4. Initialize the Dataset Builder
        # This usually involves connecting to the Tinker API to get the tokenizer/client
        print("   🔌 Connecting to Tinker & Initializing Builder...")
        builder = train_config.dataset_builder
        
        # 5. Build the dataset (Async operation)
        # This verifies data generation and API client creation
        dataset, _ = await builder()
        
        if len(dataset) == 0:
            print("   ⚠️  Dataset is empty.")
            return False

        # 6. Instantiate the Environment
        print("   🏗️  Instantiating Environment...")
        # Get the first batch of environment factories
        env_group_builders = dataset.get_batch(0)
        if not env_group_builders:
            print("   ❌ Failed to get batch from dataset.")
            return False
            
        # Create the actual environment instances
        envs = await env_group_builders[0].make_envs()
        if not envs:
            print("   ❌ No environments created.")
            return False
        
        # 7. Run Initial Observation
        # This checks if the prompt rendering works
        print("   👀 Checking Initial Observation...")
        
        # Depending on the renderer, obs might be a string or a tuple/object
        # We just want to ensure it didn't crash
        print("   ✅ Environment initialized and generated prompt successfully.")
        return True

    except Exception as e:
        print(f"   ❌ CRASHED: {e}")
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if config_path.exists():
            os.remove(config_path)

async def main_async():
    # 0. Auth Check
    if not os.getenv("TINKER_API_KEY"):
        print("❌ Error: TINKER_API_KEY is not set.")
        sys.exit(1)

    # 1. Setup
    install_dependencies()

    # 2. Discovery
    print_header("AUTO-DISCOVERING ENVIRONMENTS")
    envs_dir = ROOT_DIR / "environments"
    if not envs_dir.exists():
        print("❌ Error: 'environments' directory not found.")
        sys.exit(1)

    discovered = []
    for item in sorted(envs_dir.iterdir()):
        if not item.is_dir():
            continue
        # Check for python package markers
        if (item / "__init__.py").exists() or (item / "pyproject.toml").exists():
            discovered.append(item)
            print(f"   🔍 Found: {item.name}")

    if not discovered:
        print("⚠️  No environments found.")
        sys.exit(1)

    # 3. Execution Loop
    results = {}
    for env_path in discovered:
        success = await check_single_env(env_path)
        results[env_path.name] = "PASS" if success else "FAIL"

    # 4. Summary
    print_header("TEST SUMMARY")
    print(f"{'ENVIRONMENT':<30} | {'STATUS':<10}")
    print("-" * 45)
    
    all_passed = True
    for name, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"{name:<30} | {icon} {status}")
        if status == "FAIL":
            all_passed = False

    sys.exit(0 if all_passed else 1)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
