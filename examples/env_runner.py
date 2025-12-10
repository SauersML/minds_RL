import os
import sys
import subprocess
import time
from pathlib import Path
import importlib
import json

# Ensure project root is in sys.path so 'custom_utils' (flat layout) is importable
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# --- Configuration ---
TINKER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_STEPS = 1
ROLLOUTS = 2
MAX_TOKENS = 256
ROOT_DIR = Path(os.getcwd())

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
    

    run_cmd(["pip", "install", "tinker"], "Installing Tinker SDK")

    run_cmd(["pip", "install", "tinker_cookbook"], "Installing tinker_cookbook")
    
    utils_path = ROOT_DIR / "custom_utils"
    run_cmd(["pip", "install", "-e", str(utils_path)], "Installing custom_utils")


def discover_and_install_envs():
    print_header("DISCOVERING ENVIRONMENTS")
    envs_dir = ROOT_DIR / "environments"
    if not envs_dir.exists():
        print("❌ Error: 'environments' directory not found.")
        sys.exit(1)

    discovered = []
    
    # Look for any directory containing a pyproject.toml or setup.py
    for item in envs_dir.iterdir():
        if item.is_dir() and (item / "pyproject.toml").exists():
            discovered.append(item)
            run_cmd(["pip", "install", "-e", str(item)], f"Installing env: {item.name}")
    
    if not discovered:
        print("⚠️  No environments found in environments/ directory.")
    
    return discovered

def generate_config(env_path, output_path):
    # Relative path for the config
    rel_path = f"./environments/{env_path.name}"

    if env_path.name == "gradient_intuition":
        env_args_block = (
            "[env.args]\n"
            "inner_env_id = \"./environments/ghost_trace\"\n"
            "inner_env_args = { num_examples = 3 }\n"
            "alpha = 0.3\n"
            "shadow_rank = 4\n"
            "shadow_learning_rate = 5e-5\n\n"
        )
    else:
        env_args_block = "[env.args]\nnum_examples = 5  # Small dataset for speed\n\n"

    toml_content = f"""
[env]
id = "{rel_path}"

{env_args_block}[trainer]
rollouts_per_example = {ROLLOUTS}
loss_fn = "importance_sampling"

  [trainer.args]
  max_new_tokens = {MAX_TOKENS}
  training_rank = 8
  learning_rate = 3.162e-6

[model]
base_model = "{TINKER_MODEL}"

[tinker]
api_key_env = "TINKER_API_KEY"
"""
    output_path.write_text(toml_content)

def run_integration_test(env_path):
    env_name = env_path.name
    print(f"\n🧪 TESTING ENVIRONMENT: [ {env_name} ]")
    
    # Dynamic imports to ensure we pick up the installed packages
    try:
        # Force reload of trainer to clear previous states if any
        if "custom_utils.trainer" in sys.modules:
            import custom_utils.trainer
            importlib.reload(custom_utils.trainer)
        from custom_utils.trainer import Trainer
    except ImportError as e:
        print(f"   ❌ Import Failed: {e}")
        return False

    # Create run-specific config and output dir
    run_dir = ROOT_DIR / "runs" / f"test_{env_name}_{int(time.time())}"
    config_file = run_dir / "config.toml"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    generate_config(env_path, config_file)

    try:
        print("   ⚙️  Configuring Trainer...")
        trainer = Trainer.from_config(config_file)
        
        print(f"   ⚡ Sending Job to Tinker (Steps: {MAX_STEPS})...")
        trainer.train(max_steps=MAX_STEPS, output_dir=run_dir)
        
        # Validation
        metrics_file = run_dir / "metrics.jsonl"
        if metrics_file.exists():
            with metrics_file.open() as f:
                data = [json.loads(line) for line in f if line.strip()]

            if data:
                last_reward = data[-1].get("reward", "N/A")
                print(f"   ✅ SUCCESS! Final Reward: {last_reward}")
                return True

            print("   ⚠️  Metrics file is empty.")
            return False
        else:
            print("   ⚠️  Finished, but no metrics file generated.")
            return False

    except Exception as e:
        print(f"   ❌ CRASHED: {e}")
        # import traceback
        # traceback.print_exc()
        return False
    finally:
        # Cleanup config to keep repo clean
        if config_file.exists():
            os.remove(config_file)

def main():
    # 0. Auth Check
    if not os.getenv("TINKER_API_KEY"):
        print("❌ Error: TINKER_API_KEY is not set.")
        sys.exit(1)

    # 1. Setup
    install_dependencies()
    
    # 2. Discovery & Install
    env_paths = discover_and_install_envs()
    
    # 3. Execution Loop
    print_header("STARTING INTEGRATION TESTS")
    results = {}
    
    for env_path in env_paths:
        success = run_integration_test(env_path)
        results[env_path.name] = "PASS" if success else "FAIL"

    # 4. Final Report
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

if __name__ == "__main__":
    main()
