# Minds RL: Tinker Client Training Harness

A lightweight client-side harness for reinforcement learning (RL) over large language models using the Tinker API. Local code handles environment logic, reward shaping, rendering/tokenization, and data assembly while the remote Tinker service performs distributed sampling, forward/backward passes, and optimizer steps.

## Installation

1. **Python**: requires Python 3.10+.
2. **Install environments** (editable installs keep code changes live):
   ```bash
   pip install -e environments/ghost_trace
   pip install -e environments/self_prediction
   pip install -e environments/gradient_prophet
   ```
3. **Core dependencies**: the environments depend on `verifiers`, `torch`, `transformers>=4.44`, `datasets`, and `peft`; the utilities expect the `tinker` SDK to be available at runtime.
4. **Authentication**: export your key before training: `export TINKER_API_KEY=<your_key>` (or set a custom env var referenced by `[tinker.api_key_env]` in configs).

## Architecture: What It Does

1. **Configuration & setup**
   * `RunnerConfig` converts the lightweight TOML configs in `configs/` into a `tinker_cookbook.rl.train.Config`, wiring in the right `RLDatasetBuilder` for each environment type.
   * The cookbook's `train.main` entry point handles all client creation (service/training/sampling), logging, and checkpoint management.

2. **Training loop**
   * The Tinker cookbook drives async, off-policy training with streaming minibatches so sampling continues while optimization runs.
   * Dataset builders for verifiers-based environments use the built-in `VerifiersRLDatasetBuilder`; Prophet and Gradient Intuition environments use custom builders in `environments/rl_datasets.py` with dedicated service clients for reward-side sampling.

3. **Data flow summary**
   * Local code provides prompts, reward computation, and dataset definitions.
   * Remote Tinker servers perform sampling plus overlapped forward/backward and optimizer steps, scheduled on clock cycles.

## Running Experiments

Run experiments from the repository root with the continuous runner:
```bash
python continuous_runner.py
```
The provided configs in `configs/` specify:
* `[env]`: module path for `load_environment` plus optional `[env.args]` forwarded as `**kwargs`.
* `[trainer]` and `[trainer.args]`: rollout count, loss_fn, and `max_new_tokens` for sampling.
* `[model]`: the base model ID expected by Tinker.
* `[tinker]`: either a direct `api_key` or an `api_key_env` that names the environment variable holding your key.

## Adding New Environments

1. **Directory layout**: add a package under `environments/<name>/` with an `__init__.py` and your main module.
2. **Entry point**: export a `load_environment(**kwargs)` function; any `[env.args]` values from the TOML config are passed to it.
3. **Minimum interface**:
   * `initial_observation(self) -> str`: returns the prompt text.
   * `step(self, action_text) -> StepResult|Mapping`: returns an object/dict with a `reward` field (numeric).  
   Environments may instead expose a `dataset` attribute with `prompt`/`question` fields; the trainer will sample across it.
4. **Rubric-based environments**: to use verifiers, attach a `rubric` with `funcs` and `weights`; the trainer will accumulate each function’s result as the reward.
5. **Model-aware rewards**: if the environment object defines `build(self, sampling_client)`, the trainer will call it and pass in the active sampler; use this to compute logprobs or additional samples during reward calculation (see `GradientProphetEnv`).
6. **Checklist**:
   * [ ] `load_environment` returns an object implementing the interface above.  
   * [ ] `initial_observation` returns a string prompt; tokenization/rendering is handled downstream.  
   * [ ] `step`/reward helpers return numeric rewards (float-compatible).  
   * [ ] Async Tinker calls (`sample_async`, `compute_logprobs_async`) are awaited inside reward code when used.
7. **Register the environment**: reference the module path in a config file, e.g.
   ```toml
   [env]
   id = "./environments/my_new_task"

   [env.args]
   difficulty = "hard"
   ```

## Development Notes

* Ensure the installed renderer/tokenizer matches your `base_model` string when configuring Tinker.  
* The trainer accepts both sync and async environment methods via `_maybe_await`, so prefer async reward computations that invoke remote APIs.  
* The `tinker` package must be installed and importable; otherwise `_require_tinker` raises an explicit error before training begins.

## Troubleshooting

* **Missing dependencies**: install the editable packages above and ensure `tinker` is available.  
* **Authentication**: verify `TINKER_API_KEY` (or your configured env var) is set before launching runs.  
* **Sampling/shape issues**: rewards are converted to NumPy arrays and normalized; non-numeric rewards will be coerced to zero.  
* **Logging**: per-step metrics are appended to `<output_dir>/metrics.json` so you can inspect reward trends during a run.
