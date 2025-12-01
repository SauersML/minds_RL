# Minds RL: Tinker Client Training Harness

A lightweight client-side harness for reinforcement learning (RL) over large language models using the Tinker API. Local code handles environment logic, reward shaping, rendering/tokenization, and data assembly while the remote Tinker service performs distributed sampling, forward/backward passes, and optimizer steps.

## Installation

1. **Python**: requires Python 3.10+ (per the local utility package metadata).  
2. **Install local utilities and environments** (editable installs keep code changes live):
   ```bash
   pip install -e custom_utils
   pip install -e environments/ghost_trace
   pip install -e environments/self_prediction
   pip install -e environments/gradient_prophet
   ```
3. **Core dependencies**: the utilities and environments depend on `verifiers`, `torch`, `transformers>=4.44`, `datasets`, and `peft`; the utilities expect the `tinker` SDK to be available at runtime.  
4. **Authentication**: export your key before training: `export TINKER_API_KEY=<your_key>` (or set a custom env var referenced by `[tinker.api_key_env]` in configs).

## Architecture: What It Does

1. **Configuration & setup**
   * `Trainer.from_config(<path>)` loads a TOML config, imports the `load_environment` function from the configured module, and captures trainer/model/tinker settings (including API key resolution from `TINKER_API_KEY` or a custom env var).  
   * The trainer lazily imports the `tinker` package, then builds a `ServiceClient` (for sampling) and `TrainingClient` (for optimization) using the chosen base model and loss function.

2. **Training loop (per step)**
   * Selects an environment: if the loaded object exposes `build(sampling_client)`, the trainer materializes a list of env instances with access to the active sampler; otherwise it reuses the provided dataset/env.  
   * Obtains a prompt via `initial_observation()` or a dataset entry (`prompt`/`question`).  
   * Calls `sampling_client.sample(prompt, num_samples, max_tokens)` to generate multiple completions.  
   * Scores each completion: environments with a `rubric` run the weighted verification functions; otherwise `env.step(text)` is awaited and its `reward` is used.  
   * Computes a baseline mean reward, converts centered rewards to advantages, and packages each prompt/completion/advantage into `tinker.Datum` objects.  
   * Pipelines `forward_backward` and `optim_step` calls to the Tinker service to update weights efficiently, then logs step metrics to `outputs/metrics.json` (or the provided `output_dir`).

3. **Data flow summary**
   * Local code manages prompts, sampling, reward computation, and data assembly.  
   * Remote Tinker servers perform sampling (when using `ServiceClient`) and the paired forward/backward + optimizer clock cycles issued by `TrainingClient`.

## Running Experiments

Run experiments from the repository root:
```bash
python run_experiment.py configs/train_ghost.toml
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
