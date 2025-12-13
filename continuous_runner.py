from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersEnvGroupBuilder

from rl_config import RunnerConfig
from verifiers_adapter import make_custom_do_group_rollout

# --- Monkeypatch to fix noisy shutdown error in Tinker library ---
# The InternalClientHolder.__del__ method can raise AttributeError/RuntimeWarning
# during interpreter shutdown if the event loop is already closed or the object
# was not fully initialized. We patch it to suppress these errors.
try:
    from tinker.lib.internal_client_holder import InternalClientHolder

    _original_del = getattr(InternalClientHolder, "__del__", None)

    def _safe_del(self):
        try:
            if _original_del:
                _original_del(self)
        except (AttributeError, RuntimeError):
            pass

    InternalClientHolder.__del__ = _safe_del
except ImportError:
    pass
# -----------------------------------------------------------------

logger = logging.getLogger(__name__)

# --- Helper for robust saving ---
async def save_checkpoint_async(**kwargs):
    """
    Wraps save_checkpoint_async. If a checkpoint name collision occurs
    (server has it, but we don't), retry with suffixes _1, _2, etc.
    """
    base_name = kwargs.get("name", "ckpt")
    
    # Try exact name first
    try:
        return await checkpoint_utils.save_checkpoint_async(**kwargs)
    except tinker.ConflictError:
        print(f"⚠️ Conflict: Checkpoint '{base_name}' exists on server. Attempting resolution...")

    # Iterate until we find a free slot
    for i in range(1, 1000):
        new_name = f"{base_name}_{i}"
        print(f"🔄 Retrying save as: '{new_name}'")
        kwargs["name"] = new_name
        try:
            return await checkpoint_utils.save_checkpoint_async(**kwargs)
        except tinker.ConflictError:
            continue # Name taken, try next
            
    raise RuntimeError(f"Could not save checkpoint '{base_name}' after 1000 attempts.")
# --------------------------------

def _install_advantage_normalization() -> None:
    # Tinker automatically centers advantages within each TrajectoryGroup.
    # We patch this if needed, but currently rely on Tinker's default behavior
    # for normalization, while ensuring logging tags are propagated correctly via the wrapper.
    pass


def _install_deadline_guard(stop_time: float | None) -> None:
    if stop_time is None:
        return

    async def _timed_stream_training(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
    ):
        sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
            training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
        )

        for i_batch in range(start_batch, end_batch):
            if _deadline_reached(stop_time):
                await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch)
                break

            metrics = {
                "progress/batch": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            if (cfg.eval_every > 0 and i_batch % cfg.eval_every == 0) or i_batch == end_batch - 1:
                with train.timed("run_evals", metrics):
                    eval_metrics = await train.run_evaluations_parallel(
                        evaluators, sampling_client, cfg, i_batch
                    )
                    metrics.update(eval_metrics)

            with train._get_logtree_scope(
                cfg.log_path,
                cfg.num_groups_to_log,
                f"train_iteration_{i_batch:06d}",
                f"RL Iteration {i_batch}",
            ):
                trajectory_groups_queue = asyncio.Queue()
                env_group_builders_P = dataset.get_batch(i_batch)

                @train.scope
                async def trajectory_group_worker_task(builder, enable_logging: bool) -> None:
                    local_metrics = {}
                    local_start = time.time()
                    trajectory_group = await train.do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        enable_logging=enable_logging,
                    )
                    local_metrics["time/trajectory_group_worker_loop/total"] = time.time() - local_start
                    if trajectory_group is not None:
                        trajectory_groups_queue.put_nowait(
                            train.WrappedTrajectoryGroup(
                                trajectory_group=trajectory_group,
                                env_group_builder=builder,
                                sampling_client_step=i_batch,
                                metrics=local_metrics,
                            )
                        )
                    else:
                        trajectory_groups_queue.put_nowait(None)

                for i, builder in enumerate(env_group_builders_P):
                    asyncio.create_task(
                        trajectory_group_worker_task(builder, enable_logging=i < cfg.num_groups_to_log),
                        name=f"trajectory_group_worker_task_{i}",
                    )

                sampling_client, full_batch_metrics = await train.do_train_step_streaming_and_get_sampling_client(
                    cfg,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    service_client,
                    tokenizer,
                )

            metrics.update(full_batch_metrics)
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)

            if _deadline_reached(stop_time):
                await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch + 1)
                break

    async def _timed_sync_training(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
    ):
        sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
            training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
        )

        for i_batch in range(start_batch, end_batch):
            if _deadline_reached(stop_time):
                await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch)
                break

            metrics = {
                "progress/batch": i_batch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }
            t_start = time.time()

            if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
                with train.timed("run_evals", metrics):
                    eval_metrics = await train.run_evaluations_parallel(
                        evaluators, sampling_client, cfg, i_batch
                    )
                    metrics.update(eval_metrics)

            env_group_builders_P = dataset.get_batch(i_batch)

            with train._get_logtree_scope(
                log_path=cfg.log_path,
                num_groups_to_log=cfg.num_groups_to_log,
                f_name=f"train_iteration_{i_batch:06d}",
                scope_name=f"RL Iteration {i_batch}",
            ):
                trajectory_groups_P = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            train.do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                max_tokens=cfg.max_tokens,
                                temperature=cfg.temperature,
                                do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                                enable_logging=i < cfg.num_groups_to_log,
                            ),
                            name=f"sample_task_{i}",
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ]
                )
            trajectory_groups_P = [
                trajectory_group
                for trajectory_group in trajectory_groups_P
                if trajectory_group is not None
            ]

            sampling_client, train_step_metrics = await train.do_train_step_and_get_sampling_client(
                cfg,
                i_batch,
                training_client,
                service_client,
                tokenizer,
                env_group_builders_P,
                trajectory_groups_P,
            )

            metrics.update(train_step_metrics)
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)

            if _deadline_reached(stop_time):
                await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch + 1)
                break

    async def _timed_async_training(
        start_batch,
        end_batch,
        num_batches,
        cfg,
        training_client,
        service_client,
        evaluators,
        dataset,
        ml_logger,
        tokenizer,
    ):
        """Modified version of train.do_async_training that respects deadlines."""
        assert cfg.async_config is not None

        shutdown_event = asyncio.Event()
        env_group_builders_queue = asyncio.Queue(maxsize=cfg.async_config.groups_per_batch)
        trajectory_groups_queue = asyncio.Queue()

        path_dict = await save_checkpoint_async(
            training_client=training_client,
            name=f"{start_batch:06d}",
            log_path=cfg.log_path,
            loop_state={"batch": start_batch},
            kind="both",
        )

        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        
        # Shared state
        state = {
            "sampling_client": sampling_client,
            "sampling_client_step": start_batch
        }
        sampling_client_updated_event = asyncio.Event()
        sampling_client_updated_event.set()

        @train.scope
        def shutdown_loops():
            shutdown_event.set()
            for _ in range(cfg.async_config.groups_per_batch):
                env_group_builders_queue.put_nowait(None)
            sampling_client_updated_event.set()

        @train.scope
        async def dataloader_loop():
            i_batch = start_batch
            while not shutdown_event.is_set() and i_batch < end_batch:
                env_group_builders_P = dataset.get_batch(i_batch)
                for env_group_builder in env_group_builders_P:
                    await env_group_builders_queue.put(env_group_builder)
                i_batch += 1

        @train.scope
        async def trajectory_group_worker_loop():
            while not shutdown_event.is_set():
                env_group_builder = await env_group_builders_queue.get()
                if env_group_builder is None:
                    break

                metrics = {}
                t_start = time.time()
                current_step = state["sampling_client_step"]
                
                trajectory_group = await train.do_group_rollout_and_filter_constant_reward(
                    state["sampling_client"],
                    env_group_builder,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                )
                
                if trajectory_group is None:
                    trajectory_groups_queue.put_nowait(None)
                else:
                    metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
                    trajectory_groups_queue.put_nowait(
                        train.WrappedTrajectoryGroup(
                            trajectory_group=trajectory_group,
                            env_group_builder=env_group_builder,
                            sampling_client_step=current_step,
                            metrics=metrics,
                        )
                    )

        @train.scope
        async def training_loop():
            i_batch = start_batch
            wrapped_trajectory_groups = []
            while i_batch < end_batch:
                # DEADLINE CHECK
                if _deadline_reached(stop_time):
                    await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch)
                    shutdown_loops()
                    return

                # PERIODIC CHECKPOINT
                if i_batch > 0 and i_batch % cfg.save_every == 0:
                    await save_checkpoint_async(
                        training_client=training_client,
                        name=f"{i_batch:06d}",
                        log_path=cfg.log_path,
                        loop_state={"batch": i_batch},
                        kind="state",
                    )

                wrapped_trajectory_group = await trajectory_groups_queue.get()
                if wrapped_trajectory_group is None:
                    logger.warning("Dropped a trajectory group (likely constant reward or rollout failure). If this happens often, training will hang.")
                    continue

                @train.scope
                def filter_stale_trajectory_group(wtg):
                    if wtg is None:
                        return False
                    if (i_batch - wtg.sampling_client_step > cfg.async_config.max_steps_off_policy):
                        asyncio.create_task(
                            env_group_builders_queue.put(wtg.env_group_builder),
                            name="requeue_stale_sample_task",
                        )
                        return False
                    return True

                metrics = {
                    "training_client/step": i_batch,
                    "optim/lr": cfg.learning_rate,
                    "progress/done_frac": (i_batch + 1) / num_batches,
                }
                t_start = time.time()

                if cfg.stream_minibatch_config is not None:
                    new_sampling_client, train_step_metrics = await train.do_train_step_streaming_and_get_sampling_client(
                        cfg, i_batch, trajectory_groups_queue, training_client, service_client, tokenizer, filter_stale_trajectory_group
                    )
                    state["sampling_client"] = new_sampling_client
                else:
                    if not filter_stale_trajectory_group(wrapped_trajectory_group):
                        continue
                    
                    wrapped_trajectory_groups.append(wrapped_trajectory_group)
                    if len(wrapped_trajectory_groups) < cfg.async_config.groups_per_batch:
                        continue
                    
                    metrics.update(train.compute_sampling_client_metrics(wrapped_trajectory_groups))
                    
                    new_sampling_client, train_step_metrics = await train.do_train_step_and_get_sampling_client(
                        cfg, i_batch, training_client, service_client, tokenizer,
                        [g.env_group_builder for g in wrapped_trajectory_groups],
                        [g.trajectory_group for g in wrapped_trajectory_groups],
                    )
                    state["sampling_client"] = new_sampling_client
                
                state["sampling_client_step"] = i_batch + 1
                sampling_client_updated_event.set()

                metrics.update(train_step_metrics)
                metrics["time/training_loop/total"] = time.time() - t_start
                ml_logger.log_metrics(metrics, step=i_batch)
                
                # Check deadline after step completion as well
                if _deadline_reached(stop_time):
                    await _save_checkpoint_on_deadline(training_client, cfg.log_path, i_batch + 1)
                    shutdown_loops()
                    return

                i_batch += 1
                wrapped_trajectory_groups = []

            shutdown_loops()

        @train.scope
        async def evaluation_loop():
            if len(evaluators) == 0 or cfg.eval_every == 0:
                return
            while not shutdown_event.is_set():
                await sampling_client_updated_event.wait()
                sampling_client_updated_event.clear()
                
                if _deadline_reached(stop_time):
                    break

                metrics = {}
                t_start = time.time()
                eval_step = state["sampling_client_step"]
                eval_client = state["sampling_client"]
                
                if cfg.eval_every > 0 and eval_step % cfg.eval_every == 0:
                    with train.timed("run_evals", metrics):
                        metrics.update(await train.run_evaluations_parallel(evaluators, eval_client, cfg, eval_step))
                    metrics["time/evaluation_loop/total"] = time.time() - t_start
                    ml_logger.log_metrics(metrics, step=eval_step)

        await asyncio.gather(
            asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
            *[asyncio.create_task(trajectory_group_worker_loop(), name=f"worker_{i}") 
              for i in range(cfg.async_config.groups_per_batch)],
            asyncio.create_task(training_loop(), name="training_loop"),
            asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
        )

    train.do_sync_training_with_stream_minibatch = _timed_stream_training
    train.do_sync_training = _timed_sync_training
    train.do_async_training = _timed_async_training


def _deadline_reached(stop_time: float) -> bool:
    return time.time() >= stop_time


async def _save_checkpoint_on_deadline(training_client, log_path, i_batch) -> None:
    print(f"Deadline reached at batch {i_batch}. Saving checkpoint and exiting...")
    await save_checkpoint_async(
        training_client=training_client,
        name=f"{i_batch:06d}_deadline",
        log_path=log_path,
        loop_state={"batch": i_batch},
        kind="state",
    )


def _load_last_state_checkpoint(log_dir: Path) -> str | None:
    """Find the most recent training checkpoint in the log directory.

    Parses the ``checkpoints.jsonl`` file to find the last entry with a
    ``state_path``. This path is used to resume training in the next iteration.

    Args:
        log_dir: The directory containing the training logs.

    Returns:
        The Tinker URI of the last checkpoint, or None if not found.
    """
    checkpoints_file = log_dir / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return None

    last_checkpoint: str | None = None
    try:
        with checkpoints_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if isinstance(record, dict) and "state_path" in record:
                    last_checkpoint = str(record["state_path"])
    except Exception:
        return last_checkpoint

    return last_checkpoint


def _wrap_rollout_with_tags(rollout_fn):
    async def wrapper(builder, *args, **kwargs):
        traj_group = await rollout_fn(builder, *args, **kwargs)
        if traj_group is not None:
            try:
                tags = builder.logging_tags()
                if tags:
                    setattr(traj_group, "logging_tags", tags)
            except AttributeError:
                pass
        return traj_group
    return wrapper


def main() -> None:
    """Main entry point for the continuous runner."""
    state_file = Path("run_state.json")
    run_id = None

    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                run_id = state.get("run_id")
                wandb_run_id = state.get("wandb_run_id")
                if wandb_run_id:
                    os.environ["WANDB_RUN_ID"] = wandb_run_id
                    os.environ["WANDB_RESUME"] = "allow"
                    print(f"Resuming run_id: {run_id}, wandb_id: {wandb_run_id}")
        except Exception as e:
            print(f"Failed to read run state: {e}")

    try:
        import wandb
        if not os.getenv("WANDB_RUN_ID"):
            generated_id = wandb.util.generate_id()
            os.environ["WANDB_RUN_ID"] = generated_id
            os.environ.setdefault("WANDB_RESUME", "allow")
    except Exception:
        pass

    if run_id is None:
        run_id = os.getenv("RUN_ID") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(state_file, "w") as f:
            json.dump({
                "run_id": run_id,
                "wandb_run_id": os.environ.get("WANDB_RUN_ID")
            }, f)
        print(f"Started new run_id: {run_id}")

    master_config = Path(os.getenv("MASTER_CONFIG_PATH", "configs/multi_task.toml"))
    time_budget_seconds = float(os.getenv("CURRICULUM_TIME_LIMIT_SECONDS", 21000))
    deadline = time.time() + time_budget_seconds

    base_url = os.getenv("TINKER_BASE_URL")
    wandb_project = os.getenv("WANDB_PROJECT")

    log_root = Path("outputs") / run_id
    log_root.mkdir(parents=True, exist_ok=True)

    initial_checkpoint = _load_last_state_checkpoint(log_root / master_config.stem)

    cfg = RunnerConfig(
        config_path=master_config,
        log_root=log_root,
        base_url=base_url,
        wandb_project=wandb_project,
        initial_checkpoint=initial_checkpoint,
    ).build()

    _install_advantage_normalization()
    _install_deadline_guard(deadline)

    # Determine group size (fallback to config default if not available)
    # The default is 8 in the config. We can access it via cfg.dataset_builder if available or assume consistent group size.
    # cfg.dataset_builder might be CompositeRLDatasetBuilder.
    # The verifiers adapter needs a fixed group size for concurrency control.
    group_size = getattr(cfg.dataset_builder, "group_size", 8)

    verifiers_rollout_fn = make_custom_do_group_rollout(
        cfg,
        group_size=group_size,
    )
    original_do_group_rollout = train.do_group_rollout

    async def dispatch_rollout(builder, policy):
        # Unwrap the builder to find the true underlying builder
        inner_builder = builder
        while hasattr(inner_builder, "_base"):
            inner_builder = inner_builder._base

        is_verifiers_builder = False
        # Check if VerifiersEnvGroupBuilder is a valid type before using isinstance
        if isinstance(VerifiersEnvGroupBuilder, type):
            if isinstance(inner_builder, VerifiersEnvGroupBuilder):
                is_verifiers_builder = True

        if not is_verifiers_builder:
            # Fallback: check class name or attribute
            if type(inner_builder).__name__ == "VerifiersEnvGroupBuilder":
                is_verifiers_builder = True
            elif hasattr(inner_builder, "vf_env"):
                is_verifiers_builder = True

        if is_verifiers_builder:
            # Use the custom Verifiers rollout logic with the UNWRAPPED builder
            return await verifiers_rollout_fn(inner_builder, policy)
        else:
            # Use the standard Tinker rollout logic with the ORIGINAL (possibly wrapped) builder
            return await original_do_group_rollout(builder, policy)

    train.do_group_rollout = _wrap_rollout_with_tags(dispatch_rollout)

    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
