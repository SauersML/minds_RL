import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import data_processing, train
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersRLDatasetBuilder

from rl_config import RunnerConfig
from verifiers_adapter import make_custom_do_group_rollout


def _wrap_rollout_with_tags(rollout_fn):
    async def _wrapped(env_group_builder, policy):
        trajectory_group = await rollout_fn(env_group_builder, policy)
        try:
            tags = env_group_builder.logging_tags()
        except Exception:
            tags = []
        setattr(trajectory_group, "logging_tags", list(tags or ["default"]))
        return trajectory_group

    return _wrapped


def _install_advantage_normalization() -> None:
    original_compute = data_processing.compute_advantages

    def _compute_with_whitening(trajectory_groups_P):
        if not trajectory_groups_P:
            return original_compute(trajectory_groups_P)

        rewards_by_tag: dict[tuple[str, ...], list[torch.Tensor]] = {}
        for trajectory_group in trajectory_groups_P:
            tag = tuple(getattr(trajectory_group, "logging_tags", []) or ["default"])
            rewards_by_tag.setdefault(tag, []).append(
                torch.tensor(trajectory_group.get_total_rewards(), dtype=torch.float32)
            )

        stats = {
            tag: torch.cat(values)
            for tag, values in rewards_by_tag.items()
            if len(values) > 0
        }
        means = {tag: tensor.mean() for tag, tensor in stats.items()}
        stds = {tag: tensor.std(unbiased=False) for tag, tensor in stats.items()}

        normalized: list[torch.Tensor] = []
        for trajectory_group in trajectory_groups_P:
            tag = tuple(getattr(trajectory_group, "logging_tags", []) or ["default"])
            rewards = torch.tensor(trajectory_group.get_total_rewards(), dtype=torch.float32)
            mean = means.get(tag, rewards.mean())
            std = stds.get(tag, torch.tensor(0.0))
            normalized.append((rewards - mean) / (std + 1e-8))

        return normalized

    data_processing.compute_advantages = _compute_with_whitening


def _deadline_reached(stop_time: float | None) -> bool:
    return stop_time is not None and time.time() >= stop_time


def _save_checkpoint_on_deadline(training_client, log_path: str, batch_idx: int) -> asyncio.Future:
    return asyncio.create_task(
        checkpoint_utils.save_checkpoint_async(
            training_client,
            name="deadline",
            log_path=log_path,
            loop_state={"batch": batch_idx},
            kind="both",
        )
    )


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

    train.do_sync_training_with_stream_minibatch = _timed_stream_training
    train.do_sync_training = _timed_sync_training


def _load_last_state_checkpoint(log_dir: Path) -> str | None:
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


def main() -> None:
    try:
        import wandb  # type: ignore

        if not os.getenv("WANDB_RUN_ID"):
            os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
            os.environ.setdefault("WANDB_RESUME", "allow")
    except Exception:
        pass

    master_config = Path(os.getenv("MASTER_CONFIG_PATH", "configs/multi_task.toml"))
    time_budget_seconds = float(os.getenv("CURRICULUM_TIME_LIMIT_SECONDS", 21000))
    deadline = time.time() + time_budget_seconds

    base_url = os.getenv("TINKER_BASE_URL")
    wandb_project = os.getenv("WANDB_PROJECT")

    run_id = os.getenv("RUN_ID") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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

    rollout_fn = train.do_group_rollout
    if isinstance(cfg.dataset_builder, VerifiersRLDatasetBuilder):
        group_size = getattr(cfg.dataset_builder, "group_size", 1)
        rollout_fn = make_custom_do_group_rollout(
            cfg,
            group_size=group_size,
        )

    train.do_group_rollout = _wrap_rollout_with_tags(rollout_fn)

    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
