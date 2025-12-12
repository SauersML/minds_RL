from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
import json

from tinker_cookbook.rl import train
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import VerifiersRLDatasetBuilder

from rl_config import RunnerConfig
from verifiers_adapter import make_custom_do_group_rollout


async def _run_config(cfg: train.Config) -> None:
    if isinstance(cfg.dataset_builder, VerifiersRLDatasetBuilder):
        original_do_group_rollout = train.do_group_rollout
        group_size = getattr(cfg.dataset_builder, "group_size", 1)
        rollout_fn = make_custom_do_group_rollout(
            cfg,
            group_size=group_size,
        )

        train.do_group_rollout = rollout_fn
        try:
            await train.main(cfg)
        finally:
            train.do_group_rollout = original_do_group_rollout
        return

    await train.main(cfg)


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

    curriculum = [
        Path("configs/train_ghost.toml"),
        Path("configs/train_prophet.toml"),
        Path("configs/train_self_pred.toml"),
        Path("configs/train_gradient_intuition.toml"),
    ]
    time_budget_seconds = float(os.getenv("CURRICULUM_TIME_LIMIT_SECONDS", 21000))
    deadline = time.time() + time_budget_seconds

    base_url = os.getenv("TINKER_BASE_URL")
    wandb_project = os.getenv("WANDB_PROJECT")
    log_root = Path("outputs")
    log_root.mkdir(parents=True, exist_ok=True)

    latest_checkpoint: str | None = None

    while time.time() < deadline:
        config_path = random.choice(curriculum)
        cfg_log_dir = log_root / config_path.stem
        cfg_log_dir.mkdir(parents=True, exist_ok=True)

        cfg = RunnerConfig(
            config_path=config_path,
            log_root=log_root,
            base_url=base_url,
            wandb_project=wandb_project,
            initial_checkpoint=latest_checkpoint,
        ).build()

        # Respect the time budget by breaking early if the next run would exceed it.
        if time.time() >= deadline:
            break
        asyncio.run(_run_config(cfg))

        updated_checkpoint = _load_last_state_checkpoint(Path(cfg.log_path)) if hasattr(cfg, "log_path") else None
        if updated_checkpoint:
            latest_checkpoint = updated_checkpoint


if __name__ == "__main__":
    main()
