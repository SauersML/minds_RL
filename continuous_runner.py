from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path

from tinker_cookbook.rl import train

from rl_config import RunnerConfig


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

    while time.time() < deadline:
        config_path = random.choice(curriculum)
        cfg_log_dir = log_root / config_path.stem
        cfg_log_dir.mkdir(parents=True, exist_ok=True)

        cfg = RunnerConfig(
            config_path=config_path,
            log_root=log_root,
            base_url=base_url,
            wandb_project=wandb_project,
        ).build()

        # Respect the time budget by breaking early if the next run would exceed it.
        if time.time() >= deadline:
            break
        asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
