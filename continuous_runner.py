from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

from custom_utils.trainer import Trainer

DATA_DIR = Path("data")
CHECKPOINT_FILE = DATA_DIR / "current_checkpoint.txt"
HISTORY_FILE = DATA_DIR / "training_history.jsonl"
SAMPLES_FILE = DATA_DIR / "samples.tsv"


def _read_checkpoint() -> str | None:
    if not CHECKPOINT_FILE.exists():
        return None
    content = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    return content or None


def _write_checkpoint(checkpoint_id: str) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(checkpoint_id, encoding="utf-8")


def _append_summary(results: list[tuple[str, Mapping[str, Any]]], final_checkpoint: str | None) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "## Curriculum run",
        "",
        "| Stage | Reward | Loss | Checkpoint |",
        "| --- | --- | --- | --- |",
    ]
    for stage, metrics in results:
        reward = metrics.get("reward", "n/a")
        loss = metrics.get("loss", "n/a")
        checkpoint = metrics.get("checkpoint_id") or final_checkpoint or ""
        lines.append(f"| {stage} | {reward} | {loss} | {checkpoint} |")

    if final_checkpoint:
        lines.extend([
            "",
            f"Final checkpoint: `{final_checkpoint}`",
        ])

    try:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        return


def _init_shared_clients(
    config_path: Path, checkpoint_id: str | None
) -> tuple[Trainer, Any, Any]:
    trainer = Trainer.from_config(config_path)
    if checkpoint_id:
        trainer.config.resume_checkpoint_id = checkpoint_id
    asyncio.run(trainer._build_clients())
    trainer._ensure_tokenizer()
    trainer._ensure_renderer()
    return trainer, trainer.service_client, trainer.training_client


def _run_stage(
    config_path: Path,
    checkpoint_id: str | None,
    *,
    max_steps: int,
    service_client: Any,
    training_client: Any,
    tokenizer: Any,
    renderer: Any,
    sampling_client: Any | None,
    deadline: float,
) -> tuple[Mapping[str, Any], str | None, Any | None]:
    trainer = Trainer.from_config(config_path)
    trainer.config.resume_checkpoint_id = checkpoint_id
    output_dir = Path("outputs") / config_path.stem
    metrics = trainer.train(
        max_steps=max_steps,
        output_dir=output_dir,
        checkpoint_path=CHECKPOINT_FILE,
        history_path=HISTORY_FILE,
        samples_path=SAMPLES_FILE,
        stage_name=config_path.stem,
        service_client=service_client,
        training_client=training_client,
        tokenizer=tokenizer,
        renderer=renderer,
        sampling_client=sampling_client,
        deadline=deadline,
    )
    next_checkpoint = None
    next_sampling_client = metrics.get("sampling_client") if isinstance(metrics, Mapping) else None
    if isinstance(metrics, Mapping):
        next_checkpoint = metrics.get("checkpoint_id")
    if next_checkpoint:
        _write_checkpoint(next_checkpoint)
    return metrics, next_checkpoint, next_sampling_client


def main() -> None:
    checkpoint_id = _read_checkpoint()
    curriculum = [
        Path("configs/train_ghost.toml"),
        Path("configs/train_prophet.toml"),
        Path("configs/train_self_pred.toml"),
    ]
    max_steps = int(os.getenv("CURRICULUM_STEPS", "10"))
    time_budget_seconds = float(os.getenv("CURRICULUM_TIME_LIMIT_SECONDS", 21000))
    buffer_seconds = float(os.getenv("CURRICULUM_TIME_BUFFER", 300))
    deadline = time.time() + time_budget_seconds

    base_trainer, service_client, training_client = _init_shared_clients(
        curriculum[0], checkpoint_id
    )
    tokenizer = base_trainer.tokenizer
    renderer = base_trainer.renderer
    sampling_client = None

    results: list[tuple[str, Mapping[str, Any]]] = []
    while time.time() < deadline:
        config_path = random.choice(curriculum)
        metrics, checkpoint_id, sampling_client = _run_stage(
            config_path,
            checkpoint_id,
            max_steps=max_steps,
            service_client=service_client,
            training_client=training_client,
            tokenizer=tokenizer,
            renderer=renderer,
            sampling_client=sampling_client,
            deadline=deadline,
        )
        if isinstance(metrics, Mapping):
            results.append((config_path.stem, metrics))
        if checkpoint_id:
            _write_checkpoint(checkpoint_id)

    if checkpoint_id:
        _write_checkpoint(checkpoint_id)
    _append_summary(results, checkpoint_id)


if __name__ == "__main__":
    main()
