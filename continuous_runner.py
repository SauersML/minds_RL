from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from custom_utils.trainer import Trainer

CHECKPOINT_FILE = Path(".tinker_checkpoint_id")


def _read_checkpoint() -> str | None:
    if not CHECKPOINT_FILE.exists():
        return None
    content = CHECKPOINT_FILE.read_text(encoding="utf-8").strip()
    return content or None


def _write_checkpoint(checkpoint_id: str) -> None:
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


def _run_stage(config_path: Path, checkpoint_id: str | None, *, max_steps: int) -> tuple[Mapping[str, Any], str | None]:
    trainer = Trainer.from_config(config_path)
    trainer.config.resume_checkpoint_id = checkpoint_id
    output_dir = Path("outputs") / config_path.stem
    metrics = trainer.train(max_steps=max_steps, output_dir=output_dir)
    next_checkpoint = None
    if isinstance(metrics, Mapping):
        next_checkpoint = metrics.get("checkpoint_id")
    if next_checkpoint:
        _write_checkpoint(next_checkpoint)
    return metrics, next_checkpoint


def main() -> None:
    checkpoint_id = _read_checkpoint()
    curriculum = [
        Path("configs/train_ghost.toml"),
        Path("configs/train_prophet.toml"),
        Path("configs/train_self_pred.toml"),
    ]
    max_steps = int(os.getenv("CURRICULUM_STEPS", "100"))

    results: list[tuple[str, Mapping[str, Any]]] = []
    for config_path in curriculum:
        metrics, checkpoint_id = _run_stage(config_path, checkpoint_id, max_steps=max_steps)
        if isinstance(metrics, Mapping):
            results.append((config_path.stem, metrics))
    if checkpoint_id:
        _write_checkpoint(checkpoint_id)
    _append_summary(results, checkpoint_id)


if __name__ == "__main__":
    main()
