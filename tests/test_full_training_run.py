from __future__ import annotations

from pathlib import Path

import pytest
import json

from custom_utils import Trainer


@pytest.mark.slow
@pytest.mark.parametrize("config_path", [Path("configs/ci_test.toml")])
def test_full_training_loop_runs(config_path: Path, tmp_path: Path) -> None:
    assert config_path.exists(), "CI config must exist"
    trainer = Trainer.from_config(config_path)
    metrics = trainer.train(max_steps=1, output_dir=tmp_path)
    metrics_path = tmp_path / "metrics.jsonl"

    assert metrics_path.exists()
    lines = metrics_path.read_text().strip().splitlines()
    assert lines, "metrics.jsonl should not be empty"
    last_metrics = json.loads(lines[-1])

    assert "loss" in last_metrics and "reward" in last_metrics
    assert last_metrics["loss"] >= 0
    assert last_metrics["loss"] == metrics["loss"]
    assert last_metrics["reward"] == metrics["reward"]
