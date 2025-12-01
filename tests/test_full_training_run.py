from __future__ import annotations

from pathlib import Path

import pytest

from custom_utils import Trainer


@pytest.mark.slow
@pytest.mark.parametrize("config_path", [Path("configs/ci_test.toml")])
def test_full_training_loop_runs(config_path: Path, tmp_path: Path) -> None:
    assert config_path.exists(), "CI config must exist"
    trainer = Trainer.from_config(config_path)
    metrics = trainer.train(max_steps=1, output_dir=tmp_path)
    metrics_path = tmp_path / "metrics.json"

    assert metrics_path.exists()
    content = metrics_path.read_text().strip()
    assert "loss" in content and "reward" in content
    assert metrics["loss"] >= 0
