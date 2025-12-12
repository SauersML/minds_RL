from __future__ import annotations

import os
from pathlib import Path

from rl_config import RunnerConfig


def test_runner_config_parses_ci_config(tmp_path: Path) -> None:
    config_path = Path("configs/ci_test.toml")
    assert config_path.exists(), "CI config must exist"

    # Provide a stub API key so RunnerConfig validation passes without hitting the network.
    os.environ.setdefault("TINKER_API_KEY", "dummy-key")

    runner_cfg = RunnerConfig(config_path=config_path, log_root=tmp_path)
    cfg = runner_cfg.build()

    assert cfg.model_name == "meta-llama/Llama-3.2-1B"
    assert cfg.max_tokens == 512
    assert cfg.async_config is not None
    assert cfg.stream_minibatch_config is not None
