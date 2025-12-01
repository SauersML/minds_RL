from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import importlib.util
import sys

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class TrainerConfig:
    model_name: str
    batch_size: int = 1
    gradient_accumulation_steps: int = 1


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
        )
        self.model.to("cpu")
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "Trainer":
        path = Path(config_path)
        data = tomllib.loads(path.read_text())
        model_name = data.get("model", {}).get("name")
        if not model_name:
            raise ValueError("Config missing model.name")
        trainer_cfg = data.get("trainer", {})
        env_cfg = data.get("env", {})
        env_id = env_cfg.get("id")
        env_args = env_cfg.get("args", {})
        if not env_id:
            raise ValueError("Config missing env.id")
        env_path = Path(env_id).resolve()
        if env_path.is_dir():
            if str(env_path) not in sys.path:
                sys.path.append(str(env_path))
            module_spec = importlib.util.spec_from_file_location(
                env_path.name,
                env_path / "__init__.py",
                submodule_search_locations=[str(env_path)],
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Unable to import environment from {env_path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(env_id)
        if not hasattr(module, "load_environment"):
            raise AttributeError("Environment module must define load_environment")
        env = module.load_environment(**env_args)
        config = TrainerConfig(
            model_name=model_name,
            batch_size=int(trainer_cfg.get("batch_size", 1)),
            gradient_accumulation_steps=int(trainer_cfg.get("gradient_accumulation_steps", 1)),
        )
        return cls(config, env)

    def train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / "metrics.json"

        losses: list[float] = []
        rewards: list[float] = []

        dataset = self.env.dataset
        steps = max_steps if max_steps > 0 else 1

        for step in range(steps):
            sample = dataset[step % len(dataset)]
            prompt = sample.get("prompt") or sample.get("question")
            if not isinstance(prompt, str):
                raise ValueError("Sample prompt missing")
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            labels = inputs["input_ids"].clone()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(float(loss.detach()))

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )
            completion_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            completion = [{"role": "assistant", "content": completion_text}]
            prompt_messages = [{"role": "user", "content": prompt}]
            state: dict[str, Any] = {}
            reward_values: Sequence[float] = [
                func(
                    prompt_messages,
                    completion,
                    sample.get("answer"),
                    state,
                    sample.get("metadata"),
                )
                for func in self.env.rubric.funcs
            ]
            rewards.append(float(sum(reward_values)))

        metrics = {
            "loss": sum(losses) / len(losses),
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
        }
        metrics_path.write_text(
            "{\n  \"loss\": %.6f,\n  \"reward\": %.6f\n}" % (metrics["loss"], metrics["reward"])
        )
        return metrics

