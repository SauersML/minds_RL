from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import importlib.util
import sys
import json
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class TrainerConfig:
    model_name: str
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    rollouts_per_example: int = 4
    max_new_tokens: int = 256
    kl_beta: float = 0.1
    micro_batch_size: int = 1
    use_lora: bool = False
    gpus: int = 0


class Trainer:
    def __init__(self, config: TrainerConfig, env: Any) -> None:
        self.config = config
        self.env = env
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = (
            "cuda"
            if torch.cuda.is_available() and config.gpus and config.gpus > 0
            else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.ref_model: Optional[AutoModelForCausalLM] = None
        if config.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
            )
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=1e-4)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "Trainer":
        path = Path(config_path)
        data = tomllib.loads(path.read_text())
        model_name = data.get("model", {}).get("name")
        if not model_name:
            raise ValueError("Config missing model.name")
        trainer_cfg = data.get("trainer", {})
        trainer_args = trainer_cfg.get("args", {}) if isinstance(trainer_cfg, dict) else {}
        env_cfg = data.get("env", {})
        inference_cfg = data.get("inference", {})
        inference_args = inference_cfg.get("args", {}) if isinstance(inference_cfg, dict) else {}
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
            rollouts_per_example=int(trainer_cfg.get("rollouts_per_example", 4)),
            max_new_tokens=int(inference_args.get("max_new_tokens", 256)),
            kl_beta=float(trainer_cfg.get("kl_beta", 0.1)),
            micro_batch_size=int(trainer_args.get("micro_batch_size", trainer_cfg.get("micro_batch_size", 1))),
            use_lora=bool(trainer_cfg.get("use_lora", False)),
            gpus=int(inference_cfg.get("gpus", 0)),
        )
        return cls(config, env)

    def train(self, *, max_steps: int = 1, output_dir: Path | str | None = None) -> dict[str, Any]:
        output_path = Path(output_dir) if output_dir is not None else Path("outputs")
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = output_path / "metrics.json"

        losses: list[float] = []
        rewards: list[float] = []
        metrics_history: list[dict[str, float | int]] = []

        dataset = self.env.dataset
        steps = max_steps if max_steps > 0 else 1

        grad_accum_steps = max(self.config.gradient_accumulation_steps, 1)
        micro_step = 0

        for step in range(steps):
            batch_samples = [
                dataset[(step * self.config.batch_size + i) % len(dataset)]
                for i in range(self.config.batch_size)
            ]

            prompts = []
            prompt_lengths: list[int] = []
            prompt_samples: list[Mapping[str, Any]] = []
            for sample in batch_samples:
                prompt = sample.get("prompt") or sample.get("question")
                if not isinstance(prompt, str):
                    raise ValueError("Sample prompt missing")
                prompts.append(prompt)
                prompt_samples.append(sample)
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
                prompt_lengths.append(int(prompt_inputs["input_ids"].shape[1]))

            completions: list[tuple[torch.Tensor, str, dict[str, Any]]] = []
            for prompt, sample in zip(prompts, prompt_samples):
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
                prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
                with torch.no_grad():
                    generated = self.model.generate(
                        **prompt_inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        num_return_sequences=self.config.rollouts_per_example,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                for seq in generated:
                    completions.append(
                        (
                            seq,
                            self.tokenizer.decode(
                                seq.detach().cpu(), skip_special_tokens=True
                            ),
                            {"prompt": prompt, "sample": sample},
                        )
                    )

            if not completions:
                continue

            prompt_messages = [
                [{"role": "user", "content": info.get("prompt", "")}]
                for _, _, info in completions
            ]
            completion_rewards: list[float] = []
            for (_, completion_text, info), prompt_msgs in zip(completions, prompt_messages):
                completion = [{"role": "assistant", "content": completion_text}]
                state: dict[str, Any] = {}
                answer_value = ""
                sample_info = info.get("sample") if isinstance(info, dict) else None
                if isinstance(sample_info, Mapping):
                    raw_answer = sample_info.get("answer")
                    if isinstance(raw_answer, str):
                        answer_value = raw_answer
                reward_values: Sequence[float] = [
                    func(
                        prompt_msgs,
                        completion,
                        answer_value,
                        state,
                        info,
                        model=self.model,
                        tokenizer=self.tokenizer,
                    )
                    for func in self.env.rubric.funcs
                ]
                completion_rewards.append(float(sum(reward_values)))

            padded_ids = pad_sequence(
                [ids.to(self.device) for ids, _, _ in completions],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            attention_mask = (padded_ids != self.tokenizer.pad_token_id).long()
            labels = padded_ids.clone()
            prompt_lengths_expanded = []
            for pl in prompt_lengths:
                prompt_lengths_expanded.extend([pl] * self.config.rollouts_per_example)
            if hasattr(self.env, "apply_loss_mask"):
                labels = self.env.apply_loss_mask(
                    self.tokenizer,
                    {"input_ids": padded_ids, "attention_mask": attention_mask},
                    labels,
                    prompt_length=prompt_lengths_expanded,
                )
            labels_shift = labels[:, 1:]

            micro = max(int(self.config.micro_batch_size), 1)
            policy_logits_chunks = []
            for start in range(0, padded_ids.size(0), micro):
                end = start + micro
                outputs = self.model(
                    input_ids=padded_ids[start:end],
                    attention_mask=attention_mask[start:end],
                )
                policy_logits_chunks.append(outputs.logits)
            logits = torch.cat(policy_logits_chunks, dim=0)[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, padded_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            ref_logits_chunks = []
            ref_model = self.ref_model if self.ref_model is not None else self.model
            ref_context = (
                ref_model.disable_adapter()
                if hasattr(ref_model, "disable_adapter")
                else nullcontext()
            )
            with torch.no_grad(), ref_context:
                for start in range(0, padded_ids.size(0), micro):
                    end = start + micro
                    ref_outputs = ref_model(
                        input_ids=padded_ids[start:end],
                        attention_mask=attention_mask[start:end],
                    )
                    ref_logits_chunks.append(ref_outputs.logits)
            ref_logits = torch.cat(ref_logits_chunks, dim=0)[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs.gather(2, padded_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

            mask = (labels_shift != -100) & (attention_mask[:, 1:] > 0)
            mask_f = mask.float()
            mask_tokens = mask_f.sum(dim=1)
            mask_tokens = torch.clamp(mask_tokens, min=1.0)

            kl_values = (token_log_probs - ref_token_log_probs) * mask_f
            kl_mean = kl_values.sum(dim=1) / mask_tokens
            adjusted_rewards = [r - float(self.config.kl_beta) * float(k) for r, k in zip(completion_rewards, kl_mean.tolist())]

            reward_mean = sum(adjusted_rewards) / len(adjusted_rewards)
            reward_var = sum((r - reward_mean) ** 2 for r in adjusted_rewards) / max(len(adjusted_rewards) - 1, 1)
            reward_std = reward_var ** 0.5 if reward_var > 0 else 1e-6
            advantages = [(r - reward_mean) / reward_std for r in adjusted_rewards]

            self.model.train()

            advantages_tensor = torch.tensor(
                advantages, device=self.device, dtype=token_log_probs.dtype
            )
            mask_sum = mask_f.sum(dim=1).clamp(min=1.0)
            masked_mean_logprob = (token_log_probs * mask_f).sum(dim=1) / mask_sum
            loss_per_example = -advantages_tensor * masked_mean_logprob
            loss = loss_per_example.mean() / grad_accum_steps
            loss.backward()
            micro_step += 1
            loss_value = float(loss.detach())
            if micro_step % grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                micro_step = 0

            losses.append(loss_value)
            rewards.append(float(reward_mean))

            metrics_history.append(
                {"loss": losses[-1], "reward": rewards[-1], "step": step}
            )
            metrics_path.write_text(json.dumps(metrics_history, indent=2))

            if step % 10 == 0 or step == steps - 1:
                self.save_checkpoint(output_path / f"checkpoint-{step}")

        if micro_step % grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        metrics = {
            "loss": sum(losses) / len(losses) if losses else 0.0,
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
        }
        metrics_history.append({"loss": metrics["loss"], "reward": metrics["reward"]})
        metrics_path.write_text(json.dumps(metrics_history, indent=2))
        self.save_checkpoint(output_path / "checkpoint-final")
        return metrics

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

