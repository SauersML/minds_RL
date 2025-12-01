from __future__ import annotations

from contextlib import AbstractContextManager
from typing import List, Sequence

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class EphemeralShadow(AbstractContextManager):
    """Context manager that attaches a temporary LoRA adapter for shadow updates."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Sequence[str] | None = None,
        lr: float = 1e-2,
        steps: int = 3,
        adapter_name: str = "shadow",
    ) -> None:
        self.model = model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = list(target_modules) if target_modules is not None else None
        self.lr = lr
        self.steps = steps
        self.adapter_name = adapter_name
        self._optimizer: torch.optim.Optimizer | None = None
        self._requires_grad_backup: dict[str, bool] = {}

    def __enter__(self) -> "EphemeralShadow":
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
        )
        if not hasattr(self.model, "add_adapter"):
            self.model = get_peft_model(self.model, config, adapter_name=self.adapter_name)
        else:
            self.model.add_adapter(config, adapter_name=self.adapter_name)
        self.model.train()
        self._requires_grad_backup = {
            name: param.requires_grad for name, param in self.model.named_parameters()
        }
        for name, param in self.model.named_parameters():
            param.requires_grad = self.adapter_name in name
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.SGD(trainable_params, lr=self.lr)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "delete_adapter"):
            try:
                self.model.delete_adapter(self.adapter_name)
            except Exception:
                pass
        if self._requires_grad_backup:
            for name, param in self.model.named_parameters():
                if name in self._requires_grad_backup:
                    param.requires_grad = self._requires_grad_backup[name]
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    def run_shadow_update(
        self,
        lesson_input_ids: torch.Tensor,
        lesson_target_ids: torch.Tensor,
        *,
        pad_token_id: int,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        if self._optimizer is None:
            raise RuntimeError("Shadow optimizer not initialized")

        _ = pad_token_id  # unused placeholder for future padding strategies

        device = next(self.model.parameters()).device
        lesson_input_ids = lesson_input_ids.to(device)
        lesson_target_ids = lesson_target_ids.to(device)

        combined = torch.cat([lesson_input_ids, lesson_target_ids], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(combined)
        else:
            attention_mask = torch.cat(
                [attention_mask.to(device), torch.ones_like(lesson_target_ids, device=device)], dim=1
            )

        labels = combined.clone()
        labels[:, : lesson_input_ids.size(1)] = -100

        for _ in range(self.steps):
            self._optimizer.zero_grad()
            outputs = self.model(input_ids=combined, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            self._optimizer.step()


def get_seq_logprob(
    model: torch.nn.Module,
    prompt_ids: Sequence[torch.Tensor],
    target_ids: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    attention_masks: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    if len(prompt_ids) != len(target_ids):
        raise ValueError("prompt_ids and target_ids must have the same length")
    device = next(model.parameters()).device

    combined: List[torch.Tensor] = []
    combined_masks: List[torch.Tensor] = []
    prompt_lens: List[int] = []
    target_lens: List[int] = []
    for idx, (p, t) in enumerate(zip(prompt_ids, target_ids)):
        p = p.to(device)
        t = t.to(device)
        prompt_lens.append(p.size(1))
        target_lens.append(t.size(1))
        combined.append(torch.cat([p, t], dim=1))
        if attention_masks is not None and idx < len(attention_masks):
            combined_masks.append(
                torch.cat([
                    attention_masks[idx].to(device),
                    torch.ones_like(t, device=device),
                ], dim=1)
            )
        else:
            combined_masks.append(torch.ones_like(combined[-1], device=device))

    max_len = max(t.size(1) for t in combined)
    padded = []
    padded_mask = []
    for seq, mask in zip(combined, combined_masks):
        pad_len = max_len - seq.size(1)
        if pad_len > 0:
            pad_tensor = torch.full((seq.size(0), pad_len), pad_token_id, device=device)
            pad_mask = torch.zeros((mask.size(0), pad_len), device=device, dtype=mask.dtype)
            padded.append(torch.cat([pad_tensor, seq], dim=1))
            padded_mask.append(torch.cat([pad_mask, mask], dim=1))
        else:
            padded.append(seq)
            padded_mask.append(mask)

    padded = torch.stack(padded, dim=0)
    padded_mask = torch.stack(padded_mask, dim=0)

    outputs = model(input_ids=padded, attention_mask=padded_mask)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    shifted_labels = padded[:, 1:]
    token_log_probs = log_probs.gather(2, shifted_labels.unsqueeze(-1)).squeeze(-1)

    target_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    for idx, (pl, tl) in enumerate(zip(prompt_lens, target_lens)):
        start = max(pl - 1, 0)
        end = start + tl
        target_mask[idx, start:end] = True

    valid_mask = target_mask & (padded_mask[:, 1:] > 0)
    masked_log_probs = token_log_probs * valid_mask.float()
    return masked_log_probs.sum(dim=1)


__all__ = ["EphemeralShadow", "get_seq_logprob"]
