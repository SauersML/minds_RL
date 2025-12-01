from __future__ import annotations

from typing import Any, Sequence

from .tokenizers import get_tokenizer


class DefaultRenderer:
    def __init__(self, base_model: str):
        self.base_model = base_model
        self.tokenizer = get_tokenizer(base_model)

    def build_generation_prompt(self, prompt: str) -> Sequence[int]:
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            if hasattr(self.tokenizer, "encode"):
                prompt_ids = self.tokenizer.encode(prompt)
            else:  # pragma: no cover
                prompt_ids = [hash(prompt) % 100]
        return list(prompt_ids)

    def get_stop_sequences(self) -> list[str]:
        stops: list[str] = []
        eos = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos, str):
            stops.append(eos)

        chat_template = getattr(self.tokenizer, "chat_template", "") or ""
        candidates: Sequence[str] = (
            "<|im_end|>",
            "<|im_start|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|user|>",
        )
        for marker in candidates:
            if marker in chat_template:
                stops.append(marker)

        if "llama" in self.base_model.lower():
            stops.append("<|eot_id|>")
        if "qwen" in self.base_model.lower():
            stops.append("<|im_end|>")

        deduped: list[str] = []
        for val in stops:
            if val and val not in deduped:
                deduped.append(val)
        return deduped


def get_renderer(base_model: str, **_: Any):
    return DefaultRenderer(base_model)


__all__ = ["get_renderer", "DefaultRenderer"]
