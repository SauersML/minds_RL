from __future__ import annotations

from typing import Any, Sequence


class ChatRenderer:
    """Render chat prompts and derive reasonable stop sequences."""

    def __init__(self, tokenizer: Any, *, base_model: str | None = None) -> None:
        self.tokenizer = tokenizer
        self.base_model = base_model or ""

    def build_generation_prompt(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            # Fallback to naive encoding if the tokenizer cannot handle chat templates.
            if hasattr(self.tokenizer, "encode"):
                prompt_ids = self.tokenizer.encode(prompt)
            else:  # pragma: no cover - extremely defensive
                prompt_ids = [hash(prompt) % 100]
        return list(prompt_ids)

    def get_stop_sequences(self) -> list[str]:
        stops: list[str] = []
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos_token, str):
            stops.append(eos_token)

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

        # Deduplicate while preserving order
        deduped: list[str] = []
        for value in stops:
            if value and value not in deduped:
                deduped.append(value)
        return deduped


__all__ = ["ChatRenderer"]
