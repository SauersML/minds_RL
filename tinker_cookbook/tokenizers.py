from __future__ import annotations

import importlib
import importlib.util
from typing import Any


class _DummyTokenizer:
    def apply_chat_template(self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = True):
        del add_generation_prompt, tokenize
        content = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, dict))
        return [len(content) % 7, len(content) % 5, len(content) % 3]

    def encode(self, text: str):
        return [hash(text) % 101]

    def decode(self, tokens: Any, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(str(t) for t in tokens)


def get_tokenizer(base_model: str, **kwargs: Any):
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        return _DummyTokenizer()

    transformers = importlib.import_module("transformers")
    auto_tokenizer = getattr(transformers, "AutoTokenizer", None)
    if auto_tokenizer is None:
        return _DummyTokenizer()

    try:
        return auto_tokenizer.from_pretrained(base_model, trust_remote_code=True, **kwargs)
    except Exception:
        return _DummyTokenizer()


__all__ = ["get_tokenizer"]
