from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


class Parser:
    """Base parser interface used by environments."""

    def parse(self, text: str) -> dict[str, Any] | None:  # pragma: no cover - base impl
        return None

    def parse_answer(self, completion: list[Mapping[str, Any]]) -> str | None:  # pragma: no cover - base impl
        if not completion:
            return None
        content = completion[-1].get("content")
        return content if isinstance(content, str) else None


@dataclass
class Rubric:
    funcs: Sequence[Callable[..., float]]
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = [1.0 / len(self.funcs)] * len(self.funcs)


class SingleTurnEnv:
    """Minimal single-turn environment for running verifiers locally."""

    def __init__(
        self,
        *,
        dataset: Sequence[Mapping[str, Any]],
        parser: Parser,
        rubric: Rubric,
        system_prompt: str = "",
        few_shot: Iterable[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.dataset = dataset
        self.parser = parser
        self.rubric = rubric
        self.system_prompt = system_prompt
        self.few_shot = list(few_shot) if few_shot is not None else []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Mapping[str, Any]:
        return self.dataset[idx]


from .trainer import Trainer  # noqa: E402

__all__ = ["Parser", "Rubric", "SingleTurnEnv", "Trainer"]
