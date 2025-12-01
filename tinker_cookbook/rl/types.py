from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class Env:
    """Minimal Env base class placeholder for CI."""

    def reset(self):  # pragma: no cover - optional
        return None

    async def step(self, action: Any, **kwargs: Any):  # pragma: no cover - must be overridden
        raise NotImplementedError


@dataclass
class StepResult:  # type: ignore[misc]
    observation: Any
    reward: float
    done: bool
    info: Mapping[str, Any] | None = None


__all__ = ["Env", "StepResult"]
