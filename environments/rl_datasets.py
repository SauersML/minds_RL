from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Sequence

import chz
import tinker
from tinker_cookbook.rl.train import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.types import Env, EnvGroupBuilder

from environments.gradient_intuition.gradient_intuition import GradientIntuitionBuilder
from environments.gradient_prophet.gradient_prophet import GradientProphetDatasetBuilder, GradientProphetEnv


class _SingleEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self, factory: Callable[[], Env], *, group_size: int, tags: Sequence[str] | None = None
    ) -> None:
        self._factory = factory
        self._group_size = max(1, int(group_size))
        self._tags = list(tags or [])

    async def make_envs(self) -> Sequence[Env]:  # type: ignore[override]
        return [self._factory() for _ in range(self._group_size)]

    def logging_tags(self) -> list[str]:
        return list(self._tags)


class _StaticEnvDataset(RLDataset):
    def __init__(
        self,
        factories: Sequence[Callable[[], Env]],
        *,
        batch_size: int,
        group_size: int,
    ) -> None:
        self._factories = list(factories)
        self._batch_size = max(1, int(batch_size))
        self._group_size = max(1, int(group_size))

    def __len__(self) -> int:  # type: ignore[override]
        return (len(self._factories) + self._batch_size - 1) // self._batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:  # type: ignore[override]
        start = index * self._batch_size
        end = min(len(self._factories), start + self._batch_size)
        batch_factories = self._factories[start:end]
        return [
            _SingleEnvGroupBuilder(factory, group_size=self._group_size)
            for factory in batch_factories
        ]


@chz.chz
class GradientProphetRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    batch_size: int
    group_size: int
    renderer: Any
    base_url: str | None = None
    seed: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = await service_client.create_sampling_client_async(base_model=self.model_name)

        dataset_builder = GradientProphetDatasetBuilder(seed=self.seed)
        envs = dataset_builder.build(sampling_client)

        factories = [
            (
                lambda sample=env.sample, seed=self.seed: GradientProphetEnv(
                    sample,
                    sampling_client,
                    renderer=self.renderer,
                    seed=seed,
                )
            )
            for env in envs
        ]
        dataset = _StaticEnvDataset(
            factories, batch_size=self.batch_size, group_size=self.group_size
        )
        return dataset, None


@chz.chz
class GradientIntuitionRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    inner_env_id: str
    inner_env_args: Mapping[str, Any]
    batch_size: int
    group_size: int
    renderer: Any
    base_url: str | None = None
    alpha: float = 0.3
    seed: int | None = None
    shadow_rank: int = 8
    shadow_learning_rate: float = 1e-4

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        shadow_service_client = tinker.ServiceClient(base_url=self.base_url)
        builder = GradientIntuitionBuilder(
            inner_env_id=self.inner_env_id,
            inner_env_args=self.inner_env_args,
            alpha=self.alpha,
            seed=self.seed,
            shadow_rank=self.shadow_rank,
            shadow_learning_rate=self.shadow_learning_rate,
            renderer=self.renderer,
        )

        envs = builder.build(
            sampling_client=None,
            service_client=shadow_service_client,
            base_model=self.model_name,
            training_client=None,
        )

        if not envs:
            return _StaticEnvDataset([], batch_size=self.batch_size, group_size=self.group_size), None

        env_count = len(envs)
        if env_count < self.batch_size:
            envs = list(itertools.islice(itertools.cycle(envs), self.batch_size))

        factories: list[Callable[[], Env]] = []
        for idx, _ in enumerate(envs):
            env_seed = (self.seed or 0) + idx

            def _factory(i: int = idx, seed_val: int = env_seed) -> Env:
                rebuilt = builder.build(
                    sampling_client=None,
                    service_client=shadow_service_client,
                    base_model=self.model_name,
                    training_client=None,
                )
                env = rebuilt[i % len(rebuilt)]
                if hasattr(env, "rng"):
                    try:
                        env.rng.seed(seed_val)
                    except Exception:
                        pass
                env.renderer = self.renderer  # type: ignore[attr-defined]
                return env

            factories.append(_factory)

        dataset = _StaticEnvDataset(
            factories, batch_size=self.batch_size, group_size=self.group_size
        )
        return dataset, None


__all__ = [
    "GradientIntuitionRLDatasetBuilder",
    "GradientProphetRLDatasetBuilder",
]
