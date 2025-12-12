from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import tinker
from tinker_cookbook.rl.train import RLDataset, RLDatasetBuilder
from tinker_cookbook.rl.types import Env, EnvGroupBuilder

from environments.gradient_intuition.gradient_intuition import GradientIntuitionBuilder
from environments.gradient_prophet.gradient_prophet import GradientProphetDatasetBuilder, GradientProphetEnv


class _SingleEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, factory: Callable[[], Env], *, tags: Sequence[str] | None = None) -> None:
        self._factory = factory
        self._tags = list(tags or [])

    async def make_envs(self) -> Sequence[Env]:  # type: ignore[override]
        return [self._factory()]

    def logging_tags(self) -> list[str]:
        return list(self._tags)


class _StaticEnvDataset(RLDataset):
    def __init__(self, factories: Sequence[Callable[[], Env]], groups_per_batch: int) -> None:
        self._factories = list(factories)
        self._groups_per_batch = max(1, int(groups_per_batch))

    def __len__(self) -> int:  # type: ignore[override]
        return (len(self._factories) + self._groups_per_batch - 1) // self._groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:  # type: ignore[override]
        start = index * self._groups_per_batch
        end = min(len(self._factories), start + self._groups_per_batch)
        batch_factories = self._factories[start:end]
        return [_SingleEnvGroupBuilder(factory) for factory in batch_factories]


@dataclass
class GradientProphetRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    groups_per_batch: int
    base_url: str | None = None
    seed: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = await service_client.create_sampling_client_from_model_async(self.model_name)

        dataset_builder = GradientProphetDatasetBuilder(seed=self.seed)
        envs = dataset_builder.build(sampling_client)

        factories = [
            (lambda sample=env.sample: GradientProphetEnv(sample, sampling_client))
            for env in envs
        ]
        dataset = _StaticEnvDataset(factories, self.groups_per_batch)
        return dataset, None


@dataclass
class GradientIntuitionRLDatasetBuilder(RLDatasetBuilder):
    model_name: str
    inner_env_id: str
    inner_env_args: Mapping[str, Any]
    groups_per_batch: int
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
        )

        envs = builder.build(
            sampling_client=None,
            service_client=shadow_service_client,
            base_model=self.model_name,
            training_client=None,
        )

        if not envs:
            return _StaticEnvDataset([], self.groups_per_batch), None

        # Repeat environments if the dataset is smaller than a batch so batches are fully populated.
        env_count = len(envs)
        if env_count < self.groups_per_batch:
            envs = list(itertools.islice(itertools.cycle(envs), self.groups_per_batch))
            env_count = len(envs)

        factories: list[Callable[[], Env]] = []
        for idx, _ in enumerate(envs):
            def _factory(i: int = idx) -> Env:
                rebuilt = builder.build(
                    sampling_client=None,
                    service_client=shadow_service_client,
                    base_model=self.model_name,
                    training_client=None,
                )
                return rebuilt[i % len(rebuilt)]

            factories.append(_factory)

        dataset = _StaticEnvDataset(factories, self.groups_per_batch)
        return dataset, None


__all__ = [
    "GradientIntuitionRLDatasetBuilder",
    "GradientProphetRLDatasetBuilder",
]
