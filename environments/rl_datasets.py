from __future__ import annotations

import random
import itertools
from dataclasses import dataclass
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


@dataclass
class _CompositeComponent:
    name: str
    dataset: RLDataset
    weight: float


class _TaggedEnvGroupBuilder(EnvGroupBuilder):
    def __init__(self, base: EnvGroupBuilder, tag: str) -> None:
        self._base = base
        self._tag = tag

    async def make_envs(self) -> Sequence[Env]:  # type: ignore[override]
        return await self._base.make_envs()

    async def compute_group_rewards(  # type: ignore[override]
        self, trajectory_group: list[Any], env_group: Sequence[Env]
    ) -> list[tuple[float, Mapping[str, Any]]]:
        return await self._base.compute_group_rewards(trajectory_group, env_group)

    def logging_tags(self) -> list[str]:
        base_tags = []
        try:
            base_tags = self._base.logging_tags()
        except Exception:
            base_tags = []
        return [self._tag, *base_tags]


class CompositeRLDataset(RLDataset):
    def __init__(
        self,
        components: Sequence[_CompositeComponent],
        *,
        groups_per_batch: int,
        total_batches: int,
        seed: int | None = None,
    ) -> None:
        if not components:
            raise ValueError("CompositeRLDataset requires at least one component")

        weight_sum = sum(c.weight for c in components)
        if weight_sum <= 0:
            raise ValueError("Composite component weights must sum to a positive value")

        self._components = list(components)
        self._groups_per_batch = max(1, int(groups_per_batch))
        self._total_batches = max(1, int(total_batches))
        self._seed = seed

        self._normalized_weights = [c.weight / weight_sum for c in self._components]
        self._queues: dict[str, list[EnvGroupBuilder]] = {c.name: [] for c in self._components}
        self._indices: dict[str, int] = {c.name: 0 for c in self._components}

    def __len__(self) -> int:  # type: ignore[override]
        return self._total_batches

    def _rng(self, index: int) -> random.Random:
        seed = self._seed if self._seed is not None else 0
        return random.Random(seed + index)

    def _choose_component(self, rng: random.Random) -> _CompositeComponent:
        return rng.choices(self._components, weights=self._normalized_weights, k=1)[0]

    def _refresh_queue(self, component: _CompositeComponent) -> None:
        queue = self._queues[component.name]
        if queue:
            return

        dataset_len = len(component.dataset)
        if dataset_len == 0:
            return
        batch_idx = self._indices[component.name] % dataset_len
        self._indices[component.name] += 1
        try:
            new_builders = component.dataset.get_batch(batch_idx)
        except Exception:
            new_builders = []

        queue.extend([_TaggedEnvGroupBuilder(b, component.name) for b in new_builders])

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:  # type: ignore[override]
        rng = self._rng(index)
        batch: list[EnvGroupBuilder] = []

        for _ in range(self._groups_per_batch):
            component = self._choose_component(rng)
            self._refresh_queue(component)
            queue = self._queues[component.name]
            if not queue:
                # If the queue is still empty (e.g., dataset failure), fall back to the next component
                fallback = self._choose_component(rng)
                self._refresh_queue(fallback)
                queue = self._queues[fallback.name]
                if not queue:
                    continue

            batch.append(queue.pop())

        return batch


class _RoundRobinDataset(RLDataset):
    def __init__(self, components: Sequence[_CompositeComponent]) -> None:
        self._components = [c for c in components if len(c.dataset) > 0]
        self._offsets: list[int] = []
        running_total = 0
        for comp in self._components:
            self._offsets.append(running_total)
            running_total += len(comp.dataset)
        self._total = running_total

    def __len__(self) -> int:  # type: ignore[override]
        return self._total

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:  # type: ignore[override]
        if not self._components or index >= self._total:
            return []

        remaining = index
        for comp, offset in zip(self._components, self._offsets, strict=False):
            local_index = remaining - offset
            if 0 <= local_index < len(comp.dataset):
                builders = comp.dataset.get_batch(local_index)
                return [_TaggedEnvGroupBuilder(b, comp.name) for b in builders]
            remaining -= len(comp.dataset)
        return []


@chz.chz
class CompositeRLDatasetBuilder(RLDatasetBuilder):
    components: Sequence[tuple[str, RLDatasetBuilder, float]]
    groups_per_batch: int
    total_batches: int
    seed: int | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        if not self.components:
            return _StaticEnvDataset([], batch_size=1, group_size=1), None

        train_components: list[_CompositeComponent] = []
        eval_components: list[_CompositeComponent] = []

        for name, builder, weight in self.components:
            train_ds, eval_ds = await builder()
            train_components.append(_CompositeComponent(name, train_ds, weight))
            if eval_ds is not None:
                eval_components.append(_CompositeComponent(name, eval_ds, weight))

        train_dataset = CompositeRLDataset(
            train_components,
            groups_per_batch=self.groups_per_batch,
            total_batches=self.total_batches,
            seed=self.seed,
        )

        eval_dataset: RLDataset | None = None
        if eval_components:
            eval_dataset = _RoundRobinDataset(eval_components)

        return train_dataset, eval_dataset


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

        # Pass renderer to builder so it can instantiate Envs correctly
        dataset_builder = GradientProphetDatasetBuilder(seed=self.seed, renderer=self.renderer)
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
    "CompositeRLDatasetBuilder",
    "GradientIntuitionRLDatasetBuilder",
    "GradientProphetRLDatasetBuilder",
]
