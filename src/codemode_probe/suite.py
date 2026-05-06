from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.executor_factory import available_executor_ids, build_executor, normalize_executor_id
from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.runner import BenchmarkRunner

ArmOrder = Literal["fixed", "randomized"]


class BenchmarkSuiteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    arms: tuple[str, ...] = ("deterministic_oracle_client",)
    repetitions: int = Field(default=1, ge=1)
    arm_order: ArmOrder = "fixed"
    random_seed: int = 1

    @property
    def normalized_arms(self) -> tuple[str, ...]:
        return tuple(normalize_executor_id(arm) for arm in self.arms)

    def validate_arms(self) -> None:
        valid = set(available_executor_ids())
        unknown = [arm for arm in self.normalized_arms if arm not in valid]
        if unknown:
            raise ValueError(f"unknown executor id: {unknown[0]}")


def run_benchmark_suite(
    tasks: list[ProbeTask],
    config: BenchmarkSuiteConfig,
) -> list[ArmResult]:
    results: list[ArmResult] = []
    rng = random.Random(config.random_seed)
    config.validate_arms()
    normalized_arms = config.normalized_arms

    for repetition in range(1, config.repetitions + 1):
        for task in tasks:
            arms = list(normalized_arms)
            if config.arm_order == "randomized":
                rng.shuffle(arms)
            for arm in arms:
                executor = build_executor(arm, task)
                results.append(BenchmarkRunner(executor).run_task(task, repetition=repetition))

    return results
