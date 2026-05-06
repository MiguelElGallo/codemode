from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.executor_factory import available_executor_ids, build_executor, normalize_executor_id
from codemode_probe.models import ArmResult, CachePolicy, CacheState, ExecutionContext, ProbeTask
from codemode_probe.runner import BenchmarkRunner

ArmOrder = Literal["fixed", "randomized"]


class BenchmarkSuiteConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    arms: tuple[str, ...] = ("deterministic_oracle_client",)
    repetitions: int = Field(default=1, ge=1)
    arm_order: ArmOrder = "fixed"
    random_seed: int = 1
    paired_baseline_arm: str = "direct_mcp_agent_parallel"
    cache_policy: CachePolicy = CachePolicy.UNSPECIFIED
    cache_namespace: str | None = None
    cache_warmup_repetitions: int = Field(default=0, ge=0)

    @property
    def normalized_arms(self) -> tuple[str, ...]:
        return tuple(normalize_executor_id(arm) for arm in self.arms)

    @property
    def normalized_paired_baseline_arm(self) -> str:
        return normalize_executor_id(self.paired_baseline_arm)

    def validate_arms(self) -> None:
        valid = set(available_executor_ids())
        unknown = [arm for arm in self.normalized_arms if arm not in valid]
        if unknown:
            raise ValueError(f"unknown executor id: {unknown[0]}")
        if self.normalized_paired_baseline_arm not in valid:
            raise ValueError(
                f"unknown paired baseline executor id: {self.paired_baseline_arm}"
            )


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
            trial_id = f"{task.id}:rep-{repetition}"
            arm_order = tuple(arms)
            cache_state = _cache_state_for_repetition(repetition, config)
            context = ExecutionContext(
                cache_policy=config.cache_policy,
                cache_state=cache_state,
                cache_namespace=config.cache_namespace,
                cache_warmup_run=cache_state == CacheState.WARMUP,
            )
            for arm_order_index, arm in enumerate(arms):
                executor = build_executor(arm, task)
                result = BenchmarkRunner(executor).run_task(
                    task,
                    repetition=repetition,
                    context=context,
                )
                results.append(
                    result.model_copy(
                        update={
                            "trial_id": trial_id,
                            "arm_order_index": arm_order_index,
                            "arm_order": arm_order,
                            "cache_policy": config.cache_policy,
                            "cache_state": cache_state,
                            "cache_namespace": config.cache_namespace,
                            "cache_warmup_run": cache_state == CacheState.WARMUP,
                        }
                    )
                )

    return results


def _cache_state_for_repetition(
    repetition: int,
    config: BenchmarkSuiteConfig,
) -> CacheState:
    if config.cache_policy == CachePolicy.UNSPECIFIED:
        return CacheState.UNSPECIFIED
    if config.cache_policy == CachePolicy.DISABLED:
        return CacheState.DISABLED
    if config.cache_policy == CachePolicy.COLD:
        return CacheState.COLD
    if config.cache_policy == CachePolicy.WARM:
        if repetition <= config.cache_warmup_repetitions:
            return CacheState.WARMUP
        return CacheState.WARM
    if config.cache_policy == CachePolicy.COLD_THEN_WARM:
        if repetition == 1:
            return CacheState.COLD
        if repetition <= config.cache_warmup_repetitions + 1:
            return CacheState.WARMUP
        return CacheState.WARM
    return CacheState.UNSPECIFIED
