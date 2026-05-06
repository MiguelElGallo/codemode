from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.models import ProbeTask, TaskFamily, ToolShape
from codemode_probe.workload import make_probe_task

CasePreset = Literal["smoke", "orchestration_matrix"]


class CaseMatrixConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    preset: CasePreset = "smoke"
    base_seed: int = 1
    payload_bytes: int = Field(default=256, ge=0)
    relevant_fraction: float = Field(default=0.2, ge=0.0, le=1.0)


def generate_case_tasks(config: CaseMatrixConfig) -> list[ProbeTask]:
    if config.preset == "smoke":
        return [_task(config, "smoke_single_lookup", TaskFamily.SINGLE_LOOKUP, ToolShape.SCALAR, 1, 1, 1)]
    if config.preset == "orchestration_matrix":
        return [
            _task(config, "single_lookup", TaskFamily.SINGLE_LOOKUP, ToolShape.SCALAR, 1, 1, 1),
            _task(
                config,
                "small_parallel_lookup",
                TaskFamily.SMALL_PARALLEL_LOOKUP,
                ToolShape.SCALAR,
                4,
                1,
                2,
            ),
            _task(
                config,
                "scalar_large_fanout_25",
                TaskFamily.SCALAR_LARGE_FANOUT,
                ToolShape.SCALAR,
                5,
                5,
                5,
            ),
            _task(
                config,
                "scalar_large_fanout_100",
                TaskFamily.SCALAR_LARGE_FANOUT,
                ToolShape.SCALAR,
                10,
                10,
                5,
            ),
            _task(
                config,
                "batch_large_fanout_100",
                TaskFamily.BATCH_LARGE_FANOUT,
                ToolShape.BATCH,
                10,
                10,
                5,
            ),
            _task(
                config,
                "deep_branching_filter_rank",
                TaskFamily.DEEP_BRANCHING_FILTER_RANK,
                ToolShape.SCALAR,
                8,
                12,
                5,
            ),
        ]
    raise ValueError(f"unknown case preset: {config.preset}")


def _task(
    config: CaseMatrixConfig,
    suffix: str,
    task_family: TaskFamily,
    tool_shape: ToolShape,
    shard_count: int,
    candidates_per_shard: int,
    top_k: int,
) -> ProbeTask:
    seed = config.base_seed + len(suffix)
    task = make_probe_task(
        f"{config.preset}_{suffix}",
        seed=seed,
        task_family=task_family,
        tool_shape=tool_shape,
        shard_count=shard_count,
        candidates_per_shard=candidates_per_shard,
        payload_bytes=config.payload_bytes,
        relevant_fraction=config.relevant_fraction,
        top_k=top_k,
    )
    required_tool_calls = shard_count + (
        1 if tool_shape == ToolShape.BATCH else shard_count * candidates_per_shard
    )
    return task.model_copy(update={"max_tool_calls": max(task.max_tool_calls, required_tool_calls)})
