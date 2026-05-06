from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.models import ProbeTask, StructuredAnswer, ToolSpec
from codemode_probe.synthetic_tools import SYNTHETIC_TOOL_SPECS, canonical_json_bytes


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str
    task_parameters: dict[str, object]
    tool_specs: list[ToolSpec]
    answer_schema: dict[str, object]
    max_tool_calls: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    canonical_hash: str


def render_prompt(
    task: ProbeTask,
    *,
    tool_specs: tuple[ToolSpec, ...] = SYNTHETIC_TOOL_SPECS,
) -> RenderedPrompt:
    answer_schema = StructuredAnswer.model_json_schema()
    payload = {
        "task_id": task.id,
        "prompt": task.prompt,
        "task_parameters": _task_parameters(task),
        "tool_specs": [tool.model_dump(mode="json") for tool in tool_specs],
        "answer_schema": answer_schema,
        "max_tool_calls": task.max_tool_calls,
        "timeout_seconds": task.timeout_seconds,
    }
    return RenderedPrompt(
        task_id=task.id,
        prompt=task.prompt,
        task_parameters=_task_parameters(task),
        tool_specs=list(tool_specs),
        answer_schema=answer_schema,
        max_tool_calls=task.max_tool_calls,
        timeout_seconds=task.timeout_seconds,
        canonical_hash=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )


def _task_parameters(task: ProbeTask) -> dict[str, object]:
    return {
        "seed": task.workload.seed,
        "task_family": task.workload.task_family.value,
        "tool_shape": task.workload.tool_shape.value,
        "shard_count": task.workload.shard_count,
        "candidates_per_shard": task.workload.candidates_per_shard,
        "payload_bytes": task.workload.payload_bytes,
        "relevant_fraction": task.workload.relevant_fraction,
        "top_k": task.workload.top_k,
    }
