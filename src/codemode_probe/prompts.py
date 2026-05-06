from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.models import ProbeTask, StructuredAnswer, ToolSpec
from codemode_probe.synthetic_tools import SYNTHETIC_TOOL_SPECS, canonical_json_bytes


class RenderedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    prompt: str
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
        "tool_specs": [tool.model_dump(mode="json") for tool in tool_specs],
        "answer_schema": answer_schema,
        "max_tool_calls": task.max_tool_calls,
        "timeout_seconds": task.timeout_seconds,
    }
    return RenderedPrompt(
        task_id=task.id,
        prompt=task.prompt,
        tool_specs=list(tool_specs),
        answer_schema=answer_schema,
        max_tool_calls=task.max_tool_calls,
        timeout_seconds=task.timeout_seconds,
        canonical_hash=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    )
