from __future__ import annotations

import hashlib

from codemode_probe.models import StructuredAnswer, ToolSpec
from codemode_probe.prompts import render_prompt
from codemode_probe.synthetic_tools import SYNTHETIC_TOOL_SPECS, canonical_json_bytes
from codemode_probe.workload import make_probe_task


def tiny_task(task_id: str = "task-1"):
    return make_probe_task(
        task_id,
        seed=17,
        shard_count=2,
        candidates_per_shard=2,
        payload_bytes=4,
        relevant_fraction=0.5,
        top_k=2,
    )


def test_render_prompt_canonical_hash_is_deterministic_and_canonical() -> None:
    task = tiny_task()
    tool_specs = tuple(
        ToolSpec.model_validate(tool.model_dump(mode="json"))
        for tool in SYNTHETIC_TOOL_SPECS
    )

    first = render_prompt(task, tool_specs=tool_specs)
    second = render_prompt(task, tool_specs=tool_specs)

    payload = {
        "task_id": task.id,
        "prompt": task.prompt,
        "task_parameters": {
            "seed": task.workload.seed,
            "task_family": task.workload.task_family.value,
            "tool_shape": task.workload.tool_shape.value,
            "shard_count": task.workload.shard_count,
            "candidates_per_shard": task.workload.candidates_per_shard,
            "payload_bytes": task.workload.payload_bytes,
            "relevant_fraction": task.workload.relevant_fraction,
            "top_k": task.workload.top_k,
        },
        "tool_specs": [tool.model_dump(mode="json") for tool in tool_specs],
        "answer_schema": StructuredAnswer.model_json_schema(),
        "max_tool_calls": task.max_tool_calls,
        "timeout_seconds": task.timeout_seconds,
    }
    expected_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    assert first.canonical_hash == second.canonical_hash == expected_hash
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_render_prompt_hash_changes_when_prompt_tool_specs_or_limits_change() -> None:
    task = tiny_task()
    base_hash = render_prompt(task).canonical_hash

    changed_prompt = task.model_copy(update={"prompt": task.prompt + "\nUse a terse answer."})
    changed_tools = (
        *SYNTHETIC_TOOL_SPECS,
        ToolSpec(name="inspect_candidate_history", description="Return historical review data."),
    )
    changed_max_tool_calls = task.model_copy(update={"max_tool_calls": task.max_tool_calls + 1})
    changed_timeout = task.model_copy(update={"timeout_seconds": task.timeout_seconds + 1.0})

    assert render_prompt(changed_prompt).canonical_hash != base_hash
    assert render_prompt(task, tool_specs=changed_tools).canonical_hash != base_hash
    assert render_prompt(changed_max_tool_calls).canonical_hash != base_hash
    assert render_prompt(changed_timeout).canonical_hash != base_hash


def test_render_prompt_includes_answer_schema_and_tool_specs() -> None:
    task = tiny_task()
    tool_specs = (
        ToolSpec(name="visible_lookup", description="Visible result.", model_visible=True),
        ToolSpec(name="hidden_lookup", description="Hidden result.", model_visible=False),
    )

    rendered = render_prompt(task, tool_specs=tool_specs)
    dumped = rendered.model_dump(mode="json")

    assert dumped["answer_schema"] == StructuredAnswer.model_json_schema()
    assert dumped["tool_specs"] == [
        {"name": "visible_lookup", "description": "Visible result.", "model_visible": True},
        {"name": "hidden_lookup", "description": "Hidden result.", "model_visible": False},
    ]
    assert dumped["max_tool_calls"] == task.max_tool_calls
    assert dumped["timeout_seconds"] == task.timeout_seconds
