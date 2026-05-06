from __future__ import annotations

import pytest

from codemode_probe.executor_factory import (
    available_executor_ids,
    build_executor,
    normalize_executor_id,
)
from codemode_probe.models import ProbeTask, ToolShape
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.workload import make_probe_task


def tiny_task(*, tool_shape: ToolShape = ToolShape.BATCH) -> ProbeTask:
    return make_probe_task(
        "factory-task",
        seed=23,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        relevant_fraction=0.5,
        top_k=2,
    )


def test_available_executor_ids_and_aliases_cover_cli_friendly_names() -> None:
    assert available_executor_ids() == (
        "deterministic_oracle_client",
        "in_process_tool_oracle",
        "direct_mcp_tool_oracle",
        "direct_mcp_agent_parallel",
    )
    assert normalize_executor_id("deterministic_oracle") == "deterministic_oracle_client"
    assert normalize_executor_id("in_process") == "in_process_tool_oracle"
    assert normalize_executor_id("direct_mcp") == "direct_mcp_tool_oracle"
    assert normalize_executor_id("direct_agent") == "direct_mcp_agent_parallel"
    assert normalize_executor_id("direct_mcp_agent_parallel") == "direct_mcp_agent_parallel"


@pytest.mark.parametrize(
    ("executor_id", "expected_arm", "expected_tool_calls"),
    [
        ("direct_mcp_tool_oracle", "direct_mcp_tool_oracle", 3),
        ("direct_mcp_agent_parallel", "direct_mcp_agent_parallel", 3),
        ("direct_mcp", "direct_mcp_tool_oracle", 3),
        ("direct_agent", "direct_mcp_agent_parallel", 3),
    ],
)
def test_build_executor_returns_fresh_direct_mcp_state_for_each_build(
    executor_id: str,
    expected_arm: str,
    expected_tool_calls: int,
) -> None:
    task = tiny_task(tool_shape=ToolShape.BATCH)

    first = BenchmarkRunner(build_executor(executor_id, task)).run_task(task)
    second = BenchmarkRunner(build_executor(executor_id, task)).run_task(task)

    assert first.arm_name == expected_arm
    assert second.arm_name == expected_arm
    assert first.execution.usage.tool_calls == expected_tool_calls
    assert second.execution.usage.tool_calls == expected_tool_calls
    assert len(first.execution.tool_calls) == expected_tool_calls
    assert len(second.execution.tool_calls) == expected_tool_calls
    assert first.score.top_k_overlap == 1.0
    assert second.score.top_k_overlap == 1.0


def test_build_executor_rejects_unknown_executor_id() -> None:
    with pytest.raises(ValueError, match="unknown executor id: not-an-arm"):
        build_executor("not-an-arm", tiny_task())
