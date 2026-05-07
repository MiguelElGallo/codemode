from __future__ import annotations

import pytest

from codemode_probe.executor_factory import (
    available_executor_ids,
    build_executor,
    normalize_executor_id,
)
from codemode_probe.executors import CodeModeSyntheticScriptedExecutor
from codemode_probe.models import ProbeTask, ToolShape
from codemode_probe.oracle import rank_candidates
from codemode_probe.provider import ProviderTurnResponse
from codemode_probe.reporting import summarize_paired_deltas
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.suite import BenchmarkSuiteConfig, run_benchmark_suite
from codemode_probe.workload import generate_candidates, make_probe_task


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
        "code_mode_synthetic_scripted",
    )
    assert normalize_executor_id("deterministic_oracle") == "deterministic_oracle_client"
    assert normalize_executor_id("in_process") == "in_process_tool_oracle"
    assert normalize_executor_id("direct_mcp") == "direct_mcp_tool_oracle"
    assert normalize_executor_id("direct_agent") == "direct_mcp_agent_parallel"
    assert normalize_executor_id("code_mode") == "code_mode_synthetic_scripted"
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


def test_build_executor_can_inject_provider_client_for_direct_agent() -> None:
    class FakeProviderClient:
        provider_name = "fake-provider"
        model_name = "fake-model"

        async def run_provider_turn(self, request):
            return ProviderTurnResponse(
                final_answer={"task_id": request.rendered_prompt.task_id, "candidates": []},
                stop_reason="final_answer",
                raw={"request_id": "fake-request"},
            )

    task = tiny_task(tool_shape=ToolShape.BATCH)

    result = BenchmarkRunner(
        build_executor(
            "direct_agent",
            task,
            provider_client=FakeProviderClient(),
        )
    ).run_task(task)

    assert result.arm_name == "direct_mcp_agent_parallel"
    assert result.execution.error is None
    assert result.execution.raw["model_turns"][0]["provider_name"] == "fake-provider"
    assert result.execution.raw["model_turns"][0]["model_name"] == "fake-model"
    assert result.execution.raw["model_turns"][0]["provider_raw"] == {
        "request_id": "fake-request"
    }


@pytest.mark.parametrize("tool_shape", [ToolShape.SCALAR, ToolShape.BATCH])
def test_code_mode_synthetic_scripted_executor_matches_oracle_with_hidden_tool_outputs(
    tool_shape: ToolShape,
) -> None:
    task = tiny_task(tool_shape=tool_shape)

    result = BenchmarkRunner(CodeModeSyntheticScriptedExecutor()).run_task(task)

    assert result.arm_name == "code_mode_synthetic_scripted"
    assert result.execution.answer == rank_candidates(
        task.id,
        generate_candidates(task.workload),
        task.workload.top_k,
    )
    assert result.score.schema_valid is True
    assert result.score.top_k_overlap == 1.0
    assert result.score.precision_at_k == 1.0
    assert result.score.recall_at_k == 1.0
    assert result.score.ndcg_at_k == 1.0
    assert result.execution.trace.nested_tool_call_count == len(result.execution.tool_calls)
    assert result.execution.tool_calls
    assert all(not call.model_visible for call in result.execution.tool_calls)
    assert result.execution.usage.model_visible_bytes_total == 0
    assert result.execution.usage.tool_response_bytes_total > 0
    assert result.execution.raw == {
        "candidate_count": task.workload.candidate_count,
        "code_mode": "synthetic_scripted",
        "tool_outputs_model_visible": False,
    }


@pytest.mark.parametrize(
    ("executor_id", "expected_arm"),
    [
        ("code_mode", "code_mode_synthetic_scripted"),
        ("code_mode_synthetic_scripted", "code_mode_synthetic_scripted"),
    ],
)
def test_build_executor_supports_code_mode_alias_and_canonical_id(
    executor_id: str,
    expected_arm: str,
) -> None:
    task = tiny_task(tool_shape=ToolShape.BATCH)

    result = BenchmarkRunner(build_executor(executor_id, task)).run_task(task)

    assert result.arm_name == expected_arm
    assert result.execution.raw["code_mode"] == "synthetic_scripted"
    assert result.execution.usage.model_visible_bytes_total == 0
    assert result.execution.usage.tool_response_bytes_total > 0
    assert result.score.top_k_overlap == 1.0


def test_suite_runs_code_mode_beside_direct_agent_and_paired_deltas_show_visible_byte_suppression() -> None:
    task = tiny_task(tool_shape=ToolShape.BATCH)

    results = run_benchmark_suite(
        [task],
        BenchmarkSuiteConfig(arms=("direct_agent", "code_mode")),
    )

    assert [result.arm_name for result in results] == [
        "direct_mcp_agent_parallel",
        "code_mode_synthetic_scripted",
    ]
    assert all(result.score.top_k_overlap == 1.0 for result in results)

    direct_agent = results[0]
    code_mode = results[1]
    assert direct_agent.execution.usage.model_visible_bytes_total is not None
    assert code_mode.execution.usage.model_visible_bytes_total == 0
    assert (
        code_mode.execution.usage.model_visible_bytes_total
        <= direct_agent.execution.usage.model_visible_bytes_total
    )

    paired_deltas = summarize_paired_deltas(
        results,
        baseline_arm="direct_mcp_agent_parallel",
    )

    assert len(paired_deltas) == 1
    assert paired_deltas[0]["comparison_arm"] == "code_mode_synthetic_scripted"
    assert paired_deltas[0]["delta_model_visible_bytes"] <= 0
    assert paired_deltas[0]["payload_visible_ratio_comparison"] == 0


def test_build_executor_rejects_unknown_executor_id() -> None:
    with pytest.raises(ValueError, match="unknown executor id: not-an-arm"):
        build_executor("not-an-arm", tiny_task())
