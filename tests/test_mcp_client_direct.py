from __future__ import annotations

import asyncio

import pytest

from codemode_probe.executors import (
    DirectMcpToolOracleExecutor,
    InProcessToolOracleExecutor,
)
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import (
    DirectMcpSyntheticToolClient,
    FastMcpInProcessSession,
    extract_structured_result,
)
from codemode_probe.models import ProbeTask, ToolShape
from codemode_probe.oracle import rank_candidates
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.synthetic_tools import canonical_json_bytes
from codemode_probe.workload import generate_candidates, make_probe_task


def tiny_task(
    *,
    task_id: str = "direct-mcp-task",
    tool_shape: ToolShape = ToolShape.SCALAR,
) -> ProbeTask:
    return make_probe_task(
        task_id,
        seed=17,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=16,
        relevant_fraction=0.5,
        top_k=2,
    )


def direct_mcp_client(task: ProbeTask) -> DirectMcpSyntheticToolClient:
    return DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )


def test_extract_structured_result_accepts_supported_mcp_shapes() -> None:
    assert extract_structured_result((object(), {"result": {"ok": True}})) == {"ok": True}
    assert extract_structured_result({"result": [1, 2, 3]}) == [1, 2, 3]
    assert extract_structured_result({"plain": "object"}) == {"plain": "object"}


def test_extract_structured_result_accepts_fastmcp_tuple_object_payload() -> None:
    assert extract_structured_result((object(), {"ok": True})) == {"ok": True}


@pytest.mark.parametrize(
    "raw",
    [
        (object(), {"result": {}}, "extra"),
        ["not", "an", "mcp", "result"],
    ],
)
def test_extract_structured_result_rejects_unsupported_shapes(raw: object) -> None:
    with pytest.raises(TypeError, match="Unsupported MCP tool result shape"):
        extract_structured_result(raw)


def test_direct_mcp_synthetic_tool_client_calls_fastmcp_in_process_session() -> None:
    async def exercise() -> None:
        task = tiny_task()
        candidates = generate_candidates(task.workload)
        client = direct_mcp_client(task)

        search_response = await client.search_shard(0, limit=2)
        single_fetch_response = await client.fetch_candidate(candidates[0].id)
        batch_fetch_response = await client.fetch_candidates(
            [candidates[1].id, candidates[3].id]
        )

        assert search_response == [
            {
                "id": candidate.id,
                "shard_id": candidate.shard_id,
                "title": candidate.title,
                "category": candidate.category,
            }
            for candidate in candidates[:2]
        ]
        assert single_fetch_response == candidates[0].model_dump(mode="json")
        assert batch_fetch_response == [
            candidates[1].model_dump(mode="json"),
            candidates[3].model_dump(mode="json"),
        ]

        assert [call.tool_name for call in client.calls] == [
            "search_shard",
            "fetch_candidate",
            "fetch_candidates",
        ]
        assert [call.item_count for call in client.calls] == [2, 1, 2]
        assert all(call.model_visible for call in client.calls)
        assert [call.response_bytes for call in client.calls] == [
            len(canonical_json_bytes(search_response)),
            len(canonical_json_bytes(single_fetch_response)),
            len(canonical_json_bytes(batch_fetch_response)),
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize("tool_shape", [ToolShape.SCALAR, ToolShape.BATCH])
def test_direct_mcp_tool_oracle_executor_matches_in_process_executor(
    tool_shape: ToolShape,
) -> None:
    task = tiny_task(task_id=f"direct-parity-{tool_shape}", tool_shape=tool_shape)

    direct = DirectMcpToolOracleExecutor(direct_mcp_client(task)).execute(task)
    in_process = InProcessToolOracleExecutor().execute(task)

    assert direct.answer == in_process.answer
    assert direct.usage == in_process.usage
    assert direct.tool_calls == in_process.tool_calls
    assert direct.raw == in_process.raw


def test_benchmark_runner_scores_direct_mcp_tool_oracle_executor() -> None:
    task = tiny_task(task_id="direct-runner-task", tool_shape=ToolShape.BATCH)

    result = BenchmarkRunner(
        DirectMcpToolOracleExecutor(direct_mcp_client(task))
    ).run_task(task)

    assert result.arm_name == "direct_mcp_tool_oracle"
    assert result.execution.answer == rank_candidates(
        task.id,
        generate_candidates(task.workload),
        task.workload.top_k,
    )
    assert result.execution.raw == {"candidate_count": task.workload.candidate_count}
    assert result.score.schema_valid is True
    assert result.score.top_k_overlap == 1.0
    assert result.score.precision_at_k == 1.0
    assert result.score.recall_at_k == 1.0
    assert result.score.ndcg_at_k == 1.0
    assert result.score.failure_reason is None
