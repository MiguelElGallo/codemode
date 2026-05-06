from __future__ import annotations

import asyncio
import json
from collections import Counter

from codemode_probe.executors import InProcessToolOracleExecutor
from codemode_probe.models import Candidate, ProbeTask, ToolShape
from codemode_probe.oracle import rank_candidates
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.synthetic_tools import InProcessSyntheticTools, run_tool_oracle
from codemode_probe.workload import generate_candidates, make_probe_task


def tiny_task(
    *,
    task_id: str = "tool-task",
    tool_shape: ToolShape = ToolShape.SCALAR,
    payload_bytes: int = 16,
) -> ProbeTask:
    return make_probe_task(
        task_id,
        seed=17,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=payload_bytes,
        relevant_fraction=0.5,
        top_k=2,
    )


def json_byte_count(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def candidate_summary(candidate: Candidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "shard_id": candidate.shard_id,
        "title": candidate.title,
        "category": candidate.category,
    }


def test_synthetic_tools_search_and_fetch_response_contracts() -> None:
    candidates = generate_candidates(tiny_task().workload)
    tools = InProcessSyntheticTools(candidates)

    async def exercise_tools() -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
        search_response = await tools.search_shard(0, limit=2)
        single_fetch_response = await tools.fetch_candidate(candidates[0].id)
        batch_fetch_response = await tools.fetch_candidates(
            [candidates[1].id, candidates[3].id]
        )
        return search_response, single_fetch_response, batch_fetch_response

    search_response, single_fetch_response, batch_fetch_response = asyncio.run(exercise_tools())

    assert search_response == [candidate_summary(candidate) for candidate in candidates[:2]]
    assert set(search_response[0]) == {"id", "shard_id", "title", "category"}
    assert "payload" not in search_response[0]

    assert single_fetch_response == candidates[0].model_dump(mode="json")
    assert single_fetch_response["payload"] == candidates[0].payload
    assert batch_fetch_response == [
        candidates[1].model_dump(mode="json"),
        candidates[3].model_dump(mode="json"),
    ]

    assert [call.tool_name for call in tools.calls] == [
        "search_shard",
        "fetch_candidate",
        "fetch_candidates",
    ]
    assert [call.item_count for call in tools.calls] == [2, 1, 2]
    assert all(call.model_visible for call in tools.calls)
    assert [call.response_bytes for call in tools.calls] == [
        json_byte_count(search_response),
        json_byte_count(single_fetch_response),
        json_byte_count(batch_fetch_response),
    ]


def test_run_tool_oracle_scalar_and_batch_tool_call_counts() -> None:
    scalar_task = tiny_task(tool_shape=ToolShape.SCALAR)
    batch_task = tiny_task(tool_shape=ToolShape.BATCH)

    scalar_result = run_tool_oracle(
        scalar_task,
        InProcessSyntheticTools.from_task(scalar_task),
    )
    batch_result = run_tool_oracle(
        batch_task,
        InProcessSyntheticTools.from_task(batch_task),
    )

    assert scalar_result.usage.tool_calls == (
        scalar_task.workload.shard_count + scalar_task.workload.candidate_count
    )
    assert Counter(call.tool_name for call in scalar_result.tool_calls) == Counter(
        {
            "search_shard": scalar_task.workload.shard_count,
            "fetch_candidate": scalar_task.workload.candidate_count,
        }
    )

    assert batch_result.usage.tool_calls == batch_task.workload.shard_count + 1
    assert Counter(call.tool_name for call in batch_result.tool_calls) == Counter(
        {
            "search_shard": batch_task.workload.shard_count,
            "fetch_candidates": 1,
        }
    )
    assert batch_result.tool_calls[-1].item_count == batch_task.workload.candidate_count


def test_run_tool_oracle_accounts_payload_and_model_visible_bytes() -> None:
    task = tiny_task(tool_shape=ToolShape.BATCH, payload_bytes=64)
    candidates = generate_candidates(task.workload)
    result = run_tool_oracle(task, InProcessSyntheticTools(candidates))

    expected_search_bytes = sum(
        json_byte_count(
            [
                candidate_summary(candidate)
                for candidate in candidates
                if candidate.shard_id == shard_id
            ]
        )
        for shard_id in range(task.workload.shard_count)
    )
    expected_fetch_bytes = json_byte_count(
        [candidate.model_dump(mode="json") for candidate in candidates]
    )
    expected_total_bytes = expected_search_bytes + expected_fetch_bytes

    assert result.usage.tool_response_bytes_total == expected_total_bytes
    assert result.usage.model_visible_bytes_total == expected_total_bytes
    assert result.usage.tool_response_bytes_total == sum(
        call.response_bytes for call in result.tool_calls
    )
    assert result.usage.model_visible_bytes_total == sum(
        call.response_bytes for call in result.tool_calls if call.model_visible
    )
    assert result.usage.tool_response_bytes_total > expected_search_bytes


def test_in_process_tool_oracle_executor_scores_perfectly_through_benchmark_runner() -> None:
    task = tiny_task(task_id="runner-tool-task", tool_shape=ToolShape.BATCH)

    result = BenchmarkRunner(InProcessToolOracleExecutor()).run_task(task)

    assert result.arm_name == "in_process_tool_oracle"
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
