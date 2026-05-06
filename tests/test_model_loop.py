from __future__ import annotations

import asyncio
import time
from collections import Counter

import pytest

from codemode_probe.executors import (
    DirectMcpAgentParallelExecutor,
    InProcessToolOracleExecutor,
)
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient, FastMcpInProcessSession
from codemode_probe.model_loop import DirectMcpAgentExecutor, ScriptedFanoutModelClient
from codemode_probe.models import (
    CachePolicy,
    CacheState,
    ExecutionContext,
    ModelTurnRequest,
    ModelTurnResult,
    FailureCategory,
    NormalizedModelUsage,
    NormalizedToolRequest,
    NormalizedToolResult,
    ProbeTask,
    RankedCandidate,
    StructuredAnswer,
    ToolShape,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.synthetic_tools import InProcessSyntheticTools
from codemode_probe.workload import generate_candidates, make_probe_task


def tiny_task(
    *,
    task_id: str = "model-loop-task",
    tool_shape: ToolShape = ToolShape.SCALAR,
    max_tool_calls: int | None = None,
) -> ProbeTask:
    task = make_probe_task(
        task_id,
        seed=17,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=16,
        relevant_fraction=0.5,
        top_k=2,
    )
    if max_tool_calls is None:
        return task
    return task.model_copy(update={"max_tool_calls": max_tool_calls})


def direct_mcp_client(task: ProbeTask) -> DirectMcpSyntheticToolClient:
    return DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )


def candidate_ids_from_search_results(tool_results: list[object]) -> list[str]:
    candidate_ids: list[str] = []
    for result in tool_results:
        assert isinstance(result, list)
        candidate_ids.extend(str(item["id"]) for item in result)
    return candidate_ids


def test_scripted_fanout_model_client_scalar_turn_sequence() -> None:
    async def exercise() -> None:
        task = tiny_task(tool_shape=ToolShape.SCALAR)
        model = ScriptedFanoutModelClient()
        tools = InProcessSyntheticTools.from_task(task)

        first = await model.run_turn(ModelTurnRequest(task=task, turn_index=1))
        assert [request.name for request in first.tool_requests] == [
            "search_shard",
            "search_shard",
        ]
        assert [request.arguments for request in first.tool_requests] == [
            {"shard_id": 0},
            {"shard_id": 1},
        ]
        assert first.usage == NormalizedModelUsage(input_tokens=100, output_tokens=25)
        assert first.raw == {"scripted_turn": "search_shards"}

        search_results = [
            await tools.search_shard(**request.arguments)
            for request in first.tool_requests
        ]
        second = await model.run_turn(
            ModelTurnRequest(
                task=task,
                turn_index=2,
                tool_results=[
                    NormalizedToolResult(request=request, result=result)
                    for request, result in zip(first.tool_requests, search_results, strict=True)
                ],
            )
        )

        expected_ids = candidate_ids_from_search_results(search_results)
        assert [request.name for request in second.tool_requests] == [
            "fetch_candidate"
        ] * task.workload.candidate_count
        assert [request.arguments for request in second.tool_requests] == [
            {"candidate_id": candidate_id}
            for candidate_id in expected_ids
        ]
        assert second.usage == NormalizedModelUsage(input_tokens=150, output_tokens=35)
        assert second.raw == {"scripted_turn": "fetch_candidates"}

        fetch_results = [
            await tools.fetch_candidate(**request.arguments)
            for request in second.tool_requests
        ]
        final = await model.run_turn(
            ModelTurnRequest(
                task=task,
                turn_index=3,
                tool_results=[
                    NormalizedToolResult(request=request, result=result)
                    for request, result in zip(second.tool_requests, fetch_results, strict=True)
                ],
            )
        )

        assert final.tool_requests == []
        assert final.final_answer == rank_candidates(
            task.id,
            generate_candidates(task.workload),
            task.workload.top_k,
        )
        assert final.usage == NormalizedModelUsage(input_tokens=200, output_tokens=50)
        assert final.raw == {"scripted_turn": "final_answer"}

    asyncio.run(exercise())


def test_scripted_fanout_model_client_batch_second_turn() -> None:
    async def exercise() -> None:
        task = tiny_task(tool_shape=ToolShape.BATCH)
        model = ScriptedFanoutModelClient()
        tools = InProcessSyntheticTools.from_task(task)

        first = await model.run_turn(ModelTurnRequest(task=task, turn_index=1))
        search_results = [
            await tools.search_shard(**request.arguments)
            for request in first.tool_requests
        ]
        second = await model.run_turn(
            ModelTurnRequest(
                task=task,
                turn_index=2,
                tool_results=[
                    NormalizedToolResult(request=request, result=result)
                    for request, result in zip(first.tool_requests, search_results, strict=True)
                ],
            )
        )

        assert len(second.tool_requests) == 1
        assert second.tool_requests[0].name == "fetch_candidates"
        assert second.tool_requests[0].arguments == {
            "candidate_ids": candidate_ids_from_search_results(search_results)
        }

    asyncio.run(exercise())


@pytest.mark.parametrize("tool_shape", [ToolShape.SCALAR, ToolShape.BATCH])
def test_direct_mcp_agent_executor_matches_in_process_tool_oracle(
    tool_shape: ToolShape,
) -> None:
    task = tiny_task(task_id=f"agent-parity-{tool_shape}", tool_shape=tool_shape)

    direct = DirectMcpAgentExecutor(
        direct_mcp_client(task),
        ScriptedFanoutModelClient(),
    ).execute(task)
    in_process = InProcessToolOracleExecutor().execute(task)

    assert direct.answer == in_process.answer
    assert direct.usage.tool_calls == in_process.usage.tool_calls
    assert direct.usage.failed_tool_calls == 0
    assert direct.usage.tool_response_bytes_total == in_process.usage.tool_response_bytes_total
    assert direct.usage.model_visible_bytes_total == in_process.usage.model_visible_bytes_total
    assert direct.tool_calls == in_process.tool_calls
    assert direct.error is None
    assert direct.trace.failure_category is None


@pytest.mark.parametrize(
    ("tool_shape", "expected_tools"),
    [
        (
            ToolShape.SCALAR,
            Counter({"search_shard": 2, "fetch_candidate": 6}),
        ),
        (
            ToolShape.BATCH,
            Counter({"search_shard": 2, "fetch_candidates": 1}),
        ),
    ],
)
def test_direct_mcp_agent_executor_aggregates_usage_and_trace(
    tool_shape: ToolShape,
    expected_tools: Counter[str],
) -> None:
    task = tiny_task(task_id=f"usage-{tool_shape}", tool_shape=tool_shape)

    result = DirectMcpAgentExecutor(
        direct_mcp_client(task),
        ScriptedFanoutModelClient(),
    ).execute(task)

    assert result.usage.model_requests == 3
    assert result.usage.input_tokens == 450
    assert result.usage.output_tokens == 110
    assert result.usage.cache_read_tokens is None
    assert result.usage.cache_write_tokens is None
    assert result.usage.tool_calls == sum(expected_tools.values())
    assert Counter(call.tool_name for call in result.tool_calls) == expected_tools
    assert result.usage.tool_response_bytes_total == sum(
        call.response_bytes for call in result.tool_calls
    )
    assert result.usage.model_visible_bytes_total == sum(
        call.response_bytes for call in result.tool_calls if call.model_visible
    )
    assert result.trace.span_count == result.usage.model_requests + result.usage.tool_calls
    assert result.trace.nested_tool_call_count == result.usage.tool_calls
    assert result.raw == {
        "model_turns": [
            {"scripted_turn": "search_shards"},
            {"scripted_turn": "fetch_candidates"},
            {"scripted_turn": "final_answer"},
        ]
    }


class FailingToolModelClient:
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        if request.turn_index == 1:
            return ModelTurnResult(
                tool_requests=[
                    NormalizedToolRequest(
                        id="unknown",
                        name="missing_tool",
                        arguments={},
                    ),
                    NormalizedToolRequest(
                        id="bad-fetch",
                        name="fetch_candidate",
                        arguments={"candidate_id": "does-not-exist"},
                    ),
                ],
                usage=NormalizedModelUsage(
                    input_tokens=1,
                    output_tokens=2,
                    cache_read_tokens=3,
                ),
                raw={"turn": "bad_tools"},
            )

        assert [tool_result.error for tool_result in request.tool_results] == [
            "unknown_tool:missing_tool",
            "KeyError:'does-not-exist'",
        ]
        return ModelTurnResult(
            final_answer=StructuredAnswer(
                task_id=request.task.id,
                candidates=[
                    RankedCandidate(id="synthetic", score=1.0),
                ],
            ),
            usage=NormalizedModelUsage(
                input_tokens=4,
                output_tokens=5,
                cache_write_tokens=6,
            ),
            raw={"turn": "final"},
        )


class InvalidFinalAnswerModelClient:
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        return ModelTurnResult(
            final_answer={"task_id": request.task.id, "candidates": [{"id": "missing-score"}]},
            usage=NormalizedModelUsage(input_tokens=1, output_tokens=1),
            raw={"turn": "invalid_final"},
        )


class FailingSecondTurnModelClient:
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        if request.turn_index == 1:
            return ModelTurnResult(
                tool_requests=[
                    NormalizedToolRequest(
                        id="search-0",
                        name="search_shard",
                        arguments={"shard_id": 0},
                    )
                ],
                usage=NormalizedModelUsage(input_tokens=11, output_tokens=3),
                raw={"turn": "before_provider_failure"},
            )
        raise RuntimeError("provider unavailable")


class SlowModelClient:
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        time.sleep(request.task.timeout_seconds * 2)
        return ModelTurnResult(raw={"turn": "too_late"})


class RecordingContextModelClient:
    def __init__(self) -> None:
        self.requests: list[ModelTurnRequest] = []

    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        self.requests.append(request)
        return ModelTurnResult(
            final_answer=StructuredAnswer(task_id=request.task.id, candidates=[]),
            raw={"turn": "record_context"},
        )


def test_direct_mcp_agent_executor_counts_unknown_and_failed_tools() -> None:
    task = tiny_task(task_id="failed-tools")

    result = DirectMcpAgentExecutor(
        InProcessSyntheticTools.from_task(task),
        FailingToolModelClient(),
    ).execute(task)

    assert result.error is None
    assert result.answer == StructuredAnswer(
        task_id=task.id,
        candidates=[RankedCandidate(id="synthetic", score=1.0)],
    )
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 2
    assert result.usage.failed_tool_calls == 2
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 7
    assert result.usage.cache_read_tokens == 3
    assert result.usage.cache_write_tokens == 6
    assert result.trace.span_count == 4


def test_direct_mcp_agent_executor_counts_failed_tools_against_budget() -> None:
    task = tiny_task(task_id="failed-budget", max_tool_calls=1)

    result = DirectMcpAgentExecutor(
        InProcessSyntheticTools.from_task(task),
        FailingToolModelClient(),
    ).execute(task)

    assert result.answer is None
    assert result.error == "max_tool_calls_exceeded"
    assert result.trace.failure_category == FailureCategory.TOOL_BUDGET_EXCEEDED
    assert result.usage.model_requests == 1
    assert result.usage.tool_calls == 0
    assert result.usage.failed_tool_calls == 0


def test_direct_mcp_agent_executor_stops_before_exceeding_tool_budget() -> None:
    task = tiny_task(task_id="budget", max_tool_calls=2)

    result = DirectMcpAgentExecutor(
        direct_mcp_client(task),
        ScriptedFanoutModelClient(),
    ).execute(task)

    assert result.answer is None
    assert result.error == "max_tool_calls_exceeded"
    assert result.trace.failure_category == FailureCategory.TOOL_BUDGET_EXCEEDED
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 2
    assert result.usage.failed_tool_calls == 0
    assert result.usage.input_tokens == 250
    assert result.usage.output_tokens == 60
    assert [call.tool_name for call in result.tool_calls] == [
        "search_shard",
        "search_shard",
    ]
    assert [turn["scripted_turn"] for turn in result.raw["model_turns"]] == [
        "search_shards",
        "fetch_candidates",
    ]


def test_direct_mcp_agent_executor_classifies_invalid_final_answer_schema() -> None:
    task = tiny_task(task_id="invalid-final")

    result = DirectMcpAgentExecutor(
        InProcessSyntheticTools.from_task(task),
        InvalidFinalAnswerModelClient(),
    ).execute(task)

    assert result.answer is None
    assert result.error == "final_answer_schema_invalid"
    assert result.trace.failure_category == FailureCategory.SCHEMA_FAILURE
    assert result.usage.model_requests == 1
    assert result.usage.tool_calls == 0


def test_direct_mcp_agent_executor_contains_model_client_failures() -> None:
    task = tiny_task(task_id="provider-failure")

    result = DirectMcpAgentExecutor(
        InProcessSyntheticTools.from_task(task),
        FailingSecondTurnModelClient(),
    ).execute(task)

    assert result.answer is None
    assert result.error == "RuntimeError:provider unavailable"
    assert result.trace.failure_category == FailureCategory.PROVIDER_FAILURE
    assert result.usage.model_requests == 2
    assert result.usage.tool_calls == 1
    assert result.usage.failed_tool_calls == 0
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 3
    assert result.trace.span_count == 3
    assert result.raw == {
        "model_turns": [
            {"turn": "before_provider_failure"},
            {
                "turn_index": 2,
                "error": "RuntimeError:provider unavailable",
                "failure_category": "provider_failure",
            },
        ]
    }


def test_direct_mcp_agent_executor_does_not_contain_runner_timeout() -> None:
    task = tiny_task(task_id="runner-timeout").model_copy(update={"timeout_seconds": 0.01})

    result = BenchmarkRunner(
        DirectMcpAgentExecutor(
            InProcessSyntheticTools.from_task(task),
            SlowModelClient(),
        )
    ).run_task(task)

    assert result.timed_out is True
    assert result.execution.error == "timeout"
    assert result.execution.trace.failure_category == FailureCategory.TIMEOUT


def test_direct_mcp_agent_executor_propagates_execution_context_to_model_turns() -> None:
    task = tiny_task(task_id="cache-context")
    context = ExecutionContext(
        cache_policy=CachePolicy.WARM,
        cache_state=CacheState.WARMUP,
        cache_namespace="provider-cache",
        cache_warmup_run=True,
    )
    model = RecordingContextModelClient()

    result = DirectMcpAgentExecutor(
        InProcessSyntheticTools.from_task(task),
        model,
    ).execute(task, context=context)

    assert result.error is None
    assert len(model.requests) == 1
    assert model.requests[0].context == context


def test_benchmark_runner_scores_direct_mcp_agent_parallel_executor() -> None:
    task = tiny_task(task_id="parallel-runner", tool_shape=ToolShape.BATCH)

    result = BenchmarkRunner(
        DirectMcpAgentParallelExecutor(
            direct_mcp_client(task),
            ScriptedFanoutModelClient(),
        )
    ).run_task(task, repetition=3)

    assert result.task_id == task.id
    assert result.arm_name == "direct_mcp_agent_parallel"
    assert result.repetition == 3
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
    assert result.score.failure_reason is None
