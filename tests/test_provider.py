from __future__ import annotations

import asyncio
import json

import pytest

from codemode_probe.executors import InProcessToolOracleExecutor
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient, FastMcpInProcessSession
from codemode_probe.model_loop import DirectMcpAgentExecutor
from codemode_probe.models import (
    ModelTurnRequest,
    NormalizedModelUsage,
    NormalizedToolRequest,
    NormalizedToolResult,
    ProbeTask,
    RankedCandidate,
    StructuredAnswer,
    ToolShape,
)
from codemode_probe.prompts import RenderedPrompt, render_prompt
from codemode_probe.provider import (
    ProviderBackedModelClient,
    ProviderTurnRequest,
    ProviderTurnResponse,
    ScriptedProviderClient,
)
from codemode_probe.workload import make_probe_task


def tiny_task(
    *,
    task_id: str = "provider-task",
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


class RecordingProviderClient:
    provider_name = "recording-provider"
    model_name = "recording-model"

    def __init__(self, response: ProviderTurnResponse) -> None:
        self.response = response
        self.requests: list[ProviderTurnRequest] = []

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        self.requests.append(request)
        return self.response


def test_provider_backed_model_client_sends_rendered_prompt_with_task_parameters_and_hash() -> None:
    task = tiny_task()
    tool_result = NormalizedToolResult(
        request=NormalizedToolRequest(id="search-0", name="search_shard", arguments={"shard_id": 0}),
        result=[{"id": "cand-0000"}],
    )
    provider = RecordingProviderClient(ProviderTurnResponse(stop_reason="done"))

    asyncio.run(
        ProviderBackedModelClient(provider).run_turn(
            ModelTurnRequest(task=task, turn_index=2, tool_results=[tool_result])
        )
    )

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.turn_index == 2
    assert request.tool_results == [tool_result]
    assert isinstance(request.rendered_prompt, RenderedPrompt)
    assert request.rendered_prompt.task_id == task.id
    assert request.rendered_prompt.task_parameters == {
        "seed": task.workload.seed,
        "task_family": task.workload.task_family.value,
        "tool_shape": task.workload.tool_shape.value,
        "shard_count": task.workload.shard_count,
        "candidates_per_shard": task.workload.candidates_per_shard,
        "payload_bytes": task.workload.payload_bytes,
        "relevant_fraction": task.workload.relevant_fraction,
        "top_k": task.workload.top_k,
    }
    assert request.rendered_prompt.canonical_hash == render_prompt(task).canonical_hash


def test_provider_backed_model_client_returns_normalized_response_and_serializable_raw() -> None:
    task = tiny_task()
    tool_request = NormalizedToolRequest(
        id="lookup-1",
        name="fetch_candidate",
        arguments={"candidate_id": "cand-0001"},
    )
    answer = StructuredAnswer(
        task_id=task.id,
        candidates=[RankedCandidate(id="cand-0001", score=0.92)],
    ).model_dump(mode="json")
    usage = NormalizedModelUsage(
        input_tokens=10,
        output_tokens=4,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    provider = RecordingProviderClient(
        ProviderTurnResponse(
            tool_requests=[tool_request],
            final_answer=answer,
            usage=usage,
            stop_reason="tool_requests",
            raw={"request_id": "req-123", "latency_ms": 12.5},
        )
    )

    result = asyncio.run(
        ProviderBackedModelClient(provider).run_turn(
            ModelTurnRequest(task=task, turn_index=1)
        )
    )

    assert result.tool_requests == [tool_request]
    assert result.final_answer == StructuredAnswer.model_validate(answer)
    assert result.usage == usage
    assert result.raw == {
        "provider_name": "recording-provider",
        "model_name": "recording-model",
        "prompt_hash": render_prompt(task).canonical_hash,
        "stop_reason": "tool_requests",
        "provider_raw": {"request_id": "req-123", "latency_ms": 12.5},
    }
    json.dumps(result.raw)


def test_provider_backed_model_client_missing_token_fields_stay_none() -> None:
    task = tiny_task()
    provider = RecordingProviderClient(
        ProviderTurnResponse(
            usage=NormalizedModelUsage(input_tokens=7),
            stop_reason="final_answer",
        )
    )

    result = asyncio.run(
        ProviderBackedModelClient(provider).run_turn(
            ModelTurnRequest(task=task, turn_index=1)
        )
    )

    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens is None
    assert result.usage.cache_read_tokens is None
    assert result.usage.cache_write_tokens is None


def direct_mcp_client(task: ProbeTask) -> DirectMcpSyntheticToolClient:
    return DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )


@pytest.mark.parametrize("tool_shape", [ToolShape.SCALAR, ToolShape.BATCH])
def test_scripted_provider_client_drives_direct_mcp_agent_scalar_batch_parity(
    tool_shape: ToolShape,
) -> None:
    task = tiny_task(task_id=f"scripted-provider-{tool_shape}", tool_shape=tool_shape)

    direct = DirectMcpAgentExecutor(
        direct_mcp_client(task),
        ProviderBackedModelClient(ScriptedProviderClient()),
    ).execute(task)
    in_process = InProcessToolOracleExecutor().execute(task)

    assert direct.answer == in_process.answer
    assert direct.usage.tool_calls == in_process.usage.tool_calls
    assert direct.usage.failed_tool_calls == 0
    assert direct.usage.input_tokens == 450
    assert direct.usage.output_tokens == 110
    assert direct.usage.cache_read_tokens is None
    assert direct.usage.cache_write_tokens is None
    assert direct.usage.tool_response_bytes_total == in_process.usage.tool_response_bytes_total
    assert direct.usage.model_visible_bytes_total == in_process.usage.model_visible_bytes_total
    assert direct.tool_calls == in_process.tool_calls
    assert direct.error is None
    assert direct.trace.failure_category is None
    assert [turn["provider_name"] for turn in direct.raw["model_turns"]] == ["scripted"] * 3
    assert [turn["model_name"] for turn in direct.raw["model_turns"]] == [
        "scripted-fanout"
    ] * 3
    assert {turn["prompt_hash"] for turn in direct.raw["model_turns"]} == {
        render_prompt(task).canonical_hash
    }
    assert [turn["stop_reason"] for turn in direct.raw["model_turns"]] == [
        "tool_requests",
        "tool_requests",
        "final_answer",
    ]
    json.dumps(direct.raw)
