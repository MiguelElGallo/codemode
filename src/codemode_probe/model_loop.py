from __future__ import annotations

import asyncio
from typing import Protocol

from codemode_probe.models import (
    Candidate,
    ExecutionContext,
    ExecutionResult,
    FailureCategory,
    ModelTurnRequest,
    ModelTurnResult,
    NormalizedModelUsage,
    NormalizedToolRequest,
    NormalizedToolResult,
    ProbeTask,
    StructuredAnswer,
    TraceSummary,
    UsageStats,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.synthetic_tools import SyntheticToolClient


class ModelClient(Protocol):
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        ...


class DirectMcpAgentExecutor:
    name = "direct_mcp_agent_parallel"

    def __init__(self, tool_client: SyntheticToolClient, model_client: ModelClient) -> None:
        self._tool_client = tool_client
        self._model_client = model_client

    def execute(
        self,
        task: ProbeTask,
        *,
        context: ExecutionContext | None = None,
    ) -> ExecutionResult:
        return asyncio.run(self._execute_async(task, context or ExecutionContext()))

    async def _execute_async(self, task: ProbeTask, context: ExecutionContext) -> ExecutionResult:
        tool_results: list[NormalizedToolResult] = []
        model_usage = _UsageAccumulator()
        model_turns: list[dict[str, object]] = []
        failed_tool_calls = 0
        attempted_tool_calls = 0

        for turn_index in range(1, task.max_tool_calls + 2):
            try:
                model_result = await self._model_client.run_turn(
                    ModelTurnRequest(
                        task=task,
                        turn_index=turn_index,
                        tool_results=tool_results,
                        context=context,
                    )
                )
            except Exception as exc:
                failure_category = _classify_model_exception(exc)
                model_turns.append(
                    {
                        "turn_index": turn_index,
                        "error": _exception_label(exc),
                        "failure_category": failure_category.value,
                    }
                )
                return self._execution_result(
                    answer=None,
                    model_requests=turn_index,
                    failed_tool_calls=failed_tool_calls,
                    attempted_tool_calls=attempted_tool_calls,
                    model_usage=model_usage,
                    model_turns=model_turns,
                    error=_exception_label(exc),
                    failure_category=failure_category,
                )
            model_usage.add(model_result.usage)
            model_turns.append(model_result.raw)

            if model_result.final_answer is not None:
                try:
                    answer = _parse_final_answer(model_result.final_answer)
                except Exception:
                    return self._execution_result(
                        answer=None,
                        model_requests=turn_index,
                        failed_tool_calls=failed_tool_calls,
                        model_usage=model_usage,
                        model_turns=model_turns,
                        error="final_answer_schema_invalid",
                        failure_category=FailureCategory.SCHEMA_FAILURE,
                    )
                return self._execution_result(
                    answer=answer,
                    model_requests=turn_index,
                    failed_tool_calls=failed_tool_calls,
                    model_usage=model_usage,
                    model_turns=model_turns,
                )

            requested_count = len(model_result.tool_requests)
            if requested_count == 0:
                return self._execution_result(
                    answer=None,
                    model_requests=turn_index,
                    failed_tool_calls=failed_tool_calls,
                    model_usage=model_usage,
                    model_turns=model_turns,
                    error="model_returned_no_tool_requests_or_final_answer",
                    failure_category=FailureCategory.MODEL_PROTOCOL_ERROR,
                )

            if attempted_tool_calls + requested_count > task.max_tool_calls:
                return self._execution_result(
                    answer=None,
                    model_requests=turn_index,
                    failed_tool_calls=failed_tool_calls,
                    attempted_tool_calls=attempted_tool_calls,
                    model_usage=model_usage,
                    model_turns=model_turns,
                    error="max_tool_calls_exceeded",
                    failure_category=FailureCategory.TOOL_BUDGET_EXCEEDED,
                )

            turn_tool_results = await asyncio.gather(
                *[self._dispatch_tool(request) for request in model_result.tool_requests]
            )
            attempted_tool_calls += requested_count
            failed_tool_calls += sum(1 for result in turn_tool_results if result.error is not None)
            tool_results.extend(turn_tool_results)

        return self._execution_result(
            answer=None,
            model_requests=task.max_tool_calls + 1,
            failed_tool_calls=failed_tool_calls,
            attempted_tool_calls=attempted_tool_calls,
            model_usage=model_usage,
            model_turns=model_turns,
            error="model_loop_exhausted",
            failure_category=FailureCategory.MODEL_PROTOCOL_ERROR,
        )

    async def _dispatch_tool(self, request: NormalizedToolRequest) -> NormalizedToolResult:
        try:
            if request.name == "search_shard":
                result = await self._tool_client.search_shard(**request.arguments)
            elif request.name == "fetch_candidate":
                result = await self._tool_client.fetch_candidate(**request.arguments)
            elif request.name == "fetch_candidates":
                result = await self._tool_client.fetch_candidates(**request.arguments)
            else:
                return NormalizedToolResult(request=request, error=f"unknown_tool:{request.name}")
        except Exception as exc:
            return NormalizedToolResult(request=request, error=f"{type(exc).__name__}:{exc}")
        return NormalizedToolResult(request=request, result=result)

    def _execution_result(
        self,
        *,
        answer: StructuredAnswer | None,
        model_requests: int,
        failed_tool_calls: int,
        attempted_tool_calls: int | None = None,
        model_usage: "_UsageAccumulator",
        model_turns: list[dict[str, object]],
        error: str | None = None,
        failure_category: FailureCategory | None = None,
    ) -> ExecutionResult:
        tool_response_bytes = sum(call.response_bytes for call in self._tool_client.calls)
        model_visible_bytes = sum(
            call.response_bytes for call in self._tool_client.calls if call.model_visible
        )
        return ExecutionResult(
            answer=answer,
            usage=UsageStats(
                model_requests=model_requests,
                tool_calls=attempted_tool_calls
                if attempted_tool_calls is not None
                else len(self._tool_client.calls) + failed_tool_calls,
                failed_tool_calls=failed_tool_calls,
                input_tokens=model_usage.input_tokens,
                output_tokens=model_usage.output_tokens,
                cache_read_tokens=model_usage.cache_read_tokens,
                cache_write_tokens=model_usage.cache_write_tokens,
                tool_response_bytes_total=tool_response_bytes,
                model_visible_bytes_total=model_visible_bytes,
            ),
            trace=TraceSummary(
                span_count=model_requests
                + (
                    attempted_tool_calls
                    if attempted_tool_calls is not None
                    else len(self._tool_client.calls) + failed_tool_calls
                ),
                nested_tool_call_count=attempted_tool_calls
                if attempted_tool_calls is not None
                else len(self._tool_client.calls) + failed_tool_calls,
                failure_category=failure_category,
            ),
            tool_calls=self._tool_client.calls,
            raw={"model_turns": model_turns},
            error=error,
        )


class ScriptedFanoutModelClient:
    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        if request.turn_index == 1:
            return ModelTurnResult(
                tool_requests=[
                    NormalizedToolRequest(
                        id=f"search-{shard_id}",
                        name="search_shard",
                        arguments={"shard_id": shard_id},
                    )
                    for shard_id in range(request.task.workload.shard_count)
                ],
                usage=NormalizedModelUsage(input_tokens=100, output_tokens=25),
                raw={"scripted_turn": "search_shards"},
            )

        if request.turn_index == 2:
            candidate_ids = _candidate_ids_from_tool_results(request.tool_results)
            if request.task.workload.tool_shape == "batch":
                tool_requests = [
                    NormalizedToolRequest(
                        id="fetch-batch",
                        name="fetch_candidates",
                        arguments={"candidate_ids": candidate_ids},
                    )
                ]
            else:
                tool_requests = [
                    NormalizedToolRequest(
                        id=f"fetch-{candidate_id}",
                        name="fetch_candidate",
                        arguments={"candidate_id": candidate_id},
                    )
                    for candidate_id in candidate_ids
                ]
            return ModelTurnResult(
                tool_requests=tool_requests,
                usage=NormalizedModelUsage(input_tokens=150, output_tokens=35),
                raw={"scripted_turn": "fetch_candidates"},
            )

        candidates = [
            Candidate.model_validate(item)
            for item in _full_candidate_payloads_from_tool_results(request.tool_results)
        ]
        return ModelTurnResult(
            final_answer=rank_candidates(
                request.task.id,
                candidates,
                request.task.workload.top_k,
            ),
            usage=NormalizedModelUsage(input_tokens=200, output_tokens=50),
            raw={"scripted_turn": "final_answer"},
        )


class _UsageAccumulator:
    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cache_read_tokens: int | None = None
        self.cache_write_tokens: int | None = None

    def add(self, usage: NormalizedModelUsage) -> None:
        self.input_tokens = _add_optional(self.input_tokens, usage.input_tokens)
        self.output_tokens = _add_optional(self.output_tokens, usage.output_tokens)
        self.cache_read_tokens = _add_optional(self.cache_read_tokens, usage.cache_read_tokens)
        self.cache_write_tokens = _add_optional(self.cache_write_tokens, usage.cache_write_tokens)


def _classify_model_exception(exc: Exception) -> FailureCategory:
    if isinstance(exc, TimeoutError):
        return FailureCategory.PROVIDER_FAILURE
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return FailureCategory.ADAPTER_FAILURE
    return FailureCategory.PROVIDER_FAILURE


def _exception_label(exc: Exception) -> str:
    return f"{type(exc).__name__}:{exc}"


def _parse_final_answer(answer: StructuredAnswer | dict[str, object]) -> StructuredAnswer:
    if isinstance(answer, StructuredAnswer):
        return answer
    return StructuredAnswer.model_validate(answer)


def _candidate_ids_from_tool_results(tool_results: list[NormalizedToolResult]) -> list[str]:
    candidate_ids: list[str] = []
    for tool_result in tool_results:
        if tool_result.request.name != "search_shard" or tool_result.error is not None:
            continue
        if not isinstance(tool_result.result, list):
            continue
        candidate_ids.extend(str(item["id"]) for item in tool_result.result if isinstance(item, dict))
    return candidate_ids


def _full_candidate_payloads_from_tool_results(
    tool_results: list[NormalizedToolResult],
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for tool_result in tool_results:
        if tool_result.error is not None:
            continue
        if tool_result.request.name == "fetch_candidate" and isinstance(tool_result.result, dict):
            payloads.append(tool_result.result)
        elif tool_result.request.name == "fetch_candidates" and isinstance(tool_result.result, list):
            payloads.extend(item for item in tool_result.result if isinstance(item, dict))
    return payloads


def _add_optional(current: int | None, value: int | None) -> int | None:
    if value is None:
        return current
    return (current or 0) + value
