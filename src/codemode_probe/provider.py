from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.models import (
    Candidate,
    ModelTurnRequest,
    ModelTurnResult,
    NormalizedModelUsage,
    NormalizedToolRequest,
    NormalizedToolResult,
    ToolShape,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.prompts import RenderedPrompt, render_prompt


class ProviderTurnRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    rendered_prompt: RenderedPrompt
    turn_index: int = Field(ge=1)
    tool_results: list[NormalizedToolResult] = Field(default_factory=list)


class ProviderTurnResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_requests: list[NormalizedToolRequest] = Field(default_factory=list)
    final_answer: dict[str, Any] | None = None
    usage: NormalizedModelUsage = Field(default_factory=NormalizedModelUsage)
    stop_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ProviderClient(Protocol):
    provider_name: str
    model_name: str

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        ...


class ProviderBackedModelClient:
    def __init__(self, provider_client: ProviderClient) -> None:
        self._provider_client = provider_client

    async def run_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        provider_request = ProviderTurnRequest(
            rendered_prompt=render_prompt(request.task),
            turn_index=request.turn_index,
            tool_results=request.tool_results,
        )
        provider_response = await self._provider_client.run_provider_turn(provider_request)
        return ModelTurnResult(
            tool_requests=provider_response.tool_requests,
            final_answer=provider_response.final_answer,
            usage=provider_response.usage,
            raw={
                "provider_name": self._provider_client.provider_name,
                "model_name": self._provider_client.model_name,
                "prompt_hash": provider_request.rendered_prompt.canonical_hash,
                "stop_reason": provider_response.stop_reason,
                "provider_raw": provider_response.raw,
            },
        )


class ScriptedProviderClient:
    provider_name = "scripted"
    model_name = "scripted-fanout"

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        if request.turn_index == 1:
            return ProviderTurnResponse(
                tool_requests=[
                    NormalizedToolRequest(
                        id=f"search-{shard_id}",
                        name="search_shard",
                        arguments={"shard_id": shard_id},
                    )
                    for shard_id in range(_int_param(request.rendered_prompt, "shard_count"))
                ],
                usage=NormalizedModelUsage(input_tokens=100, output_tokens=25),
                stop_reason="tool_requests",
                raw={"scripted_turn": "search_shards"},
            )

        if request.turn_index == 2:
            candidate_ids = _candidate_ids_from_tool_results(request.tool_results)
            if _tool_shape(request.rendered_prompt) == ToolShape.BATCH:
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
            return ProviderTurnResponse(
                tool_requests=tool_requests,
                usage=NormalizedModelUsage(input_tokens=150, output_tokens=35),
                stop_reason="tool_requests",
                raw={"scripted_turn": "fetch_candidates"},
            )

        candidates = [
            Candidate.model_validate(item)
            for item in _full_candidate_payloads_from_tool_results(request.tool_results)
        ]
        final_answer = rank_candidates(
            request.rendered_prompt.task_id,
            candidates,
            _int_param(request.rendered_prompt, "top_k"),
        ).model_dump(mode="json")
        return ProviderTurnResponse(
            final_answer=final_answer,
            usage=NormalizedModelUsage(input_tokens=200, output_tokens=50),
            stop_reason="final_answer",
            raw={"scripted_turn": "final_answer"},
        )


def _int_param(rendered_prompt: RenderedPrompt, name: str) -> int:
    return int(rendered_prompt.task_parameters[name])


def _tool_shape(rendered_prompt: RenderedPrompt) -> ToolShape:
    return ToolShape(str(rendered_prompt.task_parameters["tool_shape"]))


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
