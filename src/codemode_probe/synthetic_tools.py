from __future__ import annotations

import asyncio
from typing import Iterable, Protocol

from codemode_probe.models import (
    Candidate,
    ExecutionResult,
    ProbeTask,
    ToolCallRecord,
    ToolSpec,
    TraceSummary,
    UsageStats,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.workload import candidates_by_shard, generate_candidates


SYNTHETIC_TOOL_SPECS = (
    ToolSpec(
        name="search_shard",
        description="Return lightweight candidate summaries for one shard.",
    ),
    ToolSpec(
        name="fetch_candidate",
        description="Return one full candidate by id.",
    ),
    ToolSpec(
        name="fetch_candidates",
        description="Return full candidates for the requested ids in order.",
    ),
)


class InProcessSyntheticTools:
    def __init__(self, candidates: list[Candidate]) -> None:
        self._candidates = {candidate.id: candidate for candidate in candidates}
        self._shards = candidates_by_shard(candidates)
        self.calls: list[ToolCallRecord] = []

    @classmethod
    def from_task(cls, task: ProbeTask) -> InProcessSyntheticTools:
        return cls(generate_candidates(task.workload))

    async def search_shard(self, shard_id: int, *, limit: int | None = None) -> list[dict[str, object]]:
        candidates = self._shards.get(shard_id, [])
        if limit is not None:
            candidates = candidates[:limit]
        response = [_summary(candidate) for candidate in candidates]
        self._record("search_shard", response, model_visible=True)
        return response

    async def fetch_candidate(self, candidate_id: str) -> dict[str, object]:
        candidate = self._candidates[candidate_id]
        response = candidate.model_dump(mode="json")
        self._record("fetch_candidate", response, model_visible=True)
        return response

    async def fetch_candidates(self, candidate_ids: Iterable[str]) -> list[dict[str, object]]:
        response = [self._candidates[candidate_id].model_dump(mode="json") for candidate_id in candidate_ids]
        self._record("fetch_candidates", response, model_visible=True)
        return response

    def _record(self, tool_name: str, response: object, *, model_visible: bool) -> None:
        encoded = _json_bytes(response)
        item_count = len(response) if isinstance(response, list) else 1
        self.calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                response_bytes=len(encoded),
                model_visible=model_visible,
                item_count=item_count,
            )
        )


class SyntheticToolClient(Protocol):
    calls: list[ToolCallRecord]

    async def search_shard(self, shard_id: int, *, limit: int | None = None) -> list[dict[str, object]]:
        ...

    async def fetch_candidate(self, candidate_id: str) -> dict[str, object]:
        ...

    async def fetch_candidates(self, candidate_ids: Iterable[str]) -> list[dict[str, object]]:
        ...


def run_tool_oracle(task: ProbeTask, tools: SyntheticToolClient) -> ExecutionResult:
    return asyncio.run(_run_tool_oracle(task, tools))


async def _run_tool_oracle(task: ProbeTask, tools: SyntheticToolClient) -> ExecutionResult:
    shard_results = await asyncio.gather(
        *[
            tools.search_shard(shard_id)
            for shard_id in range(task.workload.shard_count)
        ]
    )
    candidate_ids = [str(item["id"]) for shard in shard_results for item in shard]

    if task.workload.tool_shape == "batch":
        fetched = await tools.fetch_candidates(candidate_ids)
    else:
        fetched = await asyncio.gather(
            *[tools.fetch_candidate(candidate_id) for candidate_id in candidate_ids]
        )

    candidates = [Candidate.model_validate(item) for item in fetched]
    answer = rank_candidates(task.id, candidates, task.workload.top_k)
    tool_response_bytes = sum(call.response_bytes for call in tools.calls)
    model_visible_bytes = sum(
        call.response_bytes for call in tools.calls if call.model_visible
    )
    return ExecutionResult(
        answer=answer,
        usage=UsageStats(
            tool_calls=len(tools.calls),
            tool_response_bytes_total=tool_response_bytes,
            model_visible_bytes_total=model_visible_bytes,
        ),
        trace=TraceSummary(
            span_count=len(tools.calls),
            nested_tool_call_count=len(tools.calls),
        ),
        tool_calls=tools.calls,
        raw={"candidate_count": len(candidates)},
    )


def _summary(candidate: Candidate) -> dict[str, object]:
    return {
        "id": candidate.id,
        "shard_id": candidate.shard_id,
        "title": candidate.title,
        "category": candidate.category,
    }


def _json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_bytes(value: object) -> bytes:
    return _json_bytes(value)
