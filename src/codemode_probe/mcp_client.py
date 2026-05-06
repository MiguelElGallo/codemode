from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from codemode_probe.models import ToolCallRecord
from codemode_probe.synthetic_tools import canonical_json_bytes


class JsonToolSession(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        ...


class DirectMcpSyntheticToolClient:
    def __init__(self, session: JsonToolSession) -> None:
        self._session = session
        self.calls: list[ToolCallRecord] = []

    async def search_shard(self, shard_id: int, *, limit: int | None = None) -> list[dict[str, object]]:
        arguments: dict[str, object] = {"shard_id": shard_id}
        if limit is not None:
            arguments["limit"] = limit
        value = await self._call_json("search_shard", arguments)
        return _ensure_list_of_dicts(value)

    async def fetch_candidate(self, candidate_id: str) -> dict[str, object]:
        value = await self._call_json("fetch_candidate", {"candidate_id": candidate_id})
        if not isinstance(value, dict):
            raise TypeError("fetch_candidate returned a non-object payload")
        return dict(value)

    async def fetch_candidates(self, candidate_ids: Iterable[str]) -> list[dict[str, object]]:
        value = await self._call_json("fetch_candidates", {"candidate_ids": list(candidate_ids)})
        return _ensure_list_of_dicts(value)

    async def _call_json(self, tool_name: str, arguments: dict[str, object]) -> object:
        raw = await self._session.call_tool(tool_name, arguments)
        value = extract_structured_result(raw)
        self._record(tool_name, value, model_visible=True)
        return value

    def _record(self, tool_name: str, value: object, *, model_visible: bool) -> None:
        item_count = len(value) if isinstance(value, list) else 1
        self.calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                response_bytes=len(canonical_json_bytes(value)),
                model_visible=model_visible,
                item_count=item_count,
            )
        )


class FastMcpInProcessSession:
    def __init__(self, server: Any) -> None:
        self._server = server

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._server.call_tool(name, arguments)


def extract_structured_result(raw: Any) -> object:
    if isinstance(raw, tuple) and len(raw) == 2:
        metadata = raw[1]
        if isinstance(metadata, dict) and "result" in metadata:
            return metadata["result"]
        if isinstance(metadata, dict) or isinstance(metadata, list):
            return metadata

    if isinstance(raw, dict) and "result" in raw:
        return raw["result"]

    if isinstance(raw, dict):
        return raw

    raise TypeError(f"Unsupported MCP tool result shape: {type(raw).__name__}")


def _ensure_list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("expected a list payload")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError("expected a list of object payloads")
    return [dict(item) for item in value]
