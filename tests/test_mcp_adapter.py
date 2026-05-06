from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.fastmcp import FastMCP

from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.models import ProbeTask
from codemode_probe.synthetic_tools import (
    InProcessSyntheticTools,
    SYNTHETIC_TOOL_SPECS,
    canonical_json_bytes,
)
from codemode_probe.workload import make_probe_task


def tiny_task() -> ProbeTask:
    return make_probe_task(
        "mcp-adapter-task",
        seed=17,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=16,
        relevant_fraction=0.5,
        top_k=2,
    )


async def call_structured_tool(
    server: FastMCP,
    name: str,
    arguments: dict[str, Any],
) -> object:
    result = await server.call_tool(name, arguments)
    if isinstance(result, tuple):
        _, structured = result
    else:
        structured = result
    if isinstance(structured, dict) and set(structured) == {"result"}:
        return structured["result"]
    return structured


def test_build_synthetic_mcp_server_lists_expected_tools() -> None:
    async def exercise() -> None:
        tools = await build_synthetic_mcp_server(tiny_task()).list_tools()

        assert [tool.name for tool in tools] == [
            spec.name for spec in SYNTHETIC_TOOL_SPECS
        ]
        assert [tool.description for tool in tools] == [
            spec.description for spec in SYNTHETIC_TOOL_SPECS
        ]
        assert [spec.model_visible for spec in SYNTHETIC_TOOL_SPECS] == [
            True,
            True,
            True,
        ]

    asyncio.run(exercise())


def test_mcp_search_shard_matches_in_process_tools() -> None:
    async def exercise() -> None:
        task = tiny_task()
        server = build_synthetic_mcp_server(task)
        in_process = InProcessSyntheticTools.from_task(task)

        expected = await in_process.search_shard(0, limit=2)
        actual = await call_structured_tool(
            server,
            "search_shard",
            {"shard_id": 0, "limit": 2},
        )

        assert actual == expected
        assert canonical_json_bytes(actual) == canonical_json_bytes(expected)

    asyncio.run(exercise())


def test_mcp_fetch_candidate_matches_in_process_tools() -> None:
    async def exercise() -> None:
        task = tiny_task()
        server = build_synthetic_mcp_server(task)
        in_process = InProcessSyntheticTools.from_task(task)
        candidate_id = (await in_process.search_shard(1))[0]["id"]

        expected = await in_process.fetch_candidate(str(candidate_id))
        actual = await call_structured_tool(
            server,
            "fetch_candidate",
            {"candidate_id": candidate_id},
        )

        assert actual == expected
        assert canonical_json_bytes(actual) == canonical_json_bytes(expected)

    asyncio.run(exercise())


def test_mcp_fetch_candidates_preserves_order_and_matches_in_process_tools() -> None:
    async def exercise() -> None:
        task = tiny_task()
        server = build_synthetic_mcp_server(task)
        in_process = InProcessSyntheticTools.from_task(task)
        shard_zero = await in_process.search_shard(0)
        shard_one = await in_process.search_shard(1)
        candidate_ids = [
            str(shard_one[1]["id"]),
            str(shard_zero[0]["id"]),
            str(shard_one[0]["id"]),
        ]

        expected = await in_process.fetch_candidates(candidate_ids)
        actual = await call_structured_tool(
            server,
            "fetch_candidates",
            {"candidate_ids": candidate_ids},
        )

        assert [candidate["id"] for candidate in actual] == candidate_ids
        assert actual == expected
        assert canonical_json_bytes(actual) == canonical_json_bytes(expected)

    asyncio.run(exercise())
