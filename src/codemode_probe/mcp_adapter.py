from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from codemode_probe.models import ProbeTask
from codemode_probe.synthetic_tools import InProcessSyntheticTools
from codemode_probe.workload import make_probe_task


def build_synthetic_mcp_server(task: ProbeTask) -> FastMCP:
    tools = InProcessSyntheticTools.from_task(task)
    server = FastMCP(
        name=f"codemode-probe-{task.id}",
        instructions=(
            "Synthetic benchmark tools for candidate fan-out/fan-in tasks. "
            "Use search_shard to discover candidate ids, then fetch_candidate "
            "or fetch_candidates for full records."
        ),
    )

    @server.tool(description="Return lightweight candidate summaries for one shard.")
    async def search_shard(shard_id: int, limit: int | None = None) -> list[dict[str, object]]:
        return await tools.search_shard(shard_id, limit=limit)

    @server.tool(description="Return one full candidate by id.")
    async def fetch_candidate(candidate_id: str) -> dict[str, object]:
        return await tools.fetch_candidate(candidate_id)

    @server.tool(description="Return full candidates for the requested ids in order.")
    async def fetch_candidates(candidate_ids: list[str]) -> list[dict[str, object]]:
        return await tools.fetch_candidates(candidate_ids)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic MCP server over stdio.")
    parser.add_argument("--task-id", default="synthetic_fanout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--candidates-per-shard", type=int, default=20)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    task = make_probe_task(
        args.task_id,
        seed=args.seed,
        shard_count=args.shards,
        candidates_per_shard=args.candidates_per_shard,
        payload_bytes=args.payload_bytes,
        top_k=args.top_k,
    )
    build_synthetic_mcp_server(task).run(transport="stdio")


if __name__ == "__main__":
    main()
