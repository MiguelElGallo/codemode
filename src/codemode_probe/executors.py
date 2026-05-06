from __future__ import annotations

from typing import Protocol

from codemode_probe.models import ExecutionResult, ProbeTask, UsageStats
from codemode_probe.model_loop import DirectMcpAgentExecutor, ModelClient
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient
from codemode_probe.oracle import rank_candidates
from codemode_probe.synthetic_tools import InProcessSyntheticTools, run_tool_oracle
from codemode_probe.workload import generate_candidates


class CandidateExecutor(Protocol):
    name: str

    def execute(self, task: ProbeTask) -> ExecutionResult:
        """Execute a benchmark task and return a normalized answer."""


class DeterministicOracleExecutor:
    name = "deterministic_oracle_client"

    def execute(self, task: ProbeTask) -> ExecutionResult:
        candidates = generate_candidates(task.workload)
        answer = rank_candidates(task.id, candidates, task.workload.top_k)
        payload_bytes = sum(len(candidate.model_dump_json()) for candidate in candidates)
        return ExecutionResult(
            answer=answer,
            usage=UsageStats(
                tool_calls=task.workload.shard_count,
                tool_response_bytes_total=payload_bytes,
                model_visible_bytes_total=0,
            ),
            raw={"candidate_count": len(candidates)},
        )


class InProcessToolOracleExecutor:
    name = "in_process_tool_oracle"

    def execute(self, task: ProbeTask) -> ExecutionResult:
        tools = InProcessSyntheticTools.from_task(task)
        return run_tool_oracle(task, tools)


class CodeModeSyntheticScriptedExecutor:
    name = "code_mode_synthetic_scripted"

    def execute(self, task: ProbeTask) -> ExecutionResult:
        tools = InProcessSyntheticTools(
            generate_candidates(task.workload),
            tool_outputs_model_visible=False,
        )
        execution = run_tool_oracle(task, tools)
        return execution.model_copy(
            update={
                "raw": {
                    **execution.raw,
                    "code_mode": "synthetic_scripted",
                    "tool_outputs_model_visible": False,
                }
            }
        )


class DirectMcpToolOracleExecutor:
    name = "direct_mcp_tool_oracle"

    def __init__(self, tool_client: DirectMcpSyntheticToolClient) -> None:
        self._tool_client = tool_client

    def execute(self, task: ProbeTask) -> ExecutionResult:
        return run_tool_oracle(task, self._tool_client)


class DirectMcpAgentParallelExecutor:
    name = "direct_mcp_agent_parallel"

    def __init__(
        self,
        tool_client: DirectMcpSyntheticToolClient,
        model_client: ModelClient,
    ) -> None:
        self._executor = DirectMcpAgentExecutor(tool_client, model_client)

    def execute(self, task: ProbeTask) -> ExecutionResult:
        return self._executor.execute(task)
