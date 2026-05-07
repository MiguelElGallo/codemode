from __future__ import annotations

from collections.abc import Callable

from codemode_probe.executors import (
    CandidateExecutor,
    CodeModeSyntheticScriptedExecutor,
    DeterministicOracleExecutor,
    DirectMcpAgentParallelExecutor,
    DirectMcpToolOracleExecutor,
    InProcessToolOracleExecutor,
)
from codemode_probe.executor_ids import available_executor_ids, normalize_executor_id
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient, FastMcpInProcessSession
from codemode_probe.models import ProbeTask
from codemode_probe.provider import ProviderBackedModelClient, ProviderClient, ScriptedProviderClient

ExecutorFactory = Callable[[ProbeTask], CandidateExecutor]


def build_executor(
    executor_id: str,
    task: ProbeTask,
    *,
    provider_client: ProviderClient | None = None,
) -> CandidateExecutor:
    normalized = normalize_executor_id(executor_id)
    if normalized == "deterministic_oracle_client":
        return DeterministicOracleExecutor()
    if normalized == "in_process_tool_oracle":
        return InProcessToolOracleExecutor()
    if normalized == "direct_mcp_tool_oracle":
        return DirectMcpToolOracleExecutor(_direct_mcp_tool_client(task))
    if normalized == "direct_mcp_agent_parallel":
        return DirectMcpAgentParallelExecutor(
            _direct_mcp_tool_client(task),
            ProviderBackedModelClient(provider_client or ScriptedProviderClient()),
        )
    if normalized == "code_mode_synthetic_scripted":
        return CodeModeSyntheticScriptedExecutor()
    raise ValueError(f"unknown executor id: {executor_id}")


def _direct_mcp_tool_client(task: ProbeTask) -> DirectMcpSyntheticToolClient:
    return DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )
