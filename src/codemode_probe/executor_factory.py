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
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient, FastMcpInProcessSession
from codemode_probe.models import ProbeTask
from codemode_probe.provider import ProviderBackedModelClient, ScriptedProviderClient

ExecutorFactory = Callable[[ProbeTask], CandidateExecutor]

EXECUTOR_ALIASES = {
    "deterministic_oracle": "deterministic_oracle_client",
    "in_process": "in_process_tool_oracle",
    "direct_mcp": "direct_mcp_tool_oracle",
    "direct_agent": "direct_mcp_agent_parallel",
    "code_mode": "code_mode_synthetic_scripted",
}


def available_executor_ids() -> tuple[str, ...]:
    return (
        "deterministic_oracle_client",
        "in_process_tool_oracle",
        "direct_mcp_tool_oracle",
        "direct_mcp_agent_parallel",
        "code_mode_synthetic_scripted",
    )


def normalize_executor_id(executor_id: str) -> str:
    return EXECUTOR_ALIASES.get(executor_id, executor_id)


def build_executor(executor_id: str, task: ProbeTask) -> CandidateExecutor:
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
            ProviderBackedModelClient(ScriptedProviderClient()),
        )
    if normalized == "code_mode_synthetic_scripted":
        return CodeModeSyntheticScriptedExecutor()
    raise ValueError(f"unknown executor id: {executor_id}")


def _direct_mcp_tool_client(task: ProbeTask) -> DirectMcpSyntheticToolClient:
    return DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )
