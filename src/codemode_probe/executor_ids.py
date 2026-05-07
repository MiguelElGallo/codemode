from __future__ import annotations

EXECUTOR_ALIASES = {
    "deterministic_oracle": "deterministic_oracle_client",
    "in_process": "in_process_tool_oracle",
    "direct_mcp": "direct_mcp_tool_oracle",
    "direct_agent": "direct_mcp_agent_parallel",
    "code_mode": "code_mode_synthetic_scripted",
    "code_mode_real": "code_mode_pydantic_monty",
    "pydantic_monty": "code_mode_pydantic_monty",
}


def available_executor_ids() -> tuple[str, ...]:
    return (
        "deterministic_oracle_client",
        "in_process_tool_oracle",
        "direct_mcp_tool_oracle",
        "direct_mcp_agent_parallel",
        "code_mode_synthetic_scripted",
        "code_mode_pydantic_monty",
    )


def normalize_executor_id(executor_id: str) -> str:
    return EXECUTOR_ALIASES.get(executor_id, executor_id)
