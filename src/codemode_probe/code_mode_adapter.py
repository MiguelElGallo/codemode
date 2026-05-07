from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from typing import Any

from codemode_probe.code_mode_config import (
    CodeModeConfig,
    CodeModeConfigError,
    CodeModeRuntime,
)
from codemode_probe.models import (
    ExecutionResult,
    ProbeTask,
    StructuredAnswer,
    TraceSummary,
    UsageStats,
)
from codemode_probe.synthetic_tools import InProcessSyntheticTools


class CodeModeAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class PydanticCodeModeRunMetadata:
    model_requests: int
    input_tokens: int | None
    output_tokens: int | None
    pydantic_tool_calls: int
    run_code_calls: int
    nested_tool_calls: int


def create_code_mode_capability(config: CodeModeConfig) -> Any:
    config.validate_for_code_mode_use()
    if config.runtime != CodeModeRuntime.PYDANTIC_AI_HARNESS:
        raise CodeModeConfigError(f"unsupported Code Mode runtime: {config.runtime}")

    module = importlib.import_module(config.sdk_package)
    code_mode_class = getattr(module, "CodeMode", None)
    if code_mode_class is None:
        raise CodeModeAdapterError(
            f"optional Code Mode package '{config.sdk_package}' does not expose CodeMode"
        )
    return code_mode_class(
        tools=config.tool_selector,
        max_retries=config.max_retries,
    )


def run_pydantic_code_mode_task(
    task: ProbeTask,
    *,
    config: CodeModeConfig,
) -> ExecutionResult:
    """Run a benchmark task through Pydantic AI Harness CodeMode backed by Monty.

    This uses a deterministic local Pydantic AI `FunctionModel` to issue one
    `run_code` call. The orchestration runtime is real CodeMode/Monty; the
    model policy is scripted so tests and dry benchmark runs do not spend live
    model budget.
    """
    config.validate_for_code_mode_use()
    if config.runtime != CodeModeRuntime.PYDANTIC_AI_HARNESS:
        raise CodeModeConfigError(f"unsupported Code Mode runtime: {config.runtime}")

    try:
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel
    except ImportError as exc:  # pragma: no cover
        raise CodeModeAdapterError("Pydantic AI is required for the real Code Mode arm") from exc

    capability = create_code_mode_capability(config)
    tools = InProcessSyntheticTools(
        task_candidates(task),
        tool_outputs_model_visible=False,
    )
    code = _code_mode_program(task)

    async def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        for message in messages:
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolReturnPart) and part.tool_name == "run_code":
                    return ModelResponse(
                        [
                            TextPart(
                                json.dumps(
                                    part.content,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                            )
                        ]
                    )
        return ModelResponse(
            [
                ToolCallPart(
                    "run_code",
                    {"code": code, "restart": True},
                    tool_call_id="code-mode-run-1",
                )
            ]
        )

    async def search_shard(*, shard_id: int, limit: int | None = None) -> list[dict[str, object]]:
        return await tools.search_shard(shard_id=shard_id, limit=limit)

    async def fetch_candidate(*, candidate_id: str) -> dict[str, object]:
        return await tools.fetch_candidate(candidate_id=candidate_id)

    async def fetch_candidates(*, candidate_ids: list[str]) -> list[dict[str, object]]:
        return await tools.fetch_candidates(candidate_ids=candidate_ids)

    agent = Agent(
        FunctionModel(model_function, model_name="scripted-code-mode-policy"),
        capabilities=[capability],
    )
    agent.tool_plain(search_shard)
    agent.tool_plain(fetch_candidate)
    agent.tool_plain(fetch_candidates)

    run_result = agent.run_sync(task.prompt)
    answer = StructuredAnswer.model_validate_json(str(run_result.output))
    metadata = _code_mode_metadata(run_result)
    tool_response_bytes = sum(call.response_bytes for call in tools.calls)
    model_visible_bytes = sum(call.response_bytes for call in tools.calls if call.model_visible)

    return ExecutionResult(
        answer=answer,
        usage=UsageStats(
            model_requests=metadata.model_requests,
            tool_calls=len(tools.calls),
            input_tokens=metadata.input_tokens,
            output_tokens=metadata.output_tokens,
            tool_response_bytes_total=tool_response_bytes,
            model_visible_bytes_total=model_visible_bytes,
        ),
        trace=TraceSummary(
            span_count=metadata.model_requests + len(tools.calls),
            nested_tool_call_count=metadata.nested_tool_calls,
        ),
        tool_calls=tools.calls,
        raw={
            "code_mode": "pydantic_monty",
            "model_policy": "scripted_function_model",
            "run_code_calls": metadata.run_code_calls,
            "pydantic_tool_calls": metadata.pydantic_tool_calls,
            "nested_tool_calls": metadata.nested_tool_calls,
            "tool_outputs_model_visible": False,
        },
    )


def task_candidates(task: ProbeTask):
    from codemode_probe.workload import generate_candidates

    return generate_candidates(task.workload)


def _code_mode_metadata(run_result: Any) -> PydanticCodeModeRunMetadata:
    usage = run_result.usage()
    run_code_calls = 0
    nested_tool_calls = 0
    tool_return_part_type = _tool_return_part_type()
    for message in run_result.all_messages():
        for part in getattr(message, "parts", []):
            if not isinstance(part, tool_return_part_type) or part.tool_name != "run_code":
                continue
            run_code_calls += 1
            metadata = part.metadata if isinstance(part.metadata, dict) else {}
            nested = metadata.get("tool_calls", {})
            if isinstance(nested, dict):
                nested_tool_calls += len(nested)
    return PydanticCodeModeRunMetadata(
        model_requests=int(getattr(usage, "requests", 0)),
        input_tokens=_optional_int(getattr(usage, "input_tokens", None)),
        output_tokens=_optional_int(getattr(usage, "output_tokens", None)),
        pydantic_tool_calls=int(getattr(usage, "tool_calls", 0)),
        run_code_calls=run_code_calls,
        nested_tool_calls=nested_tool_calls,
    )


def _tool_return_part_type() -> type:
    from pydantic_ai.messages import ToolReturnPart

    return ToolReturnPart


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _code_mode_program(task: ProbeTask) -> str:
    fetch_block = (
        "fetched = await fetch_candidates(candidate_ids=candidate_ids)"
        if task.workload.tool_shape == "batch"
        else "\n".join(
            [
                "fetched = await asyncio.gather(",
                "    *[fetch_candidate(candidate_id=candidate_id) for candidate_id in candidate_ids]",
                ")",
            ]
        )
    )
    return "\n".join(
        [
            "import asyncio",
            "",
            "shards = await asyncio.gather(",
            *[
                f"    search_shard(shard_id={shard_id}),"
                for shard_id in range(task.workload.shard_count)
            ],
            ")",
            "candidate_ids = [item['id'] for shard in shards for item in shard]",
            fetch_block,
            "",
            "ranked = []",
            "for candidate in fetched:",
            "    if candidate['is_draft'] or candidate['is_bot_authored']:",
            "        continue",
            "    approval_score = min(candidate['approvals'], 4) / 4",
            "    ci_score = 1.0 if candidate['failing_checks'] == 0 else max(0.0, 1 - candidate['failing_checks'] / 4)",
            "    reaction_score = min(candidate['reactions'], 50) / 50",
            "    recency_score = max(0.0, 1 - candidate['age_days'] / 60)",
            "    size_score = max(0.0, 1 - candidate['changed_files'] / 100)",
            "    breakdown = {",
            "        'relevance': candidate['relevance'] * 0.35,",
            "        'approvals': approval_score * 0.20,",
            "        'ci': ci_score * 0.20,",
            "        'reactions': reaction_score * 0.10,",
            "        'recency': recency_score * 0.10,",
            "        'size': size_score * 0.05,",
            "    }",
            "    score = round(sum(breakdown.values()), 6)",
            "    if score <= 0:",
            "        continue",
            "    ranked.append({",
            "        'id': candidate['id'],",
            "        'score': score,",
            "        'rationale': f\"{candidate['approvals']} approvals, {candidate['failing_checks']} failing checks, {candidate['reactions']} reactions, {candidate['changed_files']} changed files\",",
            "        'score_breakdown': {key: round(value, 6) for key, value in breakdown.items()},",
            "    })",
            "ranked.sort(key=lambda item: (-item['score'], item['id']))",
            f"{{'task_id': {task.id!r}, 'candidates': ranked[:{task.workload.top_k}]}}",
        ]
    )
