from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from codemode_probe.models import ToolShape
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.executor_factory import build_executor
from codemode_probe.workload import make_probe_task


class PreflightCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    details: dict[str, object]


def run_preflight_checks() -> list[PreflightCheckResult]:
    return [
        _oracle_ceiling_check(),
        *_tool_oracle_parity_checks(),
        *_direct_mcp_tool_oracle_parity_checks(),
        *_scripted_agent_parity_checks(),
        *_code_mode_scripted_parity_checks(),
    ]


def assert_preflight_checks_pass() -> None:
    failures = [check for check in run_preflight_checks() if not check.passed]
    if failures:
        failed_names = ", ".join(check.name for check in failures)
        raise RuntimeError(f"preflight checks failed: {failed_names}")


def _oracle_ceiling_check() -> PreflightCheckResult:
    task = make_probe_task(
        "preflight-oracle-ceiling",
        seed=101,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        top_k=2,
    )
    result = BenchmarkRunner(build_executor("deterministic_oracle_client", task)).run_task(task)
    passed = (
        result.score.schema_valid
        and result.score.top_k_overlap == 1.0
        and result.score.ndcg_at_k == 1.0
        and result.execution.error is None
    )
    return PreflightCheckResult(
        name="deterministic_oracle_ceiling",
        passed=passed,
        details=_quality_details(result),
    )


def _tool_oracle_parity_checks() -> list[PreflightCheckResult]:
    return [
        _parity_check(
            name=f"tool_oracle_parity_{tool_shape.value}",
            baseline_arm="deterministic_oracle_client",
            comparison_arm="in_process_tool_oracle",
            tool_shape=tool_shape,
        )
        for tool_shape in (ToolShape.SCALAR, ToolShape.BATCH)
    ]


def _direct_mcp_tool_oracle_parity_checks() -> list[PreflightCheckResult]:
    return [
        _parity_check(
            name=f"direct_mcp_tool_oracle_parity_{tool_shape.value}",
            baseline_arm="in_process_tool_oracle",
            comparison_arm="direct_mcp_tool_oracle",
            tool_shape=tool_shape,
        )
        for tool_shape in (ToolShape.SCALAR, ToolShape.BATCH)
    ]


def _scripted_agent_parity_checks() -> list[PreflightCheckResult]:
    return [
        _parity_check(
            name=f"scripted_agent_parity_{tool_shape.value}",
            baseline_arm="in_process_tool_oracle",
            comparison_arm="direct_mcp_agent_parallel",
            tool_shape=tool_shape,
        )
        for tool_shape in (ToolShape.SCALAR, ToolShape.BATCH)
    ]


def _code_mode_scripted_parity_checks() -> list[PreflightCheckResult]:
    return [
        _parity_check(
            name=f"code_mode_scripted_parity_{tool_shape.value}",
            baseline_arm="in_process_tool_oracle",
            comparison_arm="code_mode_synthetic_scripted",
            tool_shape=tool_shape,
        )
        for tool_shape in (ToolShape.SCALAR, ToolShape.BATCH)
    ]


def _parity_check(
    *,
    name: str,
    baseline_arm: str,
    comparison_arm: str,
    tool_shape: ToolShape,
) -> PreflightCheckResult:
    task = make_probe_task(
        f"preflight-{name}",
        seed=202,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        top_k=2,
    )
    baseline = BenchmarkRunner(build_executor(baseline_arm, task)).run_task(task)
    comparison = BenchmarkRunner(build_executor(comparison_arm, task)).run_task(task)
    passed = (
        baseline.execution.answer == comparison.execution.answer
        and baseline.score.top_k_overlap == 1.0
        and comparison.score.top_k_overlap == 1.0
        and baseline.execution.error is None
        and comparison.execution.error is None
    )
    return PreflightCheckResult(
        name=name,
        passed=passed,
        details={
            "baseline_arm": baseline.arm_name,
            "comparison_arm": comparison.arm_name,
            "tool_shape": tool_shape.value,
            "baseline": _quality_details(baseline),
            "comparison": _quality_details(comparison),
        },
    )


def _quality_details(result) -> dict[str, object]:
    return {
        "arm_name": result.arm_name,
        "schema_valid": result.score.schema_valid,
        "top_k_overlap": result.score.top_k_overlap,
        "ndcg_at_k": result.score.ndcg_at_k,
        "execution_error": result.execution.error,
        "failure_reason": result.score.failure_reason,
    }
