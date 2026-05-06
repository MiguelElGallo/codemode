from __future__ import annotations

import pytest

from codemode_probe import preflight
from codemode_probe.preflight import (
    PreflightCheckResult,
    assert_preflight_checks_pass,
    run_preflight_checks,
)


def test_run_preflight_checks_covers_oracle_parity_and_code_mode_controls() -> None:
    results = run_preflight_checks()

    assert [result.name for result in results] == [
        "deterministic_oracle_ceiling",
        "tool_oracle_parity_scalar",
        "tool_oracle_parity_batch",
        "direct_mcp_tool_oracle_parity_scalar",
        "direct_mcp_tool_oracle_parity_batch",
        "scripted_agent_parity_scalar",
        "scripted_agent_parity_batch",
        "code_mode_scripted_parity_scalar",
        "code_mode_scripted_parity_batch",
        "calibration_schema_invalid",
        "calibration_task_id_mismatch",
        "calibration_partial_overlap",
        "calibration_hallucinated_ids",
        "calibration_duplicate_ids",
        "calibration_degraded_ranking",
    ]
    assert all(result.passed for result in results)
    assert all(result.details for result in results)


def test_assert_preflight_checks_pass_raises_with_failed_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_preflight_checks",
        lambda: [
            PreflightCheckResult(name="ok", passed=True, details={}),
            PreflightCheckResult(name="bad", passed=False, details={}),
        ],
    )

    with pytest.raises(RuntimeError, match="preflight checks failed: bad"):
        assert_preflight_checks_pass()
