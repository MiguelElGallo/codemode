from __future__ import annotations

from datetime import date

from codemode_probe.budget import RunBudgetConfig
from codemode_probe.costs import summarize_cost_estimates
from codemode_probe.models import ArmResult, ExecutionResult, ScoreResult, UsageStats
from codemode_probe.provider_config import openai_config


def _result(
    arm_name: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> ArmResult:
    return ArmResult(
        task_id="cost-task",
        arm_name=arm_name,
        repetition=1,
        latency_ms=1.0,
        execution=ExecutionResult(
            usage=UsageStats(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
        ),
        score=ScoreResult(
            schema_valid=True,
            top_k_overlap=1.0,
            precision_at_k=1.0,
            recall_at_k=1.0,
            ndcg_at_k=1.0,
        ),
    )


def test_summarize_cost_estimates_uses_measured_tokens_when_pricing_is_source_backed() -> None:
    rows = summarize_cost_estimates(
        [
            _result(
                "direct_mcp_agent_parallel",
                input_tokens=1000,
                output_tokens=250,
                cache_read_tokens=100,
            ),
            _result(
                "direct_mcp_agent_parallel",
                input_tokens=500,
                output_tokens=100,
                cache_write_tokens=50,
            ),
        ],
        provider_config=openai_config(
            enabled=True,
            pricing_source_id="openai-pricing-2026-05-06",
            pricing_snapshot_date=date(2026, 5, 6),
            currency="USD",
        ),
        budget_config=RunBudgetConfig(
            input_cost_per_1m_tokens=2.0,
            output_cost_per_1m_tokens=8.0,
        ),
    )

    assert rows == [
        {
            "arm_name": "direct_mcp_agent_parallel",
            "runs": 2,
            "status": "estimated",
            "reason": None,
            "input_tokens": 1500,
            "output_tokens": 350,
            "cache_read_tokens": 100,
            "cache_write_tokens": 50,
            "input_cost": 0.003,
            "output_cost": 0.0028,
            "cache_cost": None,
            "total_estimated_cost": 0.0058,
            "currency": "USD",
            "pricing_source_id": "openai-pricing-2026-05-06",
            "pricing_snapshot_date": "2026-05-06",
            "cache_pricing_status": "not_estimated",
        }
    ]


def test_summarize_cost_estimates_emits_not_estimated_without_pricing_evidence() -> None:
    rows = summarize_cost_estimates(
        [_result("arm-a", input_tokens=100, output_tokens=20)]
    )

    assert rows[0]["status"] == "not_estimated"
    assert rows[0]["reason"] == "missing_provider_pricing_evidence"
    assert rows[0]["total_estimated_cost"] is None


def test_summarize_cost_estimates_emits_not_estimated_without_token_usage() -> None:
    rows = summarize_cost_estimates(
        [_result("arm-a")],
        provider_config=openai_config(
            enabled=True,
            pricing_source_id="openai-pricing-2026-05-06",
            pricing_snapshot_date=date(2026, 5, 6),
            currency="USD",
        ),
        budget_config=RunBudgetConfig(
            input_cost_per_1m_tokens=2.0,
            output_cost_per_1m_tokens=8.0,
        ),
    )

    assert rows[0]["status"] == "not_estimated"
    assert rows[0]["reason"] == "missing_token_usage"


def test_summarize_cost_estimates_does_not_estimate_partial_token_usage() -> None:
    rows = summarize_cost_estimates(
        [
            _result("arm-a", input_tokens=100, output_tokens=20),
            _result("arm-a", input_tokens=None, output_tokens=30),
        ],
        provider_config=openai_config(
            enabled=True,
            pricing_source_id="openai-pricing-2026-05-06",
            pricing_snapshot_date=date(2026, 5, 6),
            currency="USD",
        ),
        budget_config=RunBudgetConfig(
            input_cost_per_1m_tokens=2.0,
            output_cost_per_1m_tokens=8.0,
        ),
    )

    assert rows[0]["status"] == "not_estimated"
    assert rows[0]["reason"] == "missing_token_usage"
    assert rows[0]["input_tokens"] is None
    assert rows[0]["output_tokens"] == 50
    assert rows[0]["total_estimated_cost"] is None


def test_summarize_cost_estimates_does_not_estimate_dry_run_provider_config() -> None:
    rows = summarize_cost_estimates(
        [_result("arm-a", input_tokens=100, output_tokens=20)],
        provider_config=openai_config(
            enabled=False,
            pricing_source_id="openai-pricing-2026-05-06",
            pricing_snapshot_date=date(2026, 5, 6),
            currency="USD",
        ),
        budget_config=RunBudgetConfig(
            input_cost_per_1m_tokens=2.0,
            output_cost_per_1m_tokens=8.0,
        ),
    )

    assert rows[0]["status"] == "not_estimated"
    assert rows[0]["reason"] == "dry_run_provider_config"
    assert rows[0]["total_estimated_cost"] is None
