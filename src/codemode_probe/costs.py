from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from codemode_probe.models import ArmResult

if TYPE_CHECKING:
    from codemode_probe.budget import RunBudgetConfig
    from codemode_probe.provider_config import LiveProviderConfig


def summarize_cost_estimates(
    results: list[ArmResult],
    *,
    provider_config: LiveProviderConfig | None = None,
    budget_config: RunBudgetConfig | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm_name, arm_results in sorted(_group_by_arm(results).items()):
        input_tokens = _sum_required(
            result.execution.usage.input_tokens for result in arm_results
        )
        output_tokens = _sum_required(
            result.execution.usage.output_tokens for result in arm_results
        )
        cache_read_tokens = _sum_optional(
            result.execution.usage.cache_read_tokens for result in arm_results
        )
        cache_write_tokens = _sum_optional(
            result.execution.usage.cache_write_tokens for result in arm_results
        )
        reason = _not_estimated_reason(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_config=provider_config,
            budget_config=budget_config,
        )
        estimated = reason is None
        input_cost = None
        output_cost = None
        total_cost = None
        if estimated:
            assert budget_config is not None
            assert budget_config.input_cost_per_1m_tokens is not None
            assert budget_config.output_cost_per_1m_tokens is not None
            assert input_tokens is not None
            assert output_tokens is not None
            input_cost = _token_cost(input_tokens, budget_config.input_cost_per_1m_tokens)
            output_cost = _token_cost(output_tokens, budget_config.output_cost_per_1m_tokens)
            total_cost = round(input_cost + output_cost, 6)

        rows.append(
            {
                "arm_name": arm_name,
                "runs": len(arm_results),
                "status": "estimated" if estimated else "not_estimated",
                "reason": reason,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cache_cost": None,
                "total_estimated_cost": total_cost,
                "currency": _currency(provider_config, budget_config),
                "pricing_source_id": (
                    provider_config.pricing_source_id
                    if provider_config is not None
                    else None
                ),
                "pricing_snapshot_date": _date_string(
                    provider_config.pricing_snapshot_date
                    if provider_config is not None
                    else None
                ),
                "cache_pricing_status": _cache_pricing_status(
                    cache_read_tokens,
                    cache_write_tokens,
                ),
            }
        )
    return rows


def _group_by_arm(results: list[ArmResult]) -> dict[str, list[ArmResult]]:
    by_arm: dict[str, list[ArmResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm_name, []).append(result)
    return by_arm


def _not_estimated_reason(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    provider_config: LiveProviderConfig | None,
    budget_config: RunBudgetConfig | None,
) -> str | None:
    if input_tokens is None or output_tokens is None:
        return "missing_token_usage"
    if provider_config is None:
        return "missing_provider_pricing_evidence"
    if not provider_config.enabled:
        return "dry_run_provider_config"
    if (
        provider_config.pricing_source_id is None
        or provider_config.pricing_snapshot_date is None
        or provider_config.currency is None
    ):
        return "missing_provider_pricing_evidence"
    if (
        budget_config is None
        or budget_config.input_cost_per_1m_tokens is None
        or budget_config.output_cost_per_1m_tokens is None
    ):
        return "missing_token_price_rates"
    return None


def _sum_required(values) -> int | None:
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(materialized)


def _sum_optional(values) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _token_cost(tokens: int, cost_per_1m_tokens: float) -> float:
    return round(tokens * cost_per_1m_tokens / 1_000_000, 6)


def _currency(
    provider_config: LiveProviderConfig | None,
    budget_config: RunBudgetConfig | None,
) -> str | None:
    if provider_config is not None and provider_config.currency is not None:
        return provider_config.currency
    if budget_config is not None:
        return budget_config.currency
    return None


def _date_string(value: date | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _cache_pricing_status(
    cache_read_tokens: int | None,
    cache_write_tokens: int | None,
) -> str:
    if cache_read_tokens is None and cache_write_tokens is None:
        return "no_cache_tokens_reported"
    return "not_estimated"
