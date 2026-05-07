from __future__ import annotations

import math
import random
from statistics import median
from typing import TYPE_CHECKING

from codemode_probe.models import ArmResult, ProbeTask

if TYPE_CHECKING:
    from codemode_probe.preflight import PreflightCheckResult
    from codemode_probe.provider_config import LiveProviderConfig
    from codemode_probe.suite import BenchmarkSuiteConfig


def summarize_results(results: list[ArmResult]) -> dict[str, object]:
    by_arm: dict[str, list[ArmResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm_name, []).append(result)

    arms = {}
    for arm_name, arm_results in by_arm.items():
        arms[arm_name] = _summarize_arm(arm_results)
    return {"schema_version": 1, "arms": arms}


def render_summary_markdown(
    results: list[ArmResult],
    *,
    paired_baseline_arm: str = "direct_mcp_agent_parallel",
    warnings: list[dict[str, object]] | None = None,
) -> str:
    summary = summarize_results(results)
    lines = [
        "# Benchmark Summary",
        "",
        "| Arm | Runs | Success rate | Mean top-k | Mean NDCG | P95 latency ms | Model requests | Tool calls | Visible fraction | Suppression | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_name, arm_summary in summary["arms"].items():
        assert isinstance(arm_summary, dict)
        lines.append(
            "| "
            + " | ".join(
                [
                    arm_name,
                    str(arm_summary["runs"]),
                    _fmt_float(arm_summary["success_rate"]),
                    _fmt_float(arm_summary["mean_top_k_overlap"]),
                    _fmt_float(arm_summary["mean_ndcg_at_k"]),
                    _fmt_float(arm_summary["p95_latency_ms"]),
                    str(arm_summary["model_requests_total"]),
                    str(arm_summary["tool_calls_total"]),
                    _fmt_float(arm_summary["visible_fraction"]),
                    _fmt_float(arm_summary["payload_suppression_ratio"]),
                    str(arm_summary["failures"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Payload suppression is `1 - model_visible_bytes_total / tool_response_bytes_total`.",
            f"Pairwise deltas use `{paired_baseline_arm}` as the baseline when present.",
            "Cache cohorts are recorded as run/result metadata; provider cache enforcement is adapter-specific.",
            "",
        ]
    )
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(
                f"- `{warning['id']}` ({warning['severity']}): {warning['message']}"
            )
        lines.append("")
    return "\n".join(lines)


def collect_run_warnings(
    results: list[ArmResult],
    *,
    provider_config: LiveProviderConfig | None = None,
    suite_config: BenchmarkSuiteConfig | None = None,
    preflight_results: list[PreflightCheckResult] | None = None,
    paired_baseline_arm: str = "direct_mcp_agent_parallel",
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []

    def add(
        warning_id: str,
        severity: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        warnings.append(
            {
                "id": warning_id,
                "severity": severity,
                "message": message,
                "details": details or {},
            }
        )

    if provider_config is None:
        add(
            "synthetic_harness_validation",
            "info",
            "No live provider config was supplied; claims are limited to synthetic harness validation.",
        )
    elif not provider_config.enabled:
        add(
            "dry_run_provider_config",
            "warning",
            "Provider config was recorded in dry-run mode; no live provider calls were made.",
            {"provider": provider_config.provider.value, "model": provider_config.model},
        )

    if provider_config is not None:
        model_evidence_fields = (
            "model_version",
            "api_version",
            "sdk_version",
            "model_docs_source_id",
        )
        missing_model_fields = [
            field for field in model_evidence_fields if getattr(provider_config, field) is None
        ]
        if missing_model_fields:
            add(
                "missing_provider_model_evidence",
                "warning",
                "Provider config is missing model/API/SDK evidence fields.",
                {"missing_fields": missing_model_fields},
            )

        pricing_fields = (
            "pricing_source_id",
            "pricing_snapshot_date",
            "currency",
        )
        missing_pricing_fields = [
            field for field in pricing_fields if getattr(provider_config, field) is None
        ]
        if missing_pricing_fields:
            add(
                "missing_provider_pricing_evidence",
                "warning",
                "Provider config is missing pricing evidence fields; cost claims are not source-backed.",
                {"missing_fields": missing_pricing_fields},
            )
        elif (
            provider_config.provider.value == "azure_openai"
            and str(provider_config.pricing_source_id).startswith("openai-")
        ):
            add(
                "azure_pricing_source_not_verified",
                "warning",
                "Azure OpenAI run uses non-Azure pricing evidence; treat cost estimates as assumption-backed, not Azure billing evidence.",
                {"pricing_source_id": provider_config.pricing_source_id},
            )

    if preflight_results is None:
        add(
            "preflight_not_run",
            "warning",
            "Preflight checks were skipped or unavailable for this artifact set.",
        )
    else:
        failed_preflight = [result.name for result in preflight_results if not result.passed]
        if failed_preflight:
            add(
                "preflight_failed",
                "error",
                "One or more preflight checks failed.",
                {"failed_checks": failed_preflight},
            )

    if suite_config is None:
        add(
            "suite_config_missing",
            "warning",
            "No suite config was recorded; repetition and arm-order controls are unspecified.",
        )
    else:
        if suite_config.repetitions < 3:
            add(
                "low_repetition_count",
                "warning",
                "Run has fewer than 3 repetitions; treat comparisons as exploratory.",
                {"repetitions": suite_config.repetitions},
            )
        if provider_config is not None and provider_config.enabled and suite_config.arm_order == "fixed":
            add(
                "fixed_arm_order_live_provider",
                "warning",
                "Live-provider run uses fixed arm order; latency comparisons may include ordering bias.",
                {"arm_order": suite_config.arm_order},
            )

    if any(result.cache_warmup_run for result in results):
        add(
            "cache_warmup_rows_present",
            "info",
            "Cache warmup rows are present and should be excluded from warm-cache effect estimates.",
        )

    pairing = summarize_pairing_coverage(results, baseline_arm=paired_baseline_arm)
    if pairing["unpaired_comparisons_total"]:
        add(
            "unpaired_comparisons",
            "warning",
            "Some comparison rows could not be paired with the configured baseline arm.",
            {
                "baseline_arm": pairing["baseline_arm"],
                "unpaired_comparisons_total": pairing["unpaired_comparisons_total"],
            },
        )
    if pairing["duplicate_trial_arm_groups"]:
        add(
            "duplicate_trial_arm_groups",
            "warning",
            "Duplicate arm rows were found for one or more trial keys.",
            {"duplicate_trial_arm_groups": pairing["duplicate_trial_arm_groups"]},
        )

    failure_rows = summarize_failure_modes(results)
    if failure_rows:
        add(
            "run_failures_present",
            "error",
            "One or more runs failed schema, execution, scoring, or timeout checks.",
            {"failure_mode_count": len(failure_rows)},
        )
    if any(result.timed_out for result in results):
        add(
            "timeouts_present",
            "error",
            "One or more runs timed out.",
            {"timed_out_runs": sum(1 for result in results if result.timed_out)},
        )

    return sorted(warnings, key=lambda warning: str(warning["id"]))


def summarize_paired_deltas(
    results: list[ArmResult],
    *,
    baseline_arm: str,
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int, str | None], dict[str, list[ArmResult]]] = {}
    for result in results:
        by_key.setdefault(
            (result.task_id, result.repetition, result.trial_id), {}
        ).setdefault(result.arm_name, []).append(result)

    rows: list[dict[str, object]] = []
    for task_id, repetition, trial_id in sorted(
        by_key,
        key=lambda key: (key[0], key[1], key[2] or ""),
    ):
        arm_results = by_key[(task_id, repetition, trial_id)]
        if any(len(duplicates) != 1 for duplicates in arm_results.values()):
            continue
        baseline_rows = arm_results.get(baseline_arm)
        if baseline_rows is None:
            continue
        baseline = baseline_rows[0]
        for arm_name in sorted(arm_results):
            if arm_name == baseline_arm:
                continue
            comparison = arm_results[arm_name][0]
            rows.append(_paired_delta_row(task_id, repetition, baseline, comparison))
    return rows


def summarize_pairing_coverage(
    results: list[ArmResult],
    *,
    baseline_arm: str,
) -> dict[str, object]:
    by_key: dict[tuple[str, int, str | None], dict[str, list[ArmResult]]] = {}
    for result in results:
        by_key.setdefault(
            (result.task_id, result.repetition, result.trial_id),
            {},
        ).setdefault(result.arm_name, []).append(result)

    trials_missing_baseline = 0
    comparison_results_total = 0
    paired_comparisons_total = 0
    duplicate_trial_arm_groups = 0
    missing_baseline_trial_keys: list[dict[str, object]] = []

    for task_id, repetition, trial_id in sorted(
        by_key,
        key=lambda key: (key[0], key[1], key[2] or ""),
    ):
        arm_results = by_key[(task_id, repetition, trial_id)]
        baseline_count = len(arm_results.get(baseline_arm, []))
        comparison_count = sum(
            len(rows) for arm_name, rows in arm_results.items() if arm_name != baseline_arm
        )
        comparison_results_total += comparison_count
        has_duplicate_arm_rows = any(len(rows) > 1 for rows in arm_results.values())
        if baseline_count == 0:
            trials_missing_baseline += 1
            missing_baseline_trial_keys.append(
                {
                    "task_id": task_id,
                    "repetition": repetition,
                    "trial_id": trial_id,
                    "comparison_results": comparison_count,
                }
            )
        elif not has_duplicate_arm_rows:
            paired_comparisons_total += comparison_count

        duplicate_trial_arm_groups += sum(1 for rows in arm_results.values() if len(rows) > 1)

    return {
        "baseline_arm": baseline_arm,
        "trial_count": len(by_key),
        "trials_with_baseline": len(by_key) - trials_missing_baseline,
        "trials_missing_baseline": trials_missing_baseline,
        "comparison_results_total": comparison_results_total,
        "paired_comparisons_total": paired_comparisons_total,
        "unpaired_comparisons_total": comparison_results_total - paired_comparisons_total,
        "duplicate_trial_arm_groups": duplicate_trial_arm_groups,
        "missing_baseline_trial_keys": missing_baseline_trial_keys,
    }


def summarize_paired_delta_groups(
    paired_deltas: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in paired_deltas:
        groups.setdefault(
            (str(row["baseline_arm"]), str(row["comparison_arm"])),
            [],
        ).append(row)

    summaries: list[dict[str, object]] = []
    for baseline_arm, comparison_arm in sorted(groups):
        rows = groups[(baseline_arm, comparison_arm)]
        summaries.append(
            {
                "baseline_arm": baseline_arm,
                "comparison_arm": comparison_arm,
                "pairs": len(rows),
                "mean_delta_ndcg_at_k": _mean(
                    [float(row["delta_ndcg_at_k"]) for row in rows]
                ),
                "mean_delta_top_k_overlap": _mean(
                    [float(row["delta_top_k_overlap"]) for row in rows]
                ),
                "median_delta_latency_ms": round(
                    median(float(row["delta_latency_ms"]) for row in rows),
                    3,
                ),
                "mean_delta_latency_ms": _mean(
                    [float(row["delta_latency_ms"]) for row in rows]
                ),
                "mean_delta_model_requests": _mean(
                    [float(row["delta_model_requests"]) for row in rows]
                ),
                "mean_delta_tool_calls": _mean(
                    [float(row["delta_tool_calls"]) for row in rows]
                ),
                "mean_delta_tool_response_bytes": _mean(
                    [float(row["delta_tool_response_bytes"]) for row in rows]
                ),
                "mean_delta_model_visible_bytes": _mean(
                    [float(row["delta_model_visible_bytes"]) for row in rows]
                ),
            }
        )
    return summaries


def summarize_paired_uncertainty(
    paired_deltas: list[dict[str, object]],
    *,
    bootstrap_iterations: int = 1000,
    random_seed: int = 1,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in paired_deltas:
        groups.setdefault(
            (str(row["baseline_arm"]), str(row["comparison_arm"])),
            [],
        ).append(row)

    summaries: list[dict[str, object]] = []
    for baseline_arm, comparison_arm in sorted(groups):
        rows = groups[(baseline_arm, comparison_arm)]
        rng = random.Random(f"{random_seed}:{baseline_arm}:{comparison_arm}")
        summaries.append(
            {
                "baseline_arm": baseline_arm,
                "comparison_arm": comparison_arm,
                "pairs": len(rows),
                "bootstrap_iterations": bootstrap_iterations,
                "metrics": {
                    metric: _bootstrap_metric_summary(
                        [float(row[metric]) for row in rows],
                        rng=rng,
                        iterations=bootstrap_iterations,
                    )
                    for metric in (
                        "delta_ndcg_at_k",
                        "delta_top_k_overlap",
                        "delta_latency_ms",
                        "delta_model_requests",
                        "delta_tool_calls",
                        "delta_tool_response_bytes",
                        "delta_model_visible_bytes",
                    )
                },
            }
        )
    return summaries


def summarize_workload_regimes(
    tasks: list[ProbeTask],
    results: list[ArmResult],
) -> list[dict[str, object]]:
    tasks_by_id = {task.id: task for task in tasks}
    groups: dict[tuple[object, ...], list[ArmResult]] = {}
    for result in results:
        task = tasks_by_id.get(result.task_id)
        if task is None:
            continue
        key = (
            task.workload.task_family.value,
            task.workload.tool_shape.value,
            task.workload.candidate_count,
            task.workload.payload_bytes,
            task.workload.top_k,
            result.arm_name,
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, object]] = []
    for key in sorted(groups):
        task_family, tool_shape, candidate_count, payload_bytes, top_k, arm_name = key
        arm_summary = _summarize_arm(groups[key])
        rows.append(
            {
                "task_family": task_family,
                "tool_shape": tool_shape,
                "candidate_count": candidate_count,
                "payload_bytes": payload_bytes,
                "top_k": top_k,
                "arm_name": arm_name,
                "runs": arm_summary["runs"],
                "success_rate": arm_summary["success_rate"],
                "mean_ndcg_at_k": arm_summary["mean_ndcg_at_k"],
                "mean_top_k_overlap": arm_summary["mean_top_k_overlap"],
                "median_latency_ms": arm_summary["median_latency_ms"],
                "p95_latency_ms": arm_summary["p95_latency_ms"],
                "mean_tool_calls": arm_summary["mean_tool_calls"],
                "mean_model_requests": arm_summary["mean_model_requests"],
                "tool_response_bytes_total": arm_summary["tool_response_bytes_total"],
                "model_visible_bytes_total": arm_summary["model_visible_bytes_total"],
                "visible_fraction": arm_summary["visible_fraction"],
                "payload_suppression_ratio": arm_summary["payload_suppression_ratio"],
            }
        )
    return rows


def summarize_cache_cohorts(results: list[ArmResult]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[ArmResult]] = {}
    for result in results:
        key = (
            result.arm_name,
            result.cache_policy.value,
            result.cache_state.value,
            result.cache_namespace,
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, object]] = []
    for key in sorted(groups, key=_sortable_group_key):
        arm_name, cache_policy, cache_state, cache_namespace = key
        arm_summary = _summarize_arm(groups[key])
        rows.append(
            {
                "arm_name": arm_name,
                "cache_policy": cache_policy,
                "cache_state": cache_state,
                "cache_namespace": cache_namespace,
                "runs": arm_summary["runs"],
                "success_rate": arm_summary["success_rate"],
                "mean_ndcg_at_k": arm_summary["mean_ndcg_at_k"],
                "mean_top_k_overlap": arm_summary["mean_top_k_overlap"],
                "median_latency_ms": arm_summary["median_latency_ms"],
                "p95_latency_ms": arm_summary["p95_latency_ms"],
                "mean_model_requests": arm_summary["mean_model_requests"],
                "mean_tool_calls": arm_summary["mean_tool_calls"],
                "input_tokens_total": arm_summary["input_tokens_total"],
                "output_tokens_total": arm_summary["output_tokens_total"],
                "cache_read_tokens_total": arm_summary["cache_read_tokens_total"],
                "cache_write_tokens_total": arm_summary["cache_write_tokens_total"],
                "tool_response_bytes_total": arm_summary["tool_response_bytes_total"],
                "model_visible_bytes_total": arm_summary["model_visible_bytes_total"],
                "visible_fraction": arm_summary["visible_fraction"],
                "payload_suppression_ratio": arm_summary["payload_suppression_ratio"],
            }
        )
    return rows


def summarize_failure_modes(results: list[ArmResult]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[ArmResult]] = {}
    for result in results:
        if _is_success(result):
            continue
        key = (
            result.arm_name,
            (
                result.execution.trace.failure_category.value
                if result.execution.trace.failure_category is not None
                else None
            ),
            result.execution.error,
            (
                result.score.failure_reason.value
                if result.score.failure_reason is not None
                else None
            ),
            result.timed_out,
            result.score.schema_valid,
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda item: tuple("" if value is None else value for value in item)):
        (
            arm_name,
            failure_category,
            execution_error,
            score_failure_reason,
            timed_out,
            schema_valid,
        ) = key
        grouped_results = groups[key]
        rows.append(
            {
                "arm_name": arm_name,
                "failure_category": failure_category,
                "execution_error": execution_error,
                "score_failure_reason": score_failure_reason,
                "timed_out": timed_out,
                "schema_valid": schema_valid,
                "runs": len(grouped_results),
                "task_ids": sorted({result.task_id for result in grouped_results}),
                "trial_ids": sorted(
                    {result.trial_id for result in grouped_results if result.trial_id is not None}
                ),
            }
        )
    return rows


def _sortable_group_key(group_key: tuple[object, ...]) -> tuple[object, ...]:
    return tuple("" if value is None else value for value in group_key)


def _summarize_arm(results: list[ArmResult]) -> dict[str, object]:
    runs = len(results)
    latencies = [result.latency_ms for result in results]
    ndcgs = [result.score.ndcg_at_k for result in results]
    model_visible = sum(
        result.execution.usage.model_visible_bytes_total or 0 for result in results
    )
    tool_bytes = sum(result.execution.usage.tool_response_bytes_total for result in results)
    visible_fraction = model_visible / tool_bytes if tool_bytes else None
    suppression = 1 - visible_fraction if visible_fraction is not None else None
    successes = [result for result in results if _is_success(result)]
    return {
        "runs": runs,
        "schema_valid": sum(1 for result in results if result.score.schema_valid),
        "schema_valid_rate": _ratio(
            sum(1 for result in results if result.score.schema_valid), runs
        ),
        "successes": len(successes),
        "success_rate": _ratio(len(successes), runs),
        "timeout_rate": _ratio(sum(1 for result in results if result.timed_out), runs),
        "failures": sum(1 for result in results if result.execution.error is not None),
        "mean_latency_ms": _mean(latencies),
        "median_latency_ms": round(median(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_top_k_overlap": _mean([result.score.top_k_overlap for result in results]),
        "mean_precision_at_k": _mean([result.score.precision_at_k for result in results]),
        "mean_recall_at_k": _mean([result.score.recall_at_k for result in results]),
        "mean_ndcg_at_k": _mean(ndcgs),
        "median_ndcg_at_k": round(median(ndcgs), 6) if ndcgs else 0.0,
        "model_requests_total": sum(result.execution.usage.model_requests for result in results),
        "mean_model_requests": _mean(
            [float(result.execution.usage.model_requests) for result in results]
        ),
        "tool_calls_total": sum(result.execution.usage.tool_calls for result in results),
        "mean_tool_calls": _mean(
            [float(result.execution.usage.tool_calls) for result in results]
        ),
        "failed_tool_calls_total": sum(
            result.execution.usage.failed_tool_calls for result in results
        ),
        "mean_failed_tool_calls": _mean(
            [float(result.execution.usage.failed_tool_calls) for result in results]
        ),
        "input_tokens_total": _sum_optional(
            result.execution.usage.input_tokens for result in results
        ),
        "output_tokens_total": _sum_optional(
            result.execution.usage.output_tokens for result in results
        ),
        "cache_read_tokens_total": _sum_optional(
            result.execution.usage.cache_read_tokens for result in results
        ),
        "cache_write_tokens_total": _sum_optional(
            result.execution.usage.cache_write_tokens for result in results
        ),
        "tool_response_bytes_total": tool_bytes,
        "model_visible_bytes_total": model_visible,
        "hidden_bytes_total": max(0, tool_bytes - model_visible),
        "visible_fraction": round(visible_fraction, 6) if visible_fraction is not None else None,
        "payload_suppression_ratio": round(suppression, 6) if suppression is not None else None,
    }


def _is_success(result: ArmResult) -> bool:
    return (
        result.score.schema_valid
        and result.score.failure_reason is None
        and not result.timed_out
        and result.execution.error is None
    )


def _paired_delta_row(
    task_id: str,
    repetition: int,
    baseline: ArmResult,
    comparison: ArmResult,
) -> dict[str, object]:
    baseline_visible_fraction = _visible_fraction(baseline)
    comparison_visible_fraction = _visible_fraction(comparison)
    arm_order = baseline.arm_order or comparison.arm_order
    return {
        "task_id": task_id,
        "repetition": repetition,
        "trial_id": baseline.trial_id,
        "arm_order": list(arm_order),
        "baseline_arm_order_index": baseline.arm_order_index,
        "comparison_arm_order_index": comparison.arm_order_index,
        "baseline_arm": baseline.arm_name,
        "comparison_arm": comparison.arm_name,
        "delta_ndcg_at_k": round(comparison.score.ndcg_at_k - baseline.score.ndcg_at_k, 6),
        "delta_top_k_overlap": round(
            comparison.score.top_k_overlap - baseline.score.top_k_overlap, 6
        ),
        "delta_latency_ms": round(comparison.latency_ms - baseline.latency_ms, 3),
        "latency_ratio": _safe_ratio(comparison.latency_ms, baseline.latency_ms),
        "delta_tool_calls": (
            comparison.execution.usage.tool_calls - baseline.execution.usage.tool_calls
        ),
        "delta_model_requests": (
            comparison.execution.usage.model_requests
            - baseline.execution.usage.model_requests
        ),
        "delta_tool_response_bytes": (
            comparison.execution.usage.tool_response_bytes_total
            - baseline.execution.usage.tool_response_bytes_total
        ),
        "delta_model_visible_bytes": (
            (comparison.execution.usage.model_visible_bytes_total or 0)
            - (baseline.execution.usage.model_visible_bytes_total or 0)
        ),
        "payload_visible_ratio_baseline": baseline_visible_fraction,
        "payload_visible_ratio_comparison": comparison_visible_fraction,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _bootstrap_metric_summary(
    values: list[float],
    *,
    rng: random.Random,
    iterations: int,
) -> dict[str, object]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    if len(values) == 1 or iterations <= 0:
        mean_value = _mean(values)
        return {
            "mean": mean_value,
            "ci95_low": mean_value,
            "ci95_high": mean_value,
        }

    bootstrapped_means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped_means.append(_mean(sample))

    sorted_means = sorted(bootstrapped_means)
    return {
        "mean": _mean(values),
        "ci95_low": _quantile(sorted_means, 0.025),
        "ci95_high": _quantile(sorted_means, 0.975),
    }


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * quantile)))
    return round(sorted_values[index], 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _visible_fraction(result: ArmResult) -> float | None:
    tool_bytes = result.execution.usage.tool_response_bytes_total
    if tool_bytes == 0:
        return None
    return round((result.execution.usage.model_visible_bytes_total or 0) / tool_bytes, 6)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * percentile) - 1))
    return round(sorted_values[index], 3)


def _sum_optional(values) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _fmt_float(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
