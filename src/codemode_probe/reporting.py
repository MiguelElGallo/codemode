from __future__ import annotations

from statistics import median

from codemode_probe.models import ArmResult, ProbeTask


def summarize_results(results: list[ArmResult]) -> dict[str, object]:
    by_arm: dict[str, list[ArmResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm_name, []).append(result)

    arms = {}
    for arm_name, arm_results in by_arm.items():
        arms[arm_name] = _summarize_arm(arm_results)
    return {"schema_version": 1, "arms": arms}


def render_summary_markdown(results: list[ArmResult]) -> str:
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
            "Pairwise deltas use `direct_mcp_agent_parallel` as the default baseline when present.",
            "Cold/warm cache cohorts are not separated in this run metadata yet.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_paired_deltas(
    results: list[ArmResult],
    *,
    baseline_arm: str,
) -> list[dict[str, object]]:
    by_key: dict[tuple[str, int], dict[str, ArmResult]] = {}
    for result in results:
        by_key.setdefault((result.task_id, result.repetition), {})[result.arm_name] = result

    rows: list[dict[str, object]] = []
    for task_id, repetition in sorted(by_key):
        arm_results = by_key[(task_id, repetition)]
        baseline = arm_results.get(baseline_arm)
        if baseline is None:
            continue
        for arm_name in sorted(arm_results):
            if arm_name == baseline_arm:
                continue
            comparison = arm_results[arm_name]
            rows.append(_paired_delta_row(task_id, repetition, baseline, comparison))
    return rows


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
    successes = [
        result
        for result in results
        if result.score.schema_valid
        and result.score.failure_reason is None
        and not result.timed_out
        and result.execution.error is None
    ]
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


def _paired_delta_row(
    task_id: str,
    repetition: int,
    baseline: ArmResult,
    comparison: ArmResult,
) -> dict[str, object]:
    baseline_visible_fraction = _visible_fraction(baseline)
    comparison_visible_fraction = _visible_fraction(comparison)
    return {
        "task_id": task_id,
        "repetition": repetition,
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
    index = max(0, min(len(sorted_values) - 1, int((len(sorted_values) - 1) * percentile)))
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
