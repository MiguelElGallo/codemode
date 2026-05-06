from __future__ import annotations

from statistics import median

from codemode_probe.models import ArmResult


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
            "Cold/warm cache cohorts are not separated in this run metadata yet.",
            "",
        ]
    )
    return "\n".join(lines)


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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


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
