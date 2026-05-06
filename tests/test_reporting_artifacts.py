from __future__ import annotations

import json
from pathlib import Path

from codemode_probe.artifacts import write_run_artifacts
from codemode_probe.models import ArmResult, ExecutionResult, ScoreResult, UsageStats
from codemode_probe.reporting import render_summary_markdown, summarize_results
from codemode_probe.workload import make_probe_task


def _result(
    arm_name: str,
    *,
    latency_ms: float = 100.0,
    schema_valid: bool = True,
    timed_out: bool = False,
    failure_reason: str | None = None,
    top_k_overlap: float = 0.5,
    precision_at_k: float = 0.25,
    recall_at_k: float = 0.75,
    ndcg_at_k: float = 0.5,
    error: str | None = None,
    usage: UsageStats | None = None,
) -> ArmResult:
    return ArmResult(
        task_id="task-1",
        arm_name=arm_name,
        repetition=1,
        latency_ms=latency_ms,
        timed_out=timed_out,
        execution=ExecutionResult(usage=usage or UsageStats(), error=error),
        score=ScoreResult(
            schema_valid=schema_valid,
            timed_out=timed_out,
            top_k_overlap=top_k_overlap,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            failure_reason=failure_reason,
        ),
    )


def test_summarize_results_preserves_existing_keys_and_adds_quality_efficiency_fields() -> None:
    results = [
        _result(
            "arm-a",
            usage=UsageStats(
                model_requests=2,
                tool_calls=3,
                failed_tool_calls=1,
                input_tokens=11,
                output_tokens=7,
                cache_read_tokens=5,
                tool_response_bytes_total=100,
                model_visible_bytes_total=25,
            ),
        )
    ]

    arm = summarize_results(results)["arms"]["arm-a"]  # type: ignore[index]

    assert set(arm) == {
        "runs",
        "schema_valid",
        "schema_valid_rate",
        "successes",
        "success_rate",
        "timeout_rate",
        "failures",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_top_k_overlap",
        "mean_precision_at_k",
        "mean_recall_at_k",
        "mean_ndcg_at_k",
        "median_ndcg_at_k",
        "model_requests_total",
        "mean_model_requests",
        "tool_calls_total",
        "mean_tool_calls",
        "failed_tool_calls_total",
        "mean_failed_tool_calls",
        "input_tokens_total",
        "output_tokens_total",
        "cache_read_tokens_total",
        "cache_write_tokens_total",
        "tool_response_bytes_total",
        "model_visible_bytes_total",
        "hidden_bytes_total",
        "visible_fraction",
        "payload_suppression_ratio",
    }
    assert arm["runs"] == 1
    assert arm["success_rate"] == 1.0
    assert arm["mean_precision_at_k"] == 0.25
    assert arm["mean_recall_at_k"] == 0.75
    assert arm["failed_tool_calls_total"] == 1
    assert arm["input_tokens_total"] == 11


def test_summarize_results_payload_visibility_and_suppression_math() -> None:
    summary = summarize_results(
        [
            _result(
                "arm-a",
                usage=UsageStats(
                    tool_response_bytes_total=100,
                    model_visible_bytes_total=25,
                ),
            ),
            _result(
                "arm-a",
                usage=UsageStats(
                    tool_response_bytes_total=300,
                    model_visible_bytes_total=75,
                ),
            ),
        ]
    )

    arm = summary["arms"]["arm-a"]  # type: ignore[index]
    assert arm["tool_response_bytes_total"] == 400
    assert arm["model_visible_bytes_total"] == 100
    assert arm["hidden_bytes_total"] == 300
    assert arm["visible_fraction"] == 0.25
    assert arm["payload_suppression_ratio"] == 0.75


def test_summarize_results_optional_token_aggregation_ignores_none_until_all_missing() -> None:
    summary = summarize_results(
        [
            _result(
                "mixed",
                usage=UsageStats(
                    input_tokens=10,
                    output_tokens=None,
                    cache_read_tokens=1,
                    cache_write_tokens=None,
                ),
            ),
            _result(
                "mixed",
                usage=UsageStats(
                    input_tokens=None,
                    output_tokens=20,
                    cache_read_tokens=None,
                    cache_write_tokens=None,
                ),
            ),
            _result("all-none", usage=UsageStats()),
        ]
    )

    mixed = summary["arms"]["mixed"]  # type: ignore[index]
    assert mixed["input_tokens_total"] == 10
    assert mixed["output_tokens_total"] == 20
    assert mixed["cache_read_tokens_total"] == 1
    assert mixed["cache_write_tokens_total"] is None

    all_none = summary["arms"]["all-none"]  # type: ignore[index]
    assert all_none["input_tokens_total"] is None
    assert all_none["output_tokens_total"] is None
    assert all_none["cache_read_tokens_total"] is None
    assert all_none["cache_write_tokens_total"] is None


def test_summarize_results_median_and_p95_metrics() -> None:
    results = [
        _result(
            "arm-a",
            latency_ms=float(index * 10),
            ndcg_at_k=round(index / 20, 2),
        )
        for index in range(21)
    ]

    arm = summarize_results(results)["arms"]["arm-a"]  # type: ignore[index]

    assert arm["median_latency_ms"] == 100.0
    assert arm["p95_latency_ms"] == 190.0
    assert arm["median_ndcg_at_k"] == 0.5


def test_render_summary_markdown_is_deterministic_and_includes_caveats() -> None:
    results = [
        _result(
            "arm-b",
            latency_ms=100,
            top_k_overlap=0.5,
            ndcg_at_k=0.4,
            usage=UsageStats(
                model_requests=2,
                tool_calls=4,
                tool_response_bytes_total=1000,
                model_visible_bytes_total=250,
            ),
        ),
        _result(
            "arm-a",
            latency_ms=50,
            top_k_overlap=1,
            ndcg_at_k=1,
            usage=UsageStats(model_requests=1, tool_calls=1),
        ),
    ]

    assert render_summary_markdown(results) == "\n".join(
        [
            "# Benchmark Summary",
            "",
            "| Arm | Runs | Success rate | Mean top-k | Mean NDCG | P95 latency ms | Model requests | Tool calls | Visible fraction | Suppression | Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| arm-b | 1 | 1.000 | 0.500 | 0.400 | 100.000 | 2 | 4 | 0.250 | 0.750 | 0 |",
            "| arm-a | 1 | 1.000 | 1.000 | 1.000 | 50.000 | 1 | 1 | n/a | n/a | 0 |",
            "",
            "Payload suppression is `1 - model_visible_bytes_total / tool_response_bytes_total`.",
            "Cold/warm cache cohorts are not separated in this run metadata yet.",
            "",
        ]
    )


def test_write_run_artifacts_writes_report_without_replacing_summary_or_results(
    tmp_path: Path,
) -> None:
    task = make_probe_task(
        "task-1",
        seed=1,
        shard_count=1,
        candidates_per_shard=2,
        payload_bytes=4,
        top_k=1,
    )
    results = [
        _result(
            "arm-a",
            usage=UsageStats(
                model_requests=1,
                tool_calls=2,
                tool_response_bytes_total=80,
                model_visible_bytes_total=20,
            ),
        )
    ]

    write_run_artifacts(tmp_path, [task], results)

    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith(
        "# Benchmark Summary\n"
    )
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summarize_results(
        results
    )
    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["arm_name"] for row in jsonl_rows] == ["arm-a"]
