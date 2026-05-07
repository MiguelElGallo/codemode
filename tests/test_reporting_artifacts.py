from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from codemode_probe.artifacts import write_run_artifacts
from codemode_probe.models import (
    ArmResult,
    CachePolicy,
    CacheState,
    ExecutionResult,
    FailureCategory,
    ScoreResult,
    TaskFamily,
    ToolShape,
    TraceSummary,
    UsageStats,
)
from codemode_probe.preflight import PreflightCheckResult
from codemode_probe.provider_config import azure_openai_config, openai_config
from codemode_probe.reporting import (
    collect_run_warnings,
    render_summary_markdown,
    summarize_cache_cohorts,
    summarize_failure_modes,
    summarize_pairing_coverage,
    summarize_paired_delta_groups,
    summarize_paired_deltas,
    summarize_paired_uncertainty,
    summarize_results,
    summarize_workload_regimes,
)
from codemode_probe.suite import BenchmarkSuiteConfig
from codemode_probe.workload import make_probe_task


def _result(
    arm_name: str,
    *,
    task_id: str = "task-1",
    repetition: int = 1,
    trial_id: str | None = None,
    arm_order_index: int | None = None,
    arm_order: tuple[str, ...] = (),
    cache_policy: CachePolicy = CachePolicy.UNSPECIFIED,
    cache_state: CacheState = CacheState.UNSPECIFIED,
    cache_namespace: str | None = None,
    cache_warmup_run: bool = False,
    latency_ms: float = 100.0,
    schema_valid: bool = True,
    timed_out: bool = False,
    failure_reason: str | None = None,
    top_k_overlap: float = 0.5,
    precision_at_k: float = 0.25,
    recall_at_k: float = 0.75,
    ndcg_at_k: float = 0.5,
    error: str | None = None,
    failure_category: FailureCategory | None = None,
    usage: UsageStats | None = None,
    raw: dict[str, object] | None = None,
) -> ArmResult:
    return ArmResult(
        task_id=task_id,
        arm_name=arm_name,
        repetition=repetition,
        trial_id=trial_id,
        arm_order_index=arm_order_index,
        arm_order=arm_order,
        cache_policy=cache_policy,
        cache_state=cache_state,
        cache_namespace=cache_namespace,
        cache_warmup_run=cache_warmup_run,
        latency_ms=latency_ms,
        timed_out=timed_out,
        execution=ExecutionResult(
            usage=usage or UsageStats(),
            trace=TraceSummary(failure_category=failure_category),
            raw=raw or {},
            error=error,
        ),
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


def test_summarize_results_p95_uses_upper_tail_for_small_cohorts() -> None:
    arm = summarize_results(
        [
            _result("arm-a", latency_ms=10.0),
            _result("arm-a", latency_ms=30.0),
        ]
    )["arms"]["arm-a"]  # type: ignore[index]

    assert arm["p95_latency_ms"] == 30.0


def test_summarize_paired_deltas_matches_task_and_repetition_and_orders_rows() -> None:
    results = [
        _result("candidate-b", task_id="task-b", repetition=2),
        _result("baseline", task_id="task-b", repetition=2),
        _result("candidate-b", task_id="task-a", repetition=2),
        _result("candidate-a", task_id="task-a", repetition=1),
        _result("candidate-b", task_id="task-a", repetition=1),
        _result("baseline", task_id="task-a", repetition=1),
        _result("candidate-a", task_id="task-c", repetition=1),
        _result("baseline", task_id="task-a", repetition=2),
    ]

    rows = summarize_paired_deltas(results, baseline_arm="baseline")

    assert [
        (row["task_id"], row["repetition"], row["comparison_arm"]) for row in rows
    ] == [
        ("task-a", 1, "candidate-a"),
        ("task-a", 1, "candidate-b"),
        ("task-a", 2, "candidate-b"),
        ("task-b", 2, "candidate-b"),
    ]


def test_summarize_paired_deltas_calculates_deltas_and_payload_visible_ratios() -> None:
    rows = summarize_paired_deltas(
        [
            _result(
                "baseline",
                latency_ms=100.0,
                top_k_overlap=0.25,
                ndcg_at_k=0.4,
                usage=UsageStats(
                    model_requests=2,
                    tool_calls=4,
                    tool_response_bytes_total=200,
                    model_visible_bytes_total=50,
                ),
            ),
            _result(
                "comparison",
                latency_ms=150.0,
                top_k_overlap=0.75,
                ndcg_at_k=0.9,
                usage=UsageStats(
                    model_requests=5,
                    tool_calls=7,
                    tool_response_bytes_total=80,
                    model_visible_bytes_total=60,
                ),
            ),
        ],
        baseline_arm="baseline",
    )

    assert rows == [
        {
            "task_id": "task-1",
            "repetition": 1,
            "trial_id": None,
            "arm_order": [],
            "baseline_arm_order_index": None,
            "comparison_arm_order_index": None,
            "baseline_arm": "baseline",
            "comparison_arm": "comparison",
            "delta_ndcg_at_k": 0.5,
            "delta_top_k_overlap": 0.5,
            "delta_latency_ms": 50.0,
            "latency_ratio": 1.5,
            "delta_tool_calls": 3,
            "delta_model_requests": 3,
            "delta_tool_response_bytes": -120,
            "delta_model_visible_bytes": 10,
            "payload_visible_ratio_baseline": 0.25,
            "payload_visible_ratio_comparison": 0.75,
        }
    ]


def test_summarize_paired_deltas_returns_none_latency_ratio_for_zero_baseline() -> None:
    rows = summarize_paired_deltas(
        [
            _result("baseline", latency_ms=0.0),
            _result("comparison", latency_ms=25.0),
        ],
        baseline_arm="baseline",
    )

    assert rows[0]["latency_ratio"] is None


def test_summarize_paired_deltas_skips_duplicate_trial_arm_groups() -> None:
    rows = summarize_paired_deltas(
        [
            _result("baseline", task_id="duplicate-baseline"),
            _result("baseline", task_id="duplicate-baseline"),
            _result("comparison", task_id="duplicate-baseline"),
            _result("baseline", task_id="duplicate-comparison"),
            _result("comparison", task_id="duplicate-comparison"),
            _result("comparison", task_id="duplicate-comparison"),
            _result("baseline", task_id="valid"),
            _result("comparison", task_id="valid"),
        ],
        baseline_arm="baseline",
    )

    assert [(row["task_id"], row["comparison_arm"]) for row in rows] == [
        ("valid", "comparison")
    ]


def test_summarize_pairing_coverage_counts_missing_baselines_and_duplicates() -> None:
    coverage = summarize_pairing_coverage(
        [
            _result("baseline", task_id="task-a", trial_id="task-a:rep-1"),
            _result("comparison", task_id="task-a", trial_id="task-a:rep-1"),
            _result("comparison", task_id="task-b", trial_id="task-b:rep-1"),
            _result("comparison", task_id="task-b", trial_id="task-b:rep-1"),
            _result("baseline", task_id="task-c", trial_id="task-c:rep-1"),
            _result("baseline", task_id="task-d", trial_id="task-d:rep-1"),
            _result("baseline", task_id="task-d", trial_id="task-d:rep-1"),
            _result("comparison", task_id="task-d", trial_id="task-d:rep-1"),
            _result("baseline", task_id="task-e", trial_id="task-e:rep-1"),
            _result("comparison", task_id="task-e", trial_id="task-e:rep-1"),
            _result("comparison", task_id="task-e", trial_id="task-e:rep-1"),
        ],
        baseline_arm="baseline",
    )

    assert coverage == {
        "baseline_arm": "baseline",
        "trial_count": 5,
        "trials_with_baseline": 4,
        "trials_missing_baseline": 1,
        "comparison_results_total": 6,
        "paired_comparisons_total": 1,
        "unpaired_comparisons_total": 5,
        "duplicate_trial_arm_groups": 3,
        "missing_baseline_trial_keys": [
            {
                "task_id": "task-b",
                "repetition": 1,
                "trial_id": "task-b:rep-1",
                "comparison_results": 2,
            }
        ],
    }


def test_summarize_paired_deltas_preserves_trial_provenance_and_keeps_trials_separate() -> None:
    rows = summarize_paired_deltas(
        [
            _result(
                "baseline",
                trial_id="task-1:rep-1:trial-a",
                arm_order_index=1,
                arm_order=("comparison", "baseline"),
                latency_ms=100.0,
            ),
            _result(
                "comparison",
                trial_id="task-1:rep-1:trial-a",
                arm_order_index=0,
                arm_order=("comparison", "baseline"),
                latency_ms=125.0,
            ),
            _result(
                "baseline",
                trial_id="task-1:rep-1:trial-b",
                arm_order_index=0,
                arm_order=("baseline", "comparison"),
                latency_ms=200.0,
            ),
            _result(
                "comparison",
                trial_id="task-1:rep-1:trial-b",
                arm_order_index=1,
                arm_order=("baseline", "comparison"),
                latency_ms=300.0,
            ),
            _result("baseline", latency_ms=10.0),
            _result("comparison", latency_ms=15.0),
        ],
        baseline_arm="baseline",
    )

    assert [
        (
            row["trial_id"],
            row["arm_order"],
            row["baseline_arm_order_index"],
            row["comparison_arm_order_index"],
            row["delta_latency_ms"],
        )
        for row in rows
    ] == [
        (None, [], None, None, 5.0),
        ("task-1:rep-1:trial-a", ["comparison", "baseline"], 1, 0, 25.0),
        ("task-1:rep-1:trial-b", ["baseline", "comparison"], 0, 1, 100.0),
    ]


def test_summarize_paired_delta_groups_aggregates_by_arm_pair() -> None:
    rows = summarize_paired_deltas(
        [
            _result(
                "baseline",
                task_id="task-a",
                latency_ms=100.0,
                ndcg_at_k=0.3,
                top_k_overlap=0.2,
                usage=UsageStats(
                    model_requests=2,
                    tool_calls=4,
                    tool_response_bytes_total=100,
                    model_visible_bytes_total=80,
                ),
            ),
            _result(
                "comparison",
                task_id="task-a",
                latency_ms=70.0,
                ndcg_at_k=0.6,
                top_k_overlap=0.5,
                usage=UsageStats(
                    model_requests=1,
                    tool_calls=6,
                    tool_response_bytes_total=90,
                    model_visible_bytes_total=20,
                ),
            ),
            _result(
                "baseline",
                task_id="task-b",
                latency_ms=100.0,
                ndcg_at_k=0.4,
                top_k_overlap=0.3,
                usage=UsageStats(
                    model_requests=4,
                    tool_calls=8,
                    tool_response_bytes_total=200,
                    model_visible_bytes_total=100,
                ),
            ),
            _result(
                "comparison",
                task_id="task-b",
                latency_ms=130.0,
                ndcg_at_k=0.8,
                top_k_overlap=0.9,
                usage=UsageStats(
                    model_requests=2,
                    tool_calls=10,
                    tool_response_bytes_total=220,
                    model_visible_bytes_total=40,
                ),
            ),
        ],
        baseline_arm="baseline",
    )

    assert summarize_paired_delta_groups(rows) == [
        {
            "baseline_arm": "baseline",
            "comparison_arm": "comparison",
            "pairs": 2,
            "mean_delta_ndcg_at_k": 0.35,
            "mean_delta_top_k_overlap": 0.45,
            "median_delta_latency_ms": 0.0,
            "mean_delta_latency_ms": 0.0,
            "mean_delta_model_requests": -1.5,
            "mean_delta_tool_calls": 2.0,
            "mean_delta_tool_response_bytes": 5.0,
            "mean_delta_model_visible_bytes": -60.0,
        }
    ]


def test_summarize_paired_uncertainty_bootstraps_paired_delta_metrics() -> None:
    rows = summarize_paired_deltas(
        [
            _result("baseline", task_id="task-a", latency_ms=100.0, ndcg_at_k=0.5),
            _result("comparison", task_id="task-a", latency_ms=90.0, ndcg_at_k=0.7),
            _result("baseline", task_id="task-b", latency_ms=200.0, ndcg_at_k=0.5),
            _result("comparison", task_id="task-b", latency_ms=190.0, ndcg_at_k=0.7),
        ],
        baseline_arm="baseline",
    )

    uncertainty = summarize_paired_uncertainty(
        rows,
        bootstrap_iterations=50,
        random_seed=7,
    )

    assert uncertainty == [
        {
            "baseline_arm": "baseline",
            "comparison_arm": "comparison",
            "pairs": 2,
            "bootstrap_iterations": 50,
            "metrics": {
                "delta_ndcg_at_k": {"mean": 0.2, "ci95_low": 0.2, "ci95_high": 0.2},
                "delta_top_k_overlap": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
                "delta_latency_ms": {"mean": -10.0, "ci95_low": -10.0, "ci95_high": -10.0},
                "delta_model_requests": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
                "delta_tool_calls": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
                "delta_tool_response_bytes": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
                "delta_model_visible_bytes": {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
            },
        }
    ]


def test_summarize_workload_regimes_groups_by_workload_and_arm_and_skips_unknown_tasks() -> None:
    tasks = [
        make_probe_task(
            "scalar-small",
            task_family=TaskFamily.SCALAR_LARGE_FANOUT,
            tool_shape=ToolShape.SCALAR,
            shard_count=2,
            candidates_per_shard=3,
            payload_bytes=128,
            top_k=2,
        ),
        make_probe_task(
            "batch-large",
            task_family=TaskFamily.BATCH_LARGE_FANOUT,
            tool_shape=ToolShape.BATCH,
            shard_count=4,
            candidates_per_shard=5,
            payload_bytes=512,
            top_k=4,
        ),
    ]
    results = [
        _result(
            "arm-a",
            task_id="scalar-small",
            latency_ms=10.0,
            top_k_overlap=0.2,
            ndcg_at_k=0.4,
            usage=UsageStats(
                model_requests=1,
                tool_calls=2,
                tool_response_bytes_total=100,
                model_visible_bytes_total=25,
            ),
        ),
        _result(
            "arm-a",
            task_id="scalar-small",
            repetition=2,
            latency_ms=30.0,
            top_k_overlap=0.8,
            ndcg_at_k=0.6,
            usage=UsageStats(
                model_requests=3,
                tool_calls=4,
                tool_response_bytes_total=300,
                model_visible_bytes_total=75,
            ),
        ),
        _result(
            "arm-b",
            task_id="scalar-small",
            latency_ms=20.0,
            top_k_overlap=1.0,
            ndcg_at_k=1.0,
            usage=UsageStats(
                model_requests=2,
                tool_calls=6,
                tool_response_bytes_total=0,
                model_visible_bytes_total=None,
            ),
        ),
        _result("arm-a", task_id="batch-large", latency_ms=40.0),
        _result("arm-a", task_id="unknown-task", latency_ms=999.0),
    ]

    rows = summarize_workload_regimes(tasks, results)

    assert [
        (
            row["task_family"],
            row["tool_shape"],
            row["candidate_count"],
            row["payload_bytes"],
            row["top_k"],
            row["arm_name"],
        )
        for row in rows
    ] == [
        ("batch_large_fanout", "batch", 20, 512, 4, "arm-a"),
        ("scalar_large_fanout", "scalar", 6, 128, 2, "arm-a"),
        ("scalar_large_fanout", "scalar", 6, 128, 2, "arm-b"),
    ]
    assert rows[1] == {
        "task_family": "scalar_large_fanout",
        "tool_shape": "scalar",
        "candidate_count": 6,
        "payload_bytes": 128,
        "top_k": 2,
        "arm_name": "arm-a",
        "runs": 2,
        "success_rate": 1.0,
        "mean_ndcg_at_k": 0.5,
        "mean_top_k_overlap": 0.5,
        "median_latency_ms": 20.0,
        "p95_latency_ms": 30.0,
        "mean_tool_calls": 3.0,
        "mean_model_requests": 2.0,
        "tool_response_bytes_total": 400,
        "model_visible_bytes_total": 100,
        "visible_fraction": 0.25,
        "payload_suppression_ratio": 0.75,
    }
    assert rows[2]["visible_fraction"] is None
    assert rows[2]["payload_suppression_ratio"] is None


def test_summarize_cache_cohorts_groups_by_arm_policy_state_and_namespace() -> None:
    rows = summarize_cache_cohorts(
        [
            _result(
                "arm-a",
                latency_ms=10.0,
                ndcg_at_k=0.4,
                cache_policy=CachePolicy.COLD_THEN_WARM,
                cache_state=CacheState.COLD,
                cache_namespace="cohort-a",
                usage=UsageStats(
                    model_requests=1,
                    tool_calls=2,
                    input_tokens=10,
                    cache_read_tokens=0,
                    tool_response_bytes_total=100,
                    model_visible_bytes_total=25,
                ),
            ),
            _result(
                "arm-a",
                repetition=2,
                latency_ms=30.0,
                ndcg_at_k=0.8,
                cache_policy=CachePolicy.COLD_THEN_WARM,
                cache_state=CacheState.WARM,
                cache_namespace="cohort-a",
                usage=UsageStats(
                    model_requests=1,
                    tool_calls=2,
                    input_tokens=5,
                    cache_read_tokens=5,
                    tool_response_bytes_total=100,
                    model_visible_bytes_total=10,
                ),
            ),
            _result(
                "arm-b",
                latency_ms=20.0,
                cache_policy=CachePolicy.DISABLED,
                cache_state=CacheState.DISABLED,
                cache_namespace=None,
            ),
        ]
    )

    assert [
        (
            row["arm_name"],
            row["cache_policy"],
            row["cache_state"],
            row["cache_namespace"],
            row["runs"],
        )
        for row in rows
    ] == [
        ("arm-a", "cold_then_warm", "cold", "cohort-a", 1),
        ("arm-a", "cold_then_warm", "warm", "cohort-a", 1),
        ("arm-b", "disabled", "disabled", None, 1),
    ]
    assert rows[0]["input_tokens_total"] == 10
    assert rows[0]["cache_read_tokens_total"] == 0
    assert rows[0]["visible_fraction"] == 0.25
    assert rows[1]["input_tokens_total"] == 5
    assert rows[1]["cache_read_tokens_total"] == 5
    assert rows[1]["visible_fraction"] == 0.1


def test_summarize_failure_modes_groups_non_successes_by_failure_contract() -> None:
    rows = summarize_failure_modes(
        [
            _result("arm-a", task_id="success"),
            _result(
                "arm-a",
                task_id="schema-a",
                trial_id="schema-a:rep-1",
                schema_valid=False,
                failure_reason="schema_invalid",
            ),
            _result(
                "arm-a",
                task_id="schema-b",
                trial_id="schema-b:rep-1",
                schema_valid=False,
                failure_reason="schema_invalid",
            ),
            _result(
                "arm-b",
                task_id="budget",
                error="max_tool_calls_exceeded",
                failure_category=FailureCategory.TOOL_BUDGET_EXCEEDED,
            ),
            _result(
                "arm-b",
                task_id="timeout",
                timed_out=True,
                error="timeout",
                failure_category=FailureCategory.TIMEOUT,
                failure_reason="timeout",
            ),
        ]
    )

    assert rows == [
        {
            "arm_name": "arm-a",
            "failure_category": None,
            "execution_error": None,
            "score_failure_reason": "schema_invalid",
            "timed_out": False,
            "schema_valid": False,
            "runs": 2,
            "task_ids": ["schema-a", "schema-b"],
            "trial_ids": ["schema-a:rep-1", "schema-b:rep-1"],
        },
        {
            "arm_name": "arm-b",
            "failure_category": "timeout",
            "execution_error": "timeout",
            "score_failure_reason": "timeout",
            "timed_out": True,
            "schema_valid": True,
            "runs": 1,
            "task_ids": ["timeout"],
            "trial_ids": [],
        },
        {
            "arm_name": "arm-b",
            "failure_category": "tool_budget_exceeded",
            "execution_error": "max_tool_calls_exceeded",
            "score_failure_reason": None,
            "timed_out": False,
            "schema_valid": True,
            "runs": 1,
            "task_ids": ["budget"],
            "trial_ids": [],
        },
    ]


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
            "Pairwise deltas use `direct_mcp_agent_parallel` as the baseline when present.",
            "Cache cohorts are recorded as run/result metadata; provider cache enforcement is adapter-specific.",
            "",
        ]
    )


def test_render_summary_markdown_includes_warning_section_when_supplied() -> None:
    report = render_summary_markdown(
        [_result("arm-a")],
        warnings=[
            {
                "id": "low_repetition_count",
                "severity": "warning",
                "message": "Run has fewer than 3 repetitions.",
                "details": {},
            }
        ],
    )

    assert "## Warnings" in report
    assert (
        "- `low_repetition_count` (warning): Run has fewer than 3 repetitions."
        in report
    )


def test_collect_run_warnings_flags_claim_readiness_gaps() -> None:
    warnings = collect_run_warnings(
        [
            _result(
                "code_mode_synthetic_scripted",
                cache_warmup_run=True,
                timed_out=True,
                error="timeout",
                failure_category=FailureCategory.TIMEOUT,
            )
        ],
        provider_config=openai_config(model="gpt-test", enabled=False),
        suite_config=BenchmarkSuiteConfig(
            arms=("code_mode",),
            repetitions=1,
            arm_order="fixed",
        ),
        preflight_results=None,
        paired_baseline_arm="direct_mcp_agent_parallel",
    )

    by_id = {warning["id"]: warning for warning in warnings}

    assert set(by_id) == {
        "cache_warmup_rows_present",
        "dry_run_provider_config",
        "low_repetition_count",
        "missing_provider_model_evidence",
        "missing_provider_pricing_evidence",
        "preflight_not_run",
        "run_failures_present",
        "timeouts_present",
        "unpaired_comparisons",
    }
    assert by_id["missing_provider_model_evidence"]["details"] == {
        "missing_fields": [
            "model_version",
            "api_version",
            "sdk_version",
            "model_docs_source_id",
        ]
    }
    assert by_id["missing_provider_pricing_evidence"]["details"] == {
        "missing_fields": [
            "pricing_source_id",
            "pricing_snapshot_date",
            "currency",
        ]
    }


def test_collect_run_warnings_flags_azure_runs_using_openai_pricing_evidence() -> None:
    warnings = collect_run_warnings(
        [_result("direct_mcp_agent_parallel", trial_id="trial-1")],
        provider_config=azure_openai_config(
            model="gpt-4.1-mini",
            enabled=True,
            model_version="gpt-4.1-mini",
            api_version="2025-01-01-preview",
            sdk_version="2.0.0",
            pricing_source_id="openai-gpt-4-1-mini-docs-2026-05-06",
            model_docs_source_id="openai-gpt-4-1-mini-docs-2026-05-06",
            pricing_snapshot_date=date(2026, 5, 6),
            currency="USD",
        ),
        suite_config=BenchmarkSuiteConfig(
            arms=("direct_agent",),
            repetitions=3,
        ),
        preflight_results=[
            PreflightCheckResult(name="preflight-a", passed=True, details={})
        ],
        paired_baseline_arm="direct_mcp_agent_parallel",
    )

    by_id = {warning["id"]: warning for warning in warnings}

    assert by_id["azure_pricing_source_not_verified"] == {
        "id": "azure_pricing_source_not_verified",
        "severity": "warning",
        "message": (
            "Azure OpenAI run uses non-Azure pricing evidence; treat cost estimates as "
            "assumption-backed, not Azure billing evidence."
        ),
        "details": {
            "pricing_source_id": "openai-gpt-4-1-mini-docs-2026-05-06",
        },
    }


def test_collect_run_warnings_stays_empty_for_ready_synthetic_suite() -> None:
    warnings = collect_run_warnings(
        [
            _result(
                "direct_mcp_agent_parallel",
                task_id="task-1",
                repetition=1,
                trial_id="trial-1",
            ),
            _result(
                "code_mode_synthetic_scripted",
                task_id="task-1",
                repetition=1,
                trial_id="trial-1",
            ),
        ],
        suite_config=BenchmarkSuiteConfig(
            arms=("direct_agent", "code_mode"),
            repetitions=3,
            arm_order="randomized",
        ),
        preflight_results=[
            PreflightCheckResult(name="preflight-a", passed=True, details={})
        ],
    )

    assert [warning["id"] for warning in warnings] == ["synthetic_harness_validation"]


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

    warnings = json.loads((tmp_path / "warnings.json").read_text(encoding="utf-8"))
    assert [warning["id"] for warning in warnings] == [
        "preflight_not_run",
        "suite_config_missing",
        "synthetic_harness_validation",
        "unpaired_comparisons",
    ]
    assert "## Warnings" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_write_run_artifacts_writes_paired_deltas_with_direct_mcp_parallel_baseline(
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
            "direct_mcp_agent_parallel",
            latency_ms=100.0,
            top_k_overlap=0.25,
            ndcg_at_k=0.4,
        ),
        _result(
            "optimized_agent",
            latency_ms=125.0,
            top_k_overlap=0.75,
            ndcg_at_k=0.9,
        ),
        _result(
            "unpaired_agent",
            task_id="task-2",
            latency_ms=999.0,
            top_k_overlap=1.0,
            ndcg_at_k=1.0,
        ),
    ]

    write_run_artifacts(tmp_path, [task], results)

    paired_deltas = json.loads(
        (tmp_path / "paired_deltas.json").read_text(encoding="utf-8")
    )
    paired_summary = json.loads(
        (tmp_path / "paired_delta_summary.json").read_text(encoding="utf-8")
    )
    paired_uncertainty = json.loads(
        (tmp_path / "paired_uncertainty.json").read_text(encoding="utf-8")
    )
    pairing_coverage = json.loads(
        (tmp_path / "pairing_coverage.json").read_text(encoding="utf-8")
    )
    assert paired_deltas == [
        {
            "task_id": "task-1",
            "repetition": 1,
            "trial_id": None,
            "arm_order": [],
            "baseline_arm_order_index": None,
            "comparison_arm_order_index": None,
            "baseline_arm": "direct_mcp_agent_parallel",
            "comparison_arm": "optimized_agent",
            "delta_ndcg_at_k": 0.5,
            "delta_top_k_overlap": 0.5,
            "delta_latency_ms": 25.0,
            "latency_ratio": 1.25,
            "delta_tool_calls": 0,
            "delta_model_requests": 0,
            "delta_tool_response_bytes": 0,
            "delta_model_visible_bytes": 0,
            "payload_visible_ratio_baseline": None,
            "payload_visible_ratio_comparison": None,
        }
    ]
    assert paired_summary == [
        {
            "baseline_arm": "direct_mcp_agent_parallel",
            "comparison_arm": "optimized_agent",
            "pairs": 1,
            "mean_delta_ndcg_at_k": 0.5,
            "mean_delta_top_k_overlap": 0.5,
            "median_delta_latency_ms": 25.0,
            "mean_delta_latency_ms": 25.0,
            "mean_delta_model_requests": 0.0,
            "mean_delta_tool_calls": 0.0,
            "mean_delta_tool_response_bytes": 0.0,
            "mean_delta_model_visible_bytes": 0.0,
        }
    ]
    assert paired_uncertainty[0]["baseline_arm"] == "direct_mcp_agent_parallel"
    assert paired_uncertainty[0]["comparison_arm"] == "optimized_agent"
    assert paired_uncertainty[0]["pairs"] == 1
    assert paired_uncertainty[0]["metrics"]["delta_ndcg_at_k"] == {
        "mean": 0.5,
        "ci95_low": 0.5,
        "ci95_high": 0.5,
    }
    assert pairing_coverage == {
        "baseline_arm": "direct_mcp_agent_parallel",
        "trial_count": 2,
        "trials_with_baseline": 1,
        "trials_missing_baseline": 1,
        "comparison_results_total": 2,
        "paired_comparisons_total": 1,
        "unpaired_comparisons_total": 1,
        "duplicate_trial_arm_groups": 0,
        "missing_baseline_trial_keys": [
            {
                "task_id": "task-2",
                "repetition": 1,
                "trial_id": None,
                "comparison_results": 1,
            }
        ],
    }


def test_write_run_artifacts_uses_suite_paired_baseline_for_deltas_and_report(
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
        _result("in_process_tool_oracle", latency_ms=100.0, ndcg_at_k=0.8),
        _result("code_mode_synthetic_scripted", latency_ms=90.0, ndcg_at_k=1.0),
        _result("direct_mcp_agent_parallel", latency_ms=150.0, ndcg_at_k=0.5),
    ]

    write_run_artifacts(
        tmp_path,
        [task],
        results,
        suite_config=BenchmarkSuiteConfig(
            arms=("in_process", "code_mode", "direct_agent"),
            paired_baseline_arm="in_process",
        ),
    )

    paired_deltas = json.loads(
        (tmp_path / "paired_deltas.json").read_text(encoding="utf-8")
    )
    assert [row["baseline_arm"] for row in paired_deltas] == [
        "in_process_tool_oracle",
        "in_process_tool_oracle",
    ]
    assert [row["comparison_arm"] for row in paired_deltas] == [
        "code_mode_synthetic_scripted",
        "direct_mcp_agent_parallel",
    ]
    assert "Pairwise deltas use `in_process_tool_oracle` as the baseline when present." in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")


def test_write_run_artifacts_writes_empty_paired_deltas_when_baseline_missing(
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
        _result("optimized_agent", latency_ms=125.0),
        _result("direct_mcp_agent_sequential", latency_ms=150.0),
    ]

    write_run_artifacts(tmp_path, [task], results)

    assert json.loads((tmp_path / "paired_deltas.json").read_text(encoding="utf-8")) == []
    assert json.loads((tmp_path / "paired_delta_summary.json").read_text(encoding="utf-8")) == []
    assert json.loads((tmp_path / "paired_uncertainty.json").read_text(encoding="utf-8")) == []


def test_write_run_artifacts_writes_workload_regimes_joined_by_task(
    tmp_path: Path,
) -> None:
    scalar_task = make_probe_task(
        "scalar-small",
        task_family=TaskFamily.SCALAR_LARGE_FANOUT,
        tool_shape=ToolShape.SCALAR,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=128,
        top_k=2,
    )
    batch_task = make_probe_task(
        "batch-large",
        task_family=TaskFamily.BATCH_LARGE_FANOUT,
        tool_shape=ToolShape.BATCH,
        shard_count=4,
        candidates_per_shard=5,
        payload_bytes=512,
        top_k=4,
    )
    results = [
        _result("arm-a", task_id="scalar-small", latency_ms=10.0),
        _result("arm-b", task_id="scalar-small", latency_ms=20.0),
        _result("arm-a", task_id="batch-large", latency_ms=30.0),
        _result("arm-a", task_id="unknown-task", latency_ms=999.0),
    ]

    write_run_artifacts(tmp_path, [scalar_task, batch_task], results)

    workload_regimes = json.loads(
        (tmp_path / "workload_regimes.json").read_text(encoding="utf-8")
    )
    assert [
        (
            row["task_family"],
            row["tool_shape"],
            row["candidate_count"],
            row["payload_bytes"],
            row["top_k"],
            row["arm_name"],
            row["runs"],
        )
        for row in workload_regimes
    ] == [
        ("batch_large_fanout", "batch", 20, 512, 4, "arm-a", 1),
        ("scalar_large_fanout", "scalar", 6, 128, 2, "arm-a", 1),
        ("scalar_large_fanout", "scalar", 6, 128, 2, "arm-b", 1),
    ]


def test_write_run_artifacts_writes_failure_modes(
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
            "direct_mcp_agent_parallel",
            task_id="task-1",
            error="max_tool_calls_exceeded",
            failure_category=FailureCategory.TOOL_BUDGET_EXCEEDED,
        )
    ]

    write_run_artifacts(tmp_path, [task], results)

    assert json.loads((tmp_path / "failure_modes.json").read_text(encoding="utf-8")) == [
        {
            "arm_name": "direct_mcp_agent_parallel",
            "failure_category": "tool_budget_exceeded",
            "execution_error": "max_tool_calls_exceeded",
            "score_failure_reason": None,
            "timed_out": False,
            "schema_valid": True,
            "runs": 1,
            "task_ids": ["task-1"],
            "trial_ids": [],
        }
    ]


def test_write_run_artifacts_writes_redacted_bounded_transcripts(tmp_path: Path) -> None:
    task = make_probe_task("transcript-task", seed=1)
    results = [
        _result(
            "provider-arm",
            task_id=task.id,
            cache_policy=CachePolicy.WARM,
            cache_state=CacheState.WARMUP,
            cache_namespace="cache-a",
            cache_warmup_run=True,
            usage=UsageStats(
                input_tokens=11,
                output_tokens=7,
                cache_read_tokens=5,
                cache_write_tokens=3,
            ),
            raw={
                "model_turns": [
                    {
                        "provider_name": "fake",
                        "Authorization": "Bearer live-secret",
                        "provider_raw": {
                            "accessToken": "camel-access-secret",
                            "api_key": "sk-test-secret",
                            "inputTokens": 11,
                            "session_token": "plain-session-secret",
                            "sessionToken": "camel-session-secret",
                            "payload": "x" * 600,
                            "notes": [
                                "Bearer leaked-token",
                                "https://user:pass@example.test/path",
                            ],
                            "token_count": 123,
                        },
                    }
                ]
            },
        )
    ]

    write_run_artifacts(tmp_path, [task], results)

    rows = [
        json.loads(line)
        for line in (tmp_path / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == task.id
    assert row["arm_name"] == "provider-arm"
    assert row["cache_policy"] == "warm"
    assert row["cache_state"] == "warmup"
    assert row["cache_namespace"] == "cache-a"
    assert row["cache_warmup_run"] is True
    assert isinstance(row["transcript_hash"], str)
    turn = row["model_turns"][0]
    assert turn["Authorization"] == "[REDACTED]"
    assert turn["provider_raw"]["accessToken"] == "[REDACTED]"
    assert turn["provider_raw"]["api_key"] == "[REDACTED]"
    assert turn["provider_raw"]["inputTokens"] == 11
    assert turn["provider_raw"]["session_token"] == "[REDACTED]"
    assert turn["provider_raw"]["sessionToken"] == "[REDACTED]"
    assert turn["provider_raw"]["payload"]["truncated"] is True
    assert turn["provider_raw"]["payload"]["original_chars"] == 600
    assert turn["provider_raw"]["notes"] == ["[REDACTED]", "[REDACTED]"]
    assert turn["provider_raw"]["token_count"] == 123

    results_rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    result_turn = results_rows[0]["execution"]["raw"]["model_turns"][0]
    assert result_turn["Authorization"] == "[REDACTED]"
    assert result_turn["provider_raw"]["accessToken"] == "[REDACTED]"
    assert result_turn["provider_raw"]["api_key"] == "[REDACTED]"
    assert result_turn["provider_raw"]["inputTokens"] == 11
    assert result_turn["provider_raw"]["session_token"] == "[REDACTED]"
    assert result_turn["provider_raw"]["sessionToken"] == "[REDACTED]"
    assert result_turn["provider_raw"]["payload"]["truncated"] is True
    assert result_turn["provider_raw"]["notes"] == ["[REDACTED]", "[REDACTED]"]
    assert result_turn["provider_raw"]["token_count"] == 123
    assert results_rows[0]["execution"]["usage"] == {
        "cache_read_tokens": 5,
        "cache_write_tokens": 3,
        "failed_tool_calls": 0,
        "input_tokens": 11,
        "model_requests": 0,
        "model_visible_bytes_total": None,
        "output_tokens": 7,
        "tool_calls": 0,
        "tool_response_bytes_total": 0,
    }

    all_artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.iterdir()
        if path.is_file()
    )
    for forbidden in (
        "live-secret",
        "sk-test-secret",
        "camel-access-secret",
        "plain-session-secret",
        "camel-session-secret",
        "leaked-token",
        "user:pass",
    ):
        assert forbidden not in all_artifact_text


def test_write_run_artifacts_keeps_existing_summary_results_and_report_artifacts(
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
            "direct_mcp_agent_parallel",
            latency_ms=100.0,
            usage=UsageStats(
                model_requests=1,
                tool_calls=2,
                tool_response_bytes_total=80,
                model_visible_bytes_total=20,
            ),
        )
    ]

    write_run_artifacts(tmp_path, [task], results)

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summarize_results(
        results
    )
    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert jsonl_rows == [results[0].model_dump(mode="json")]
    assert "Pairwise deltas use `direct_mcp_agent_parallel` as the baseline when present." in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")
