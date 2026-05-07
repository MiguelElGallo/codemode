from __future__ import annotations

import pytest

from codemode_probe.budget import (
    BudgetError,
    RunBudgetConfig,
    enforce_run_budget,
    estimate_run_budget,
)
from codemode_probe.cases import CaseMatrixConfig, generate_case_tasks
from codemode_probe.suite import BenchmarkSuiteConfig
from codemode_probe.workload import make_probe_task


def tiny_task():
    return make_probe_task(
        "budget-task",
        seed=3,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        top_k=2,
    )


def test_estimate_run_budget_counts_result_rows_and_model_request_upper_bound() -> None:
    task = tiny_task().model_copy(update={"max_tool_calls": 4, "timeout_seconds": 2.5})
    suite_config = BenchmarkSuiteConfig(
        arms=("direct_agent", "in_process"),
        repetitions=3,
    )

    estimate = estimate_run_budget([task], suite_config)

    assert estimate.task_count == 1
    assert estimate.result_rows == 6
    assert estimate.model_request_upper_bound == 15
    assert estimate.run_seconds_upper_bound == 15.0
    assert estimate.input_tokens_heuristic > 0
    assert estimate.output_tokens_heuristic > 0
    assert estimate.cost_estimated is False


def test_enforce_run_budget_raises_before_execution_when_request_budget_is_exceeded() -> None:
    task = tiny_task().model_copy(update={"max_tool_calls": 4})
    suite_config = BenchmarkSuiteConfig(arms=("direct_agent",), repetitions=1)

    with pytest.raises(BudgetError, match="max_model_requests 4"):
        enforce_run_budget(
            [task],
            suite_config,
            RunBudgetConfig(max_model_requests=4),
        )


def test_smoke_preset_fits_documented_live_model_request_budget() -> None:
    estimate = enforce_run_budget(
        generate_case_tasks(CaseMatrixConfig(preset="smoke")),
        BenchmarkSuiteConfig(arms=("direct_agent",), repetitions=1),
        RunBudgetConfig(max_model_requests=25),
    )

    assert estimate is not None
    assert estimate.model_request_upper_bound == 3


def test_enforce_run_budget_estimates_cost_when_pricing_metadata_is_present() -> None:
    task = tiny_task().model_copy(update={"max_tool_calls": 1})
    suite_config = BenchmarkSuiteConfig(arms=("direct_agent",), repetitions=1)

    estimate = enforce_run_budget(
        [task],
        suite_config,
        RunBudgetConfig(
            max_estimated_cost=1.0,
            input_cost_per_1m_tokens=2.0,
            output_cost_per_1m_tokens=8.0,
            currency="USD",
        ),
    )

    assert estimate is not None
    assert estimate.cost_estimated is True
    assert estimate.estimated_cost is not None
    assert estimate.currency == "USD"


def test_enforce_run_budget_requires_pricing_metadata_for_cost_cap() -> None:
    with pytest.raises(BudgetError, match="requires input and output token cost metadata"):
        enforce_run_budget(
            [tiny_task()],
            BenchmarkSuiteConfig(arms=("direct_agent",), repetitions=1),
            RunBudgetConfig(max_estimated_cost=1.0),
        )
