from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from codemode_probe.executor_ids import normalize_executor_id
from codemode_probe.models import ProbeTask
from codemode_probe.prompts import render_prompt

if TYPE_CHECKING:
    from codemode_probe.suite import BenchmarkSuiteConfig

MODEL_REQUEST_ARMS = {"direct_mcp_agent_parallel"}


class BudgetError(RuntimeError):
    pass


class RunBudgetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_run_seconds: float | None = Field(default=None, gt=0)
    max_model_requests: int | None = Field(default=None, ge=0)
    max_input_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    max_estimated_cost: float | None = Field(default=None, ge=0)
    input_cost_per_1m_tokens: float | None = Field(default=None, ge=0)
    output_cost_per_1m_tokens: float | None = Field(default=None, ge=0)
    currency: str | None = None

    @property
    def is_configured(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_run_seconds,
                self.max_model_requests,
                self.max_input_tokens,
                self.max_output_tokens,
                self.max_estimated_cost,
            )
        )


class RunBudgetEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_count: int
    result_rows: int
    model_request_upper_bound: int
    run_seconds_upper_bound: float
    input_tokens_heuristic: int
    output_tokens_heuristic: int
    estimated_cost: float | None = None
    currency: str | None = None
    cost_estimated: bool = False


def estimate_run_budget(
    tasks: list[ProbeTask],
    suite_config: BenchmarkSuiteConfig,
    *,
    budget_config: RunBudgetConfig | None = None,
) -> RunBudgetEstimate:
    normalized_arms = tuple(normalize_executor_id(arm) for arm in suite_config.arms)
    result_rows = len(tasks) * suite_config.repetitions * len(normalized_arms)
    run_seconds_upper_bound = sum(task.timeout_seconds for task in tasks)
    run_seconds_upper_bound *= suite_config.repetitions * len(normalized_arms)

    model_arms = [arm for arm in normalized_arms if arm in MODEL_REQUEST_ARMS]
    model_request_upper_bound = 0
    input_tokens_heuristic = 0
    output_tokens_heuristic = 0
    for task in tasks:
        per_turn_input = _heuristic_input_tokens(task)
        per_turn_output = _heuristic_output_tokens(task)
        max_turns = task.max_tool_calls + 1
        task_model_arm_runs = suite_config.repetitions * len(model_arms)
        model_request_upper_bound += max_turns * task_model_arm_runs
        input_tokens_heuristic += per_turn_input * max_turns * task_model_arm_runs
        output_tokens_heuristic += per_turn_output * max_turns * task_model_arm_runs

    estimated_cost = _estimated_cost(
        input_tokens=input_tokens_heuristic,
        output_tokens=output_tokens_heuristic,
        budget_config=budget_config,
    )
    return RunBudgetEstimate(
        task_count=len(tasks),
        result_rows=result_rows,
        model_request_upper_bound=model_request_upper_bound,
        run_seconds_upper_bound=round(run_seconds_upper_bound, 3),
        input_tokens_heuristic=input_tokens_heuristic,
        output_tokens_heuristic=output_tokens_heuristic,
        estimated_cost=estimated_cost,
        currency=budget_config.currency if budget_config is not None else None,
        cost_estimated=estimated_cost is not None,
    )


def enforce_run_budget(
    tasks: list[ProbeTask],
    suite_config: BenchmarkSuiteConfig,
    budget_config: RunBudgetConfig | None,
) -> RunBudgetEstimate | None:
    if budget_config is None or not budget_config.is_configured:
        return None
    if budget_config.max_estimated_cost is not None and (
        budget_config.input_cost_per_1m_tokens is None
        or budget_config.output_cost_per_1m_tokens is None
    ):
        raise BudgetError(
            "max_estimated_cost requires input and output token cost metadata"
        )

    estimate = estimate_run_budget(
        tasks,
        suite_config,
        budget_config=budget_config,
    )
    violations = _budget_violations(estimate, budget_config)
    if violations:
        raise BudgetError(f"budget exceeded: {violations[0]}")
    return estimate


def _budget_violations(
    estimate: RunBudgetEstimate,
    config: RunBudgetConfig,
) -> list[str]:
    violations: list[str] = []
    if (
        config.max_run_seconds is not None
        and estimate.run_seconds_upper_bound > config.max_run_seconds
    ):
        violations.append(
            f"run_seconds upper bound {estimate.run_seconds_upper_bound} exceeds "
            f"max_run_seconds {config.max_run_seconds}"
        )
    if (
        config.max_model_requests is not None
        and estimate.model_request_upper_bound > config.max_model_requests
    ):
        violations.append(
            f"model request upper bound {estimate.model_request_upper_bound} exceeds "
            f"max_model_requests {config.max_model_requests}"
        )
    if (
        config.max_input_tokens is not None
        and estimate.input_tokens_heuristic > config.max_input_tokens
    ):
        violations.append(
            f"input token heuristic {estimate.input_tokens_heuristic} exceeds "
            f"max_input_tokens {config.max_input_tokens}"
        )
    if (
        config.max_output_tokens is not None
        and estimate.output_tokens_heuristic > config.max_output_tokens
    ):
        violations.append(
            f"output token heuristic {estimate.output_tokens_heuristic} exceeds "
            f"max_output_tokens {config.max_output_tokens}"
        )
    if (
        config.max_estimated_cost is not None
        and estimate.estimated_cost is not None
        and estimate.estimated_cost > config.max_estimated_cost
    ):
        violations.append(
            f"estimated cost {estimate.estimated_cost} exceeds "
            f"max_estimated_cost {config.max_estimated_cost}"
        )
    return violations


def _heuristic_input_tokens(task: ProbeTask) -> int:
    prompt_bytes = len(render_prompt(task).model_dump_json().encode())
    payload_bytes = task.workload.candidate_count * (task.workload.payload_bytes + 256)
    return math.ceil((prompt_bytes + payload_bytes) / 4)


def _heuristic_output_tokens(task: ProbeTask) -> int:
    return task.workload.top_k * 96 + 64


def _estimated_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    budget_config: RunBudgetConfig | None,
) -> float | None:
    if budget_config is None:
        return None
    if (
        budget_config.input_cost_per_1m_tokens is None
        or budget_config.output_cost_per_1m_tokens is None
    ):
        return None
    cost = (
        input_tokens * budget_config.input_cost_per_1m_tokens
        + output_tokens * budget_config.output_cost_per_1m_tokens
    ) / 1_000_000
    return round(cost, 6)
