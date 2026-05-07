from __future__ import annotations

import pytest
from pydantic import ValidationError

from codemode_probe.cases import CaseMatrixConfig, generate_case_tasks
from codemode_probe.model_loop import DirectMcpAgentExecutor, ScriptedFanoutModelClient
from codemode_probe.models import TaskFamily, ToolShape
from codemode_probe.prompts import render_prompt
from codemode_probe.synthetic_tools import InProcessSyntheticTools


def test_smoke_preset_shape() -> None:
    tasks = generate_case_tasks(CaseMatrixConfig(preset="smoke", base_seed=10))

    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "smoke_smoke_single_lookup"
    assert task.workload.task_family == TaskFamily.SINGLE_LOOKUP
    assert task.workload.tool_shape == ToolShape.SCALAR
    assert task.workload.shard_count == 1
    assert task.workload.candidates_per_shard == 1
    assert task.workload.top_k == 1
    assert task.workload.seed == 10 + len("smoke_single_lookup")
    assert task.max_tool_calls == 2


def test_orchestration_matrix_task_ids_families_tool_shapes_and_sizes() -> None:
    tasks = generate_case_tasks(CaseMatrixConfig(preset="orchestration_matrix"))

    assert [
        (
            task.id,
            task.workload.task_family,
            task.workload.tool_shape,
            task.workload.shard_count,
            task.workload.candidates_per_shard,
            task.workload.top_k,
        )
        for task in tasks
    ] == [
        (
            "orchestration_matrix_single_lookup",
            TaskFamily.SINGLE_LOOKUP,
            ToolShape.SCALAR,
            1,
            1,
            1,
        ),
        (
            "orchestration_matrix_small_parallel_lookup",
            TaskFamily.SMALL_PARALLEL_LOOKUP,
            ToolShape.SCALAR,
            4,
            1,
            2,
        ),
        (
            "orchestration_matrix_scalar_large_fanout_25",
            TaskFamily.SCALAR_LARGE_FANOUT,
            ToolShape.SCALAR,
            5,
            5,
            5,
        ),
        (
            "orchestration_matrix_scalar_large_fanout_100",
            TaskFamily.SCALAR_LARGE_FANOUT,
            ToolShape.SCALAR,
            10,
            10,
            5,
        ),
        (
            "orchestration_matrix_batch_large_fanout_100",
            TaskFamily.BATCH_LARGE_FANOUT,
            ToolShape.BATCH,
            10,
            10,
            5,
        ),
        (
            "orchestration_matrix_deep_branching_filter_rank",
            TaskFamily.DEEP_BRANCHING_FILTER_RANK,
            ToolShape.SCALAR,
            8,
            12,
            5,
        ),
    ]


def test_case_seed_generation_is_deterministic_and_base_seed_relative() -> None:
    config = CaseMatrixConfig(preset="orchestration_matrix", base_seed=100)
    same_config = config.model_copy()
    shifted_config = config.model_copy(update={"base_seed": 125})

    first = generate_case_tasks(config)
    second = generate_case_tasks(same_config)
    shifted = generate_case_tasks(shifted_config)

    assert [task.workload.seed for task in first] == [
        task.workload.seed for task in second
    ]
    assert [task.id for task in first] == [task.id for task in second]
    assert [task.workload.seed + 25 for task in first] == [
        task.workload.seed for task in shifted
    ]


def test_generated_max_tool_calls_allow_large_scalar_and_batch_direct_agent_runs() -> None:
    tasks = {
        task.id: task
        for task in generate_case_tasks(CaseMatrixConfig(preset="orchestration_matrix"))
    }

    for task_id, expected_tool_calls in [
        ("orchestration_matrix_scalar_large_fanout_100", 110),
        ("orchestration_matrix_batch_large_fanout_100", 11),
    ]:
        task = tasks[task_id]
        result = DirectMcpAgentExecutor(
            InProcessSyntheticTools.from_task(task),
            ScriptedFanoutModelClient(),
        ).execute(task)

        assert task.max_tool_calls >= expected_tool_calls
        assert result.error is None
        assert result.answer is not None
        assert result.usage.tool_calls == expected_tool_calls


def test_generated_prompt_task_parameters_include_seed() -> None:
    task = generate_case_tasks(CaseMatrixConfig(preset="smoke", base_seed=7))[0]

    rendered = render_prompt(task)

    assert rendered.task_parameters["seed"] == task.workload.seed
    assert rendered.task_parameters == {
        "seed": task.workload.seed,
        "task_family": task.workload.task_family.value,
        "tool_shape": task.workload.tool_shape.value,
        "shard_count": task.workload.shard_count,
        "candidates_per_shard": task.workload.candidates_per_shard,
        "payload_bytes": task.workload.payload_bytes,
        "relevant_fraction": task.workload.relevant_fraction,
        "top_k": task.workload.top_k,
    }


def test_invalid_preset_is_rejected_by_config_and_generator() -> None:
    with pytest.raises(ValidationError):
        CaseMatrixConfig(preset="unknown")

    invalid_config = CaseMatrixConfig.model_construct(preset="unknown")
    with pytest.raises(ValueError, match="unknown case preset: unknown"):
        generate_case_tasks(invalid_config)
