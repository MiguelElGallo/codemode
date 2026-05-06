from __future__ import annotations

from types import SimpleNamespace

import pytest

from codemode_probe import suite
from codemode_probe.models import ProbeTask, ToolShape
from codemode_probe.suite import BenchmarkSuiteConfig, run_benchmark_suite
from codemode_probe.workload import make_probe_task


class FakeArmResult(SimpleNamespace):
    def model_copy(self, *, update: dict[str, object]) -> "FakeArmResult":
        return FakeArmResult(**{**self.__dict__, **update})


def tiny_task(task_id: str, *, seed: int = 11, tool_shape: ToolShape = ToolShape.SCALAR) -> ProbeTask:
    return make_probe_task(
        task_id,
        seed=seed,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        relevant_fraction=0.5,
        top_k=2,
    )


def test_benchmark_suite_config_normalizes_arm_aliases() -> None:
    config = BenchmarkSuiteConfig(
        arms=(
            "deterministic_oracle",
            "in_process",
            "direct_mcp",
            "direct_agent",
            "direct_mcp_agent_parallel",
        )
    )

    assert config.normalized_arms == (
        "deterministic_oracle_client",
        "in_process_tool_oracle",
        "direct_mcp_tool_oracle",
        "direct_mcp_agent_parallel",
        "direct_mcp_agent_parallel",
    )


def test_benchmark_suite_config_normalizes_paired_baseline_alias() -> None:
    config = BenchmarkSuiteConfig(paired_baseline_arm="direct_agent")

    assert config.normalized_paired_baseline_arm == "direct_mcp_agent_parallel"


def test_suite_fixed_order_is_stable_across_tasks_and_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    class FakeRunner:
        def __init__(self, executor: SimpleNamespace) -> None:
            self.executor = executor

        def run_task(self, task: ProbeTask, *, repetition: int = 1) -> SimpleNamespace:
            calls.append((task.id, self.executor.name, repetition))
            return FakeArmResult(
                task_id=task.id,
                arm_name=self.executor.name,
                repetition=repetition,
            )

    monkeypatch.setattr(suite, "build_executor", lambda arm, task: SimpleNamespace(name=arm))
    monkeypatch.setattr(suite, "BenchmarkRunner", FakeRunner)

    run_benchmark_suite(
        [tiny_task("task-a", seed=1), tiny_task("task-b", seed=2)],
        BenchmarkSuiteConfig(
            arms=("direct_agent", "in_process", "deterministic_oracle"),
            repetitions=2,
        ),
    )

    assert calls == [
        ("task-a", "direct_mcp_agent_parallel", 1),
        ("task-a", "in_process_tool_oracle", 1),
        ("task-a", "deterministic_oracle_client", 1),
        ("task-b", "direct_mcp_agent_parallel", 1),
        ("task-b", "in_process_tool_oracle", 1),
        ("task-b", "deterministic_oracle_client", 1),
        ("task-a", "direct_mcp_agent_parallel", 2),
        ("task-a", "in_process_tool_oracle", 2),
        ("task-a", "deterministic_oracle_client", 2),
        ("task-b", "direct_mcp_agent_parallel", 2),
        ("task-b", "in_process_tool_oracle", 2),
        ("task-b", "deterministic_oracle_client", 2),
    ]


def test_suite_randomized_order_is_deterministic_by_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRunner:
        def __init__(self, executor: SimpleNamespace) -> None:
            self.executor = executor

        def run_task(self, task: ProbeTask, *, repetition: int = 1) -> SimpleNamespace:
            return FakeArmResult(
                task_id=task.id,
                arm_name=self.executor.name,
                repetition=repetition,
            )

    monkeypatch.setattr(suite, "build_executor", lambda arm, task: SimpleNamespace(name=arm))
    monkeypatch.setattr(suite, "BenchmarkRunner", FakeRunner)

    tasks = [tiny_task("task-a", seed=1), tiny_task("task-b", seed=2)]
    config = BenchmarkSuiteConfig(
        arms=(
            "deterministic_oracle_client",
            "in_process_tool_oracle",
            "direct_mcp_tool_oracle",
            "direct_mcp_agent_parallel",
        ),
        repetitions=3,
        arm_order="randomized",
        random_seed=17,
    )
    same_seed_config = config.model_copy()
    different_seed_config = config.model_copy(update={"random_seed": 18})

    first = run_benchmark_suite(tasks, config)
    second = run_benchmark_suite(tasks, same_seed_config)
    different = run_benchmark_suite(tasks, different_seed_config)

    first_order = [(result.task_id, result.repetition, result.arm_name) for result in first]
    assert first_order == [
        (result.task_id, result.repetition, result.arm_name)
        for result in second
    ]
    assert first_order != [
        (result.task_id, result.repetition, result.arm_name)
        for result in different
    ]


def test_suite_records_trial_identity_and_arm_execution_order() -> None:
    task = tiny_task("trial-task", seed=1)

    results = run_benchmark_suite(
        [task],
        BenchmarkSuiteConfig(
            arms=("direct_agent", "in_process", "deterministic_oracle"),
            repetitions=1,
            arm_order="fixed",
        ),
    )

    assert [result.trial_id for result in results] == ["trial-task:rep-1"] * 3
    assert [result.arm_order_index for result in results] == [0, 1, 2]
    assert {result.arm_order for result in results} == {
        (
            "direct_mcp_agent_parallel",
            "in_process_tool_oracle",
            "deterministic_oracle_client",
        )
    }


def test_suite_builds_fresh_direct_mcp_agent_parallel_executor_per_repetition() -> None:
    task = tiny_task("direct-agent-suite", tool_shape=ToolShape.BATCH)

    results = run_benchmark_suite(
        [task],
        BenchmarkSuiteConfig(arms=("direct_agent",), repetitions=2),
    )

    assert [(result.arm_name, result.repetition) for result in results] == [
        ("direct_mcp_agent_parallel", 1),
        ("direct_mcp_agent_parallel", 2),
    ]
    assert [result.execution.usage.tool_calls for result in results] == [3, 3]
    assert [len(result.execution.tool_calls) for result in results] == [3, 3]
    assert all(result.execution.error is None for result in results)
    assert all(result.score.top_k_overlap == 1.0 for result in results)


def test_suite_propagates_invalid_arm_errors() -> None:
    with pytest.raises(ValueError, match="unknown executor id: not-an-arm"):
        run_benchmark_suite(
            [tiny_task("bad-arm")],
            BenchmarkSuiteConfig(arms=("not-an-arm",)),
        )


def test_suite_validates_arms_before_running_any_task(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(arm: str, task: ProbeTask) -> object:
        raise AssertionError("build_executor should not be called for invalid suite config")

    monkeypatch.setattr(suite, "build_executor", fail_if_called)

    with pytest.raises(ValueError, match="unknown executor id: not-an-arm"):
        run_benchmark_suite(
            [tiny_task("bad-arm")],
            BenchmarkSuiteConfig(arms=("deterministic_oracle", "not-an-arm")),
        )


def test_suite_validates_paired_baseline_before_running_any_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(arm: str, task: ProbeTask) -> object:
        raise AssertionError("build_executor should not be called for invalid suite config")

    monkeypatch.setattr(suite, "build_executor", fail_if_called)

    with pytest.raises(
        ValueError,
        match="unknown paired baseline executor id: not-an-arm",
    ):
        run_benchmark_suite(
            [tiny_task("bad-baseline")],
            BenchmarkSuiteConfig(paired_baseline_arm="not-an-arm"),
        )
