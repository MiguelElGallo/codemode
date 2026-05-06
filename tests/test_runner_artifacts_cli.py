from __future__ import annotations

import json
from pathlib import Path

import pytest

from codemode_probe.artifacts import create_run_dir, summarize_results, write_run_artifacts
from codemode_probe.cli import main
from codemode_probe.executors import DeterministicOracleExecutor
from codemode_probe.models import ExecutionResult, ProbeTask, UsageStats
from codemode_probe.prompts import render_prompt
from codemode_probe.oracle import rank_candidates
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.suite import BenchmarkSuiteConfig
from codemode_probe.workload import generate_candidates, make_probe_task


def tiny_task(task_id: str = "task-1", *, seed: int = 11) -> ProbeTask:
    return make_probe_task(
        task_id,
        seed=seed,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        relevant_fraction=0.5,
        top_k=2,
    )


def test_deterministic_oracle_executor_returns_oracle_answer_and_usage_contract() -> None:
    task = tiny_task()
    candidates = generate_candidates(task.workload)
    expected_answer = rank_candidates(task.id, candidates, task.workload.top_k)
    expected_payload_bytes = sum(len(candidate.model_dump_json()) for candidate in candidates)

    result = DeterministicOracleExecutor().execute(task)

    assert result.answer == expected_answer
    assert result.error is None
    assert result.raw == {"candidate_count": task.workload.candidate_count}
    assert result.usage.tool_calls == task.workload.shard_count
    assert result.usage.tool_response_bytes_total == expected_payload_bytes
    assert result.usage.model_visible_bytes_total == 0


def test_benchmark_runner_repetitions_and_result_contract() -> None:
    class OracleExecutor:
        name = "contract_executor"

        def execute(self, task: ProbeTask) -> ExecutionResult:
            return ExecutionResult(
                answer=rank_candidates(
                    task.id,
                    generate_candidates(task.workload),
                    task.workload.top_k,
                ),
                usage=UsageStats(tool_calls=1),
            )

    tasks = [tiny_task("task-a", seed=1), tiny_task("task-b", seed=2)]

    results = BenchmarkRunner(OracleExecutor()).run(tasks, repetitions=2)

    assert [(result.task_id, result.repetition) for result in results] == [
        ("task-a", 1),
        ("task-b", 1),
        ("task-a", 2),
        ("task-b", 2),
    ]
    assert {result.arm_name for result in results} == {"contract_executor"}
    assert all(result.latency_ms >= 0 for result in results)
    assert all(result.timed_out is False for result in results)
    assert all(result.score.schema_valid is True for result in results)
    assert all(result.score.top_k_overlap == 1.0 for result in results)
    assert all(result.score.failure_reason is None for result in results)


def test_artifact_creation_writing_and_summary_jsonl_stability(tmp_path: Path) -> None:
    task = tiny_task()
    results = BenchmarkRunner(DeterministicOracleExecutor()).run([task], repetitions=2)

    run_dir = create_run_dir(tmp_path, run_id="fixed-run")
    assert run_dir == tmp_path / "fixed-run"
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, run_id="fixed-run")

    write_run_artifacts(run_dir, [task], results)
    first_jsonl = (run_dir / "results.jsonl").read_text()
    write_run_artifacts(run_dir, [task], results)
    second_jsonl = (run_dir / "results.jsonl").read_text()

    assert first_jsonl == second_jsonl
    result_rows = [json.loads(line) for line in first_jsonl.splitlines()]
    assert [row["repetition"] for row in result_rows] == [1, 2]
    assert [row["trial_id"] for row in result_rows] == [None, None]
    assert [row["arm_order_index"] for row in result_rows] == [None, None]
    assert [row["arm_order"] for row in result_rows] == [[], []]
    assert {row["arm_name"] for row in result_rows} == {"deterministic_oracle_client"}

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(manifest) == {
        "schema_version",
        "created_at",
        "task_count",
        "result_count",
        "environment",
        "controls",
    }
    assert manifest["schema_version"] == 1
    assert manifest["task_count"] == 1
    assert manifest["result_count"] == 2
    assert isinstance(manifest["created_at"], str)
    assert manifest["controls"] == {
        "repetitions": None,
        "arm_order": "unspecified",
        "random_seed": None,
        "paired_baseline_arm": "direct_mcp_agent_parallel",
        "cache_policy": "unspecified",
        "concurrency_policy": "sequential",
        "retry_policy": "none",
        "timeout_policy": "per-task timeout_seconds",
    }
    assert isinstance(manifest["environment"]["python_version"], str)
    assert isinstance(manifest["environment"]["python_executable"], str)
    assert isinstance(manifest["environment"]["platform"], str)
    assert set(manifest["environment"]["packages"]) == {
        "codemode-probe",
        "mcp",
        "pydantic",
        "pydantic-ai-harness",
        "pydantic-monty",
        "openai",
        "anthropic",
    }

    resolved_tasks = json.loads((run_dir / "tasks.resolved.json").read_text())
    assert [resolved_task["id"] for resolved_task in resolved_tasks] == [task.id]
    resolved_prompts = json.loads((run_dir / "prompts.resolved.json").read_text())
    assert resolved_prompts == [render_prompt(task).model_dump(mode="json")]
    assert resolved_prompts[0]["task_id"] == task.id
    assert isinstance(resolved_prompts[0]["canonical_hash"], str)
    assert json.loads((run_dir / "summary.json").read_text()) == summarize_results(results)


def test_write_run_artifacts_manifest_includes_suite_config_when_provided(
    tmp_path: Path,
) -> None:
    task = tiny_task()
    results = BenchmarkRunner(DeterministicOracleExecutor()).run([task])
    suite_config = BenchmarkSuiteConfig(
        arms=("direct_agent", "in_process"),
        repetitions=3,
        arm_order="randomized",
        random_seed=42,
    )

    run_dir = create_run_dir(tmp_path, run_id="suite-manifest")
    write_run_artifacts(run_dir, [task], results, suite_config=suite_config)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["suite"] == {
        "arms": ["direct_agent", "in_process"],
        "repetitions": 3,
        "arm_order": "randomized",
        "random_seed": 42,
        "paired_baseline_arm": "direct_mcp_agent_parallel",
        "normalized_arms": [
            "direct_mcp_agent_parallel",
            "in_process_tool_oracle",
        ],
        "normalized_paired_baseline_arm": "direct_mcp_agent_parallel",
    }
    assert manifest["controls"] == {
        "repetitions": 3,
        "arm_order": "randomized",
        "random_seed": 42,
        "paired_baseline_arm": "direct_mcp_agent_parallel",
        "cache_policy": "unspecified",
        "concurrency_policy": "sequential",
        "retry_policy": "none",
        "timeout_policy": "per-task timeout_seconds",
    }


def test_cli_writes_artifacts_without_timestamp_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "cli-run",
            "--repetitions",
            "2",
        ],
    )

    main()

    run_dir = tmp_path / "cli-run"
    assert capsys.readouterr().out.strip() == str(run_dir)
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "tasks.resolved.json").is_file()
    assert (run_dir / "prompts.resolved.json").is_file()
    assert (run_dir / "results.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["arms"]["deterministic_oracle_client"]["runs"] == 2


def test_cli_arms_selection_writes_one_result_row_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "multi-arm",
            "--arms",
            "deterministic_oracle_client,in_process",
            "--shards",
            "2",
            "--candidates-per-shard",
            "3",
            "--payload-bytes",
            "8",
            "--top-k",
            "2",
        ],
    )

    main()

    run_dir = tmp_path / "multi-arm"
    assert capsys.readouterr().out.strip() == str(run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]
    assert [row["arm_name"] for row in rows] == [
        "deterministic_oracle_client",
        "in_process_tool_oracle",
    ]
    assert [row["trial_id"] for row in rows] == [
        "synthetic_fanout_smoke:rep-1",
        "synthetic_fanout_smoke:rep-1",
    ]
    assert [row["arm_order_index"] for row in rows] == [0, 1]
    assert [row["arm_order"] for row in rows] == [
        ["deterministic_oracle_client", "in_process_tool_oracle"],
        ["deterministic_oracle_client", "in_process_tool_oracle"],
    ]
    assert all(row["score"]["top_k_overlap"] == 1.0 for row in rows)

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["arms"]["deterministic_oracle_client"]["runs"] == 1
    assert summary["arms"]["in_process_tool_oracle"]["runs"] == 1


def test_cli_workload_knobs_propagate_to_tasks_and_prompts_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "knobs",
            "--task-id",
            "knob-task",
            "--seed",
            "99",
            "--task-family",
            "batch_large_fanout",
            "--tool-shape",
            "batch",
            "--shards",
            "3",
            "--candidates-per-shard",
            "4",
            "--payload-bytes",
            "12",
            "--relevant-fraction",
            "0.75",
            "--top-k",
            "3",
            "--max-tool-calls",
            "17",
            "--timeout-seconds",
            "9.5",
        ],
    )

    main()

    run_dir = tmp_path / "knobs"
    tasks = json.loads((run_dir / "tasks.resolved.json").read_text())
    prompts = json.loads((run_dir / "prompts.resolved.json").read_text())

    assert tasks[0]["id"] == "knob-task"
    assert tasks[0]["workload"] == {
        "seed": 99,
        "task_family": "batch_large_fanout",
        "tool_shape": "batch",
        "shard_count": 3,
        "candidates_per_shard": 4,
        "payload_bytes": 12,
        "relevant_fraction": 0.75,
        "top_k": 3,
    }
    assert tasks[0]["max_tool_calls"] == 17
    assert tasks[0]["timeout_seconds"] == 9.5
    assert prompts[0]["task_parameters"] == {
        "seed": 99,
        "task_family": "batch_large_fanout",
        "tool_shape": "batch",
        "shard_count": 3,
        "candidates_per_shard": 4,
        "payload_bytes": 12,
        "relevant_fraction": 0.75,
        "top_k": 3,
    }
    assert prompts[0]["max_tool_calls"] == 17
    assert prompts[0]["timeout_seconds"] == 9.5


def test_cli_preset_smoke_writes_one_generated_task_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "preset-smoke",
            "--preset",
            "smoke",
        ],
    )

    main()

    run_dir = tmp_path / "preset-smoke"
    assert capsys.readouterr().out.strip() == str(run_dir)

    tasks = json.loads((run_dir / "tasks.resolved.json").read_text())
    prompts = json.loads((run_dir / "prompts.resolved.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]

    assert [task["id"] for task in tasks] == ["smoke_smoke_single_lookup"]
    assert [prompt["task_id"] for prompt in prompts] == ["smoke_smoke_single_lookup"]
    assert [row["task_id"] for row in rows] == ["smoke_smoke_single_lookup"]
    assert prompts[0]["task_parameters"] == tasks[0]["workload"]


def test_cli_preset_orchestration_matrix_writes_all_generated_tasks_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "preset-matrix",
            "--preset",
            "orchestration_matrix",
            "--arms",
            "deterministic_oracle_client,in_process",
            "--repetitions",
            "2",
        ],
    )

    main()

    run_dir = tmp_path / "preset-matrix"
    tasks = json.loads((run_dir / "tasks.resolved.json").read_text())
    prompts = json.loads((run_dir / "prompts.resolved.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]
    expected_task_ids = [
        "orchestration_matrix_single_lookup",
        "orchestration_matrix_small_parallel_lookup",
        "orchestration_matrix_scalar_large_fanout_25",
        "orchestration_matrix_scalar_large_fanout_100",
        "orchestration_matrix_batch_large_fanout_100",
        "orchestration_matrix_deep_branching_filter_rank",
    ]

    assert [task["id"] for task in tasks] == expected_task_ids
    assert [prompt["task_id"] for prompt in prompts] == expected_task_ids
    assert len(rows) == len(expected_task_ids) * 2 * 2
    assert {row["arm_name"] for row in rows} == {
        "deterministic_oracle_client",
        "in_process_tool_oracle",
    }
    assert {row["repetition"] for row in rows} == {1, 2}

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["task_count"] == len(expected_task_ids)
    assert manifest["result_count"] == len(rows)


def test_cli_preset_uses_seed_as_base_seed_and_payload_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "preset-seed-payload",
            "--preset",
            "orchestration_matrix",
            "--seed",
            "70",
            "--payload-bytes",
            "13",
            "--relevant-fraction",
            "0.6",
        ],
    )

    main()

    tasks = json.loads((tmp_path / "preset-seed-payload" / "tasks.resolved.json").read_text())

    assert [
        (task["id"], task["workload"]["seed"])
        for task in tasks
    ] == [
        ("orchestration_matrix_single_lookup", 70 + len("single_lookup")),
        (
            "orchestration_matrix_small_parallel_lookup",
            70 + len("small_parallel_lookup"),
        ),
        (
            "orchestration_matrix_scalar_large_fanout_25",
            70 + len("scalar_large_fanout_25"),
        ),
        (
            "orchestration_matrix_scalar_large_fanout_100",
            70 + len("scalar_large_fanout_100"),
        ),
        (
            "orchestration_matrix_batch_large_fanout_100",
            70 + len("batch_large_fanout_100"),
        ),
        (
            "orchestration_matrix_deep_branching_filter_rank",
            70 + len("deep_branching_filter_rank"),
        ),
    ]
    assert {task["workload"]["payload_bytes"] for task in tasks} == {13}
    assert {task["workload"]["relevant_fraction"] for task in tasks} == {0.6}


def test_cli_manual_mode_still_uses_manual_task_family_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "manual-family",
            "--task-id",
            "manual-task",
            "--task-family",
            "single_lookup",
            "--tool-shape",
            "scalar",
            "--shards",
            "1",
            "--candidates-per-shard",
            "1",
            "--top-k",
            "1",
        ],
    )

    main()

    tasks = json.loads((tmp_path / "manual-family" / "tasks.resolved.json").read_text())

    assert len(tasks) == 1
    assert tasks[0]["id"] == "manual-task"
    assert tasks[0]["workload"]["task_family"] == "single_lookup"
    assert tasks[0]["workload"]["tool_shape"] == "scalar"
    assert tasks[0]["workload"]["shard_count"] == 1
    assert tasks[0]["workload"]["candidates_per_shard"] == 1
    assert tasks[0]["workload"]["top_k"] == 1


def test_cli_preset_ignores_manual_task_knobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "preset-ignores-manual",
            "--preset",
            "smoke",
            "--task-id",
            "manual-task-id",
            "--task-family",
            "batch_large_fanout",
            "--tool-shape",
            "batch",
            "--shards",
            "9",
            "--candidates-per-shard",
            "9",
            "--top-k",
            "9",
            "--max-tool-calls",
            "9",
            "--timeout-seconds",
            "9",
        ],
    )

    main()

    tasks = json.loads((tmp_path / "preset-ignores-manual" / "tasks.resolved.json").read_text())
    prompts = json.loads((tmp_path / "preset-ignores-manual" / "prompts.resolved.json").read_text())

    assert len(tasks) == 1
    assert tasks[0]["id"] == "smoke_smoke_single_lookup"
    assert tasks[0]["workload"]["task_family"] == "single_lookup"
    assert tasks[0]["workload"]["tool_shape"] == "scalar"
    assert tasks[0]["workload"]["shard_count"] == 1
    assert tasks[0]["workload"]["candidates_per_shard"] == 1
    assert tasks[0]["workload"]["top_k"] == 1
    assert tasks[0]["max_tool_calls"] != 9
    assert tasks[0]["timeout_seconds"] != 9
    assert prompts[0]["task_id"] == "smoke_smoke_single_lookup"


def test_cli_direct_agent_alias_run_scores_successfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "direct-agent",
            "--arms",
            "direct_agent",
            "--tool-shape",
            "batch",
            "--shards",
            "2",
            "--candidates-per-shard",
            "3",
            "--payload-bytes",
            "8",
            "--top-k",
            "2",
        ],
    )

    main()

    row = json.loads((tmp_path / "direct-agent" / "results.jsonl").read_text())
    assert row["arm_name"] == "direct_mcp_agent_parallel"
    assert row["execution"]["error"] is None
    assert row["score"]["schema_valid"] is True
    assert row["score"]["top_k_overlap"] == 1.0
    assert row["score"]["failure_reason"] is None


def test_cli_invalid_arm_raises_before_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "bad-arm",
            "--arms",
            "not-an-arm",
        ],
    )

    with pytest.raises(ValueError, match="unknown executor id: not-an-arm"):
        main()

    assert not (tmp_path / "bad-arm").exists()


def test_cli_delegates_arm_order_and_random_seed_to_suite_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run_benchmark_suite(tasks: list[ProbeTask], config: object) -> list[object]:
        captured["task_ids"] = [task.id for task in tasks]
        captured["arms"] = config.arms
        captured["repetitions"] = config.repetitions
        captured["arm_order"] = config.arm_order
        captured["random_seed"] = config.random_seed
        captured["paired_baseline_arm"] = config.paired_baseline_arm
        return []

    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "delegated-order",
            "--task-id",
            "delegation-task",
            "--arms",
            "direct_agent,in_process",
            "--repetitions",
            "3",
            "--arm-order",
            "randomized",
            "--random-seed",
            "42",
            "--paired-baseline-arm",
            "direct_agent",
        ],
    )
    monkeypatch.setattr(
        "codemode_probe.cli.run_benchmark_suite",
        fake_run_benchmark_suite,
    )

    main()

    manifest = json.loads((tmp_path / "delegated-order" / "manifest.json").read_text())
    assert captured == {
        "task_ids": ["delegation-task"],
        "arms": ("direct_agent", "in_process"),
        "repetitions": 3,
        "arm_order": "randomized",
        "random_seed": 42,
        "paired_baseline_arm": "direct_agent",
    }
    assert manifest["suite"] == {
        "arms": ["direct_agent", "in_process"],
        "repetitions": 3,
        "arm_order": "randomized",
        "random_seed": 42,
        "paired_baseline_arm": "direct_agent",
        "normalized_arms": [
            "direct_mcp_agent_parallel",
            "in_process_tool_oracle",
        ],
        "normalized_paired_baseline_arm": "direct_mcp_agent_parallel",
    }
