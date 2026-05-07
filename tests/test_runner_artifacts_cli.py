from __future__ import annotations

import json
import signal
import sys
import time
from datetime import date
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from codemode_probe.artifacts import create_run_dir, summarize_results, write_run_artifacts
from codemode_probe.cli import main
from codemode_probe.executors import DeterministicOracleExecutor
from codemode_probe.models import (
    CachePolicy,
    CacheState,
    ExecutionResult,
    FailureCategory,
    ProbeTask,
    ScoreFailureReason,
    UsageStats,
)
from codemode_probe.prompts import render_prompt
from codemode_probe.provider import ProviderTurnResponse
from codemode_probe.provider_config import anthropic_config
from codemode_probe.provenance import hash_candidate_set, hash_oracle_answer
from codemode_probe.oracle import rank_candidates
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.suite import BenchmarkSuiteConfig
from codemode_probe.workload import generate_candidates, make_probe_task

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PROTOCOL = REPO_ROOT / "docs" / "benchmark_protocol.md"
EVIDENCE_REGISTER = REPO_ROOT / "docs" / "evidence_register.md"


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
        config_metadata = {"mode": "unit-test"}

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
    assert all(result.provenance.executor_name == "contract_executor" for result in results)
    assert all(result.provenance.executor_config == {"mode": "unit-test"} for result in results)
    assert all(result.provenance.prompt_hash == render_prompt(tiny_task(result.task_id, seed=1 if result.task_id == "task-a" else 2)).canonical_hash for result in results)
    assert all(
        result.provenance.candidate_set_hash
        == hash_candidate_set(
            generate_candidates(
                tiny_task(
                    result.task_id,
                    seed=1 if result.task_id == "task-a" else 2,
                ).workload
            )
        )
        for result in results
    )


def test_benchmark_runner_enforces_task_timeout() -> None:
    class SlowExecutor:
        name = "slow_executor"

        def execute(self, task: ProbeTask) -> ExecutionResult:
            time.sleep(1)
            raise AssertionError("sleep should be interrupted by runner timeout")

    task = tiny_task().model_copy(update={"timeout_seconds": 0.01})

    result = BenchmarkRunner(SlowExecutor()).run_task(task)

    assert result.timed_out is True
    assert result.execution.error == "timeout"
    assert result.execution.trace.failure_category == FailureCategory.TIMEOUT
    assert result.score.timed_out is True
    assert result.score.failure_reason == ScoreFailureReason.TIMEOUT
    assert result.score.schema_valid is False
    assert result.latency_ms < 500


def test_benchmark_runner_restores_existing_signal_timer_without_extending_it() -> None:
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        pytest.skip("SIGALRM interval timers are unavailable on this platform")

    class BriefExecutor:
        name = "brief_executor"

        def execute(self, task: ProbeTask) -> ExecutionResult:
            time.sleep(0.05)
            return ExecutionResult(
                answer=rank_candidates(
                    task.id,
                    generate_candidates(task.workload),
                    task.workload.top_k,
                )
            )

    task = tiny_task().model_copy(update={"timeout_seconds": 1.0})
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, lambda signum, frame: None)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)

        result = BenchmarkRunner(BriefExecutor()).run_task(task)
        restored_timer = signal.setitimer(signal.ITIMER_REAL, 0)

        assert result.timed_out is False
        assert 0.1 < restored_timer[0] < 0.49
        assert restored_timer[1] == 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


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
    assert all(row["provenance"]["schema_version"] == 1 for row in result_rows)
    assert all(row["provenance"]["executor_name"] == "deterministic_oracle_client" for row in result_rows)
    assert all(row["provenance"]["prompt_hash"] == render_prompt(task).canonical_hash for row in result_rows)
    assert all(isinstance(row["provenance"]["task_hash"], str) for row in result_rows)
    assert all(isinstance(row["provenance"]["tool_spec_hash"], str) for row in result_rows)
    candidates = generate_candidates(task.workload)
    oracle_answer = rank_candidates(task.id, candidates, task.workload.top_k)
    assert all(row["provenance"]["candidate_set_hash"] == hash_candidate_set(candidates) for row in result_rows)
    assert all(row["provenance"]["oracle_answer_hash"] == hash_oracle_answer(oracle_answer) for row in result_rows)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert set(manifest) == {
        "schema_version",
        "created_at",
        "task_count",
        "result_count",
        "environment",
        "source",
        "protocol",
        "documentation",
        "claim_scope",
        "controls",
        "artifacts",
    }
    assert manifest["schema_version"] == 1
    assert manifest["task_count"] == 1
    assert manifest["result_count"] == 2
    assert isinstance(manifest["created_at"], str)
    assert manifest["claim_scope"] == "synthetic_harness_validation"
    assert set(manifest["documentation"]) == {"benchmark_protocol", "evidence_register"}
    assert manifest["documentation"]["benchmark_protocol"]["path"] == (
        "docs/benchmark_protocol.md"
    )
    assert manifest["documentation"]["benchmark_protocol"]["sha256"] == sha256(
        BENCHMARK_PROTOCOL.read_bytes()
    ).hexdigest()
    assert manifest["documentation"]["benchmark_protocol"]["bytes"] == (
        BENCHMARK_PROTOCOL.stat().st_size
    )
    assert manifest["documentation"]["evidence_register"]["path"] == (
        "docs/evidence_register.md"
    )
    assert manifest["documentation"]["evidence_register"]["sha256"] == sha256(
        EVIDENCE_REGISTER.read_bytes()
    ).hexdigest()
    assert manifest["documentation"]["evidence_register"]["bytes"] == (
        EVIDENCE_REGISTER.stat().st_size
    )
    assert manifest["controls"] == {
        "repetitions": None,
        "arm_order": "unspecified",
        "random_seed": None,
        "paired_baseline_arm": "direct_mcp_agent_parallel",
        "cache_policy": "unspecified",
        "cache_namespace": None,
        "cache_warmup_repetitions": 0,
        "concurrency_policy": "sequential",
        "retry_policy": "none",
        "timeout_policy": "per-task timeout_seconds",
    }
    assert isinstance(manifest["environment"]["python_version"], str)
    assert isinstance(manifest["environment"]["python_executable"], str)
    assert isinstance(manifest["environment"]["platform"], str)
    assert manifest["source"]["vcs"] == "git"
    assert set(manifest["source"]) == {
        "vcs",
        "commit",
        "branch",
        "dirty",
        "diff_hash",
    }
    assert manifest["protocol"]["protocol_version"] == "synthetic_pr_triage_v1"
    assert manifest["protocol"]["hash_algorithm"] == "sha256"
    assert set(manifest["protocol"]["module_hashes"]) == {
        "workload",
        "oracle",
        "scoring",
        "prompts",
    }
    assert all(
        isinstance(module_hash, str)
        for module_hash in manifest["protocol"]["module_hashes"].values()
    )
    assert set(manifest["environment"]["packages"]) == {
        "codemode-probe",
        "mcp",
        "pydantic",
        "pydantic-ai-harness",
        "pydantic-monty",
        "openai",
        "anthropic",
    }
    assert set(manifest["artifacts"]) == {
        "tasks.resolved.json",
        "prompts.resolved.json",
        "results.jsonl",
        "transcripts.jsonl",
        "summary.json",
        "paired_deltas.json",
        "pairing_coverage.json",
        "paired_delta_summary.json",
        "paired_uncertainty.json",
        "cache_cohorts.json",
        "failure_modes.json",
        "cost_estimates.json",
        "workload_regimes.json",
        "preflight.json",
        "warnings.json",
        "report.md",
    }
    results_path = run_dir / "results.jsonl"
    assert manifest["artifacts"]["results.jsonl"] == {
        "sha256": sha256(results_path.read_bytes()).hexdigest(),
        "bytes": results_path.stat().st_size,
    }
    assert "manifest.json" not in manifest["artifacts"]

    resolved_tasks = json.loads((run_dir / "tasks.resolved.json").read_text())
    assert [resolved_task["id"] for resolved_task in resolved_tasks] == [task.id]
    resolved_prompts = json.loads((run_dir / "prompts.resolved.json").read_text())
    assert resolved_prompts == [render_prompt(task).model_dump(mode="json")]
    assert resolved_prompts[0]["task_id"] == task.id
    assert isinstance(resolved_prompts[0]["canonical_hash"], str)
    assert json.loads((run_dir / "summary.json").read_text()) == summarize_results(results)


@pytest.mark.parametrize(
    "bad_run_id",
    [
        "",
        "   ",
        "../escape",
        "/tmp/escape",
        "nested/run",
        "nested\\run",
    ],
)
def test_create_run_dir_rejects_unsafe_run_ids(tmp_path: Path, bad_run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        create_run_dir(tmp_path, run_id=bad_run_id)

    assert list(tmp_path.iterdir()) == []


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
        cache_policy=CachePolicy.WARM,
        cache_namespace="suite-test",
        cache_warmup_repetitions=1,
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
        "cache_policy": "warm",
        "cache_namespace": "suite-test",
        "cache_warmup_repetitions": 1,
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
        "cache_policy": "warm",
        "cache_namespace": "suite-test",
        "cache_warmup_repetitions": 1,
        "concurrency_policy": "sequential",
        "retry_policy": "none",
        "timeout_policy": "per-task timeout_seconds",
    }


def test_write_run_artifacts_manifest_includes_provider_config_when_provided(
    tmp_path: Path,
) -> None:
    task = tiny_task()
    results = BenchmarkRunner(DeterministicOracleExecutor()).run([task])
    provider_config = anthropic_config(
        model="claude-test",
        enabled=False,
        api_key_env_var="ANTHROPIC_TEST_KEY",
        timeout_seconds=12.5,
        temperature=0.2,
        model_version="2026-02-14",
        api_version="messages-v1",
        sdk_version="0.74.0",
        pricing_source_id="anthropic-pricing-2026-05-06",
        model_docs_source_id="anthropic-model-docs-2026-05-06",
        pricing_snapshot_date=date(2026, 5, 6),
        currency="USD",
    )

    run_dir = create_run_dir(tmp_path, run_id="provider-manifest")
    write_run_artifacts(run_dir, [task], results, provider_config=provider_config)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["provider"] == {
        "provider": "anthropic",
        "model": "claude-test",
        "enabled": False,
        "api_key_env_var": "ANTHROPIC_TEST_KEY",
        "timeout_seconds": 12.5,
        "temperature": 0.2,
        "model_version": "2026-02-14",
        "api_version": "messages-v1",
        "sdk_version": "0.74.0",
        "pricing_source_id": "anthropic-pricing-2026-05-06",
        "model_docs_source_id": "anthropic-model-docs-2026-05-06",
        "pricing_snapshot_date": "2026-05-06",
        "currency": "USD",
    }
    assert manifest["claim_scope"] == "dry_run_provider_config"


def test_write_run_artifacts_sources_git_metadata_from_process_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = tiny_task()
    results = BenchmarkRunner(DeterministicOracleExecutor()).run([task])
    calls: list[Path | None] = []

    def fake_git_source_metadata(repo_dir: Path | None = None) -> dict[str, object]:
        calls.append(repo_dir)
        return {
            "vcs": "git",
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
            "diff_hash": None,
        }

    monkeypatch.setattr(
        "codemode_probe.artifacts.git_source_metadata",
        fake_git_source_metadata,
    )

    run_dir = create_run_dir(tmp_path, run_id="source-context")
    write_run_artifacts(run_dir, [task], results)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert calls == [None]
    assert manifest["source"]["commit"] == "abc123"


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
    preflight = json.loads((run_dir / "preflight.json").read_text())
    assert preflight["status"] == "passed"
    assert preflight["passed"] is True
    assert preflight["checks"]

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["arms"]["deterministic_oracle_client"]["runs"] == 2


def test_cli_skip_preflight_records_not_run_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "skip-preflight",
            "--skip-preflight",
        ],
    )

    main()

    preflight = json.loads((tmp_path / "skip-preflight" / "preflight.json").read_text())
    assert preflight == {"status": "not_run", "passed": None, "checks": []}


def test_cli_preflight_failure_raises_before_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "failed-preflight",
        ],
    )
    monkeypatch.setattr(
        "codemode_probe.cli.run_preflight_checks",
        lambda: [
            SimpleNamespace(
                name="bad-preflight",
                passed=False,
                details={},
                model_dump=lambda mode="json": {},
            )
        ],
    )

    with pytest.raises(RuntimeError, match="preflight checks failed: bad-preflight"):
        main()

    assert not (tmp_path / "failed-preflight").exists()


def test_cli_existing_run_id_raises_before_running_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "already-exists"
    existing.mkdir()

    def fail_if_called(tasks: list[ProbeTask], config: object) -> list[object]:
        raise AssertionError("benchmark should not run when run directory already exists")

    monkeypatch.setattr("codemode_probe.cli.run_benchmark_suite", fail_if_called)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "already-exists",
            "--skip-preflight",
        ],
    )

    with pytest.raises(FileExistsError):
        main()

    assert list(existing.iterdir()) == []


def test_cli_provider_dry_run_records_config_without_sdk_or_env_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_TEST_KEY", raising=False)

    def fail_find_spec(package: str) -> None:
        raise AssertionError(f"dry-run should not inspect SDK package {package}")

    monkeypatch.setattr("importlib.util.find_spec", fail_find_spec)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "provider-dry-run",
            "--skip-preflight",
            "--provider",
            "openai",
            "--provider-model",
            "gpt-test",
            "--provider-api-key-env-var",
            "OPENAI_TEST_KEY",
            "--provider-timeout-seconds",
            "9.5",
            "--provider-temperature",
            "0.1",
            "--provider-model-version",
            "2026-04-01",
            "--provider-api-version",
            "responses-v1",
            "--provider-sdk-version",
            "2.9.0",
            "--provider-pricing-source-id",
            "openai-pricing-2026-05-06",
            "--provider-model-docs-source-id",
            "openai-model-docs-2026-05-06",
            "--provider-pricing-snapshot-date",
            "2026-05-06",
            "--provider-currency",
            "USD",
            "--provider-dry-run",
        ],
    )

    main()

    manifest = json.loads((tmp_path / "provider-dry-run" / "manifest.json").read_text())
    assert manifest["provider"] == {
        "provider": "openai",
        "model": "gpt-test",
        "enabled": False,
        "api_key_env_var": "OPENAI_TEST_KEY",
        "timeout_seconds": 9.5,
        "temperature": 0.1,
        "model_version": "2026-04-01",
        "api_version": "responses-v1",
        "sdk_version": "2.9.0",
        "pricing_source_id": "openai-pricing-2026-05-06",
        "model_docs_source_id": "openai-model-docs-2026-05-06",
        "pricing_snapshot_date": "2026-05-06",
        "currency": "USD",
    }
    assert manifest["claim_scope"] == "dry_run_provider_config"


def test_cli_provider_without_dry_run_requires_explicit_live_enable_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "provider-disabled",
            "--provider",
            "anthropic",
        ],
    )

    with pytest.raises(RuntimeError, match="live provider 'anthropic' is disabled"):
        main()

    assert not (tmp_path / "provider-disabled").exists()


def test_cli_live_provider_requires_compatible_sdk_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: SimpleNamespace(name=package),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace())
    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "provider-incompatible-sdk",
            "--skip-preflight",
            "--provider",
            "openai",
            "--provider-model",
            "gpt-test",
            "--provider-api-key-env-var",
            "OPENAI_TEST_KEY",
            "--enable-live",
        ],
    )

    with pytest.raises(RuntimeError, match="does not expose AsyncOpenAI"):
        main()

    assert not (tmp_path / "provider-incompatible-sdk").exists()


def test_cli_budget_guard_runs_before_live_provider_validation_or_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_find_spec(package: str) -> None:
        raise AssertionError(f"budget failure should not inspect SDK package {package}")

    monkeypatch.setattr("importlib.util.find_spec", fail_find_spec)
    monkeypatch.delenv("OPENAI_TEST_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "budget-before-provider",
            "--skip-preflight",
            "--arms",
            "direct_agent",
            "--max-tool-calls",
            "2",
            "--max-model-requests",
            "2",
            "--provider",
            "openai",
            "--provider-model",
            "gpt-test",
            "--provider-api-key-env-var",
            "OPENAI_TEST_KEY",
            "--enable-live",
        ],
    )

    with pytest.raises(RuntimeError, match="budget exceeded: model request upper bound"):
        main()

    assert not (tmp_path / "budget-before-provider").exists()


def test_cli_records_budget_config_and_estimate_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "budget-manifest",
            "--skip-preflight",
            "--arms",
            "direct_agent",
            "--max-tool-calls",
            "1",
            "--max-model-requests",
            "10",
            "--max-run-seconds",
            "1000",
            "--max-estimated-cost",
            "1.0",
            "--budget-input-cost-per-1m",
            "2.0",
            "--budget-output-cost-per-1m",
            "8.0",
            "--budget-currency",
            "USD",
        ],
    )

    main()

    manifest = json.loads((tmp_path / "budget-manifest" / "manifest.json").read_text())
    assert manifest["budget"]["config"] == {
        "max_run_seconds": 1000.0,
        "max_model_requests": 10,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_estimated_cost": 1.0,
        "input_cost_per_1m_tokens": 2.0,
        "output_cost_per_1m_tokens": 8.0,
        "currency": "USD",
    }
    assert manifest["budget"]["estimate"]["model_request_upper_bound"] == 2
    assert manifest["budget"]["estimate"]["cost_estimated"] is True
    assert manifest["budget"]["estimate"]["currency"] == "USD"


def test_cli_live_provider_uses_provider_client_in_direct_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProviderClient:
        provider_name = "fake-live"
        model_name = "fake-model"

        async def run_provider_turn(self, request):
            return ProviderTurnResponse(
                final_answer={"task_id": request.rendered_prompt.task_id, "candidates": []},
                stop_reason="final_answer",
                raw={"request_id": "fake-1"},
            )

    captured = {}

    def fake_build_provider_client(config):
        captured["provider"] = config.provider.value
        captured["enabled"] = config.enabled
        captured["model"] = config.model
        return FakeProviderClient()

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: SimpleNamespace(name=package),
    )
    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setattr("codemode_probe.cli.build_provider_client", fake_build_provider_client)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codemode-probe",
            "--out",
            str(tmp_path),
            "--run-id",
            "provider-live",
            "--skip-preflight",
            "--arms",
            "direct_agent",
            "--provider",
            "openai",
            "--provider-model",
            "gpt-test",
            "--provider-api-key-env-var",
            "OPENAI_TEST_KEY",
            "--enable-live",
        ],
    )

    main()

    assert captured == {"provider": "openai", "enabled": True, "model": "gpt-test"}
    manifest = json.loads((tmp_path / "provider-live" / "manifest.json").read_text())
    assert manifest["claim_scope"] == "live_provider_config_validated"
    row = json.loads((tmp_path / "provider-live" / "results.jsonl").read_text())
    assert row["execution"]["raw"]["model_turns"][0]["provider_name"] == "fake-live"
    assert row["execution"]["raw"]["model_turns"][0]["model_name"] == "fake-model"


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
    assert [row["cache_policy"] for row in rows] == ["unspecified", "unspecified"]
    assert [row["cache_state"] for row in rows] == ["unspecified", "unspecified"]
    assert [row["cache_namespace"] for row in rows] == [None, None]
    assert [row["cache_warmup_run"] for row in rows] == [False, False]
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

    def fake_run_benchmark_suite(tasks: list[ProbeTask], config: object, **kwargs) -> list[object]:
        captured["task_ids"] = [task.id for task in tasks]
        captured["arms"] = config.arms
        captured["repetitions"] = config.repetitions
        captured["arm_order"] = config.arm_order
        captured["random_seed"] = config.random_seed
        captured["paired_baseline_arm"] = config.paired_baseline_arm
        captured["cache_policy"] = config.cache_policy
        captured["cache_namespace"] = config.cache_namespace
        captured["cache_warmup_repetitions"] = config.cache_warmup_repetitions
        captured["has_executor_factory"] = callable(kwargs["executor_factory"])
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
            "--cache-policy",
            "warm",
            "--cache-namespace",
            "cli-cache",
            "--cache-warmup-repetitions",
            "1",
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
        "cache_policy": CachePolicy.WARM,
        "cache_namespace": "cli-cache",
        "cache_warmup_repetitions": 1,
        "has_executor_factory": True,
    }
    assert manifest["suite"] == {
        "arms": ["direct_agent", "in_process"],
        "repetitions": 3,
        "arm_order": "randomized",
        "random_seed": 42,
        "paired_baseline_arm": "direct_agent",
        "cache_policy": "warm",
        "cache_namespace": "cli-cache",
        "cache_warmup_repetitions": 1,
        "normalized_arms": [
            "direct_mcp_agent_parallel",
            "in_process_tool_oracle",
        ],
        "normalized_paired_baseline_arm": "direct_mcp_agent_parallel",
    }
