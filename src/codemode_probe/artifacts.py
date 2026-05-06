from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.preflight import PreflightCheckResult
from codemode_probe.provider_config import LiveProviderConfig
from codemode_probe.prompts import render_prompt
from codemode_probe.provenance import benchmark_protocol_metadata, git_source_metadata
from codemode_probe.reporting import (
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

SCHEMA_VERSION = 1
DEFAULT_PAIRED_BASELINE_ARM = "direct_mcp_agent_parallel"
REDACTED_VALUE = "[REDACTED]"
MAX_TRANSCRIPT_STRING_CHARS = 512
MAX_TRANSCRIPT_COLLECTION_ITEMS = 50


def create_run_dir(base_dir: Path, *, run_id: str | None = None) -> Path:
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_artifacts(
    run_dir: Path,
    tasks: list[ProbeTask],
    results: list[ArmResult],
    *,
    suite_config: BenchmarkSuiteConfig | None = None,
    preflight_results: list[PreflightCheckResult] | None = None,
    provider_config: LiveProviderConfig | None = None,
) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "task_count": len(tasks),
        "result_count": len(results),
        "environment": _environment_metadata(),
        "source": git_source_metadata(),
        "protocol": benchmark_protocol_metadata(),
        "controls": _controls_metadata(suite_config),
    }
    paired_baseline_arm = DEFAULT_PAIRED_BASELINE_ARM
    if suite_config is not None:
        manifest["suite"] = suite_config.model_dump(mode="json")
        manifest["suite"]["normalized_arms"] = list(suite_config.normalized_arms)
        manifest["suite"]["normalized_paired_baseline_arm"] = (
            suite_config.normalized_paired_baseline_arm
        )
        paired_baseline_arm = suite_config.normalized_paired_baseline_arm
    if provider_config is not None:
        manifest["provider"] = provider_config.model_dump(mode="json")
    _write_json(run_dir / "tasks.resolved.json", [task.model_dump(mode="json") for task in tasks])
    _write_json(
        run_dir / "prompts.resolved.json",
        [render_prompt(task).model_dump(mode="json") for task in tasks],
    )
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_jsonl_rows(run_dir / "transcripts.jsonl", _transcript_rows(results))
    _write_json(run_dir / "summary.json", summarize_results(results))
    paired_deltas = summarize_paired_deltas(results, baseline_arm=paired_baseline_arm)
    _write_json(
        run_dir / "paired_deltas.json",
        paired_deltas,
    )
    _write_json(
        run_dir / "pairing_coverage.json",
        summarize_pairing_coverage(results, baseline_arm=paired_baseline_arm),
    )
    _write_json(run_dir / "paired_delta_summary.json", summarize_paired_delta_groups(paired_deltas))
    _write_json(run_dir / "paired_uncertainty.json", summarize_paired_uncertainty(paired_deltas))
    _write_json(run_dir / "workload_regimes.json", summarize_workload_regimes(tasks, results))
    _write_json(run_dir / "cache_cohorts.json", summarize_cache_cohorts(results))
    _write_json(run_dir / "failure_modes.json", summarize_failure_modes(results))
    _write_json(run_dir / "preflight.json", _preflight_payload(preflight_results))
    (run_dir / "report.md").write_text(
        render_summary_markdown(results, paired_baseline_arm=paired_baseline_arm),
        encoding="utf-8",
    )
    manifest["artifacts"] = _artifact_hashes(run_dir)
    _write_json(
        run_dir / "manifest.json",
        manifest,
    )


def _controls_metadata(suite_config: BenchmarkSuiteConfig | None) -> dict[str, object]:
    if suite_config is None:
        return {
            "repetitions": None,
            "arm_order": "unspecified",
            "random_seed": None,
            "paired_baseline_arm": DEFAULT_PAIRED_BASELINE_ARM,
            "cache_policy": "unspecified",
            "cache_namespace": None,
            "cache_warmup_repetitions": 0,
            "concurrency_policy": "sequential",
            "retry_policy": "none",
            "timeout_policy": "per-task timeout_seconds",
        }
    return {
        "repetitions": suite_config.repetitions,
        "arm_order": suite_config.arm_order,
        "random_seed": suite_config.random_seed,
        "paired_baseline_arm": suite_config.normalized_paired_baseline_arm,
        "cache_policy": suite_config.cache_policy.value,
        "cache_namespace": suite_config.cache_namespace,
        "cache_warmup_repetitions": suite_config.cache_warmup_repetitions,
        "concurrency_policy": "sequential",
        "retry_policy": "none",
        "timeout_policy": "per-task timeout_seconds",
    }


def _environment_metadata() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            package_name: _package_version(package_name)
            for package_name in (
                "codemode-probe",
                "mcp",
                "pydantic",
                "pydantic-ai-harness",
                "pydantic-monty",
                "openai",
                "anthropic",
            )
        },
    }


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _artifact_hashes(run_dir: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }


def _preflight_payload(
    preflight_results: list[PreflightCheckResult] | None,
) -> dict[str, object]:
    if preflight_results is None:
        return {
            "status": "not_run",
            "passed": None,
            "checks": [],
        }
    passed = all(result.passed for result in preflight_results)
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": [result.model_dump(mode="json") for result in preflight_results],
    }


def _transcript_rows(results: list[ArmResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        row = {
            "schema_version": SCHEMA_VERSION,
            "task_id": result.task_id,
            "arm_name": result.arm_name,
            "repetition": result.repetition,
            "trial_id": result.trial_id,
            "cache_policy": result.cache_policy.value,
            "cache_state": result.cache_state.value,
            "cache_namespace": result.cache_namespace,
            "cache_warmup_run": result.cache_warmup_run,
            "timed_out": result.timed_out,
            "execution_error": result.execution.error,
            "score_failure_reason": (
                result.score.failure_reason.value
                if result.score.failure_reason is not None
                else None
            ),
            "provenance": {
                "task_hash": result.provenance.task_hash,
                "prompt_hash": result.provenance.prompt_hash,
                "tool_spec_hash": result.provenance.tool_spec_hash,
                "candidate_set_hash": result.provenance.candidate_set_hash,
                "oracle_answer_hash": result.provenance.oracle_answer_hash,
                "executor_name": result.provenance.executor_name,
            },
            "usage": result.execution.usage.model_dump(mode="json"),
            "trace": result.execution.trace.model_dump(mode="json"),
            "tool_calls": [
                _redact_for_transcript(call.model_dump(mode="json"))
                for call in result.execution.tool_calls
            ],
            "model_turns": _redact_for_transcript(
                result.execution.raw.get("model_turns", [])
            ),
            "final_answer": (
                _redact_for_transcript(result.execution.answer.model_dump(mode="json"))
                if result.execution.answer is not None
                else None
            ),
        }
        rows.append(
            {
                **row,
                "transcript_hash": _canonical_hash(row),
            }
        )
    return rows


def _redact_for_transcript(
    value: Any,
    *,
    depth: int = 0,
) -> object:
    if depth > 8:
        return {"truncated": True, "reason": "max_depth"}
    if isinstance(value, BaseModel):
        return _redact_for_transcript(value.model_dump(mode="json"), depth=depth)
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for index, (key, item) in enumerate(sorted(value.items(), key=lambda pair: str(pair[0]))):
            if index >= MAX_TRANSCRIPT_COLLECTION_ITEMS:
                redacted["__truncated_items__"] = len(value) - MAX_TRANSCRIPT_COLLECTION_ITEMS
                break
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = REDACTED_VALUE
            else:
                redacted[key_text] = _redact_for_transcript(item, depth=depth + 1)
        return redacted
    if isinstance(value, (list, tuple)):
        items = [
            _redact_for_transcript(item, depth=depth + 1)
            for item in value[:MAX_TRANSCRIPT_COLLECTION_ITEMS]
        ]
        if len(value) > MAX_TRANSCRIPT_COLLECTION_ITEMS:
            items.append(
                {"truncated": True, "remaining_items": len(value) - MAX_TRANSCRIPT_COLLECTION_ITEMS}
            )
        return items
    if isinstance(value, str):
        if len(value) <= MAX_TRANSCRIPT_STRING_CHARS:
            return value
        return {
            "truncated": True,
            "original_chars": len(value),
            "preview": value[:MAX_TRANSCRIPT_STRING_CHARS],
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    secret_markers = (
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "token",
    )
    return any(marker in normalized for marker in secret_markers)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _write_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
