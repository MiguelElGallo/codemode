from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.prompts import render_prompt
from codemode_probe.provenance import benchmark_protocol_metadata, git_source_metadata
from codemode_probe.reporting import (
    render_summary_markdown,
    summarize_pairing_coverage,
    summarize_paired_delta_groups,
    summarize_paired_deltas,
    summarize_results,
    summarize_workload_regimes,
)
from codemode_probe.suite import BenchmarkSuiteConfig

SCHEMA_VERSION = 1
DEFAULT_PAIRED_BASELINE_ARM = "direct_mcp_agent_parallel"


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
    _write_json(run_dir / "tasks.resolved.json", [task.model_dump(mode="json") for task in tasks])
    _write_json(
        run_dir / "prompts.resolved.json",
        [render_prompt(task).model_dump(mode="json") for task in tasks],
    )
    _write_jsonl(run_dir / "results.jsonl", results)
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
    _write_json(run_dir / "workload_regimes.json", summarize_workload_regimes(tasks, results))
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
            "concurrency_policy": "sequential",
            "retry_policy": "none",
            "timeout_policy": "per-task timeout_seconds",
        }
    return {
        "repetitions": suite_config.repetitions,
        "arm_order": suite_config.arm_order,
        "random_seed": suite_config.random_seed,
        "paired_baseline_arm": suite_config.normalized_paired_baseline_arm,
        "cache_policy": "unspecified",
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


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
