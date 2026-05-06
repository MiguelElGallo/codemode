from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from codemode_probe.models import ArmResult, ProbeTask
from codemode_probe.prompts import render_prompt
from codemode_probe.suite import BenchmarkSuiteConfig

SCHEMA_VERSION = 1


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
    }
    if suite_config is not None:
        manifest["suite"] = suite_config.model_dump(mode="json")
        manifest["suite"]["normalized_arms"] = list(suite_config.normalized_arms)
    _write_json(
        run_dir / "manifest.json",
        manifest,
    )
    _write_json(run_dir / "tasks.resolved.json", [task.model_dump(mode="json") for task in tasks])
    _write_json(
        run_dir / "prompts.resolved.json",
        [render_prompt(task).model_dump(mode="json") for task in tasks],
    )
    _write_jsonl(run_dir / "results.jsonl", results)
    _write_json(run_dir / "summary.json", summarize_results(results))


def summarize_results(results: list[ArmResult]) -> dict[str, object]:
    by_arm: dict[str, list[ArmResult]] = {}
    for result in results:
        by_arm.setdefault(result.arm_name, []).append(result)

    arms = {}
    for arm_name, arm_results in by_arm.items():
        arms[arm_name] = {
            "runs": len(arm_results),
            "schema_valid": sum(1 for result in arm_results if result.score.schema_valid),
            "mean_latency_ms": round(
                sum(result.latency_ms for result in arm_results) / max(1, len(arm_results)),
                3,
            ),
            "mean_top_k_overlap": round(
                sum(result.score.top_k_overlap for result in arm_results)
                / max(1, len(arm_results)),
                6,
            ),
            "tool_response_bytes_total": sum(
                result.execution.usage.tool_response_bytes_total for result in arm_results
            ),
        }
    return {"schema_version": SCHEMA_VERSION, "arms": arms}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
