from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from codemode_probe import oracle, prompts, scoring, workload
from codemode_probe.models import (
    Candidate,
    ProbeTask,
    ResultProvenance,
    StructuredAnswer,
    ToolSpec,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.prompts import render_prompt
from codemode_probe.synthetic_tools import SYNTHETIC_TOOL_SPECS, canonical_json_bytes
from codemode_probe.workload import generate_candidates

BENCHMARK_PROTOCOL_VERSION = "synthetic_pr_triage_v1"
PROTOCOL_MODULES = {
    "workload": workload,
    "oracle": oracle,
    "scoring": scoring,
    "prompts": prompts,
}


def build_result_provenance(
    task: ProbeTask,
    *,
    executor_name: str,
    executor_config: dict[str, Any] | None = None,
    candidates: list[Candidate] | None = None,
    oracle_answer: StructuredAnswer | None = None,
    tool_specs: tuple[ToolSpec, ...] = SYNTHETIC_TOOL_SPECS,
) -> ResultProvenance:
    rendered_prompt = render_prompt(task, tool_specs=tool_specs)
    resolved_candidates = (
        candidates if candidates is not None else generate_candidates(task.workload)
    )
    resolved_oracle = oracle_answer if oracle_answer is not None else rank_candidates(
        task.id,
        resolved_candidates,
        task.workload.top_k,
    )
    return ResultProvenance(
        task_hash=hash_task(task),
        prompt_hash=rendered_prompt.canonical_hash,
        tool_spec_hash=hash_tool_specs(tool_specs),
        candidate_set_hash=hash_candidate_set(resolved_candidates),
        oracle_answer_hash=hash_oracle_answer(resolved_oracle),
        executor_name=executor_name,
        executor_config=executor_config or {},
        benchmark_version=_package_version("codemode-probe"),
    )


def hash_task(task: ProbeTask) -> str:
    return _hash_json(task.model_dump(mode="json"))


def hash_tool_specs(tool_specs: tuple[ToolSpec, ...] = SYNTHETIC_TOOL_SPECS) -> str:
    return _hash_json([tool.model_dump(mode="json") for tool in tool_specs])


def hash_candidate_set(candidates: list[Candidate]) -> str:
    return _hash_json([candidate.model_dump(mode="json") for candidate in candidates])


def hash_oracle_answer(answer: StructuredAnswer) -> str:
    return _hash_json(answer.model_dump(mode="json"))


def git_source_metadata(repo_dir: Path | None = None) -> dict[str, object]:
    cwd = repo_dir or Path.cwd()
    commit = _git_output(["rev-parse", "HEAD"], cwd=cwd)
    branch = _git_output(["branch", "--show-current"], cwd=cwd)
    porcelain = _git_output(["status", "--porcelain"], cwd=cwd)
    diff = _git_output(["diff", "--no-ext-diff"], cwd=cwd)
    staged_diff = _git_output(["diff", "--cached", "--no-ext-diff"], cwd=cwd)

    combined_diff = "\n".join(part for part in (porcelain, diff, staged_diff) if part)
    return {
        "vcs": "git",
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(porcelain),
        "diff_hash": hashlib.sha256(combined_diff.encode("utf-8")).hexdigest()
        if combined_diff
        else None,
    }


def benchmark_protocol_metadata(
    *,
    module_reader: Callable[[object], bytes | None] | None = None,
) -> dict[str, object]:
    reader = module_reader or _module_source_bytes
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "hash_algorithm": "sha256",
        "module_hashes": {
            module_name: _hash_source(reader(module))
            for module_name, module in PROTOCOL_MODULES.items()
        },
    }


def _hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _git_output(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n")


def _module_source_bytes(module: object) -> bytes | None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    path = Path(module_file)
    try:
        return path.read_bytes()
    except OSError:
        return None


def _hash_source(source: bytes | None) -> str | None:
    if source is None:
        return None
    return hashlib.sha256(source).hexdigest()
