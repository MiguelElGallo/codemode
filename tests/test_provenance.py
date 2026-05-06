from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from codemode_probe.models import ToolSpec
from codemode_probe.prompts import render_prompt
from codemode_probe.provenance import (
    BENCHMARK_PROTOCOL_VERSION,
    PROTOCOL_MODULES,
    benchmark_protocol_metadata,
    build_result_provenance,
    git_source_metadata,
    hash_candidate_set,
    hash_oracle_answer,
    hash_task,
    hash_tool_specs,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.synthetic_tools import SYNTHETIC_TOOL_SPECS, canonical_json_bytes
from codemode_probe.workload import generate_candidates, make_probe_task


def tiny_task():
    return make_probe_task(
        "provenance-task",
        seed=5,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        top_k=2,
    )


def test_hash_task_is_canonical_and_changes_with_task_contract() -> None:
    task = tiny_task()
    expected = hashlib.sha256(
        canonical_json_bytes(task.model_dump(mode="json"))
    ).hexdigest()

    assert hash_task(task) == expected
    assert hash_task(task.model_copy(update={"timeout_seconds": 10.0})) != expected


def test_hash_tool_specs_is_canonical_and_changes_with_tool_contract() -> None:
    expected = hashlib.sha256(
        canonical_json_bytes(
            [tool.model_dump(mode="json") for tool in SYNTHETIC_TOOL_SPECS]
        )
    ).hexdigest()
    changed_tools = (
        *SYNTHETIC_TOOL_SPECS,
        ToolSpec(name="extra_lookup", description="Return extra data."),
    )

    assert hash_tool_specs(SYNTHETIC_TOOL_SPECS) == expected
    assert hash_tool_specs(changed_tools) != expected


def test_candidate_and_oracle_hashes_are_canonical() -> None:
    task = tiny_task()
    candidates = generate_candidates(task.workload)
    oracle_answer = rank_candidates(task.id, candidates, task.workload.top_k)

    assert hash_candidate_set(candidates) == hashlib.sha256(
        canonical_json_bytes(
            [candidate.model_dump(mode="json") for candidate in candidates]
        )
    ).hexdigest()
    assert hash_oracle_answer(oracle_answer) == hashlib.sha256(
        canonical_json_bytes(oracle_answer.model_dump(mode="json"))
    ).hexdigest()
    assert hash_candidate_set(list(reversed(candidates))) != hash_candidate_set(candidates)


def test_build_result_provenance_links_prompt_task_tools_and_executor() -> None:
    task = tiny_task()
    candidates = generate_candidates(task.workload)
    oracle_answer = rank_candidates(task.id, candidates, task.workload.top_k)
    provenance = build_result_provenance(
        task,
        executor_name="direct_mcp_agent_parallel",
        executor_config={"mode": "scripted"},
        candidates=candidates,
        oracle_answer=oracle_answer,
    )

    assert provenance.schema_version == 1
    assert provenance.task_hash == hash_task(task)
    assert provenance.prompt_hash == render_prompt(task).canonical_hash
    assert provenance.tool_spec_hash == hash_tool_specs()
    assert provenance.candidate_set_hash == hash_candidate_set(candidates)
    assert provenance.oracle_answer_hash == hash_oracle_answer(oracle_answer)
    assert provenance.executor_name == "direct_mcp_agent_parallel"
    assert provenance.executor_config == {"mode": "scripted"}
    assert provenance.benchmark_version == "0.1.0"


def test_git_source_metadata_records_commit_branch_dirty_and_diff_hash(
    monkeypatch,
) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("branch", "--show-current"): "main\n",
        ("status", "--porcelain"): " M file.py\n",
        ("diff", "--no-ext-diff"): "diff --git a/file.py b/file.py\n",
        ("diff", "--cached", "--no-ext-diff"): "",
    }

    def fake_run(
        args: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert args[0] == "git"
        assert cwd == Path("/repo")
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=outputs[tuple(args[1:])],
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    metadata = git_source_metadata(Path("/repo"))

    assert metadata == {
        "vcs": "git",
        "commit": "abc123",
        "branch": "main",
        "dirty": True,
        "diff_hash": hashlib.sha256(
            b" M file.py\ndiff --git a/file.py b/file.py"
        ).hexdigest(),
    }


def test_git_source_metadata_fingerprints_untracked_dirty_state(monkeypatch) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("branch", "--show-current"): "main\n",
        ("status", "--porcelain"): "?? src/new_protocol.py\n",
        ("diff", "--no-ext-diff"): "",
        ("diff", "--cached", "--no-ext-diff"): "",
    }

    def fake_run(
        args: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=outputs[tuple(args[1:])],
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    metadata = git_source_metadata(Path("/repo"))

    assert metadata["dirty"] is True
    assert metadata["diff_hash"] == hashlib.sha256(
        b"?? src/new_protocol.py"
    ).hexdigest()


def test_git_source_metadata_handles_non_git_directory(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr("subprocess.run", fail_run)

    assert git_source_metadata(Path("/not-a-repo")) == {
        "vcs": "git",
        "commit": None,
        "branch": None,
        "dirty": False,
        "diff_hash": None,
    }


def test_benchmark_protocol_metadata_hashes_protocol_modules() -> None:
    def fake_reader(module: object) -> bytes:
        return f"source:{module.__name__}".encode("utf-8")

    metadata = benchmark_protocol_metadata(module_reader=fake_reader)

    assert metadata == {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "hash_algorithm": "sha256",
        "module_hashes": {
            module_name: hashlib.sha256(
                f"source:{module.__name__}".encode("utf-8")
            ).hexdigest()
            for module_name, module in PROTOCOL_MODULES.items()
        },
    }


def test_benchmark_protocol_metadata_handles_missing_module_source() -> None:
    metadata = benchmark_protocol_metadata(module_reader=lambda module: None)

    assert metadata == {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "hash_algorithm": "sha256",
        "module_hashes": {
            module_name: None for module_name in PROTOCOL_MODULES
        },
    }
