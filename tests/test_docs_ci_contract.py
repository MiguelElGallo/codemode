from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from codemode_probe.artifacts import write_run_artifacts
from codemode_probe.cli import main
from codemode_probe.models import ProbeTask
from codemode_probe.suite import BenchmarkSuiteConfig
from codemode_probe.workload import make_probe_task

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BENCHMARK_PROTOCOL = REPO_ROOT / "docs" / "benchmark_protocol.md"
EVIDENCE_REGISTER = REPO_ROOT / "docs" / "evidence_register.md"


def _readme_fenced_blocks(language: str) -> list[str]:
    text = README.read_text(encoding="utf-8")
    return re.findall(rf"```{language}\n(.*?)\n```", text, flags=re.DOTALL)


def _continued_shell_words(command: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", command.strip())
    return shlex.split(joined)


def _cli_args_from_readme_command(command: str, tmp_path: Path) -> list[str]:
    words = _continued_shell_words(command)
    assert words[:5] == ["uv", "run", "python", "-m", "codemode_probe.cli"]
    args = words[5:]
    out_index = args.index("--out") + 1
    args[out_index] = str(tmp_path)
    return args


def test_readme_cli_snippets_parse_and_delegate_to_benchmark_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = [
        block
        for block in _readme_fenced_blocks("bash")
        if "python -m codemode_probe.cli" in block
    ]
    assert len(commands) == 2
    captured: list[dict[str, object]] = []

    def fake_run_benchmark_suite(
        tasks: list[ProbeTask],
        config: BenchmarkSuiteConfig,
    ) -> list[object]:
        captured.append(
            {
                "task_ids": [task.id for task in tasks],
                "arms": config.arms,
                "normalized_arms": config.normalized_arms,
                "repetitions": config.repetitions,
                "arm_order": config.arm_order,
                "random_seed": config.random_seed,
            }
        )
        return []

    monkeypatch.setattr(
        "codemode_probe.cli.run_benchmark_suite",
        fake_run_benchmark_suite,
    )

    for index, command in enumerate(commands, start=1):
        monkeypatch.setattr(
            "sys.argv",
            [
                "codemode-probe",
                *_cli_args_from_readme_command(command, tmp_path),
                "--run-id",
                f"docs-{index}",
            ],
        )
        main()

    assert captured == [
        {
            "task_ids": ["smoke_smoke_single_lookup"],
            "arms": (
                "deterministic_oracle_client",
                "in_process",
                "direct_mcp",
                "direct_agent",
            ),
            "normalized_arms": (
                "deterministic_oracle_client",
                "in_process_tool_oracle",
                "direct_mcp_tool_oracle",
                "direct_mcp_agent_parallel",
            ),
            "repetitions": 1,
            "arm_order": "fixed",
            "random_seed": 1,
        },
        {
            "task_ids": [
                "orchestration_matrix_single_lookup",
                "orchestration_matrix_small_parallel_lookup",
                "orchestration_matrix_scalar_large_fanout_25",
                "orchestration_matrix_scalar_large_fanout_100",
                "orchestration_matrix_batch_large_fanout_100",
                "orchestration_matrix_deep_branching_filter_rank",
            ],
            "arms": (
                "direct_mcp_agent_parallel",
                "direct_mcp_tool_oracle",
                "in_process_tool_oracle",
            ),
            "normalized_arms": (
                "direct_mcp_agent_parallel",
                "direct_mcp_tool_oracle",
                "in_process_tool_oracle",
            ),
            "repetitions": 3,
            "arm_order": "randomized",
            "random_seed": 17,
        },
    ]


def test_readme_artifact_layout_matches_writer_outputs(tmp_path: Path) -> None:
    text_blocks = _readme_fenced_blocks("text")
    documented_artifacts = {
        line.strip()
        for block in text_blocks
        for line in block.splitlines()
        if line.strip().endswith((".json", ".jsonl", ".md"))
    }
    task = make_probe_task(
        "docs-artifacts",
        seed=1,
        shard_count=1,
        candidates_per_shard=1,
        payload_bytes=4,
        top_k=1,
    )

    write_run_artifacts(tmp_path, [task], [])

    assert documented_artifacts == {
        path.name for path in tmp_path.iterdir() if path.is_file()
    }


def test_readme_links_protocol_and_evidence_register() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "[docs/benchmark_protocol.md](docs/benchmark_protocol.md)" in readme
    assert "[docs/evidence_register.md](docs/evidence_register.md)" in readme
    assert BENCHMARK_PROTOCOL.is_file()
    assert EVIDENCE_REGISTER.is_file()


def test_protocol_doc_declares_current_claim_boundaries() -> None:
    protocol = BENCHMARK_PROTOCOL.read_text(encoding="utf-8")

    assert "synthetic harness" in protocol
    assert "not a real Pydantic Code Mode/Monty runtime" in protocol
    assert "do not support claims about live model quality" in protocol
    assert "(task_id, repetition, trial_id)" in protocol


def test_evidence_register_has_required_columns() -> None:
    evidence = EVIDENCE_REGISTER.read_text(encoding="utf-8")

    assert "| Claim | Source URL | Retrieved At | Version/Date | Used By | Notes |" in evidence
    assert "Provider pricing" in evidence
    assert "Code Mode/Monty docs" in evidence


def test_readme_setup_command_matches_ci_test_command() -> None:
    setup_block = next(
        block for block in _readme_fenced_blocks("bash") if "pytest -q" in block
    )
    readme_commands = setup_block.splitlines()
    ci_text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "uv sync --extra dev" in readme_commands
    assert "uv run --extra dev pytest -q" in readme_commands
    assert 'python-version: ["3.11", "3.12", "3.13"]' in ci_text
    assert re.search(r"run:\s*uv run --extra dev pytest -q", ci_text)
    assert "Python 3.11, 3.12, and 3.13" in README.read_text(encoding="utf-8")
    assert "uv build" in ci_text
    assert "uv run --extra providers pytest -q" in ci_text
    assert "uv run --extra code-mode pytest -q" in ci_text
