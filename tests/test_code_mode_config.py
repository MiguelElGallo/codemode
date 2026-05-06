from __future__ import annotations

import importlib.machinery
import tomllib
from pathlib import Path

import pytest

from codemode_probe.code_mode_config import (
    CodeModeConfigError,
    CodeModeRuntime,
    pydantic_code_mode_config,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_pydantic_code_mode_config_defaults_to_disabled_runtime() -> None:
    config = pydantic_code_mode_config()

    assert config.runtime == CodeModeRuntime.PYDANTIC_AI_HARNESS
    assert config.enabled is False
    assert config.timeout_seconds == 60.0
    assert config.tool_selector == "all"
    assert config.max_retries == 3
    assert config.sdk_package == "pydantic_ai_harness"


def test_disabled_code_mode_config_errors_before_sdk_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_find_spec(package: str) -> None:
        raise AssertionError(f"disabled config should not inspect SDK package {package}")

    monkeypatch.setattr("importlib.util.find_spec", fail_find_spec)

    with pytest.raises(CodeModeConfigError, match="Code Mode runtime is disabled"):
        pydantic_code_mode_config(enabled=False).validate_for_code_mode_use()


def test_missing_code_mode_sdk_error_uses_runtime_package_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda package: None)

    with pytest.raises(
        CodeModeConfigError,
        match="optional Code Mode package 'pydantic_ai_harness' is not installed",
    ):
        pydantic_code_mode_config(enabled=True).validate_for_code_mode_use()


def test_code_mode_success_path_with_mocked_sdk_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_packages: list[str] = []

    def fake_find_spec(candidate: str) -> importlib.machinery.ModuleSpec:
        checked_packages.append(candidate)
        return importlib.machinery.ModuleSpec(candidate, loader=None)

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    pydantic_code_mode_config(enabled=True).validate_for_code_mode_use()

    assert checked_packages == ["pydantic_ai_harness"]


def test_code_mode_extra_is_declared_in_project_metadata_and_lockfile() -> None:
    with (repo_root() / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    with (repo_root() / "uv.lock").open("rb") as lock_file:
        uv_lock = tomllib.load(lock_file)

    assert pyproject["project"]["optional-dependencies"]["code-mode"] == [
        "pydantic-ai-harness[code-mode]>=0.2.0",
    ]

    package = next(
        package
        for package in uv_lock["package"]
        if package["name"] == "codemode-probe"
    )
    assert package["optional-dependencies"]["code-mode"] == [
        {"name": "pydantic-ai-harness", "extra": ["code-mode"]},
    ]
    assert {
        "name": "pydantic-ai-harness",
        "extras": ["code-mode"],
        "marker": "extra == 'code-mode'",
        "specifier": ">=0.2.0",
    } in package["metadata"]["requires-dist"]
    assert "code-mode" in package["metadata"]["provides-extras"]
