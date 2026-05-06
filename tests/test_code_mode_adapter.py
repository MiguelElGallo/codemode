from __future__ import annotations

import builtins
import importlib
import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from codemode_probe.code_mode_adapter import (
    CodeModeAdapterError,
    create_code_mode_capability,
)
from codemode_probe.code_mode_config import CodeModeConfigError, pydantic_code_mode_config


def test_code_mode_adapter_import_does_not_import_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in (
        "codemode_probe.code_mode_adapter",
        "pydantic_ai_harness",
    ):
        sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "pydantic_ai_harness" or name.startswith("pydantic_ai_harness."):
            raise AssertionError(f"{name} must stay behind explicit Code Mode setup")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("codemode_probe.code_mode_adapter")


def test_create_code_mode_capability_errors_before_import_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import_module(package: str) -> None:
        raise AssertionError(f"disabled config should not import {package}")

    monkeypatch.setattr("importlib.import_module", fail_import_module)

    with pytest.raises(CodeModeConfigError, match="Code Mode runtime is disabled"):
        create_code_mode_capability(pydantic_code_mode_config(enabled=False))


def test_create_code_mode_capability_requires_runtime_code_mode_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: importlib.machinery.ModuleSpec(package, loader=None),
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda package: SimpleNamespace(),
    )

    with pytest.raises(CodeModeAdapterError, match="does not expose CodeMode"):
        create_code_mode_capability(pydantic_code_mode_config(enabled=True))


def test_create_code_mode_capability_constructs_runtime_with_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[dict[str, object]] = []

    class FakeCodeMode:
        def __init__(self, *, tools: str, max_retries: int) -> None:
            constructed.append({"tools": tools, "max_retries": max_retries})

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: importlib.machinery.ModuleSpec(package, loader=None),
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda package: SimpleNamespace(CodeMode=FakeCodeMode),
    )

    capability = create_code_mode_capability(
        pydantic_code_mode_config(
            enabled=True,
            tool_selector="benchmark_tools",
            max_retries=1,
        )
    )

    assert isinstance(capability, FakeCodeMode)
    assert constructed == [{"tools": "benchmark_tools", "max_retries": 1}]
