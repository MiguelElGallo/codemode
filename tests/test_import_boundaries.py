from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any


def test_runner_scoring_and_reporting_do_not_import_live_provider_sdks(monkeypatch) -> None:
    for module_name in (
        "codemode_probe.code_mode_config",
        "codemode_probe.code_mode_adapter",
        "codemode_probe.runner",
        "codemode_probe.scoring",
        "codemode_probe.reporting",
        "codemode_probe.provider",
        "pydantic_ai_harness",
        "pydantic_monty",
        "openai",
        "anthropic",
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
        code_mode_packages = ("pydantic_ai_harness", "pydantic_monty")
        if name in code_mode_packages or name.startswith(
            ("pydantic_ai_harness.", "pydantic_monty.")
        ):
            raise AssertionError(f"{name} must stay behind explicit Code Mode setup")
        if name in {"openai", "anthropic"} or name.startswith(("openai.", "anthropic.")):
            raise AssertionError(f"{name} must stay behind explicit live-provider setup")
        if name == "codemode_probe.provider":
            raise AssertionError("runner/scoring must not import provider.py")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("codemode_probe.runner")
    importlib.import_module("codemode_probe.scoring")
    importlib.import_module("codemode_probe.reporting")
    importlib.import_module("codemode_probe.code_mode_config")
    importlib.import_module("codemode_probe.code_mode_adapter")
    assert "codemode_probe.provider" not in sys.modules
    assert "pydantic_ai_harness" not in sys.modules
    assert "pydantic_monty" not in sys.modules
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules
