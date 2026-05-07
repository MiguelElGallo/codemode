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


def test_budget_import_does_not_import_provider_stack(monkeypatch) -> None:
    module_names = (
        "codemode_probe.budget",
        "codemode_probe.executor_factory",
        "codemode_probe.provider",
        "codemode_probe.provider_adapters",
        "openai",
        "anthropic",
    )
    existing_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in module_names
        if module_name in sys.modules
    }
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        guarded_packages = (
            "codemode_probe.executor_factory",
            "codemode_probe.provider",
            "codemode_probe.provider_adapters",
            "openai",
            "anthropic",
        )
        if name in guarded_packages or name.startswith(
            tuple(f"{package}." for package in guarded_packages)
        ):
            raise AssertionError(f"{name} must stay behind post-budget execution setup")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        importlib.import_module("codemode_probe.budget")
        assert "codemode_probe.executor_factory" not in sys.modules
        assert "codemode_probe.provider" not in sys.modules
        assert "codemode_probe.provider_adapters" not in sys.modules
        assert "openai" not in sys.modules
        assert "anthropic" not in sys.modules
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        sys.modules.update(existing_modules)


def test_costs_import_does_not_import_budget_or_provider_config(monkeypatch) -> None:
    module_names = (
        "codemode_probe.costs",
        "codemode_probe.budget",
        "codemode_probe.provider_config",
        "codemode_probe.executor_factory",
        "codemode_probe.provider",
        "codemode_probe.provider_adapters",
        "openai",
        "anthropic",
    )
    existing_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in module_names
        if module_name in sys.modules
    }
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        guarded_packages = (
            "codemode_probe.budget",
            "codemode_probe.provider_config",
            "codemode_probe.executor_factory",
            "codemode_probe.provider",
            "codemode_probe.provider_adapters",
            "openai",
            "anthropic",
        )
        if name in guarded_packages or name.startswith(
            tuple(f"{package}." for package in guarded_packages)
        ):
            raise AssertionError(
                f"{name} must stay behind artifact writer call-site configuration"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    try:
        importlib.import_module("codemode_probe.costs")
        assert "codemode_probe.budget" not in sys.modules
        assert "codemode_probe.provider_config" not in sys.modules
        assert "codemode_probe.executor_factory" not in sys.modules
        assert "codemode_probe.provider" not in sys.modules
        assert "codemode_probe.provider_adapters" not in sys.modules
        assert "openai" not in sys.modules
        assert "anthropic" not in sys.modules
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        sys.modules.update(existing_modules)
