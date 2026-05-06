from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any


def test_runner_and_scoring_do_not_import_provider(monkeypatch) -> None:
    for module_name in (
        "codemode_probe.runner",
        "codemode_probe.scoring",
        "codemode_probe.provider",
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
        if name == "codemode_probe.provider":
            raise AssertionError("runner/scoring must not import provider.py")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("codemode_probe.runner")
    importlib.import_module("codemode_probe.scoring")
    assert "codemode_probe.provider" not in sys.modules
