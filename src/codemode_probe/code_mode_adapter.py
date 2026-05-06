from __future__ import annotations

import importlib
from typing import Any

from codemode_probe.code_mode_config import (
    CodeModeConfig,
    CodeModeConfigError,
    CodeModeRuntime,
)


class CodeModeAdapterError(RuntimeError):
    pass


def create_code_mode_capability(config: CodeModeConfig) -> Any:
    config.validate_for_code_mode_use()
    if config.runtime != CodeModeRuntime.PYDANTIC_AI_HARNESS:
        raise CodeModeConfigError(f"unsupported Code Mode runtime: {config.runtime}")

    module = importlib.import_module(config.sdk_package)
    code_mode_class = getattr(module, "CodeMode", None)
    if code_mode_class is None:
        raise CodeModeAdapterError(
            f"optional Code Mode package '{config.sdk_package}' does not expose CodeMode"
        )
    return code_mode_class(
        tools=config.tool_selector,
        max_retries=config.max_retries,
    )
