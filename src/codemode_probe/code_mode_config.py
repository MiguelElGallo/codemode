from __future__ import annotations

import importlib.util
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CodeModeRuntime(StrEnum):
    PYDANTIC_AI_HARNESS = "pydantic_ai_harness"


class CodeModeConfigError(RuntimeError):
    pass


class CodeModeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime: CodeModeRuntime = CodeModeRuntime.PYDANTIC_AI_HARNESS
    enabled: bool = False
    timeout_seconds: float = Field(default=60.0, gt=0)
    tool_selector: str = "all"
    max_retries: int = Field(default=3, ge=0)

    @property
    def sdk_package(self) -> str:
        if self.runtime == CodeModeRuntime.PYDANTIC_AI_HARNESS:
            return "pydantic_ai_harness"
        raise CodeModeConfigError(f"unsupported Code Mode runtime: {self.runtime}")

    def validate_for_code_mode_use(self) -> None:
        if not self.enabled:
            raise CodeModeConfigError(
                "Code Mode runtime is disabled; pass an explicit Code Mode config"
            )
        if importlib.util.find_spec(self.sdk_package) is None:
            raise CodeModeConfigError(
                f"optional Code Mode package '{self.sdk_package}' is not installed"
            )


def pydantic_code_mode_config(
    *,
    enabled: bool = False,
    timeout_seconds: float = 60.0,
    tool_selector: str = "all",
    max_retries: int = 3,
) -> CodeModeConfig:
    return CodeModeConfig(
        runtime=CodeModeRuntime.PYDANTIC_AI_HARNESS,
        enabled=enabled,
        timeout_seconds=timeout_seconds,
        tool_selector=tool_selector,
        max_retries=max_retries,
    )
