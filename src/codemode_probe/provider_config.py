from __future__ import annotations

import importlib.util
import os
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LiveProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ProviderConfigError(RuntimeError):
    pass


class LiveProviderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LiveProvider
    model: str
    enabled: bool = False
    api_key_env_var: str
    timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    model_version: str | None = None
    api_version: str | None = None
    sdk_version: str | None = None
    pricing_source_id: str | None = None
    model_docs_source_id: str | None = None
    pricing_snapshot_date: date | None = None
    currency: str | None = None

    @property
    def sdk_package(self) -> str:
        if self.provider == LiveProvider.OPENAI:
            return "openai"
        if self.provider == LiveProvider.ANTHROPIC:
            return "anthropic"
        raise ProviderConfigError(f"unsupported provider: {self.provider}")

    def validate_for_live_use(self) -> None:
        if not self.enabled:
            raise ProviderConfigError(
                f"live provider '{self.provider.value}' is disabled; pass an explicit live config"
            )
        if importlib.util.find_spec(self.sdk_package) is None:
            raise ProviderConfigError(
                f"optional SDK package '{self.sdk_package}' is not installed"
            )
        if not os.environ.get(self.api_key_env_var):
            raise ProviderConfigError(
                f"required API key environment variable '{self.api_key_env_var}' is not set"
            )


def openai_config(
    *,
    model: str = "gpt-4.1-mini",
    enabled: bool = False,
    api_key_env_var: str = "OPENAI_API_KEY",
    timeout_seconds: float = 60.0,
    temperature: float = 0.0,
    model_version: str | None = None,
    api_version: str | None = None,
    sdk_version: str | None = None,
    pricing_source_id: str | None = None,
    model_docs_source_id: str | None = None,
    pricing_snapshot_date: date | None = None,
    currency: str | None = None,
) -> LiveProviderConfig:
    return LiveProviderConfig(
        provider=LiveProvider.OPENAI,
        model=model,
        enabled=enabled,
        api_key_env_var=api_key_env_var,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        model_version=model_version,
        api_version=api_version,
        sdk_version=sdk_version,
        pricing_source_id=pricing_source_id,
        model_docs_source_id=model_docs_source_id,
        pricing_snapshot_date=pricing_snapshot_date,
        currency=currency,
    )


def anthropic_config(
    *,
    model: str = "claude-sonnet-4-5",
    enabled: bool = False,
    api_key_env_var: str = "ANTHROPIC_API_KEY",
    timeout_seconds: float = 60.0,
    temperature: float = 0.0,
    model_version: str | None = None,
    api_version: str | None = None,
    sdk_version: str | None = None,
    pricing_source_id: str | None = None,
    model_docs_source_id: str | None = None,
    pricing_snapshot_date: date | None = None,
    currency: str | None = None,
) -> LiveProviderConfig:
    return LiveProviderConfig(
        provider=LiveProvider.ANTHROPIC,
        model=model,
        enabled=enabled,
        api_key_env_var=api_key_env_var,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        model_version=model_version,
        api_version=api_version,
        sdk_version=sdk_version,
        pricing_source_id=pricing_source_id,
        model_docs_source_id=model_docs_source_id,
        pricing_snapshot_date=pricing_snapshot_date,
        currency=currency,
    )
