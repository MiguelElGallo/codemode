from __future__ import annotations

import importlib.machinery
import tomllib
from datetime import date
from pathlib import Path

import pytest

from codemode_probe.provider_config import (
    LiveProvider,
    ProviderConfigError,
    anthropic_config,
    azure_openai_config,
    openai_config,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_openai_config_defaults_to_disabled_live_provider() -> None:
    config = openai_config()

    assert config.provider == LiveProvider.OPENAI
    assert config.model == "gpt-4.1-mini"
    assert config.enabled is False
    assert config.api_key_env_var == "OPENAI_API_KEY"
    assert config.endpoint_env_var is None
    assert config.timeout_seconds == 60.0
    assert config.temperature == 0.0
    assert config.sdk_package == "openai"
    assert config.model_version is None
    assert config.api_version is None
    assert config.sdk_version is None
    assert config.pricing_source_id is None
    assert config.model_docs_source_id is None
    assert config.pricing_snapshot_date is None
    assert config.currency is None


def test_anthropic_config_defaults_to_disabled_live_provider() -> None:
    config = anthropic_config()

    assert config.provider == LiveProvider.ANTHROPIC
    assert config.model == "claude-sonnet-4-5"
    assert config.enabled is False
    assert config.api_key_env_var == "ANTHROPIC_API_KEY"
    assert config.endpoint_env_var is None
    assert config.timeout_seconds == 60.0
    assert config.temperature == 0.0
    assert config.sdk_package == "anthropic"
    assert config.model_version is None
    assert config.api_version is None
    assert config.sdk_version is None
    assert config.pricing_source_id is None
    assert config.model_docs_source_id is None
    assert config.pricing_snapshot_date is None
    assert config.currency is None


def test_provider_config_serializes_reproducibility_evidence_fields() -> None:
    config = openai_config(
        model="gpt-test",
        model_version="2026-04-01",
        api_version="responses-v1",
        sdk_version="2.9.0",
        pricing_source_id="openai-pricing-2026-05-06",
        model_docs_source_id="openai-model-docs-2026-05-06",
        pricing_snapshot_date=date(2026, 5, 6),
        currency="USD",
    )

    assert config.model_dump(mode="json") == {
        "provider": "openai",
        "model": "gpt-test",
        "enabled": False,
        "api_key_env_var": "OPENAI_API_KEY",
        "endpoint_env_var": None,
        "timeout_seconds": 60.0,
        "temperature": 0.0,
        "model_version": "2026-04-01",
        "api_version": "responses-v1",
        "sdk_version": "2.9.0",
        "pricing_source_id": "openai-pricing-2026-05-06",
        "model_docs_source_id": "openai-model-docs-2026-05-06",
        "pricing_snapshot_date": "2026-05-06",
        "currency": "USD",
    }


def test_azure_openai_config_defaults_to_disabled_live_provider() -> None:
    config = azure_openai_config()

    assert config.provider == LiveProvider.AZURE_OPENAI
    assert config.model == "gpt-4.1-mini"
    assert config.enabled is False
    assert config.api_key_env_var == "AZURE_OPENAI_API_KEY"
    assert config.endpoint_env_var == "AZURE_OPENAI_ENDPOINT"
    assert config.sdk_package == "openai"


def test_azure_openai_config_requires_endpoint_env_var_after_sdk_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: importlib.machinery.ModuleSpec(package, loader=None),
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    with pytest.raises(
        ProviderConfigError,
        match="required endpoint environment variable 'AZURE_OPENAI_ENDPOINT' is not set",
    ):
        azure_openai_config(enabled=True).validate_for_live_use()


def test_disabled_live_config_errors_before_sdk_or_env_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_find_spec(package: str) -> None:
        raise AssertionError(f"disabled config should not inspect SDK package {package}")

    monkeypatch.setattr("importlib.util.find_spec", fail_find_spec)

    with pytest.raises(ProviderConfigError, match="live provider 'openai' is disabled"):
        openai_config(enabled=False).validate_for_live_use()


def test_missing_sdk_error_uses_provider_package_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda package: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(
        ProviderConfigError,
        match="optional SDK package 'anthropic' is not installed",
    ):
        anthropic_config(enabled=True).validate_for_live_use()


def test_missing_env_var_error_after_sdk_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda package: importlib.machinery.ModuleSpec(package, loader=None),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(
        ProviderConfigError,
        match="required API key environment variable 'OPENAI_API_KEY' is not set",
    ):
        openai_config(enabled=True).validate_for_live_use()


@pytest.mark.parametrize(
    ("config", "package", "env_var"),
    [
        (openai_config(enabled=True), "openai", "OPENAI_API_KEY"),
        (azure_openai_config(enabled=True), "openai", "AZURE_OPENAI_API_KEY"),
        (anthropic_config(enabled=True), "anthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_env_var_success_path_with_mocked_sdk_presence(
    monkeypatch: pytest.MonkeyPatch,
    config,
    package: str,
    env_var: str,
) -> None:
    checked_packages: list[str] = []

    def fake_find_spec(candidate: str) -> importlib.machinery.ModuleSpec:
        checked_packages.append(candidate)
        return importlib.machinery.ModuleSpec(candidate, loader=None)

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    monkeypatch.setenv(env_var, "test-key")
    if config.endpoint_env_var is not None:
        monkeypatch.setenv(config.endpoint_env_var, "https://example.openai.azure.com/")

    config.validate_for_live_use()

    assert checked_packages == [package]


def test_providers_extra_is_declared_in_project_metadata_and_lockfile() -> None:
    with (repo_root() / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    with (repo_root() / "uv.lock").open("rb") as lock_file:
        uv_lock = tomllib.load(lock_file)

    assert pyproject["project"]["optional-dependencies"]["providers"] == [
        "anthropic>=0.74",
        "openai>=2.0",
    ]

    package = next(
        package
        for package in uv_lock["package"]
        if package["name"] == "codemode-probe"
    )
    assert package["optional-dependencies"]["providers"] == [
        {"name": "anthropic"},
        {"name": "openai"},
    ]
    assert "providers" in package["metadata"]["provides-extras"]
