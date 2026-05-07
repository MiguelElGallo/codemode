from __future__ import annotations

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from codemode_probe.executor_factory import build_executor
from codemode_probe.mcp_adapter import build_synthetic_mcp_server
from codemode_probe.mcp_client import DirectMcpSyntheticToolClient, FastMcpInProcessSession
from codemode_probe.model_loop import DirectMcpAgentExecutor
from codemode_probe.models import (
    FailureCategory,
    NormalizedModelUsage,
    NormalizedToolRequest,
    NormalizedToolResult,
    ToolShape,
)
from codemode_probe.oracle import rank_candidates
from codemode_probe.provider import ProviderBackedModelClient
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.prompts import render_prompt
from codemode_probe.provider import ProviderTurnRequest
from codemode_probe.provider_adapters import (
    AnthropicProviderClient,
    AnthropicSdkTransport,
    AzureOpenAIProviderClient,
    AzureOpenAISdkTransport,
    OpenAIProviderClient,
    OpenAISdkTransport,
    ProviderAdapterError,
    build_provider_client,
)
from codemode_probe.provider_config import anthropic_config, azure_openai_config, openai_config
from codemode_probe.workload import generate_candidates, make_probe_task


class RecordingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.payloads: list[dict[str, Any]] = []

    async def send_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.response


def tiny_request() -> ProviderTurnRequest:
    task = make_probe_task(
        "provider-adapter-task",
        seed=17,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        top_k=2,
    )
    return ProviderTurnRequest(rendered_prompt=render_prompt(task), turn_index=1)


def test_openai_provider_client_normalizes_tool_requests_usage_and_raw_metadata() -> None:
    transport = RecordingTransport(
        {
            "id": "resp-123",
            "status": "requires_action",
            "model": "gpt-test",
            "usage": {
                "input_tokens": 17,
                "output_tokens": 5,
                "input_token_details": {
                    "cached_tokens": 3,
                    "cache_write_tokens": 2,
                },
            },
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_shard",
                    "arguments": '{"shard_id": 0}',
                }
            ],
            "unbounded_raw_payload": {"Authorization": "Bearer should-not-be-kept"},
        }
    )
    config = openai_config(model="gpt-test", enabled=True, temperature=0.2)

    response = asyncio.run(
        OpenAIProviderClient(config, transport).run_provider_turn(tiny_request())
    )

    assert response.tool_requests[0].model_dump(mode="json") == {
        "id": "call-1",
        "name": "search_shard",
        "arguments": {"shard_id": 0},
    }
    assert response.final_answer is None
    assert response.usage == NormalizedModelUsage(
        input_tokens=17,
        output_tokens=5,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    assert response.stop_reason == "requires_action"
    assert response.raw == {"id": "resp-123", "status": "requires_action", "model": "gpt-test"}
    assert transport.payloads[0]["provider"] == "openai"
    assert transport.payloads[0]["model"] == "gpt-test"
    assert transport.payloads[0]["temperature"] == 0.2


def test_openai_provider_client_normalizes_final_json_answer() -> None:
    transport = RecordingTransport(
        {
            "id": "resp-456",
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"task_id": "provider-adapter-task", "candidates": []}',
                        }
                    ],
                }
            ],
        }
    )

    response = asyncio.run(
        OpenAIProviderClient(openai_config(model="gpt-test", enabled=True), transport)
        .run_provider_turn(tiny_request())
    )

    assert response.tool_requests == []
    assert response.final_answer == {"task_id": "provider-adapter-task", "candidates": []}
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4


def test_anthropic_provider_client_normalizes_tool_requests_usage_and_raw_metadata() -> None:
    transport = RecordingTransport(
        {
            "id": "msg-123",
            "model": "claude-test",
            "role": "assistant",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 6,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 1,
            },
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "fetch_candidate",
                    "input": {"candidate_id": "cand-0001"},
                }
            ],
            "raw_headers": {"x-api-key": "should-not-be-kept"},
        }
    )

    response = asyncio.run(
        AnthropicProviderClient(
            anthropic_config(model="claude-test", enabled=True),
            transport,
        ).run_provider_turn(tiny_request())
    )

    assert response.tool_requests[0].model_dump(mode="json") == {
        "id": "toolu-1",
        "name": "fetch_candidate",
        "arguments": {"candidate_id": "cand-0001"},
    }
    assert response.usage == NormalizedModelUsage(
        input_tokens=20,
        output_tokens=6,
        cache_read_tokens=4,
        cache_write_tokens=1,
    )
    assert response.stop_reason == "tool_use"
    assert response.raw == {
        "id": "msg-123",
        "stop_reason": "tool_use",
        "model": "claude-test",
        "role": "assistant",
    }


def test_anthropic_provider_client_normalizes_final_json_answer() -> None:
    transport = RecordingTransport(
        {
            "id": "msg-456",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 8, "output_tokens": 3},
            "content": [
                {
                    "type": "text",
                    "text": '{"task_id": "provider-adapter-task", "candidates": []}',
                }
            ],
        }
    )

    response = asyncio.run(
        AnthropicProviderClient(
            anthropic_config(model="claude-test", enabled=True),
            transport,
        ).run_provider_turn(tiny_request())
    )

    assert response.tool_requests == []
    assert response.final_answer == {"task_id": "provider-adapter-task", "candidates": []}
    assert response.stop_reason == "end_turn"


def test_provider_client_factory_uses_explicit_transport() -> None:
    transport = RecordingTransport({"output": []})

    client = build_provider_client(
        openai_config(model="gpt-test", enabled=True),
        transport=transport,
    )

    assert isinstance(client, OpenAIProviderClient)


def test_provider_client_factory_uses_azure_provider_client_with_explicit_transport() -> None:
    transport = RecordingTransport({"output": []})

    client = build_provider_client(
        azure_openai_config(model="deployment-test", enabled=True),
        transport=transport,
    )

    assert isinstance(client, AzureOpenAIProviderClient)
    assert client.provider_name == "azure_openai"


def test_azure_provider_client_normalizes_chat_completion_tool_requests() -> None:
    transport = RecordingTransport(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4.1-mini-2025-04-14",
            "created": 1778130456,
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_shard",
                                    "arguments": '{"shard_id": 0}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    )

    response = asyncio.run(
        AzureOpenAIProviderClient(
            azure_openai_config(model="gpt-test", enabled=True),
            transport,
        ).run_provider_turn(tiny_request())
    )

    assert response.tool_requests[0].model_dump(mode="json") == {
        "id": "call-1",
        "name": "search_shard",
        "arguments": {"shard_id": 0},
    }
    assert response.usage == NormalizedModelUsage(
        input_tokens=17,
        output_tokens=5,
        cache_read_tokens=3,
    )
    assert response.stop_reason == "tool_calls"
    assert response.raw == {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4.1-mini-2025-04-14",
        "created": 1778130456,
    }


def test_azure_provider_client_normalizes_chat_completion_final_answer() -> None:
    transport = RecordingTransport(
        {
            "id": "chatcmpl-456",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "```json\n"
                            '{"task_id": "provider-adapter-task", "candidates": []}\n'
                            "```"
                        ),
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )

    response = asyncio.run(
        AzureOpenAIProviderClient(
            azure_openai_config(model="gpt-test", enabled=True),
            transport,
        ).run_provider_turn(tiny_request())
    )

    assert response.tool_requests == []
    assert response.final_answer == {"task_id": "provider-adapter-task", "candidates": []}
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4


def test_provider_adapters_reject_malformed_tool_arguments() -> None:
    transport = RecordingTransport(
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_shard",
                    "arguments": "{not-json}",
                }
            ]
        }
    )

    with pytest.raises(ProviderAdapterError, match="tool arguments must be valid JSON"):
        asyncio.run(
            OpenAIProviderClient(openai_config(model="gpt-test", enabled=True), transport)
            .run_provider_turn(tiny_request())
        )


def test_provider_adapters_reject_malformed_response_items() -> None:
    transport = RecordingTransport({"output": ["not-an-object"]})

    with pytest.raises(
        ProviderAdapterError,
        match="provider response field 'output' item must be an object",
    ):
        asyncio.run(
            OpenAIProviderClient(openai_config(model="gpt-test", enabled=True), transport)
            .run_provider_turn(tiny_request())
        )


def test_provider_adapters_reject_malformed_message_content_items() -> None:
    transport = RecordingTransport(
        {
            "output": [
                {
                    "type": "message",
                    "content": ["not-an-object"],
                }
            ]
        }
    )

    with pytest.raises(
        ProviderAdapterError,
        match="provider response field 'message.content' item must be an object",
    ):
        asyncio.run(
            OpenAIProviderClient(openai_config(model="gpt-test", enabled=True), transport)
            .run_provider_turn(tiny_request())
        )


def test_provider_adapters_reject_non_integer_usage_tokens() -> None:
    transport = RecordingTransport(
        {
            "output": [],
            "usage": {"input_tokens": 3.5},
        }
    )

    with pytest.raises(
        ProviderAdapterError,
        match="provider response field 'usage.input_tokens' must be an integer",
    ):
        asyncio.run(
            OpenAIProviderClient(openai_config(model="gpt-test", enabled=True), transport)
            .run_provider_turn(tiny_request())
        )


def test_provider_adapter_errors_are_reported_as_adapter_failures() -> None:
    task = make_probe_task(
        "provider-adapter-failure-task",
        seed=19,
        shard_count=1,
        candidates_per_shard=1,
        payload_bytes=8,
        top_k=1,
    )
    transport = RecordingTransport(
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_shard",
                    "arguments": "{not-json}",
                }
            ]
        }
    )

    result = BenchmarkRunner(
        build_executor(
            "direct_agent",
            task,
            provider_client=OpenAIProviderClient(
                openai_config(model="gpt-test", enabled=True),
                transport,
            ),
        )
    ).run_task(task)

    assert result.execution.error == "ProviderAdapterError:tool arguments must be valid JSON"
    assert result.execution.trace.failure_category == FailureCategory.ADAPTER_FAILURE


def test_openai_sdk_transport_builds_responses_request_and_serializes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {}

    class FakeResponses:
        async def create(self, **kwargs):
            created.setdefault("requests", []).append(kwargs)
            return SimpleNamespace(
                model_dump=lambda mode="json": {
                    "id": "resp-sdk",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            )

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    transport = OpenAISdkTransport(
        openai_config(
            model="gpt-test",
            enabled=True,
            api_key_env_var="OPENAI_TEST_KEY",
            timeout_seconds=12.0,
            temperature=0.3,
        )
    )
    response = asyncio.run(transport.send_turn(_payload_from_tiny_request("openai")))

    assert created["client"] == {"api_key": "test-key", "timeout": 12.0}
    request = created["requests"][0]
    assert request["model"] == "gpt-test"
    assert request["temperature"] == 0.3
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["name"] == "search_shard"
    assert '"task_id":"provider-adapter-task"' in request["input"]
    assert response["id"] == "resp-sdk"


def test_openai_sdk_transport_sends_new_tool_outputs_with_previous_response_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeResponses:
        async def create(self, **kwargs):
            requests.append(kwargs)
            response_id = "resp-1" if len(requests) == 1 else "resp-2"
            return {"id": response_id, "status": "completed", "output": []}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    transport = OpenAISdkTransport(
        openai_config(enabled=True, api_key_env_var="OPENAI_TEST_KEY")
    )

    asyncio.run(transport.send_turn(_payload_from_tiny_request("openai")))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request(
                "openai",
                turn_index=2,
                tool_results=[
                    NormalizedToolResult(
                        request=NormalizedToolRequest(
                            id="call-1",
                            name="search_shard",
                            arguments={"shard_id": 0},
                        ),
                        result=[{"id": "cand-1"}],
                    )
                ],
            )
        )
    )

    assert requests[1]["previous_response_id"] == "resp-1"
    assert requests[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '[{"id":"cand-1"}]',
        }
    ]


def test_azure_openai_sdk_transport_builds_client_with_endpoint_and_api_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {}

    class FakeResponses:
        async def create(self, **kwargs):
            created.setdefault("requests", []).append(kwargs)
            return {"id": "azure-resp-1", "status": "completed", "output": []}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://foundry-argus.cognitiveservices.azure.com/",
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncAzureOpenAI=FakeAsyncAzureOpenAI),
    )

    transport = AzureOpenAISdkTransport(
        azure_openai_config(
            model="deployment-test",
            enabled=True,
            api_version="2024-12-01-preview",
        )
    )
    response = asyncio.run(transport.send_turn(_payload_from_tiny_request("azure_openai")))

    assert created["client"] == {
        "api_key": "azure-key",
        "azure_endpoint": "https://foundry-argus.cognitiveservices.azure.com/",
        "api_version": "2024-12-01-preview",
        "timeout": 60.0,
    }
    assert created["requests"][0]["model"] == "deployment-test"
    assert response["id"] == "azure-resp-1"


def test_azure_openai_sdk_transport_uses_chat_completions_when_endpoint_is_full_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {}

    class FakeChatCompletions:
        async def create(self, **kwargs):
            created.setdefault("requests", []).append(kwargs)
            return {
                "id": f"chatcmpl-{len(created['requests'])}",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_shard",
                                        "arguments": '{"shard_id": 0}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://foundry-argus.cognitiveservices.azure.com/openai/deployments/"
        "gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview",
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncAzureOpenAI=FakeAsyncAzureOpenAI),
    )
    transport = AzureOpenAISdkTransport(
        azure_openai_config(model="deployment-test", enabled=True)
    )
    tool_results = [
        NormalizedToolResult(
            request=NormalizedToolRequest(
                id="call-1",
                name="search_shard",
                arguments={"shard_id": 0},
            ),
            result=[{"id": "cand-1"}],
        )
    ]

    asyncio.run(transport.send_turn(_payload_from_tiny_request("azure_openai")))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request(
                "azure_openai",
                turn_index=2,
                tool_results=tool_results,
            )
        )
    )

    assert created["client"] == {
        "api_key": "azure-key",
        "azure_endpoint": "https://foundry-argus.cognitiveservices.azure.com/",
        "api_version": "2025-01-01-preview",
        "timeout": 60.0,
    }
    assert created["requests"][0]["model"] == "gpt-4.1-mini"
    assert created["requests"][0]["tools"][0]["function"]["name"] == "search_shard"
    assert created["requests"][1]["messages"][-2]["role"] == "assistant"
    assert created["requests"][1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '[{"id":"cand-1"}]',
    }


@pytest.mark.parametrize("tool_shape", [ToolShape.SCALAR, ToolShape.BATCH])
def test_azure_openai_chat_replay_runs_through_direct_agent_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tool_shape: ToolShape,
) -> None:
    task = make_probe_task(
        f"azure-chat-replay-{tool_shape.value}",
        seed=29,
        tool_shape=tool_shape,
        shard_count=2,
        candidates_per_shard=3,
        payload_bytes=8,
        relevant_fraction=0.5,
        top_k=2,
    )
    created: dict[str, Any] = {"requests": []}
    expected_answer = rank_candidates(
        task.id,
        generate_candidates(task.workload),
        task.workload.top_k,
    ).model_dump(mode="json")

    class FakeChatCompletions:
        async def create(self, **kwargs):
            created["requests"].append(kwargs)
            turn_index = len(created["requests"])
            if turn_index == 1:
                return _chat_response(
                    turn_index,
                    finish_reason="tool_calls",
                    tool_calls=[
                        _chat_tool_call(
                            f"search-{shard_id}",
                            "search_shard",
                            {"shard_id": shard_id},
                        )
                        for shard_id in range(task.workload.shard_count)
                    ],
                )
            if turn_index == 2:
                search_ids = _candidate_ids_from_chat_tool_messages(kwargs["messages"])
                if tool_shape == ToolShape.BATCH:
                    tool_calls = [
                        _chat_tool_call(
                            "fetch-batch",
                            "fetch_candidates",
                            {"candidate_ids": search_ids},
                        )
                    ]
                else:
                    tool_calls = [
                        _chat_tool_call(
                            f"fetch-{candidate_id}",
                            "fetch_candidate",
                            {"candidate_id": candidate_id},
                        )
                        for candidate_id in search_ids
                    ]
                return _chat_response(
                    turn_index,
                    finish_reason="tool_calls",
                    tool_calls=tool_calls,
                )
            return _chat_response(
                turn_index,
                finish_reason="stop",
                content=f"```json\n{json.dumps(expected_answer, sort_keys=True)}\n```",
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://foundry-argus.cognitiveservices.azure.com/openai/deployments/"
        "gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview",
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncAzureOpenAI=FakeAsyncAzureOpenAI),
    )

    transport = AzureOpenAISdkTransport(azure_openai_config(enabled=True))
    provider_client = AzureOpenAIProviderClient(
        azure_openai_config(model="unused-deployment-name", enabled=True),
        transport,
    )
    tool_client = DirectMcpSyntheticToolClient(
        FastMcpInProcessSession(build_synthetic_mcp_server(task))
    )

    result = BenchmarkRunner(
        DirectMcpAgentExecutor(
            tool_client,
            ProviderBackedModelClient(provider_client),
        )
    ).run_task(task)

    assert result.execution.error is None
    assert result.score.schema_valid is True
    assert result.score.top_k_overlap == 1.0
    assert result.execution.usage.model_requests == 3
    assert result.execution.usage.input_tokens == 60
    assert result.execution.usage.output_tokens == 30
    assert result.execution.usage.tool_calls == (
        task.workload.shard_count
        + (1 if tool_shape == ToolShape.BATCH else task.workload.candidate_count)
    )
    assert len(created["requests"]) == 3
    assert created["client"]["azure_endpoint"] == (
        "https://foundry-argus.cognitiveservices.azure.com/"
    )
    assert created["client"]["api_version"] == "2025-01-01-preview"
    assert all(request["model"] == "gpt-4.1-mini" for request in created["requests"])
    assert [message["role"] for message in created["requests"][1]["messages"][-3:]] == [
        "assistant",
        "tool",
        "tool",
    ]
    assert created["requests"][2]["messages"][-1]["role"] == "tool"
    assert result.execution.raw["model_turns"][0]["provider_raw"] == {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4.1-mini-2025-04-14",
        "created": 1778131001,
    }


def test_openai_sdk_transport_resets_state_between_repeated_task_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeResponses:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return {"id": f"resp-{len(requests)}", "status": "completed", "output": []}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    transport = OpenAISdkTransport(
        openai_config(enabled=True, api_key_env_var="OPENAI_TEST_KEY")
    )
    tool_results = [
        NormalizedToolResult(
            request=NormalizedToolRequest(
                id="call-1",
                name="search_shard",
                arguments={"shard_id": 0},
            ),
            result=[{"id": "cand-1"}],
        )
    ]

    asyncio.run(transport.send_turn(_payload_from_tiny_request("openai", turn_index=1)))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request("openai", turn_index=2, tool_results=tool_results)
        )
    )
    asyncio.run(transport.send_turn(_payload_from_tiny_request("openai", turn_index=1)))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request("openai", turn_index=2, tool_results=tool_results)
        )
    )

    assert "previous_response_id" not in requests[2]
    assert requests[3]["previous_response_id"] == "resp-3"
    assert requests[3]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '[{"id":"cand-1"}]',
        }
    ]


def test_openai_sdk_transport_keeps_tool_output_pending_when_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []
    fail_once = {"enabled": True}

    class FakeResponses:
        async def create(self, **kwargs):
            requests.append(kwargs)
            if fail_once["enabled"] and len(requests) == 2:
                fail_once["enabled"] = False
                raise RuntimeError("temporary provider failure")
            return {"id": f"resp-{len(requests)}", "status": "completed", "output": []}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    transport = OpenAISdkTransport(
        openai_config(enabled=True, api_key_env_var="OPENAI_TEST_KEY")
    )
    tool_results = [
        NormalizedToolResult(
            request=NormalizedToolRequest(
                id="call-1",
                name="search_shard",
                arguments={"shard_id": 0},
            ),
            result=[{"id": "cand-1"}],
        )
    ]

    asyncio.run(transport.send_turn(_payload_from_tiny_request("openai", turn_index=1)))
    with pytest.raises(RuntimeError, match="temporary provider failure"):
        asyncio.run(
            transport.send_turn(
                _payload_from_tiny_request("openai", turn_index=2, tool_results=tool_results)
            )
        )
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request("openai", turn_index=2, tool_results=tool_results)
        )
    )

    assert requests[2]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '[{"id":"cand-1"}]',
        }
    ]


def test_anthropic_sdk_transport_builds_messages_request_and_serializes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = {}

    class FakeMessages:
        async def create(self, **kwargs):
            created.setdefault("requests", []).append(kwargs)
            return SimpleNamespace(
                model_dump=lambda mode="json": {
                    "id": "msg-sdk",
                    "stop_reason": "end_turn",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            )

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            created["client"] = kwargs
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )

    transport = AnthropicSdkTransport(
        anthropic_config(
            model="claude-test",
            enabled=True,
            api_key_env_var="ANTHROPIC_TEST_KEY",
            timeout_seconds=13.0,
            temperature=0.4,
        )
    )
    response = asyncio.run(transport.send_turn(_payload_from_tiny_request("anthropic")))

    assert created["client"] == {"api_key": "test-key", "timeout": 13.0}
    request = created["requests"][0]
    assert request["model"] == "claude-test"
    assert request["temperature"] == 0.4
    assert request["tools"][0]["name"] == "search_shard"
    assert "input_schema" in request["tools"][0]
    assert '"task_id":"provider-adapter-task"' in request["messages"][0]["content"]
    assert response["id"] == "msg-sdk"


def test_anthropic_sdk_transport_appends_tool_results_after_assistant_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeMessages:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return {
                "id": f"msg-{len(requests)}",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu-1",
                        "name": "search_shard",
                        "input": {"shard_id": 0},
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )
    transport = AnthropicSdkTransport(
        anthropic_config(enabled=True, api_key_env_var="ANTHROPIC_TEST_KEY")
    )

    asyncio.run(transport.send_turn(_payload_from_tiny_request("anthropic")))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request(
                "anthropic",
                turn_index=2,
                tool_results=[
                    NormalizedToolResult(
                        request=NormalizedToolRequest(
                            id="toolu-1",
                            name="search_shard",
                            arguments={"shard_id": 0},
                        ),
                        result=[{"id": "cand-1"}],
                    )
                ],
            )
        )
    )

    assert requests[1]["messages"][-2]["role"] == "assistant"
    assert requests[1]["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu-1",
                "content": '[{"id":"cand-1"}]',
            }
        ],
    }


def test_anthropic_sdk_transport_resets_message_state_between_repeated_task_executions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = []

    class FakeMessages:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return {
                "id": f"msg-{len(requests)}",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu-1",
                        "name": "search_shard",
                        "input": {"shard_id": 0},
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_TEST_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )
    transport = AnthropicSdkTransport(
        anthropic_config(enabled=True, api_key_env_var="ANTHROPIC_TEST_KEY")
    )

    asyncio.run(transport.send_turn(_payload_from_tiny_request("anthropic", turn_index=1)))
    asyncio.run(
        transport.send_turn(
            _payload_from_tiny_request(
                "anthropic",
                turn_index=2,
                tool_results=[
                    NormalizedToolResult(
                        request=NormalizedToolRequest(
                            id="toolu-1",
                            name="search_shard",
                            arguments={"shard_id": 0},
                        ),
                        result=[{"id": "cand-1"}],
                    )
                ],
            )
        )
    )
    asyncio.run(transport.send_turn(_payload_from_tiny_request("anthropic", turn_index=1)))

    assert len(requests[2]["messages"]) == 1
    assert requests[2]["messages"][0]["role"] == "user"
    assert '"task_id":"provider-adapter-task"' in requests[2]["messages"][0]["content"]


def test_provider_adapters_import_without_live_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("codemode_probe.provider_adapters", None)

    real_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"openai", "anthropic"} or name.startswith(("openai.", "anthropic.")):
            raise AssertionError(f"{name} must stay behind a transport implementation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    importlib.import_module("codemode_probe.provider_adapters")


def _chat_response(
    turn_index: int,
    *,
    finish_reason: str,
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if content is not None:
        message["content"] = content
    return {
        "id": f"chatcmpl-{turn_index}",
        "object": "chat.completion",
        "model": "gpt-4.1-mini-2025-04-14",
        "created": 1778131000 + turn_index,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 10 * turn_index,
            "completion_tokens": 5 * turn_index,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def _chat_tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, sort_keys=True),
        },
    }


def _candidate_ids_from_chat_tool_messages(messages: list[dict[str, Any]]) -> list[str]:
    candidate_ids: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        payload = json.loads(str(message["content"]))
        if isinstance(payload, list):
            candidate_ids.extend(str(item["id"]) for item in payload)
    return candidate_ids


def _payload_from_tiny_request(
    provider: str,
    *,
    turn_index: int = 1,
    tool_results: list[NormalizedToolResult] | None = None,
) -> dict[str, Any]:
    request = tiny_request()
    return {
        "provider": provider,
        "model": "model-test",
        "temperature": 0.0,
        "timeout_seconds": 60.0,
        "turn_index": turn_index,
        "prompt": request.rendered_prompt.model_dump(mode="json"),
        "tool_results": [
            tool_result.model_dump(mode="json")
            for tool_result in (tool_results or [])
        ],
        "context": request.context.model_dump(mode="json"),
    }
