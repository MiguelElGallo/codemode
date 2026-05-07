from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from codemode_probe.models import NormalizedModelUsage, NormalizedToolRequest
from codemode_probe.provider import ProviderClient, ProviderTurnRequest, ProviderTurnResponse
from codemode_probe.provider_config import LiveProvider, LiveProviderConfig, ProviderConfigError


class ProviderAdapterError(ValueError):
    pass


class ProviderTransport(Protocol):
    async def send_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AzureEndpointSettings:
    base_endpoint: str
    api_version: str | None
    chat_deployment: str | None = None


def build_provider_client(
    config: LiveProviderConfig,
    *,
    transport: ProviderTransport | None = None,
) -> ProviderClient:
    if config.provider == LiveProvider.OPENAI:
        transport = transport or OpenAISdkTransport(config)
        return OpenAIProviderClient(config, transport)
    if config.provider == LiveProvider.AZURE_OPENAI:
        transport = transport or AzureOpenAISdkTransport(config)
        return AzureOpenAIProviderClient(config, transport)
    if config.provider == LiveProvider.ANTHROPIC:
        transport = transport or AnthropicSdkTransport(config)
        return AnthropicProviderClient(config, transport)
    raise ProviderConfigError(f"unsupported provider: {config.provider}")


class OpenAIProviderClient:
    provider_name = "openai"

    def __init__(self, config: LiveProviderConfig, transport: ProviderTransport) -> None:
        self._config = config
        self._transport = transport
        self.model_name = config.model

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        response = await self._transport.send_turn(_provider_payload(self._config, request))
        return _normalize_openai_response(response)


class AzureOpenAIProviderClient(OpenAIProviderClient):
    provider_name = "azure_openai"

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        response = await self._transport.send_turn(_provider_payload(self._config, request))
        if "choices" in response:
            return _normalize_openai_chat_completion_response(response)
        return _normalize_openai_response(response)


class AnthropicProviderClient:
    provider_name = "anthropic"

    def __init__(self, config: LiveProviderConfig, transport: ProviderTransport) -> None:
        self._config = config
        self._transport = transport
        self.model_name = config.model

    async def run_provider_turn(self, request: ProviderTurnRequest) -> ProviderTurnResponse:
        response = await self._transport.send_turn(_provider_payload(self._config, request))
        return _normalize_anthropic_response(response)


class OpenAISdkTransport:
    def __init__(self, config: LiveProviderConfig) -> None:
        self._config = config
        self._state: dict[str, dict[str, Any]] = {}
        module = importlib.import_module("openai")
        async_client = getattr(module, "AsyncOpenAI", None)
        if async_client is None:
            raise ProviderConfigError("optional SDK package 'openai' does not expose AsyncOpenAI")
        self._client = async_client(
            api_key=_api_key(config),
            timeout=config.timeout_seconds,
        )

    async def send_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = _task_id(payload)
        state = self._state.get(task_id)
        if state is None or _is_initial_turn(payload):
            state = {"seen_tool_result_ids": set()}
            self._state[task_id] = state
        request_kwargs: dict[str, Any] = {
            "model": self._config.model,
            "tools": _openai_tools(payload),
            "temperature": self._config.temperature,
        }
        previous_response_id = state.get("previous_response_id")
        tool_outputs = _openai_tool_outputs(
            payload,
            seen_tool_result_ids=state["seen_tool_result_ids"],
        )
        sent_tool_result_ids = {
            str(output["call_id"])
            for output in tool_outputs
            if output.get("call_id") is not None
        }
        if previous_response_id and tool_outputs:
            request_kwargs["previous_response_id"] = previous_response_id
            request_kwargs["input"] = tool_outputs
        else:
            request_kwargs["input"] = _instruction_text(payload)

        response = await self._client.responses.create(**request_kwargs)
        response_dict = _response_dict(response)
        state["seen_tool_result_ids"].update(sent_tool_result_ids)
        if response_dict.get("id") is not None:
            state["previous_response_id"] = str(response_dict["id"])
        return response_dict


class AzureOpenAISdkTransport(OpenAISdkTransport):
    def __init__(self, config: LiveProviderConfig) -> None:
        self._config = config
        self._state: dict[str, dict[str, Any]] = {}
        module = importlib.import_module("openai")
        async_client = getattr(module, "AsyncAzureOpenAI", None)
        if async_client is None:
            raise ProviderConfigError(
                "optional SDK package 'openai' does not expose AsyncAzureOpenAI"
            )
        endpoint = _azure_endpoint_settings(_endpoint(config), config.api_version)
        if not endpoint.api_version:
            raise ProviderConfigError("Azure OpenAI requires provider api_version")
        self._chat_deployment = endpoint.chat_deployment
        self._client = async_client(
            api_key=_api_key(config),
            azure_endpoint=endpoint.base_endpoint,
            api_version=endpoint.api_version,
            timeout=config.timeout_seconds,
        )

    async def send_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._chat_deployment is None:
            return await super().send_turn(payload)

        task_id = _task_id(payload)
        state = self._state.get(task_id)
        if state is None or _is_initial_turn(payload):
            state = {
                "messages": [{"role": "user", "content": _instruction_text(payload)}],
                "seen_tool_result_ids": set(),
            }
            self._state[task_id] = state
        messages = list(state["messages"])
        tool_messages = _openai_chat_tool_messages(
            payload,
            seen_tool_result_ids=state["seen_tool_result_ids"],
        )
        sent_tool_result_ids = {
            str(message["tool_call_id"])
            for message in tool_messages
            if message.get("tool_call_id") is not None
        }
        messages.extend(tool_messages)
        request_kwargs: dict[str, Any] = {
            "model": self._chat_deployment,
            "messages": messages,
            "tools": _openai_chat_tools(payload),
            "temperature": self._config.temperature,
        }

        response = await self._client.chat.completions.create(**request_kwargs)
        response_dict = _response_dict(response)
        state["seen_tool_result_ids"].update(sent_tool_result_ids)
        assistant_message = _openai_chat_assistant_message(response_dict)
        if assistant_message is not None:
            state["messages"] = [*messages, assistant_message]
        return response_dict


class AnthropicSdkTransport:
    def __init__(self, config: LiveProviderConfig) -> None:
        self._config = config
        self._state: dict[str, dict[str, Any]] = {}
        module = importlib.import_module("anthropic")
        async_client = getattr(module, "AsyncAnthropic", None)
        if async_client is None:
            raise ProviderConfigError(
                "optional SDK package 'anthropic' does not expose AsyncAnthropic"
            )
        self._client = async_client(
            api_key=_api_key(config),
            timeout=config.timeout_seconds,
        )

    async def send_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = _task_id(payload)
        state = self._state.get(task_id)
        if state is None or _is_initial_turn(payload):
            state = {
                "messages": [{"role": "user", "content": _instruction_text(payload)}],
                "seen_tool_result_ids": set(),
            }
            self._state[task_id] = state
        messages = list(state["messages"])
        tool_results = _anthropic_tool_results(
            payload,
            seen_tool_result_ids=state["seen_tool_result_ids"],
        )
        sent_tool_result_ids = {
            str(result["tool_use_id"])
            for result in tool_results
            if result.get("tool_use_id") is not None
        }
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        response = await self._client.messages.create(
            model=self._config.model,
            max_tokens=4096,
            messages=messages,
            tools=_anthropic_tools(payload),
            temperature=self._config.temperature,
        )
        response_dict = _response_dict(response)
        content = response_dict.get("content")
        if isinstance(content, list):
            state["seen_tool_result_ids"].update(sent_tool_result_ids)
            state["messages"] = [
                *messages,
                {"role": "assistant", "content": content},
            ]
        return response_dict


def _provider_payload(
    config: LiveProviderConfig,
    request: ProviderTurnRequest,
) -> dict[str, Any]:
    return {
        "provider": config.provider.value,
        "model": config.model,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
        "turn_index": request.turn_index,
        "prompt": request.rendered_prompt.model_dump(mode="json"),
        "tool_results": [
            tool_result.model_dump(mode="json")
            for tool_result in request.tool_results
        ],
        "context": request.context.model_dump(mode="json"),
    }


def _task_id(payload: dict[str, Any]) -> str:
    prompt = _dict(payload.get("prompt", {}))
    return str(prompt.get("task_id", "unknown-task"))


def _is_initial_turn(payload: dict[str, Any]) -> bool:
    return payload.get("turn_index") == 1


def _api_key(config: LiveProviderConfig) -> str:
    value = os.environ.get(config.api_key_env_var)
    if not value:
        raise ProviderConfigError(
            f"required API key environment variable '{config.api_key_env_var}' is not set"
        )
    return value


def _endpoint(config: LiveProviderConfig) -> str:
    if config.endpoint_env_var is None:
        raise ProviderConfigError("provider endpoint_env_var is required")
    value = os.environ.get(config.endpoint_env_var)
    if not value:
        raise ProviderConfigError(
            f"required endpoint environment variable '{config.endpoint_env_var}' is not set"
        )
    return value


def _azure_endpoint_settings(endpoint: str, api_version: str | None) -> AzureEndpointSettings:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ProviderConfigError("Azure OpenAI endpoint must be an absolute URL")

    parts = [part for part in parsed.path.split("/") if part]
    chat_deployment: str | None = None
    if len(parts) >= 5 and parts[:2] == ["openai", "deployments"]:
        if parts[3:5] == ["chat", "completions"]:
            chat_deployment = parts[2]
        else:
            raise ProviderConfigError(
                "Azure OpenAI deployment endpoint must end with /chat/completions"
            )
    query_api_version = parse_qs(parsed.query).get("api-version", [None])[0]
    base_endpoint = f"{parsed.scheme}://{parsed.netloc}/"
    return AzureEndpointSettings(
        base_endpoint=base_endpoint,
        api_version=api_version or query_api_version,
        chat_deployment=chat_deployment,
    )


def _instruction_text(payload: dict[str, Any]) -> str:
    prompt = _dict(payload.get("prompt", {}))
    return "\n".join(
        [
            str(prompt.get("prompt", "")),
            "",
            "Return either tool calls or a final JSON object matching the answer schema.",
            "Do not include prose outside the final JSON object.",
            "",
            "Task context:",
            json.dumps(
                {
                    "task_id": prompt.get("task_id"),
                    "task_parameters": prompt.get("task_parameters", {}),
                    "answer_schema": prompt.get("answer_schema", {}),
                    "tool_results": payload.get("tool_results", []),
                    "execution_context": payload.get("context", {}),
                    "turn_index": payload.get("turn_index"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )


def _openai_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": _tool_parameters(tool["name"]),
        }
        for tool in _tool_specs(payload)
    ]


def _openai_tool_outputs(
    payload: dict[str, Any],
    *,
    seen_tool_result_ids: set[str],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for tool_result in _tool_results(payload):
        request = _dict(tool_result.get("request", {}))
        call_id = request.get("id")
        if call_id is None:
            continue
        call_id_text = str(call_id)
        if call_id_text in seen_tool_result_ids:
            continue
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call_id_text,
                "output": json.dumps(
                    tool_result.get("result")
                    if tool_result.get("error") is None
                    else {"error": tool_result.get("error")},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return outputs


def _openai_chat_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": _tool_parameters(tool["name"]),
            },
        }
        for tool in _tool_specs(payload)
    ]


def _openai_chat_tool_messages(
    payload: dict[str, Any],
    *,
    seen_tool_result_ids: set[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for tool_result in _tool_results(payload):
        request = _dict(tool_result.get("request", {}))
        call_id = request.get("id")
        if call_id is None:
            continue
        call_id_text = str(call_id)
        if call_id_text in seen_tool_result_ids:
            continue
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id_text,
                "content": json.dumps(
                    tool_result.get("result")
                    if tool_result.get("error") is None
                    else {"error": tool_result.get("error")},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return messages


def _openai_chat_assistant_message(response: dict[str, Any]) -> dict[str, Any] | None:
    choice = _first_chat_choice(response)
    if choice is None:
        return None
    message = _dict(choice.get("message"))
    if not message:
        return None
    assistant_message: dict[str, Any] = {"role": "assistant"}
    if message.get("content") is not None:
        assistant_message["content"] = str(message.get("content"))
    if isinstance(message.get("tool_calls"), list):
        assistant_message["tool_calls"] = message["tool_calls"]
    return assistant_message


def _anthropic_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": _tool_parameters(tool["name"]),
        }
        for tool in _tool_specs(payload)
    ]


def _anthropic_tool_results(
    payload: dict[str, Any],
    *,
    seen_tool_result_ids: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tool_result in _tool_results(payload):
        request = _dict(tool_result.get("request", {}))
        tool_use_id = request.get("id")
        if tool_use_id is None:
            continue
        tool_use_id_text = str(tool_use_id)
        if tool_use_id_text in seen_tool_result_ids:
            continue
        content: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id_text,
        }
        if tool_result.get("error") is None:
            content["content"] = json.dumps(
                tool_result.get("result"),
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            content["content"] = str(tool_result.get("error"))
            content["is_error"] = True
        results.append(content)
    return results


def _tool_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("tool_results", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _tool_specs(payload: dict[str, Any]) -> list[dict[str, str]]:
    prompt = _dict(payload.get("prompt", {}))
    specs = prompt.get("tool_specs", [])
    if not isinstance(specs, list):
        return []
    return [
        {"name": str(spec.get("name")), "description": str(spec.get("description"))}
        for spec in specs
        if isinstance(spec, dict) and spec.get("name") is not None
    ]


def _tool_parameters(tool_name: str) -> dict[str, Any]:
    if tool_name == "search_shard":
        return {
            "type": "object",
            "properties": {"shard_id": {"type": "integer", "minimum": 0}},
            "required": ["shard_id"],
            "additionalProperties": False,
        }
    if tool_name == "fetch_candidate":
        return {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}},
            "required": ["candidate_id"],
            "additionalProperties": False,
        }
    if tool_name == "fetch_candidates":
        return {
            "type": "object",
            "properties": {
                "candidate_ids": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["candidate_ids"],
            "additionalProperties": False,
        }
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _response_dict(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped
    raise ProviderAdapterError("provider SDK response must be serializable as an object")


def _normalize_openai_response(response: dict[str, Any]) -> ProviderTurnResponse:
    output = _list_field(response, "output")
    tool_requests: list[NormalizedToolRequest] = []
    final_answer: dict[str, Any] | None = None
    for item in output:
        item_type = str(item.get("type", ""))
        if item_type in {"function_call", "tool_call"}:
            tool_requests.append(
                NormalizedToolRequest(
                    id=_optional_str(item.get("call_id") or item.get("id")),
                    name=_required_str(item, "name"),
                    arguments=_object_arguments(item.get("arguments")),
                )
            )
        elif item_type == "message":
            final_answer = _first_json_text(item.get("content")) or final_answer

    return ProviderTurnResponse(
        tool_requests=tool_requests,
        final_answer=final_answer,
        usage=_openai_usage(response.get("usage", {})),
        stop_reason=_optional_str(response.get("status") or response.get("stop_reason")),
        raw=_allowlisted_raw(response, ("id", "status", "model", "created_at")),
    )


def _normalize_openai_chat_completion_response(response: dict[str, Any]) -> ProviderTurnResponse:
    choice = _first_chat_choice(response)
    message = _dict(choice.get("message")) if choice is not None else {}
    tool_requests: list[NormalizedToolRequest] = []
    final_answer: dict[str, Any] | None = None

    for item in _object_list(message.get("tool_calls", []), "choices.message.tool_calls"):
        function = _dict(item.get("function"))
        tool_requests.append(
            NormalizedToolRequest(
                id=_optional_str(item.get("id")),
                name=_required_str(function, "name"),
                arguments=_object_arguments(function.get("arguments")),
            )
        )

    content = message.get("content")
    if content is not None:
        final_answer = _json_object_from_text(str(content))

    return ProviderTurnResponse(
        tool_requests=tool_requests,
        final_answer=final_answer,
        usage=_openai_chat_usage(response.get("usage", {})),
        stop_reason=_optional_str(choice.get("finish_reason") if choice else None),
        raw=_allowlisted_raw(response, ("id", "object", "model", "created")),
    )


def _normalize_anthropic_response(response: dict[str, Any]) -> ProviderTurnResponse:
    content = _list_field(response, "content")
    tool_requests: list[NormalizedToolRequest] = []
    final_answer: dict[str, Any] | None = None
    for item in content:
        item_type = str(item.get("type", ""))
        if item_type == "tool_use":
            tool_requests.append(
                NormalizedToolRequest(
                    id=_optional_str(item.get("id")),
                    name=_required_str(item, "name"),
                    arguments=_object_arguments(item.get("input", {})),
                )
            )
        elif item_type == "text":
            final_answer = _json_object_from_text(str(item.get("text", ""))) or final_answer

    return ProviderTurnResponse(
        tool_requests=tool_requests,
        final_answer=final_answer,
        usage=_anthropic_usage(response.get("usage", {})),
        stop_reason=_optional_str(response.get("stop_reason")),
        raw=_allowlisted_raw(response, ("id", "stop_reason", "model", "role")),
    )


def _openai_usage(raw_usage: object) -> NormalizedModelUsage:
    usage = _dict(raw_usage)
    input_details = _dict(usage.get("input_token_details", {}))
    return NormalizedModelUsage(
        input_tokens=_optional_int(usage.get("input_tokens"), "usage.input_tokens"),
        output_tokens=_optional_int(usage.get("output_tokens"), "usage.output_tokens"),
        cache_read_tokens=_optional_int(
            input_details.get("cached_tokens"),
            "usage.input_token_details.cached_tokens",
        ),
        cache_write_tokens=_optional_int(
            input_details.get("cache_write_tokens"),
            "usage.input_token_details.cache_write_tokens",
        ),
    )


def _openai_chat_usage(raw_usage: object) -> NormalizedModelUsage:
    usage = _dict(raw_usage)
    prompt_details = _dict(usage.get("prompt_tokens_details", {}))
    return NormalizedModelUsage(
        input_tokens=_optional_int(usage.get("prompt_tokens"), "usage.prompt_tokens"),
        output_tokens=_optional_int(
            usage.get("completion_tokens"),
            "usage.completion_tokens",
        ),
        cache_read_tokens=_optional_int(
            prompt_details.get("cached_tokens"),
            "usage.prompt_tokens_details.cached_tokens",
        ),
    )


def _anthropic_usage(raw_usage: object) -> NormalizedModelUsage:
    usage = _dict(raw_usage)
    return NormalizedModelUsage(
        input_tokens=_optional_int(usage.get("input_tokens"), "usage.input_tokens"),
        output_tokens=_optional_int(usage.get("output_tokens"), "usage.output_tokens"),
        cache_read_tokens=_optional_int(
            usage.get("cache_read_input_tokens"),
            "usage.cache_read_input_tokens",
        ),
        cache_write_tokens=_optional_int(
            usage.get("cache_creation_input_tokens"),
            "usage.cache_creation_input_tokens",
        ),
    )


def _first_json_text(content: object) -> dict[str, Any] | None:
    for item in _object_list(content, "message.content"):
        if str(item.get("type", "")) in {"output_text", "text"}:
            parsed = _json_object_from_text(str(item.get("text", "")))
            if parsed is not None:
                return parsed
    return None


def _first_chat_choice(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = _object_list(response.get("choices", []), "choices")
    if not choices:
        return None
    return choices[0]


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        raise ProviderAdapterError("final answer text must decode to a JSON object")
    return parsed


def _json_object_from_text(value: str) -> dict[str, Any] | None:
    parsed = _json_object(value)
    if parsed is not None:
        return parsed

    stripped = value.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            parsed = _json_object("\n".join(lines[1:-1]))
            if parsed is not None:
                return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _object_arguments(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProviderAdapterError("tool arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ProviderAdapterError("tool arguments must be a JSON object")
    return dict(value)


def _list_field(response: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    return _object_list(response.get(field_name, []), field_name)


def _object_list(value: object, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProviderAdapterError(f"provider response field '{field_name}' must be a list")
    for item in value:
        if not isinstance(item, dict):
            raise ProviderAdapterError(
                f"provider response field '{field_name}' item must be an object"
            )
    return value


def _dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _required_str(item: dict[str, Any], field_name: str) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value:
        raise ProviderAdapterError(f"provider response item missing '{field_name}'")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderAdapterError(f"provider response field '{field_name}' must be an integer")
    return value


def _allowlisted_raw(
    response: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    return {key: response[key] for key in keys if key in response}
