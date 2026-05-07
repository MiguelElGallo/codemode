# Evidence Register

External claims used in demos, reports, or papers should be backed by a filled
entry in this register. Leave `source_url` as `TBD` until the source has been
checked for the exact version/date used by the run.

Required evidence categories include provider pricing, provider model docs, MCP
protocol/runtime docs, Code Mode/Monty docs, and benchmark methodology sources.

| ID | Claim | Source URL | Retrieved At | Version/Date | Used By | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pydantic-code-mode-docs-2026-05-06 | Pydantic Code Mode supports code-driven tool orchestration and has documented runtime constraints. | https://pydantic.dev/docs/ai/harness/code-mode/ | 2026-05-06 | Current docs page crawled 2026-05-01 | Benchmark introduction, Code Mode arm docs | Use before making Code Mode orchestration claims; still pair with actual harness implementation evidence. |
| pydantic-monty-article-2026-05-06 | Monty is a minimal Python runtime for agent-written code with deny-by-default host access. | https://pydantic.dev/articles/pydantic-monty | 2026-05-06 | Published 2026-03 | Runtime comparison discussion | Use for runtime positioning, not as independent benchmark evidence. |
| openai-gpt-4-1-mini-docs-2026-05-06 | OpenAI documents `gpt-4.1-mini` model capabilities, context/output limits, endpoints, and token pricing. | https://platform.openai.com/docs/models/gpt-4.1-mini | 2026-05-06 | Pricing page snapshot retrieved 2026-05-06 | Live Provider Smoke, provider model docs, cost normalization | Pricing observed: USD 0.40 / 1M input tokens, USD 0.10 / 1M cached input tokens, USD 1.60 / 1M output tokens. |
| openai-responses-function-calling-2026-05-06 | OpenAI Responses API supports tool definitions and returning `function_call_output` items for function-call results. | https://platform.openai.com/docs/guides/function-calling?api-mode=responses&lang=python | 2026-05-06 | Current docs page crawled 2026-01 | OpenAI provider transport | Use with `--provider-api-version responses-v1`. |
| anthropic-tool-use-docs-2026-05-06 | Anthropic Messages API represents tool calls as assistant `tool_use` blocks and client results as user `tool_result` blocks. | https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use | 2026-05-06 | Current docs page crawled 2025-09 | Anthropic provider transport | Use for transport formatting, not OpenAI smoke pricing. |
| mcp-protocol-docs | MCP supports tool use through server/client transports relevant to the direct MCP baseline. | TBD | TBD | TBD | Direct MCP arm docs | Fill with MCP protocol/runtime documentation before external MCP transport claims. |
| paired-bootstrap-methodology | Statistical analysis method is appropriate for paired benchmark deltas. | TBD | TBD | TBD | Paired uncertainty report | Fill with benchmark/evaluation methodology source. |
