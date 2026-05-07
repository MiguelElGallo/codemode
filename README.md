# Code Mode Probe

[![CI](https://github.com/MiguelElGallo/codemode/actions/workflows/ci.yml/badge.svg)](https://github.com/MiguelElGallo/codemode/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Tests](https://img.shields.io/badge/tests-235%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-local%20only-lightgrey)
![Version](https://img.shields.io/badge/version-0.1.0-blue)

This project is a small benchmark harness for one question:

Can code-driven orchestration over MCP-shaped tools reduce model round trips,
tokens, latency, and model-visible payloads compared with direct tool calling?

It does not try to prove that Code Mode is always better. It tries to find the
point where each approach is useful.

## What The Benchmark Tests

### The case

Imagine a repository triage task.

The model has to rank candidates that are ready to merge. It should exclude
drafts and bot-authored candidates. It should look at approvals, CI status,
reactions, recency, changed-file count, and relevance. Then it should return
structured JSON.

That is a good benchmark shape because it is not just one lookup.

The agent has to:

- search for candidate summaries
- fetch full candidate payloads
- filter irrelevant candidates
- score the remaining candidates
- return only the ranked result

This is the kind of workflow where direct parallel tool calls help, but they do
not remove every cost. If each intermediate result has to go back through the
model context, large fan-out can still become expensive.

### The two arms

The benchmark compares two arms.

The first arm is `direct_mcp_agent_parallel`.

It is a normal model-driven tool loop over MCP-shaped synthetic tools. The
model receives the task prompt and tool definitions. It asks for a tool call.
The harness executes the synthetic tool. The tool result is sent back to the
model. The loop continues until the model returns final JSON.

The second arm is `code_mode_pydantic_monty`.

It uses Pydantic AI Harness `CodeMode()` with Monty as the runtime. The model
asks for one `run_code` call. The generated Python runs inside Monty and calls
the same synthetic tools from code. The intermediate tool results stay inside
the code execution step as nested metadata. They are not sent back as individual
tool-result messages. The model sees the `run_code` return value.

That is the distinction being tested:

```text
Direct model-driven tool calling:
model -> tool call -> tool result back to model -> more tool calls -> final JSON

Code Mode:
model -> run_code -> Python calls tools inside Monty -> final JSON
```

### What is sent in the example smoke run

For the example bounded smoke run, the first user message is built from this
task:

```text
Rank the top candidates most ready to merge. Exclude drafts and bot-authored candidates. Consider approvals, CI status, reactions, recency, changed-file count, and relevance. Return structured JSON.
```

The resolved task has one shard, one candidate, scalar tool calls, and `top_k =
1`.

```json
{
  "task_id": "smoke_smoke_single_lookup",
  "task_parameters": {
    "shard_count": 1,
    "candidates_per_shard": 1,
    "tool_shape": "scalar",
    "top_k": 1
  }
}
```

This smoke is intentionally tiny. It checks plumbing and accounting. Larger
fan-out runs are needed to test the cost and latency crossover.

### Direct model-driven arm

In the example Azure run, the direct arm uses the full deployment
chat-completions URL shown later in this README. Each model turn sends Azure
OpenAI a request with:

- the task prompt
- the answer schema
- the tool definitions
- the current turn index
- any previous tool results

On the first turn, Azure returned a tool call:

```json
{"name": "search_shard", "arguments": {"shard_id": 0}}
```

The harness executed the tool and sent this result back to Azure:

```json
{
  "category": "infra",
  "id": "cand-0000",
  "shard_id": 0,
  "title": "tests candidate 0"
}
```

On the second turn, Azure returned another tool call:

```json
{"name": "fetch_candidate", "arguments": {"candidate_id": "cand-0000"}}
```

The harness executed the tool and sent the full candidate payload back to Azure.
This abbreviated snippet shows the fields used for scoring. The actual full
payload also includes fields such as `category`, `shard_id`, and the synthetic
`payload` body. Those omitted fields still count toward
`tool_response_bytes_total` and model-visible bytes.

```json
{
  "age_days": 45,
  "approvals": 0,
  "changed_files": 38,
  "failing_checks": 1,
  "id": "cand-0000",
  "is_bot_authored": false,
  "is_draft": false,
  "reactions": 8,
  "relevance": 0.4528,
  "title": "tests candidate 0"
}
```

On the third turn, Azure returned the final answer:

```json
{
  "task_id": "smoke_smoke_single_lookup",
  "candidates": [
    {
      "id": "cand-0000",
      "score": 0.4528
    }
  ]
}
```

The important part is visibility.

In this arm, the synthetic tool results are visible to the model. In the
example three-repetition smoke run, that was `567` model-visible
tool-response bytes per repetition.

### Code Mode and Monty arm

The Code Mode arm uses the same task and the same synthetic tools.

The current benchmark implementation uses a deterministic local Pydantic
`FunctionModel` for the model policy. That keeps the run reproducible and avoids
spending provider budget on this arm. The runtime being tested is still real
Pydantic Code Mode with Monty.

The arms do not yet use the same live model policy. Quality, latency, and cost
comparisons should be read as harness/runtime evidence, not as causal
live-model evidence for Code Mode.

The local model returns one `run_code` call:

```json
{
  "tool_name": "run_code",
  "arguments": {
    "restart": true,
    "code": "import asyncio\n\nshards = await asyncio.gather(...)\n..."
  }
}
```

The Python code runs inside Monty and calls the same tools:

```python
shards = await asyncio.gather(search_shard(shard_id=0))
candidate_ids = [item["id"] for shard in shards for item in shard]
fetched = await asyncio.gather(
    *[fetch_candidate(candidate_id=candidate_id) for candidate_id in candidate_ids]
)
```

Then the code filters, scores, sorts, and returns the final structured answer:

```json
{
  "task_id": "smoke_smoke_single_lookup",
  "candidates": [
    {
      "id": "cand-0000",
      "score": 0.38048
    }
  ]
}
```

Notice that the tool payloads were still fetched. They were not ignored. They
were just processed inside `run_code` instead of being sent back as model-visible
tool messages. The `run_code` return is still model-visible.

In the example three-repetition smoke run, this arm fetched the same `567`
tool-response bytes per repetition, but `0` of those bytes were model-visible
tool-response bytes.

### What the example smoke run showed

The bounded smoke run compared:

```bash
--preset smoke --arms direct_agent,code_mode_real --repetitions 3 --arm-order randomized
```

Both arms returned schema-valid answers and selected the same top candidate.
The smoke success criterion is ranking agreement, not score equality. The shown
scores are produced by different policies and should not be treated as
calibrated probabilities.

The direct Azure arm used `3` model requests per trial.

The Code Mode/Monty arm used `2` model requests per trial.

The direct Azure arm exposed tool results to the model.

The Code Mode/Monty arm kept tool results inside nested Code Mode metadata.

That shows the harness can route direct live tool calls and local
Code Mode/Monty execution while accounting for different payload visibility. It
does not isolate model-policy effects, and it is not a publishable benchmark
claim yet. For that, run more repetitions over larger fan-out workloads and
predeclare the scoring protocol.

## Run It On Your Machine

### Requirements

You need Python `3.11` or newer and `uv`.

Install the development dependencies:

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```

CI runs the same test command on Python 3.11, 3.12, and 3.13.

### Run a local synthetic smoke

Start with the local run. It does not use a live model key.

```bash
uv run python -m codemode_probe.cli \
  --preset smoke \
  --arms deterministic_oracle_client,in_process,direct_mcp,direct_agent \
  --repetitions 1 \
  --out benchmarks/outputs
```

This checks that the workload, tools, scoring, and artifact writer work on your
machine.

### Run the real Code Mode arm locally

Install the Code Mode extra:

```bash
uv sync --extra code-mode
```

Run the direct synthetic agent beside the real Pydantic Code Mode/Monty arm:

```bash
uv run --extra code-mode python -m codemode_probe.cli \
  --preset smoke \
  --arms direct_agent,code_mode_real \
  --repetitions 1 \
  --out benchmarks/outputs
```

This run validates the real Code Mode runtime path without using a live Azure
OpenAI model for the Code Mode arm.

### Prepare Azure OpenAI credentials

Install the provider extra:

```bash
uv sync --extra providers
```

Create a local environment file. This file is ignored by git.

```bash
cat > .env.local <<'EOF'
AZURE_OPENAI_API_KEY=YOUR_KEY
AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE_NAME.cognitiveservices.azure.com/openai/deployments/YOUR_AZURE_DEPLOYMENT_NAME/chat/completions?api-version=2025-01-01-preview"
EOF
```

Load it in your shell:

```bash
set -a
source .env.local
set +a
```

The endpoint can be either the Azure OpenAI resource endpoint or the full
deployment chat-completions URL from Azure AI Foundry. If you use the full URL,
the harness extracts the deployment name from the path. You still pass the
deployment name with `--provider-model`.

Set these helper values before copying the live commands:

```bash
export AZURE_OPENAI_DEPLOYMENT=YOUR_AZURE_DEPLOYMENT_NAME
export PROVIDER_SDK_VERSION=$(uv run --extra providers python -c 'import openai; print(openai.__version__)')
```

### Run a bounded Azure smoke

Use strict budget guards first.

The pricing source and token rates below are OpenAI public-pricing assumptions
used as a smoke-run budget guard. Replace them with Azure-backed pricing
evidence before treating `cost_estimates.json` as source-backed billing
evidence.

```bash
uv run --extra providers python -m codemode_probe.cli \
  --preset smoke \
  --arms direct_agent \
  --repetitions 1 \
  --provider azure_openai \
  --provider-model "$AZURE_OPENAI_DEPLOYMENT" \
  --provider-api-key-env-var AZURE_OPENAI_API_KEY \
  --provider-endpoint-env-var AZURE_OPENAI_ENDPOINT \
  --provider-model-version gpt-4.1-mini \
  --provider-api-version 2025-01-01-preview \
  --provider-sdk-version "$PROVIDER_SDK_VERSION" \
  --provider-pricing-source-id openai-gpt-4-1-mini-docs-2026-05-06 \
  --provider-model-docs-source-id openai-gpt-4-1-mini-docs-2026-05-06 \
  --provider-pricing-snapshot-date 2026-05-06 \
  --provider-currency USD \
  --enable-live \
  --max-model-requests 25 \
  --max-run-seconds 300 \
  --max-estimated-cost 1.00 \
  --budget-input-cost-per-1m 0.40 \
  --budget-output-cost-per-1m 1.60 \
  --budget-currency USD \
  --out benchmarks/outputs
```

### Run Azure direct beside local Code Mode

Install both optional extras:

```bash
uv sync --extra providers --extra code-mode
```

Then run the comparison:

```bash
uv run --extra providers --extra code-mode python -m codemode_probe.cli \
  --preset smoke \
  --arms direct_agent,code_mode_real \
  --repetitions 3 \
  --arm-order randomized \
  --provider azure_openai \
  --provider-model "$AZURE_OPENAI_DEPLOYMENT" \
  --provider-api-key-env-var AZURE_OPENAI_API_KEY \
  --provider-endpoint-env-var AZURE_OPENAI_ENDPOINT \
  --provider-model-version gpt-4.1-mini \
  --provider-api-version 2025-01-01-preview \
  --provider-sdk-version "$PROVIDER_SDK_VERSION" \
  --provider-pricing-source-id openai-gpt-4-1-mini-docs-2026-05-06 \
  --provider-model-docs-source-id openai-gpt-4-1-mini-docs-2026-05-06 \
  --provider-pricing-snapshot-date 2026-05-06 \
  --provider-currency USD \
  --enable-live \
  --max-model-requests 25 \
  --max-run-seconds 300 \
  --max-estimated-cost 1.00 \
  --budget-input-cost-per-1m 0.40 \
  --budget-output-cost-per-1m 1.60 \
  --budget-currency USD \
  --out benchmarks/outputs
```

The direct arm spends live Azure OpenAI budget.

The Code Mode arm uses the deterministic local model policy and the real
Pydantic Code Mode/Monty runtime.

This is not a like-for-like live-model comparison for the Code Mode arm. It is
a bounded comparison of a live direct model loop beside a local scripted
Code Mode/Monty runtime.

### Inspect the result

Each run creates a timestamped folder under `benchmarks/outputs`.

Open these files first:

- `report.md` for a readable summary
- `summary.json` for aggregate metrics
- `results.jsonl` for canonical per-arm result rows
- `transcripts.jsonl` for normalized model turns and tool activity
- `paired_deltas.json` for paired comparisons against the direct baseline
- `warnings.json` for claim and readiness caveats
- `cost_estimates.json` for measured-token cost estimates

The run folder contains these artifacts:

```text
manifest.json
tasks.resolved.json
prompts.resolved.json
results.jsonl
transcripts.jsonl
summary.json
paired_deltas.json
pairing_coverage.json
paired_delta_summary.json
paired_uncertainty.json
cache_cohorts.json
failure_modes.json
cost_estimates.json
preflight.json
warnings.json
workload_regimes.json
report.md
```

The main serialized tool-response byte suppression metric is:

```text
1 - model_visible_bytes_total / tool_response_bytes_total
```

If the value is close to `1`, most fetched tool payload stayed out of the model
context. If it is `0`, the fetched tool payload was fully model-visible.
If `tool_response_bytes_total` is `0`, the metric is not meaningful.

### Keep the claim narrow

A successful smoke run means the harness works and the arms can be compared.

It does not prove general Code Mode superiority.

For external claims, use the protocol in
[docs/benchmark_protocol.md](docs/benchmark_protocol.md), fill the source
register in [docs/evidence_register.md](docs/evidence_register.md), and use the
handoff checklist in
[docs/tomorrow_run_checklist.md](docs/tomorrow_run_checklist.md). Then run more
repetitions and sweep larger fan-out workloads.

### Example longer run results

This example run used one scalar fan-out task with `5` shards, `5` candidates
per shard, `top_k = 5`, and `3` repetitions.

The direct arm used live Azure OpenAI. The Code Mode arm used the local
deterministic model policy with real Pydantic Code Mode/Monty execution.

| Arm | Runs | Success | Mean top-k | Mean NDCG | P95 latency ms | Model requests | Tool calls | Model-visible tool bytes | Suppression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_mcp_agent_parallel` | 3 | 1.000 | 0.400 | 0.295 | 25122.239 | 17 | 90 | 42405 | 0.000 |
| `code_mode_pydantic_monty` | 3 | 1.000 | 1.000 | 1.000 | 271.053 | 6 | 90 | 0 | 1.000 |

Estimated cost rows for that run were:

| Arm | Input tokens | Output tokens | Estimated cost |
| --- | ---: | ---: | ---: |
| `direct_mcp_agent_parallel` | 71328 | 3496 | `$0.034125` |
| `code_mode_pydantic_monty` | 1029 | 1710 | `$0.003148` |

These cost rows use OpenAI public-pricing assumptions as a budget guard, not
verified Azure billing evidence.
