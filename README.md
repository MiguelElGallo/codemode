# Code Mode Probe

This repository contains a benchmark harness for probing orchestration cost,
latency, quality, and payload visibility under synthetic MCP-shaped workloads.

The implementation starts with a deterministic synthetic workload and oracle.
MCP and Code Mode adapters are intentionally layered on top of those contracts
so the benchmark can separate orchestration effects from runtime and transport
effects.

## Current Status

Implemented:

- deterministic synthetic workloads and oracle scoring
- in-process synthetic tools and a FastMCP adapter
- direct MCP tool-client normalization
- scripted direct-MCP agent loop
- provider-neutral model-client boundary
- suite orchestration with fixed or seeded randomized arm order
- smoke and orchestration-matrix presets
- JSONL artifacts, aggregate summaries, paired deltas, workload regimes, and Markdown reports
- run environment/control metadata, configurable paired baseline, trial provenance, and typed failure categories
- cache cohort labels and paired bootstrap uncertainty artifacts
- optional OpenAI, Azure OpenAI, and Anthropic provider transports plus Code Mode runtime dependency boundaries
- budget guards, readiness warnings, and measured-token cost estimate artifacts
- a benchmark protocol document and evidence register for future live claims

Not yet implemented:

- real Pydantic Code Mode/Monty execution arm
- provider-enforced cold/warm cache behavior
- documented minimum sample-size protocol for publishable live-model claims

Synthetic runs should be interpreted as deterministic orchestration harness
validation, not as a claim that Code Mode beats direct MCP with live models.
Live provider runs are possible for bounded smoke testing, but publishable claims
still require filled evidence-register entries, a predeclared sample-size
protocol, and a real Code Mode/Monty arm.

See [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for the formal
benchmark protocol and [docs/evidence_register.md](docs/evidence_register.md)
for the source register required before external cost/performance claims. See
[docs/tomorrow_run_checklist.md](docs/tomorrow_run_checklist.md) for a bounded
live-smoke handoff checklist.

## Setup

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```

Optional integrations are installed only when needed:

```bash
uv sync --extra providers
uv sync --extra code-mode
```

The package requires Python 3.11 or newer. The CI workflow runs the test suite
on Python 3.11, 3.12, and 3.13, then builds the package and checks optional
provider and Code Mode extras.

## Run A Smoke Benchmark

```bash
uv run python -m codemode_probe.cli \
  --preset smoke \
  --arms deterministic_oracle_client,in_process,direct_mcp,direct_agent \
  --repetitions 1 \
  --out benchmarks/outputs
```

## Run A Live Provider Smoke

Install provider extras, export the Azure OpenAI API key and endpoint, and use strict budget
guards. The Azure endpoint may be either the resource base URL or the full
deployment chat-completions URL from Azure AI Foundry. Fill the pricing and documentation source IDs from
[docs/evidence_register.md](docs/evidence_register.md) before treating cost rows
as source-backed. `--provider-model` is the Azure OpenAI deployment name.

```console
uv sync --extra providers
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-01-01-preview"
uv run --extra providers python -m codemode_probe.cli \
  --preset smoke \
  --arms direct_agent \
  --repetitions 1 \
  --provider azure_openai \
  --provider-model <azure-deployment-name> \
  --provider-api-key-env-var AZURE_OPENAI_API_KEY \
  --provider-endpoint-env-var AZURE_OPENAI_ENDPOINT \
  --provider-model-version gpt-4.1-mini \
  --provider-api-version 2025-01-01-preview \
  --provider-sdk-version <installed-openai-version> \
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

## Run The Orchestration Matrix

```bash
uv run python -m codemode_probe.cli \
  --preset orchestration_matrix \
  --arms direct_mcp_agent_parallel,direct_mcp_tool_oracle,in_process_tool_oracle \
  --repetitions 3 \
  --arm-order randomized \
  --random-seed 17 \
  --paired-baseline-arm direct_mcp_agent_parallel \
  --out benchmarks/outputs
```

## Artifact Layout

Each run writes a timestamped directory under `benchmarks/outputs` unless
`--run-id` is provided.

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

`results.jsonl` is the canonical machine-readable record. `transcripts.jsonl`
is a normalized, redacted transcript view for inspecting model turns and tool
activity. `summary.json`, `paired_deltas.json`, `pairing_coverage.json`,
`paired_delta_summary.json`, `paired_uncertainty.json`, `cache_cohorts.json`,
`failure_modes.json`, `cost_estimates.json`, `preflight.json`, `warnings.json`,
and `workload_regimes.json` are derived from the run results and run controls.
`cost_estimates.json` uses measured token fields only when source-backed pricing
metadata is present; otherwise rows are explicit `not_estimated` entries.
`report.md` is presentation-only.

`manifest.json` records the normalized arms, selected paired-delta baseline,
arm-order seed, retry/concurrency/cache policy labels, claim scope,
Python/platform metadata, git source metadata, benchmark protocol version/module
hashes, protocol/evidence document hashes, optional integration package
versions when installed, optional budget controls/estimates, and SHA-256
checksums for the emitted artifact files.

Optional provider settings can be recorded without credentials by passing
`--provider`, `--provider-model`, and `--provider-dry-run`. Without
`--provider-dry-run`, provider configs require `--enable-live` and pass SDK plus
API-key environment checks before any run artifacts are created. Live provider
turns use provider-native tool-calling transcripts: OpenAI Responses API
`function_call_output` items, Azure OpenAI chat-completions tool messages when
the endpoint is a deployment chat URL, and Anthropic Messages API `tool_result`
blocks.

Optional budget guards can be set with `--max-run-seconds`,
`--max-model-requests`, `--max-input-tokens`, `--max-output-tokens`, and
`--max-estimated-cost`. Budget checks run before live provider validation and
before artifact directories are created. Token and cost budgets use deterministic
planning heuristics; pass `--budget-input-cost-per-1m`,
`--budget-output-cost-per-1m`, and `--budget-currency` to source cost estimates.

## Interpreting Results

The most defensible comparisons are paired by `(task_id, repetition, trial_id)`.
The default paired-delta baseline is `direct_mcp_agent_parallel`; override it
with `--paired-baseline-arm`. Suite-generated result rows include `trial_id`,
`arm_order`, and `arm_order_index` so latency deltas can be audited against the
actual execution order.

Payload suppression is:

```text
1 - model_visible_bytes_total / tool_response_bytes_total
```

Cache cohorts are recorded in `results.jsonl`, `manifest.json`, and
`cache_cohorts.json`. Provider adapters are still responsible for enforcing any
real provider-side cache behavior. Cache warmup repetitions are only valid with
`warm` or `cold_then_warm` cache policies, and those configurations must leave
at least one measured warm repetition outside warmup rows. `warm` requires at
least one warmup repetition; `cold` is restricted to one repetition until cache
busting is implemented.

`pairing_coverage.json` records how many trial groups had the configured
paired-delta baseline, how many comparison results were paired, and how many
were skipped because a baseline row was missing.

## Claims Not Supported Yet

This benchmark does not yet support claims about live model quality, live
provider cost, production MCP workloads, provider cache behavior, or general
Code Mode superiority. Those claims require a real Code Mode/Monty arm, filled
source-register entries, live-run repetitions over a predeclared task/seed set,
and an analysis protocol.
