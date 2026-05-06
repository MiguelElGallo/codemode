# Code Mode Probe

This repository contains a benchmark harness for probing when code-driven
orchestration over MCP becomes cheaper or faster than direct model-driven MCP
tool use.

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
- optional provider and Code Mode runtime dependency boundaries

Not yet implemented:

- real provider SDK adapter
- real Pydantic Code Mode/Monty execution arm
- cold/warm cache cohort instrumentation
- confidence intervals or bootstrap statistics

Until those pieces exist, results should be interpreted as deterministic
orchestration harness validation, not as a claim that Code Mode beats direct
MCP with live models.

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

The package requires Python 3.11 or newer. The CI workflow currently runs the
test suite on Python 3.13.

## Run A Smoke Benchmark

```bash
uv run python -m codemode_probe.cli \
  --preset smoke \
  --arms deterministic_oracle_client,in_process,direct_mcp,direct_agent \
  --repetitions 1 \
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
summary.json
paired_deltas.json
paired_delta_summary.json
workload_regimes.json
report.md
```

`results.jsonl` is the canonical machine-readable record. `summary.json`,
`paired_deltas.json`, `paired_delta_summary.json`, and `workload_regimes.json`
are derived from it. `report.md` is presentation-only.

`manifest.json` records the normalized arms, selected paired-delta baseline,
arm-order seed, retry/concurrency/cache policy labels, Python/platform metadata,
and optional integration package versions when installed.

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

Cold/warm cache state is not instrumented yet. Repetitions are repeated
measurements, not cache-warm trials.
