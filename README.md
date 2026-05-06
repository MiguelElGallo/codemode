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
- optional provider and Code Mode runtime dependency boundaries
- a benchmark protocol document and evidence register for future live claims

Not yet implemented:

- real provider SDK adapter
- real Pydantic Code Mode/Monty execution arm
- provider-enforced cold/warm cache behavior
- documented minimum sample-size protocol for publishable live-model claims

Until those pieces exist, results should be interpreted as deterministic
orchestration harness validation, not as a claim that Code Mode beats direct
MCP with live models.

See [docs/benchmark_protocol.md](docs/benchmark_protocol.md) for the formal
benchmark protocol and [docs/evidence_register.md](docs/evidence_register.md)
for the source register required before external cost/performance claims.

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
preflight.json
workload_regimes.json
report.md
```

`results.jsonl` is the canonical machine-readable record. `transcripts.jsonl`
is a normalized, redacted transcript view for inspecting model turns and tool
activity. `summary.json`, `paired_deltas.json`, `pairing_coverage.json`,
`paired_delta_summary.json`, `paired_uncertainty.json`, `cache_cohorts.json`,
`failure_modes.json`, `preflight.json`, and `workload_regimes.json` are derived
from the run results. `report.md` is presentation-only.

`manifest.json` records the normalized arms, selected paired-delta baseline,
arm-order seed, retry/concurrency/cache policy labels, claim scope,
Python/platform metadata, git source metadata, benchmark protocol version/module
hashes, protocol/evidence document hashes, optional integration package
versions when installed, and SHA-256 checksums for the emitted artifact files.

Optional provider settings can be recorded without credentials by passing
`--provider`, `--provider-model`, and `--provider-dry-run`. Without
`--provider-dry-run`, provider configs require `--enable-live` and pass SDK plus
API-key environment checks before any run artifacts are created.

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
real provider-side cache behavior.

`pairing_coverage.json` records how many trial groups had the configured
paired-delta baseline, how many comparison results were paired, and how many
were skipped because a baseline row was missing.

## Claims Not Supported Yet

This benchmark does not yet support claims about live model quality, live
provider cost, production MCP workloads, provider cache behavior, or general
Code Mode superiority. Those claims require real provider adapters, a real Code
Mode/Monty arm, filled source-register entries, and a predeclared sample-size
and analysis protocol.
