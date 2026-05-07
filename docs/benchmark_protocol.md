# Benchmark Protocol

## Scope

This protocol defines how to interpret runs produced by `codemode-probe`.
Current no-credential runs validate the deterministic synthetic harness and
orchestration accounting. They do not support claims that Code Mode, Monty, or
any provider model is generally faster, cheaper, or higher quality than direct
MCP with live models.

## Research Question

The benchmark probes the crossover point where code-driven orchestration over an
MCP-shaped tool surface may reduce model requests, model-visible payload bytes,
latency, or cost relative to direct model-driven tool calling.

## Arms

- `direct_mcp_agent_parallel`: direct model-driven MCP-shaped agent loop. Today
  this uses a scripted provider client unless a live provider is explicitly
  selected with `--provider ... --enable-live`.
- `direct_mcp_tool_oracle`: direct MCP tool-client path with deterministic
  oracle ranking.
- `in_process_tool_oracle`: in-process tool oracle used as a deterministic
  parity control.
- `code_mode_synthetic_scripted`: synthetic Code Mode-style scripted arm with
  hidden tool outputs. This is not a real Pydantic Code Mode/Monty runtime.
- `code_mode_pydantic_monty`: real Pydantic AI Harness `CodeMode()` arm backed
  by Monty. It uses a deterministic local model policy to issue `run_code`, so
  it validates runtime orchestration without making live model-quality claims.

The default paired baseline is `direct_mcp_agent_parallel`.

## Workloads

`smoke` is a correctness check only. `orchestration_matrix` exercises a small
synthetic fan-out/fan-in matrix across task family, tool shape, candidate count,
and payload size.

Exploratory runs should use randomized arm order, at least three repetitions,
and no skipped preflight. Publishable live claims require a fixed task set,
multiple workload seeds, provider and model versions recorded in the manifest,
and source-backed pricing/model documentation in `docs/evidence_register.md`.

## Pairing

Paired deltas are grouped by `(task_id, repetition, trial_id)`. Trial groups are
excluded from paired deltas when the baseline is missing or any arm has
duplicate rows in the trial group. `pairing_coverage.json` is the audit artifact
for skipped or unpaired comparisons.

## Metrics

Primary deterministic quality metrics are schema validity, top-k overlap, and
NDCG at k. Primary orchestration metrics are model requests, tool calls, latency,
tool response bytes, model-visible bytes, payload suppression, and cache token
fields when a provider reports them.

`paired_uncertainty.json` bootstraps paired delta means. These intervals are
descriptive for deterministic synthetic runs; they are not population-level
confidence intervals until the task set and live-model sampling protocol are
predeclared.

## Controls

Each run records arm order, random seed, paired baseline, cache policy labels,
timeout policy, git source metadata, environment metadata, and artifact hashes
in `manifest.json`.

Cache policy fields are cohort labels unless a live provider or Code Mode
adapter explicitly enforces provider-side cache state. Warmup rows should not be
used as headline measurements for warm-cache claims. Warmup repetitions are
valid only for `warm` and `cold_then_warm` policies, and every warm-cache
configuration must leave at least one measured warm repetition after warmup.
`warm` requires at least one warmup repetition. `cold` is limited to one
repetition until an adapter explicitly implements cache busting or isolation.

Budget controls are pre-run guards, not measured usage. When configured, they
are evaluated before live provider validation and before artifact creation.
Model-request budgets use the planned task/repetition/arm matrix and per-task
tool-call ceilings. Token and cost budgets use deterministic planning heuristics
and require explicit pricing metadata for cost caps.

`cost_estimates.json` is post-run accounting over measured token fields. Rows
must be marked `not_estimated` when provider pricing evidence, token price
rates, or token usage fields are missing. Cache token costs are not estimated
unless a future pricing schema records cache-specific rates.

## Exclusion Rules

Timeouts, schema failures, missing baselines, duplicate trial-arm groups, and
preflight failures must be reported. Do not drop failed rows from aggregate
artifacts unless an analysis document explicitly states the exclusion and points
to the corresponding failure-mode artifact.

## Claim Boundaries

Current synthetic runs support claims about harness correctness, deterministic
artifact generation, pairing behavior, real Code Mode/Monty runtime plumbing,
and payload visibility accounting. They do not support claims about live model quality,
production MCP transport overhead, provider cache behavior, or general Code Mode
superiority. Live
provider smoke runs validate SDK transport plumbing and artifact accounting, but
publishable live cost/performance claims require filled evidence-register
sources, repeated runs over a predeclared task/seed set, and an analysis
protocol.
