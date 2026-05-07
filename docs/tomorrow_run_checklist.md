# Tomorrow Run Checklist

Use this checklist for a bounded live-provider smoke run. It is intended to
validate provider transport plumbing and artifact generation, not to make
publishable Code Mode superiority claims.

## 1. Verify The Repo

```bash
uv sync --extra dev --extra providers
uv run --extra dev pytest -q
```

Expected result: all tests pass.

## 2. Fill Evidence Fields

Before a live run, choose stable IDs from `docs/evidence_register.md` for:

- provider pricing source
- provider model documentation source

If those evidence rows are still `TBD`, keep the run as an internal smoke. Cost
artifact rows can still be generated, but they should not be treated as
source-backed external evidence.

## 3. Export Credentials

```bash
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-01-01-preview"
```

Do not put API keys in CLI arguments, config files, artifacts, or commit
history. The artifact writer redacts known secret shapes in transcripts, but the
safe path is to keep secrets only in environment variables.

## 4. Run A Bounded Smoke

Replace placeholders before running.
`--provider-model` is the Azure OpenAI deployment name, not the base model name.
The endpoint may be either the resource base URL or the full deployment
chat-completions URL from Azure AI Foundry.

```bash
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

## 5. Inspect Artifacts

Check the printed run directory:

- `manifest.json`: provider config, source metadata, budget config/estimate,
  artifact hashes
- `results.jsonl`: canonical result rows
- `transcripts.jsonl`: redacted model/tool transcript view
- `warnings.json`: readiness gaps and claim caveats
- `cost_estimates.json`: measured-token cost rows or explicit `not_estimated`
  rows
- `report.md`: presentation summary and warnings

## 6. Interpret Conservatively

A successful live smoke means the direct-agent provider transport and artifact
contracts worked for one bounded run. It does not prove model quality, provider
cost, provider cache behavior, production MCP overhead, or Code Mode superiority.
