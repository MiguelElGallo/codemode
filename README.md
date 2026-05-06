# Code Mode Probe

This repository contains a benchmark harness for probing when code-driven
orchestration over MCP becomes cheaper or faster than direct model-driven MCP
tool use.

The implementation starts with a deterministic synthetic workload and oracle.
MCP and Code Mode adapters are intentionally layered on top of those contracts
so the benchmark can separate orchestration effects from runtime and transport
effects.
