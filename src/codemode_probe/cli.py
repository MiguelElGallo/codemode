from __future__ import annotations

import argparse
from pathlib import Path

from codemode_probe.artifacts import create_run_dir, write_run_artifacts
from codemode_probe.cases import CaseMatrixConfig, generate_case_tasks
from codemode_probe.executor_factory import available_executor_ids
from codemode_probe.models import CachePolicy, ProbeTask, TaskFamily, ToolShape
from codemode_probe.preflight import run_preflight_checks
from codemode_probe.provider_config import LiveProvider, LiveProviderConfig
from codemode_probe.suite import BenchmarkSuiteConfig, run_benchmark_suite
from codemode_probe.workload import make_probe_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Code Mode probe slices.")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/outputs"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--preset",
        choices=["smoke", "orchestration_matrix"],
        default=None,
        help="Generate a named benchmark task preset instead of one manual task.",
    )
    parser.add_argument(
        "--arms",
        default="deterministic_oracle_client",
        help=f"Comma-separated executor ids. Available: {', '.join(available_executor_ids())}",
    )
    parser.add_argument("--task-id", default="synthetic_fanout_smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task-family", choices=[item.value for item in TaskFamily], default=TaskFamily.SCALAR_LARGE_FANOUT.value)
    parser.add_argument("--tool-shape", choices=[item.value for item in ToolShape], default=ToolShape.SCALAR.value)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--candidates-per-shard", type=int, default=20)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--relevant-fraction", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-tool-calls", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--arm-order", choices=["fixed", "randomized"], default="fixed")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument(
        "--paired-baseline-arm",
        default="direct_mcp_agent_parallel",
        help="Executor id used as the baseline in paired_deltas.json.",
    )
    parser.add_argument(
        "--cache-policy",
        choices=[item.value for item in CachePolicy],
        default=CachePolicy.UNSPECIFIED.value,
        help="Cache cohort label recorded in manifest and result rows.",
    )
    parser.add_argument(
        "--cache-namespace",
        default=None,
        help="Optional cache namespace label for grouping runs.",
    )
    parser.add_argument(
        "--cache-warmup-repetitions",
        type=int,
        default=0,
        help="Number of repetitions labeled as cache warmup before warm measurements.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip local synthetic preflight checks before running the benchmark.",
    )
    parser.add_argument(
        "--provider",
        choices=[item.value for item in LiveProvider],
        default=None,
        help="Optional live provider config to validate or record.",
    )
    parser.add_argument(
        "--provider-model",
        default=None,
        help="Model name for the optional live provider config.",
    )
    parser.add_argument(
        "--provider-api-key-env-var",
        default=None,
        help="Environment variable name that will hold the provider API key.",
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout recorded for the optional live provider config.",
    )
    parser.add_argument(
        "--provider-temperature",
        type=float,
        default=0.0,
        help="Temperature recorded for the optional live provider config.",
    )
    parser.add_argument(
        "--provider-dry-run",
        action="store_true",
        help="Record provider config without importing SDKs or checking credentials.",
    )
    parser.add_argument(
        "--enable-live",
        action="store_true",
        help="Allow live provider validation.",
    )
    args = parser.parse_args()

    provider_config = _provider_config_from_args(args)
    if provider_config is not None and not args.provider_dry_run:
        provider_config.validate_for_live_use()

    preflight_results = None if args.skip_preflight else run_preflight_checks()
    if preflight_results is not None and not all(result.passed for result in preflight_results):
        failed = ", ".join(result.name for result in preflight_results if not result.passed)
        raise RuntimeError(f"preflight checks failed: {failed}")

    tasks = _tasks_from_args(args)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    suite_config = BenchmarkSuiteConfig(
        arms=tuple(arms),
        repetitions=args.repetitions,
        arm_order=args.arm_order,
        random_seed=args.random_seed,
        paired_baseline_arm=args.paired_baseline_arm,
        cache_policy=CachePolicy(args.cache_policy),
        cache_namespace=args.cache_namespace,
        cache_warmup_repetitions=args.cache_warmup_repetitions,
    )
    results = run_benchmark_suite(tasks, suite_config)

    run_dir = create_run_dir(args.out, run_id=args.run_id)
    write_run_artifacts(
        run_dir,
        tasks,
        results,
        suite_config=suite_config,
        preflight_results=preflight_results,
        provider_config=provider_config,
    )
    print(run_dir)


def _tasks_from_args(args: argparse.Namespace) -> list[ProbeTask]:
    if args.preset is not None:
        return generate_case_tasks(
            CaseMatrixConfig(
                preset=args.preset,
                base_seed=args.seed,
                payload_bytes=args.payload_bytes,
                relevant_fraction=args.relevant_fraction,
            )
        )
    return [_task_from_args(args)]


def _task_from_args(args: argparse.Namespace) -> ProbeTask:
    task = make_probe_task(
        args.task_id,
        seed=args.seed,
        task_family=TaskFamily(args.task_family),
        tool_shape=ToolShape(args.tool_shape),
        shard_count=args.shards,
        candidates_per_shard=args.candidates_per_shard,
        payload_bytes=args.payload_bytes,
        relevant_fraction=args.relevant_fraction,
        top_k=args.top_k,
    )
    return task.model_copy(
        update={
            "max_tool_calls": args.max_tool_calls,
            "timeout_seconds": args.timeout_seconds,
        }
    )


def _provider_config_from_args(args: argparse.Namespace) -> LiveProviderConfig | None:
    if args.provider is None:
        return None
    provider = LiveProvider(args.provider)
    if provider == LiveProvider.OPENAI:
        default_model = "gpt-4.1-mini"
        default_env_var = "OPENAI_API_KEY"
    elif provider == LiveProvider.ANTHROPIC:
        default_model = "claude-sonnet-4-5"
        default_env_var = "ANTHROPIC_API_KEY"
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return LiveProviderConfig(
        provider=provider,
        model=args.provider_model or default_model,
        enabled=args.enable_live,
        api_key_env_var=args.provider_api_key_env_var or default_env_var,
        timeout_seconds=args.provider_timeout_seconds,
        temperature=args.provider_temperature,
    )


if __name__ == "__main__":
    main()
