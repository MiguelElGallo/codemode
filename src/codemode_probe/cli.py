from __future__ import annotations

import argparse
from pathlib import Path

from codemode_probe.artifacts import create_run_dir, write_run_artifacts
from codemode_probe.executor_factory import available_executor_ids, build_executor
from codemode_probe.models import ProbeTask, TaskFamily, ToolShape
from codemode_probe.runner import BenchmarkRunner
from codemode_probe.workload import make_probe_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Code Mode probe slices.")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/outputs"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--run-id", type=str, default=None)
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
    args = parser.parse_args()

    task = _task_from_args(args)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    results = []
    for repetition in range(1, args.repetitions + 1):
        for arm in arms:
            executor = build_executor(arm, task)
            results.append(BenchmarkRunner(executor).run_task(task, repetition=repetition))

    run_dir = create_run_dir(args.out, run_id=args.run_id)
    write_run_artifacts(run_dir, [task], results)
    print(run_dir)


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


if __name__ == "__main__":
    main()
